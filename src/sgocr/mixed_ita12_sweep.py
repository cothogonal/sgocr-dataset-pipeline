from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .dev200_eval import run_image_dependence_eval
from .mixed_ocr_frontend_canary import _q01_baseline, _run_variant
from .nemotron_frontend import nemotron_frontend_diagnostic
from .paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from .run_quality import compute_anchor_coverage, compute_answer_distribution

# ita12 goal:
#   Build the highest-probability finalization-quality dataset by combining the
#   ita08_typed_gate feature stack with NO paid frontier gate (gate disabled; inline
#   frontier eval still runs post-hoc for scoring), NO RG vdep check, and a new
#   RG candidate oversampling boost.
#
#   ita11 post-mortem:
#     Winner by sweep: ita11_iou40_rg_vdep_offset20 (sweep=15.62, 246 rows, 3 RG, 28.9% text-leaky)
#     Best quality w/ RG diversity: ita11_iou40_offset20 (sweep=15.27, Q3=57.50, 15 RG, 32.4% text-leaky)
#     Critical finding: RG vdep gate (fixed in ita11) correctly rejected ~80% of RG rows as
#     text-leaky, reducing RG from ~15 → 3 per variant. For a spatially-grounded dataset
#     this is a feature loss — the gate conflates "Gemini can't visually ground this" with
#     "this doesn't need vision."
#
#   Key design decisions for ita12:
#     1. NO paid frontier gate: SGOCR_INLINE_FRONTIER_GATE_ENABLED=0. The inline frontier
#        eval still runs (SGOCR_INLINE_FRONTIER_ENABLED=1, default) for post-hoc scoring
#        and sweep_score computation, but does NOT filter/reject candidates. This removes
#        the tautological self-grading inflation that pushed inline_mean to ~99%.
#        Hypothesis: more rows survive (especially RG), and sweep_score reflects genuine
#        quality rather than the gate's guaranteed pass rate.
#     2. NO RG vdep check: vdep gate over-rejects RG (~80% rejection); dropping it
#        preserves more spatially-grounded training signal.
#     3. RG candidate oversampling (rgboost variants): new tuning parameter
#        SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST adds a quality-score bonus to REVERSE_GROUND
#        candidates during per-image candidate selection, increasing the fraction of images
#        that include an RG candidate before QA generation. The boost (1.5) competes with
#        the novel-question-type bonus (1.1) and novel-text-node bonus (1.7) to tip
#        borderline selections in favor of RG diversity.
#     4. All variants: leaky_label_hard_reject ON, IoU=0.40 groundback, no vdep gate.
#
#   Variants:
#     ita12_full          — Baseline: ita08+ stack, no frontier gate, offset=0.15, no rgboost
#     ita12_offset20      — Same as full, but centroid_offset=0.20 for tighter spatial filter
#     ita12_rgboost       — full + RG oversampling (boost=1.5)
#     ita12_rgboost_offset20 — rgboost + centroid_offset=0.20 (max quality push)
#
#   Expected outcomes:
#     ita12_full: ~300+ rows, 20-30 RG (6-10%), Q4 target: 85+
#     ita12_offset20: ~280+ rows, slightly fewer RG, text-leaky < 30%
#     ita12_rgboost: ~280+ rows, 40+ RG (12-15%), Q4 target: 90+
#     ita12_rgboost_offset20: ~270+ rows, 35+ RG, combined quality push
#
#   Cache strategy:
#     All variants use cache_level="verified" from ita11_iou40_offset20 intermediate.
#     The upstream centroid filter is disabled in all variants (as in ita11), so
#     verified_tuples are identical across all ita11 iou40 variants. RG oversampling
#     and centroid_offset both operate post-verified_tuples (in select_candidates and
#     QA filtering), so the cache remains valid for all 4 variants.
#
#   Engineering changes:
#     - semantic_dev40_tuning.py: added rg_candidate_oversample_boost field +
#       SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST env-var
#     - dev40_complete.py: select_candidates() applies rg_candidate_oversample_boost
#       bonus to REVERSE_GROUND candidates during per-image greedy selection

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "166_sgocr_mixed_ita12_launch_2026-04-14.md"

_ITA11_IOU40_OFFSET20_INTERMEDIATE = (
    "/home/wdree/percy/vqafromscratch/data/ocr_spatial_qa/intermediate/mixed_dev150"
    "/sgocr_mixed_ita11_20260414_060042_ita11_iou40_offset20"
)

VARIANT_ORDER = [
    "ita12_full",
    "ita12_offset20",
    "ita12_rgboost",
    "ita12_rgboost_offset20",
]

VARIANT_DESCRIPTIONS = {
    "ita12_full":              "No frontier gate, IoU=0.40, offset=0.15, leaky_label, no vdep [base]",
    "ita12_offset20":          "No frontier gate, IoU=0.40, offset=0.20, leaky_label, no vdep [spatial push]",
    "ita12_rgboost":           "No frontier gate, IoU=0.40, offset=0.15, leaky_label, RG boost=1.5 [rg oversample]",
    "ita12_rgboost_offset20":  "No frontier gate, IoU=0.40, offset=0.20, leaky_label, RG boost=1.5 [rg+spatial push]",
}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="Run ita12: no frontier gate + RG oversampling, building on ita08+ feature stack."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita12_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita11_20260414_060042_report.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--qwen-batch-size", type=int, default=4)
    ap.add_argument("--qwen-gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--qwen-max-model-len", type=int, default=2048)
    ap.add_argument("--qwen-pass-count", type=int, default=1)
    ap.add_argument("--qwen-temperature", type=float, default=0.0)
    ap.add_argument("--qwen-consensus-iou", type=float, default=0.55)
    ap.add_argument("--qwen-min-support", type=int, default=1)
    ap.add_argument("--group-min-instances", type=int, default=3)
    ap.add_argument("--cheap-proxy-reject-score", type=int, default=6)
    ap.add_argument("--qwen-degenerate-threshold", type=float, default=0.92)
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.15)
    ap.add_argument("--spatial-min-centroid-offset-push", type=float, default=0.20)
    ap.add_argument("--rg-candidate-oversample-boost", type=float, default=1.5)
    ap.add_argument("--frontier-word-f1-floor", type=float, default=-1.0)
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview")
    ap.add_argument("--image-dependence-workers", type=int, default=4)
    ap.add_argument("--skip-image-dependence", action="store_true")
    return ap.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        image_id = str(row.get("image_id") or "")
        if "chartqa" in image_id.lower():
            counts["chartqa"] += 1
        elif "textocr" in image_id.lower():
            counts["textocr"] += 1
        elif "coco" in image_id.lower():
            counts["coco_text"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def _rg_leakage_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Diagnostics for RG text-leakage indicators."""
    rg_rows = [r for r in rows if str(r.get("question_type") or "") == "REVERSE_GROUND"]
    corrected = sum(1 for r in rg_rows if r.get("rg_correction_applied"))
    groundback_failed = sum(1 for r in rg_rows if r.get("groundback_failed"))
    color_tokens = frozenset({
        "red", "blue", "green", "brown", "white", "black", "gray", "grey",
        "yellow", "orange", "purple", "pink", "silver", "gold",
    })
    shape_tokens = frozenset({
        "rectangular", "circular", "square", "oval", "round",
        "triangular", "hexagonal", "cylindrical", "spherical",
        "wedge", "segment", "emblem", "badge", "bar",
    })
    leaky_label_rows = sum(
        1 for r in rg_rows
        if any(
            t in str(r.get("anchor_label") or "").lower().split()
            for t in (color_tokens | shape_tokens)
        )
    )
    return {
        "total_rg": len(rg_rows),
        "corrected_rg": corrected,
        "groundback_failed_rg": groundback_failed,
        "leaky_label_rg_candidates": leaky_label_rows,
    }


def _render_live_doc(payload: dict[str, Any]) -> str:
    perf = payload["performance_profile"]
    bundle_id = payload["bundle_id"]
    report_path = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150" / f"{bundle_id}_report.md"
    log_path = REPO_ROOT / "logs" / bundle_id / "run.log"
    results = payload["results"]

    lines = [
        "# SGOCR Mixed ITA12 Sweep",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{bundle_id}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Live report: [{bundle_id}_report.md]({report_path})",
        f"- Log: [run.log]({log_path})",
        "",
        "## Goal",
        "",
        "**No paid frontier gate + RG candidate oversampling, building on ita08+ feature stack.**",
        "",
        "ita11 post-mortem:",
        "- Winner by sweep: `ita11_iou40_rg_vdep_offset20` (sweep=15.62, 246 rows, 3 RG, 28.9% text-leaky).",
        "- Best with RG diversity: `ita11_iou40_offset20` (sweep=15.27, Q3=57.50, 15 RG, 32.4% text-leaky).",
        "- RG vdep gate (fixed in ita11) correctly rejected ~80% of RG rows, reducing from ~15 → 3.",
        "  For a spatially-grounded dataset this is a feature loss.",
        "",
        "Engineering changes:",
        "- `semantic_dev40_tuning.py`: added `rg_candidate_oversample_boost` field +",
        "  `SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST` env-var (default 0.0)",
        "- `dev40_complete.py`: `select_candidates()` applies `rg_candidate_oversample_boost`",
        "  bonus to REVERSE_GROUND candidates during per-image greedy selection",
        "",
        f"Cache: verified-level from ita11_iou40_offset20 intermediate. Workers: {perf['workers']}.",
        "",
        "## Engineering Changes",
        "",
        "| Location | Change |",
        "|---|---|",
        "| `semantic_dev40_tuning.py` | Added `rg_candidate_oversample_boost` + `SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST` env-var |",
        "| `dev40_complete.py` | `select_candidates()` adds `rg_candidate_oversample_boost` bonus to RG candidates |",
        "",
        "## Variant Summary",
        "",
        "| # | Variant | Accepted | Images | Inline mean | Sweep | DR | RG | YN | TP | AP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        status = result.get("status", "pending")
        if status == "pending":
            lines.append(f"| — | `{name}` | — | — | — | — | — | — | — | — | — |")
            continue
        if status in {"running", "image_dependence"}:
            lines.append(f"| ⏳ | `{name}` | running | — | — | — | — | — | — | — | — |")
            continue
        if status == "failed":
            lines.append(f"| ✗ | `{name}` | FAILED | — | — | — | — | — | — | — | — |")
            continue
        m = result.get("metrics") or {}
        qt = m.get("question_types") or {}
        lines.append(
            f"| ✓ | `{name}` |"
            f" {m.get('accepted_qas', 0)} |"
            f" {m.get('images_with_final_rows', 0)} |"
            f" {m.get('inline_frontier_mean', 0.0):.4f} |"
            f" {m.get('sweep_score', 0.0):.4f} |"
            f" {qt.get('DIRECT_READ', 0)} |"
            f" {qt.get('REVERSE_GROUND', 0)} |"
            f" {qt.get('YES_NO', 0)} |"
            f" {qt.get('TEXT_PROPERTY', 0)} |"
            f" {qt.get('ANCHOR_PROPERTY', 0)} |"
        )

    lines += [
        "",
        "## Image-Dependence Scores",
        "",
        "| Variant | Img acc | Text-only acc | Δ mean | Vision-nec % | Text-leaky % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        idep = result.get("image_dependence") or {}
        if not idep:
            lines.append(f"| `{name}` | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{name}` |"
            f" {idep.get('image_accuracy', 0.0):.4f} |"
            f" {idep.get('text_only_accuracy', 0.0):.4f} |"
            f" {idep.get('vision_delta_mean', 0.0):+.4f} |"
            f" {idep.get('vision_necessary_rate', 0.0)*100:.1f}% |"
            f" {idep.get('text_leaky_rate', 0.0)*100:.1f}% |"
        )

    lines += [
        "",
        "## RG Oversampling Diagnostics",
        "",
        "| Variant | RG boost | Accepted RG | RG % | Leaky-label RG |",
        "|---|---:|---:|---:|---:|",
    ]
    rg_boost_map = {
        "ita12_full": 0.0,
        "ita12_offset20": 0.0,
        "ita12_rgboost": 1.5,
        "ita12_rgboost_offset20": 1.5,
    }
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        m = result.get("metrics") or {}
        rg = result.get("rg_leakage_stats") or {}
        boost = rg_boost_map.get(name, 0.0)
        if not m:
            lines.append(f"| `{name}` | {boost} | — | — | — |")
            continue
        total_rg = rg.get("total_rg", 0)
        leaky_label = rg.get("leaky_label_rg_candidates", 0)
        total_accepted = m.get("accepted_qas", 0)
        rg_pct = (total_rg / total_accepted * 100) if total_accepted > 0 else 0.0
        lines.append(
            f"| `{name}` | {boost} | {total_rg} | {rg_pct:.1f}% | {leaky_label} |"
        )

    lines += [
        "",
        "## Centroid Offset Comparison",
        "",
        "| Variant | Offset | IoU | Frontier gate | Accepted | Yield% | Sweep | Vision-nec% | Text-leaky% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    offset_map = {
        "ita12_full": 0.15,
        "ita12_offset20": 0.20,
        "ita12_rgboost": 0.15,
        "ita12_rgboost_offset20": 0.20,
    }
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        m = result.get("metrics") or {}
        idep = result.get("image_dependence") or {}
        offset = offset_map.get(name, "?")
        accepted = m.get("accepted_qas", 0)
        yield_pct = m.get("qa_accept_rate", 0.0) * 100
        sweep = m.get("sweep_score", 0.0)
        vnec = idep.get("vision_necessary_rate", 0.0) * 100
        tleaky = idep.get("text_leaky_rate", 0.0) * 100
        if accepted > 0:
            lines.append(
                f"| `{name}` | {offset} | 0.40 | off | {accepted} | {yield_pct:.1f}% | {sweep:.4f} | {vnec:.1f}% | {tleaky:.1f}% |"
            )
        else:
            lines.append(f"| `{name}` | {offset} | 0.40 | off | — | — | — | — | — |")

    lines += [
        "",
        "## Answer Distribution & Anchor Coverage",
        "",
        "| Variant | YES rate | Top-1 conc. | Top-10 conc. | Ans entropy | Mean anchors/img | Multi-anchor img% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        adist = result.get("answer_distribution") or {}
        acov = result.get("anchor_coverage") or {}
        if not adist:
            lines.append(f"| `{name}` | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{name}` |"
            f" {adist.get('yesno_yes_rate', 0.0):.2f} |"
            f" {adist.get('top1_concentration', 0.0):.2f} |"
            f" {adist.get('top10_concentration', 0.0):.2f} |"
            f" {adist.get('answer_entropy', 0.0):.2f} |"
            f" {acov.get('mean_unique_anchors', 0.0):.2f} |"
            f" {acov.get('images_with_multiple_anchors_rate', 0.0)*100:.1f}% |"
        )

    lines += [
        "",
        "## Source Dataset Breakdown",
        "",
        "| Variant | ChartQA | TextOCR | COCOText | Other |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        sb = result.get("source_breakdown") or {}
        if not sb:
            lines.append(f"| `{name}` | — | — | — | — |")
            continue
        lines.append(
            f"| `{name}` |"
            f" {sb.get('chartqa', 0)} |"
            f" {sb.get('textocr', 0)} |"
            f" {sb.get('coco_text', 0)} |"
            f" {sb.get('other', 0)} |"
        )

    lines += [
        "",
        "## Per-Variant Analysis",
        "",
    ]
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        status = result.get("status", "pending")
        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"**Description:** {VARIANT_DESCRIPTIONS.get(name, '')}")
        lines.append("")
        if status not in {"ok", "cached"}:
            lines.append(f"Status: `{status}`")
            if result.get("note"):
                lines.append(f"Note: {result['note']}")
            lines.append("")
            continue
        m = result.get("metrics") or {}
        idep = result.get("image_dependence") or {}
        adist = result.get("answer_distribution") or {}
        acov = result.get("anchor_coverage") or {}
        sb = result.get("source_breakdown") or {}
        rg = result.get("rg_leakage_stats") or {}
        lines += [
            "| Metric | Value |",
            "|---|---|",
            f"| Accepted QAs | {m.get('accepted_qas', 0)} |",
            f"| Images with rows | {m.get('images_with_final_rows', 0)} |",
            f"| QA accept rate | {m.get('qa_accept_rate', 0.0):.3f} |",
            f"| Inline frontier mean | {m.get('inline_frontier_mean', 0.0):.4f} |",
            f"| Sweep score | {m.get('sweep_score', 0.0):.4f} |",
            f"| Q3 score | {m.get('q3_score', 0.0):.2f} |",
            f"| High ambiguity fraction | {m.get('high_ambiguity_fraction', 0.0):.3f} |",
            f"| YesNo neg | {m.get('yesno_negative', 0)} |",
            f"| Reverse local | {m.get('reverse_local', 0)} |",
            "",
        ]
        if rg:
            lines += [
                "**RG leakage diagnostics:**",
                "",
                f"- Total RG accepted: {rg.get('total_rg', 0)}",
                f"- Corrected (stripped + re-probed, passed): {rg.get('corrected_rg', 0)}",
                f"- Groundback failed: {rg.get('groundback_failed_rg', 0)}",
                f"- Leaky-label candidates (color/shape token in anchor_label): {rg.get('leaky_label_rg_candidates', 0)}",
                "",
            ]
        if idep:
            lines += [
                f"**Image-dependence ({idep.get('model', '?')}):**",
                "",
                "| Mode | All | DR | RG | YN | TP | AP |",
                "|---|---|---|---|---|---|---|",
            ]
            key_map = {
                "image_accuracy": "image_acc",
                "text_only_accuracy": "text_only_acc",
                "vision_delta": "vision_delta",
            }
            for mode_key, mode_label in [
                ("image_accuracy", "Image+Q"),
                ("text_only_accuracy", "Text-only"),
                ("vision_delta", "Δ"),
            ]:
                by_type = idep.get("by_type") or {}
                bt_key = key_map.get(mode_key, mode_key)
                row_parts = [f"| {mode_label} |", f" {idep.get(mode_key, 0.0):.4f} |"]
                for qt in ["DIRECT_READ", "REVERSE_GROUND", "YES_NO", "TEXT_PROPERTY", "ANCHOR_PROPERTY"]:
                    val = (by_type.get(qt) or {}).get(bt_key, 0.0)
                    row_parts.append(f" {val:.4f} |")
                lines.append("".join(row_parts))
            lines.append(f"| Vision-nec % | {idep.get('vision_necessary_rate', 0.0)*100:.1f}% | — | — | — | — | — |")
            lines.append(f"| Text-leaky % | {idep.get('text_leaky_rate', 0.0)*100:.1f}% | — | — | — | — | — |")
            lines.append("")
        if adist:
            lines += [
                "**Answer distribution:**",
                "",
                f"- YES/NO balance: {adist.get('yesno_yes_rate', 0.0)*100:.1f}% YES / {adist.get('yesno_no_rate', 0.0)*100:.1f}% NO",
                f"- Unique answers: {adist.get('unique_answers', 0)} across {adist.get('total_answers', 0)} rows",
                f"- Top-1 concentration: {adist.get('top1_concentration', 0.0)*100:.1f}%",
                f"- Top-10 concentration: {adist.get('top10_concentration', 0.0)*100:.1f}%",
                f"- Answer entropy: {adist.get('answer_entropy', 0.0):.3f}",
                "",
            ]
            top10 = adist.get("top10_answers") or []
            if top10:
                lines.append(f"Top answers: {', '.join(f'{chr(34)}{ans}{chr(34)} ({cnt})' for ans, cnt in top10[:5])}")
                lines.append("")
        if acov:
            lines += [
                "**Anchor coverage:**",
                "",
                f"- Mean unique anchors per image: {acov.get('mean_unique_anchors', 0.0):.2f}",
                f"- Median unique anchors per image: {acov.get('median_unique_anchors', 0.0):.1f}",
                f"- Images with ≥2 distinct anchors: {acov.get('images_with_multiple_anchors', 0)} ({acov.get('images_with_multiple_anchors_rate', 0.0)*100:.1f}%)",
                "",
            ]

    lines.append("---")
    lines.append(f"*Last updated: {payload['updated_at']}*")
    return "\n".join(lines) + "\n"


def _render_sweep_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Mixed ITA12 Sweep",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Source universe: `{payload['source_dir']}`",
        f"- Updated: `{payload['updated_at']}`",
        "",
        "## Variants",
        "",
        "| # | Variant | Description |",
        "|---|---|---|",
    ]
    for i, name in enumerate(VARIANT_ORDER, 1):
        lines.append(f"| {i} | `{name}` | {VARIANT_DESCRIPTIONS.get(name, '')} |")

    lines += [
        "",
        "## Results",
        "",
        "| Variant | Status | Accepted | Images | Inline mean | Sweep | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in VARIANT_ORDER:
        result = payload["results"].get(name) or {}
        status = result.get("status", "missing")
        if status not in {"ok", "cached"}:
            note = result.get("note") or ""
            process = result.get("process") or {}
            if not note and process:
                note = (process.get("stderr_tail") or process.get("stdout_tail") or "").replace("\n", " ")[:200]
            lines.append(f"| `{name}` | {status} | - | - | - | - | {note} |")
            continue
        metrics = result["metrics"]
        lines.append(
            f"| `{name}` | ok | {metrics['accepted_qas']} | {metrics['images_with_final_rows']} | "
            f"{metrics['inline_frontier_mean']:.4f} | {metrics['sweep_score']:.4f} | |"
        )
    return "\n".join(lines) + "\n"


def _write_all(*, payload: dict[str, Any], report_json: Path, report_md: Path, live_doc: Path) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(_render_sweep_report(payload), encoding="utf-8")
    live_doc.write_text(_render_live_doc(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150"
    source_dir = root_final / args.source_name
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))

    baseline_env, baseline_cli = _q01_baseline()

    _TYPED_GATE_QT = "DIRECT_READ,YES_NO,TEXT_PROPERTY,ANCHOR_PROPERTY"

    common_env = {
        **baseline_env,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
        "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "florence",
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "independent_raw",
        "SGOCR_QWEN_ANCHOR_INVENTORY_PASS_COUNT": str(int(args.qwen_pass_count)),
        "SGOCR_QWEN_ANCHOR_INVENTORY_TEMPERATURE": str(float(args.qwen_temperature)),
        "SGOCR_QWEN_ANCHOR_INVENTORY_CONSENSUS_IOU": str(float(args.qwen_consensus_iou)),
        "SGOCR_QWEN_ANCHOR_INVENTORY_MIN_SUPPORT": str(int(args.qwen_min_support)),
        "SGOCR_ANCHOR_RELABEL_MODE": "none",
        "SGOCR_QWEN_ANCHOR_MODEL": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": f"{float(args.qwen_gpu_mem_util):.2f}",
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": str(int(args.qwen_batch_size)),
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": str(int(args.qwen_max_model_len)),
        "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
        "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
        "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        # ita02 scene-aware flags
        "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
        "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
        "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
        "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "0",
        # ita03 structural fallback flags
        "SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD": str(float(args.qwen_degenerate_threshold)),
        # ita04 anti-OCR winner flag
        "SGOCR_QWEN_ANTI_OCR_PROMPT_ENABLED": "1",
        # Centroid offset (base) and mechanical color
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
        # ita07 quality bundle base flags
        "SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED": "1",
        # ita08 typed gate: bypass RG from frontier gate, but note we DON'T enable the gate itself
        # (SGOCR_INLINE_FRONTIER_GATE_ENABLED is intentionally NOT set — no paid gate in ita12).
        # The typed gate question-types list is still set so it takes effect if the gate is enabled
        # per-variant (it is not in ita12 base).
        "SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": _TYPED_GATE_QT,
        # ita09/ita10: leaky_label_hard_reject ON
        "SGOCR_RG_LEAKY_LABEL_HARD_REJECT_ENABLED": "1",
    }

    # Tighter centroid offset for push variants
    offset20_flags = {
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset_push)),
    }

    # RG candidate oversampling boost
    rg_boost_flags = {
        "SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": str(float(args.rg_candidate_oversample_boost)),
    }

    _cache = Path(_ITA11_IOU40_OFFSET20_INTERMEDIATE)

    variants: dict[str, dict[str, Any]] = {
        "ita12_full": {
            "env": {
                **common_env,
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
            },
            "note": VARIANT_DESCRIPTIONS["ita12_full"],
            "cache_dir": _cache,
            "cache_level": "verified",
        },
        "ita12_offset20": {
            "env": {
                **common_env,
                **offset20_flags,
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
            },
            "note": VARIANT_DESCRIPTIONS["ita12_offset20"],
            "cache_dir": _cache,
            "cache_level": "verified",
        },
        "ita12_rgboost": {
            "env": {
                **common_env,
                **rg_boost_flags,
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
            },
            "note": VARIANT_DESCRIPTIONS["ita12_rgboost"],
            "cache_dir": _cache,
            "cache_level": "verified",
        },
        "ita12_rgboost_offset20": {
            "env": {
                **common_env,
                **rg_boost_flags,
                **offset20_flags,
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
            },
            "note": VARIANT_DESCRIPTIONS["ita12_rgboost_offset20"],
            "cache_dir": _cache,
            "cache_level": "verified",
        },
    }

    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    live_doc = LIVE_DOC_DIR / LIVE_DOC_NAME

    now = datetime.now().isoformat(timespec="seconds")
    existing_payload = _load_json(report_json)
    if existing_payload:
        payload = existing_payload
        payload["updated_at"] = now
        for name, spec in variants.items():
            if name not in payload["results"]:
                payload["results"][name] = {"status": "pending", "note": spec["note"]}
        completed = [n for n, r in payload["results"].items() if r.get("status") in {"ok", "cached"}]
        if completed:
            print(f"[resume] skipping completed variants: {completed}", flush=True)
    else:
        payload = {
            "bundle_id": args.bundle_id,
            "source_dir": str(source_dir),
            "source_manifest": source_manifest,
            "reference_report_json": str(root_final / args.reference_report_json),
            "reference_report": _load_json(root_final / args.reference_report_json),
            "created_at": now,
            "updated_at": now,
            "performance_profile": {
                "workers": int(args.workers),
                "qwen_batch_size": int(args.qwen_batch_size),
                "qwen_gpu_mem_util": float(args.qwen_gpu_mem_util),
                "qwen_max_model_len": int(args.qwen_max_model_len),
                "qwen_pass_count": int(args.qwen_pass_count),
                "qwen_temperature": float(args.qwen_temperature),
                "qwen_consensus_iou": float(args.qwen_consensus_iou),
                "qwen_min_support": int(args.qwen_min_support),
                "group_min_instances": int(args.group_min_instances),
                "cheap_proxy_reject_score": int(args.cheap_proxy_reject_score),
                "qwen_degenerate_threshold": float(args.qwen_degenerate_threshold),
                "spatial_min_centroid_offset": float(args.spatial_min_centroid_offset),
                "spatial_min_centroid_offset_push": float(args.spatial_min_centroid_offset_push),
                "rg_candidate_oversample_boost": float(args.rg_candidate_oversample_boost),
                "frontier_word_f1_floor": float(args.frontier_word_f1_floor),
                "image_dependence_model": str(args.image_dependence_model),
            },
            "nemotron_diagnostic": nemotron_frontend_diagnostic(),
            "results": {name: {"status": "pending", "note": spec["note"]} for name, spec in variants.items()},
        }
    _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

    for name, spec in variants.items():
        if payload["results"].get(name, {}).get("status") in {"ok", "cached"}:
            print(f"[skip] {name} already completed — skipping", flush=True)
            continue
        payload["results"][name] = {"status": "running", "note": spec["note"]}
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

        out_dir = root_final / f"{args.bundle_id}_{name}"
        intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}"

        variant_cache_dir = spec.get("cache_dir")
        variant_cache_level = spec.get("cache_level", "none")
        if variant_cache_dir and not variant_cache_dir.exists():
            print(
                f"[warn] cache dir missing for {name}: {variant_cache_dir} — falling back to cache_level=none",
                flush=True,
            )
            variant_cache_dir = None
            variant_cache_level = "none"

        result = _run_variant(
            source_dir=source_dir,
            out_dir=out_dir,
            intermediate_dir=intermediate_dir,
            cache_intermediate_dir=variant_cache_dir,
            cache_level=variant_cache_level,
            model=str(args.model),
            workers=int(args.workers),
            max_side=int(args.max_side),
            device=str(args.device),
            env_overrides=spec["env"],
            cli_overrides=dict(baseline_cli),
        )
        result["out_dir"] = str(out_dir)
        result["intermediate_dir"] = str(intermediate_dir)
        result.setdefault("note", spec["note"])

        if result.get("status") in {"ok", "cached"}:
            rows = _load_jsonl(out_dir / "ocr_qa_dataset.jsonl")
            result["source_breakdown"] = _source_breakdown(rows)
            result["answer_distribution"] = compute_answer_distribution(rows)
            result["anchor_coverage"] = compute_anchor_coverage(rows)
            result["rg_leakage_stats"] = _rg_leakage_stats(rows)

            if not args.skip_image_dependence and rows:
                payload["results"][name] = {**result, "status": "image_dependence"}
                _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)
                try:
                    idep_dir = out_dir / "evals" / "image_dependence_ita12"
                    idep_summary = run_image_dependence_eval(
                        experiment_dir=out_dir,
                        model=str(args.image_dependence_model),
                        out_dir=idep_dir,
                        workers=int(args.image_dependence_workers),
                    )
                    result["image_dependence"] = idep_summary
                except Exception as exc:
                    result["image_dependence_error"] = str(exc)

        payload["results"][name] = result
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

    print(str(live_doc))


if __name__ == "__main__":
    main()
