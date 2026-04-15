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

# ita11 goal:
#   Fix the broken RG vision-dependence gate and probe whether tighter spatial filtering
#   can push vision-nec above 55%.
#
#   ita10 post-mortem:
#     Winner: ita10_iou40_leaky (IoU=0.40 + leaky_label_hard_reject):
#       sweep=15.7511, vision-nec=52.3%, text-leaky=33.7%, 258 rows (39.4% yield).
#     Runner-up: ita10_iou45_leaky (IoU=0.45):
#       sweep=15.2381, vision-nec=54.5%, text-leaky=31.7%, 246 rows (37.8% yield).
#     All four variants converged tightly: sweep 15.24–15.75, vision-nec 49–55%.
#
#     RG text-leakage problem:
#       rg_vdep_check was enabled in ita08 but silently non-functional: the default model
#       was "openai:gpt-5.3-codex" (non-existent), causing all API calls to error and
#       conservatively keep every RG row. In ita09/ita10 the check was dropped entirely.
#       In the ita10 winner, RG questions had 78.6% text-only accuracy — nearly 4 in 5
#       are answerable without the image. The leaky_label_hard_reject filter (structural,
#       based on color/shape tokens in anchor_label) stopped 0 rows in ita10.
#
#     Engineering fix in ita11:
#       semantic_dev40_tuning.py: default rg_vdep_model changed from
#       "openai:gpt-5.3-codex" → "gemini:gemini-3-flash-preview" so the gate can
#       actually fire when SGOCR_RG_VDEP_CHECK_ENABLED=1.
#
#     Spatial filter opportunity:
#       spatial_min_centroid_offset=0.15 is applied post-verified_tuples (during QA
#       generation), so changing it uses cache_level="verified" — cheap to test.
#       Hypothesis: offset=0.20 will filter out more center-adjacent anchors whose
#       location descriptions are trivially text-inferable, reducing text-leaky rate
#       below 30% and raising vision-nec toward 55–57%.
#
#   Variants (cement × 2, push × 2):
#     ita11_iou40_rg_vdep_cement  — IoU=0.40 + leaky + rg_vdep_check [cement]
#       Apply fixed vdep gate to ita10 winner. Expect small yield drop (RG rows filter),
#       modest vision-nec lift from eliminating the text-answerable RG questions.
#     ita11_iou45_rg_vdep_cement  — IoU=0.45 + leaky + rg_vdep_check [cement]
#       Apply fixed vdep gate to ita10 runner-up (highest vision-nec). Cross-validate
#       that the gate behaves consistently across IoU thresholds.
#     ita11_iou40_offset20        — IoU=0.40 + leaky + centroid_offset=0.20 [push]
#       Tighter spatial filter without vdep overhead. Hypothesis: removes near-center
#       anchors that inflate text-leaky rate, trading ~5% yield for +2–3pp vision-nec.
#     ita11_iou40_rg_vdep_offset20 — IoU=0.40 + leaky + rg_vdep + centroid_offset=0.20 [push]
#       Combined: spatial tightening + RG gate. Maximum quality push variant.
#
#   Key comparisons:
#     ita11_iou40_rg_vdep_cement vs ita10_iou40_leaky   → effect of fixed RG gate alone
#     ita11_iou45_rg_vdep_cement vs ita10_iou45_leaky   → same, at stricter IoU
#     ita11_iou40_offset20 vs ita10_iou40_leaky          → effect of spatial tightening
#     ita11_iou40_rg_vdep_offset20 vs ita11_iou40_offset20 → marginal effect of adding gate
#
#   Cache strategy:
#     spatial_min_centroid_offset is applied post-verified_tuples (in dev40_complete.py
#     requires_specific_location / select_candidates), so ALL variants can reuse
#     verified_tuples from the corresponding ita10 intermediates → cache_level="verified".
#     Variants with IoU=0.40 cache from ita10_iou40_leaky intermediate.
#     Variants with IoU=0.45 cache from ita10_iou45_leaky intermediate.

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "164_sgocr_mixed_ita11_launch_2026-04-14.md"

_ITA10_IOU40_INTERMEDIATE = (
    "/home/wdree/percy/vqafromscratch/data/ocr_spatial_qa/intermediate/mixed_dev150"
    "/sgocr_mixed_ita10_20260414_033941_ita10_iou40_leaky"
)
_ITA10_IOU45_INTERMEDIATE = (
    "/home/wdree/percy/vqafromscratch/data/ocr_spatial_qa/intermediate/mixed_dev150"
    "/sgocr_mixed_ita10_20260414_033941_ita10_iou45_leaky"
)

VARIANT_ORDER = [
    "ita11_iou40_rg_vdep_cement",
    "ita11_iou45_rg_vdep_cement",
    "ita11_iou40_offset20",
    "ita11_iou40_rg_vdep_offset20",
]

VARIANT_DESCRIPTIONS = {
    "ita11_iou40_rg_vdep_cement":    "IoU=0.40 + leaky_label + rg_vdep_check(gemini) [cement]",
    "ita11_iou45_rg_vdep_cement":    "IoU=0.45 + leaky_label + rg_vdep_check(gemini) [cement]",
    "ita11_iou40_offset20":          "IoU=0.40 + leaky_label + centroid_offset=0.20 [push]",
    "ita11_iou40_rg_vdep_offset20":  "IoU=0.40 + leaky_label + rg_vdep_check(gemini) + centroid_offset=0.20 [push]",
}

# All ita11 variants use cache_level="verified" (spatial_min_centroid_offset and
# rg_vdep_check both operate post-verified_tuples). Per-variant cache dirs are set
# in the variants dict and applied in the main loop.
_QWEN_MODIFYING: frozenset[str] = frozenset()  # no variants change Qwen/anchor params


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="Run ita11: fix RG vdep gate + spatial centroid tightening on IoU=0.40 baseline."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita11_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita10_20260414_033941_report.json")
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
    ap.add_argument("--frontier-word-f1-floor", type=float, default=-1.0)
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview")
    ap.add_argument("--image-dependence-workers", type=int, default=4)
    ap.add_argument("--rg-vdep-model", default="gemini:gemini-3-flash-preview",
                    help="Model for RG text-only vision-dependence probe.")
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
        "# SGOCR Mixed ITA11 Sweep",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{bundle_id}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Live report: [{bundle_id}_report.md]({report_path})",
        f"- Log: [run.log]({log_path})",
        "",
        "## Goal",
        "",
        "**Fix RG vdep gate (broken since ita08) and probe spatial centroid tightening on IoU=0.40 baseline.**",
        "",
        "ita10 post-mortem:",
        "- Winner: `ita10_iou40_leaky` (IoU=0.40 + leaky_label): sweep 15.7511, vision-nec 52.3%,",
        "  text-leaky 33.7%, 258 rows (39.4% yield).",
        "- Runner-up: `ita10_iou45_leaky` (IoU=0.45): sweep 15.2381, vision-nec 54.5%,",
        "  text-leaky 31.7%, 246 rows (37.8% yield).",
        "- RG text-leakage: winner had 78.6% text-only accuracy for RG questions — gate was",
        "  non-functional (model default 'openai:gpt-5.3-codex' doesn't exist, all API calls",
        "  errored → conservative keep).",
        "",
        "Engineering changes:",
        "- `semantic_dev40_tuning.py`: `rg_vdep_model` default → `gemini:gemini-3-flash-preview`",
        "  (was `openai:gpt-5.3-codex`). Gate now fires when `SGOCR_RG_VDEP_CHECK_ENABLED=1`.",
        "",
        f"Cache: verified-level from ita10 winner/runner-up intermediates. Workers: {perf['workers']}.",
        "",
        "## Engineering Changes",
        "",
        "| Location | Change |",
        "|---|---|",
        "| `semantic_dev40_tuning.py` | `rg_vdep_model` default: `openai:gpt-5.3-codex` → `gemini:gemini-3-flash-preview` |",
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
        "## RG Vdep Gate Diagnostics",
        "",
        "| Variant | RG input | RG accepted | RG rejected | RG leaky-label | Gate model |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        m = result.get("metrics") or {}
        rg = result.get("rg_leakage_stats") or {}
        vdep = result.get("rg_vdep_gate_stats") or {}
        if not rg and not m:
            lines.append(f"| `{name}` | — | — | — | — | — |")
            continue
        total_rg = rg.get("total_rg", 0)
        leaky_label = rg.get("leaky_label_rg_candidates", 0)
        accepted_rg = (result.get("metrics") or {}).get("question_types", {}).get("REVERSE_GROUND", 0)
        rejected_rg = vdep.get("rejected_rg", total_rg - accepted_rg) if vdep else "?"
        gate_model = vdep.get("model", "none")
        lines.append(
            f"| `{name}` |"
            f" {total_rg + (vdep.get('rejected_rg', 0) if vdep else 0)} |"
            f" {total_rg} |"
            f" {vdep.get('rejected_rg', '?') if vdep else '—'} |"
            f" {leaky_label} |"
            f" {gate_model} |"
        )

    lines += [
        "",
        "## Centroid Offset Comparison",
        "",
        "| Variant | Offset | IoU | Accepted | Yield% | Sweep | Vision-nec% | Text-leaky% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    offset_map = {
        "ita11_iou40_rg_vdep_cement": 0.15,
        "ita11_iou45_rg_vdep_cement": 0.15,
        "ita11_iou40_offset20": 0.20,
        "ita11_iou40_rg_vdep_offset20": 0.20,
    }
    iou_map = {
        "ita11_iou40_rg_vdep_cement": 0.40,
        "ita11_iou45_rg_vdep_cement": 0.45,
        "ita11_iou40_offset20": 0.40,
        "ita11_iou40_rg_vdep_offset20": 0.40,
    }
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        m = result.get("metrics") or {}
        idep = result.get("image_dependence") or {}
        offset = offset_map.get(name, "?")
        iou = iou_map.get(name, "?")
        accepted = m.get("accepted_qas", 0)
        yield_pct = m.get("qa_accept_rate", 0.0) * 100
        sweep = m.get("sweep_score", 0.0)
        vnec = idep.get("vision_necessary_rate", 0.0) * 100
        tleaky = idep.get("text_leaky_rate", 0.0) * 100
        if accepted > 0:
            lines.append(
                f"| `{name}` | {offset} | {iou} | {accepted} | {yield_pct:.1f}% | {sweep:.4f} | {vnec:.1f}% | {tleaky:.1f}% |"
            )
        else:
            lines.append(f"| `{name}` | {offset} | {iou} | — | — | — | — | — |")

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

    lines += ["", "## Per-Variant Analysis", ""]
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
        vdep = result.get("rg_vdep_gate_stats") or {}
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
        if vdep:
            lines += [
                "**RG vdep gate (post-accept diagnostics):**",
                "",
                f"- Gate model: {vdep.get('model', '?')}",
                f"- Total RG probed: {vdep.get('total_rg', 0)}",
                f"- Accepted (not text-leaky): {vdep.get('accepted_rg', 0)}",
                f"- Rejected (text-leaky): {vdep.get('rejected_rg', 0)}",
                f"- Text-leaky RG rate: {vdep.get('text_leaky_rg_rate', 0.0)*100:.1f}%",
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
        "# Mixed ITA11 Sweep",
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
        # ita02_sceneaware flags
        "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
        "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
        "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
        "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "0",
        # ita03_structural_fallback flags
        "SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD": str(float(args.qwen_degenerate_threshold)),
        # ita04_anti_ocr winner flag
        "SGOCR_QWEN_ANTI_OCR_PROMPT_ENABLED": "1",
        # Centroid offset (base) and mechanical color
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
        # ita07_quality_bundle base flags
        "SGOCR_INLINE_FRONTIER_GATE_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED": "1",
        # ita08: typed gate — bypass RG from frontier gate
        "SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": _TYPED_GATE_QT,
        # ita09/ita10: leaky_label_hard_reject ON
        "SGOCR_RG_LEAKY_LABEL_HARD_REJECT_ENABLED": "1",
    }

    # RG vdep gate flags (ita11 fix: model now resolvable)
    rg_vdep_flags = {
        "SGOCR_RG_VDEP_CHECK_ENABLED": "1",
        "SGOCR_RG_VDEP_MODEL": str(args.rg_vdep_model),
    }

    # Tighter centroid offset for push variants
    offset20_flags = {
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset_push)),
    }

    _iou40_cache = Path(_ITA10_IOU40_INTERMEDIATE)
    _iou45_cache = Path(_ITA10_IOU45_INTERMEDIATE)

    variants: dict[str, dict[str, Any]] = {
        "ita11_iou40_rg_vdep_cement": {
            "env": {
                **common_env,
                **rg_vdep_flags,
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
            },
            "note": VARIANT_DESCRIPTIONS["ita11_iou40_rg_vdep_cement"],
            "cache_dir": _iou40_cache,
            "cache_level": "verified",
        },
        "ita11_iou45_rg_vdep_cement": {
            "env": {
                **common_env,
                **rg_vdep_flags,
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.45",
            },
            "note": VARIANT_DESCRIPTIONS["ita11_iou45_rg_vdep_cement"],
            "cache_dir": _iou45_cache,
            "cache_level": "verified",
        },
        "ita11_iou40_offset20": {
            "env": {
                **common_env,
                **offset20_flags,
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
            },
            "note": VARIANT_DESCRIPTIONS["ita11_iou40_offset20"],
            "cache_dir": _iou40_cache,
            "cache_level": "verified",
        },
        "ita11_iou40_rg_vdep_offset20": {
            "env": {
                **common_env,
                **rg_vdep_flags,
                **offset20_flags,
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
                "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
            },
            "note": VARIANT_DESCRIPTIONS["ita11_iou40_rg_vdep_offset20"],
            "cache_dir": _iou40_cache,
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
                "rg_vdep_model": str(args.rg_vdep_model),
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

        # Per-variant cache: all use "verified" level (spatial_min_centroid_offset and
        # rg_vdep_check both operate post-verified_tuples, so the anchor/grounding/
        # verification stages can be fully reused from the ita10 intermediates).
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
                    idep_dir = out_dir / "evals" / "image_dependence_ita11"
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
