from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .dev200_eval import run_image_dependence_eval
from .mixed_ocr_frontend_canary import _q01_baseline, _run_variant
from ..nemotron_frontend import nemotron_frontend_diagnostic
from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from ..run_quality import compute_anchor_coverage, compute_answer_distribution

# ita08 goal:
#   Test two targeted RG text-leakage fixes on top of the ita07_quality_bundle base.
#   The ita07 post-mortem identified two root causes of residual RG leakage:
#
#   Problem 1 — Frontier gate self-selection bias:
#     The gate passes RG rows where Gemini answers correctly with image. But those are
#     exactly the rows Gemini also gets right text-only (it knows where common chart
#     elements are from pre-training). Solution: typed frontier gate — apply gate only
#     to DR/YN/TP/AP; bypass for RG. Then probe RG with a cross-model text-only check.
#
#   Problem 2 — Anchor label leakage:
#     RG questions with color+shape tokens in the anchor label ("blue rectangular bar",
#     "red circular badge") expose the element's identity in the question text.
#     Solution A: strip color/shape tokens from rejected RG questions and re-probe
#     (leakage_correction). Solution B: check if the anchor label uniquely re-localizes
#     the element via Qwen IoU probe (anchor_label_groundback).
#
#   Fixes introduced in this sweep:
#     1. Typed frontier gate: SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES=DR,YN,TP,AP
#        (bypasses RG from frontier gate — removes self-selection bias)
#     2. RG vdep check: single GPT-5.3-codex text-only call per RG row
#        (cross-model adversarial probe for text leakage)
#     3. RG leakage correction: strip color/shape tokens, re-probe once before discarding
#     4. Anchor label groundback: re-run Qwen with label text only, check bbox IoU ≥ 0.35
#        (penalty-only: 0.20 score reduction for ambiguous anchors; does not hard-reject)
#
#   Cache strategy:
#     All variants share ita07_quality_bundle intermediate data (OCR + Qwen + verified tuples).
#     cache_level="verified" re-runs question generation, gate, and filtering stages only.
#     Groundback check re-uses the already-loaded Qwen GPU instance.
#
#   Variants (2×2 factorial on leakage_correction × groundback):
#     ita08_typed_gate      — typed gate + RG vdep only                 (baseline for ita08)
#     ita08_correction      — typed gate + RG vdep + leakage correction
#     ita08_groundback      — typed gate + RG vdep + anchor groundback
#     ita08_full            — typed gate + RG vdep + correction + groundback (max quality)
#
#   Hypotheses:
#     ita08_typed_gate:   RG text_only_acc drops vs ita07 (gate no longer self-selects easy RG).
#                         RG count may rise slightly (no frontier rejection on RG).
#     ita08_correction:   Leaky RG questions with color/shape tokens → corrected/retained.
#                         Expect RG yield to rise while text_leaky_rate drops further.
#     ita08_groundback:   Ambiguous/generic anchor labels receive 0.20 score penalty.
#                         Sweep_score may drop slightly vs typed_gate but quality improves.
#     ita08_full:         Maximum quality. Test if correction + groundback are additive.
#
#   Key comparison axes:
#     ita08_typed_gate vs ita07_quality_bundle → impact of removing RG from frontier gate
#     ita08_correction vs ita08_typed_gate     → value of leakage correction
#     ita08_groundback vs ita08_typed_gate     → value of groundback penalty
#     ita08_full vs ita08_correction/groundback → additive value of combining both

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "161_sgocr_mixed_ita08_launch_2026-04-13.md"

_ITA07_QUALITY_BUNDLE_INTERMEDIATE = (
    "/home/wdree/percy/vqafromscratch/data/ocr_spatial_qa/intermediate/mixed_dev150"
    "/sgocr_mixed_ita07_20260413_091119_ita07_quality_bundle"
)

VARIANT_ORDER = [
    "ita08_typed_gate",
    "ita08_correction",
    "ita08_groundback",
    "ita08_full",
]

VARIANT_DESCRIPTIONS = {
    "ita08_typed_gate":  "typed frontier gate + RG vdep check only (ita08 baseline)",
    "ita08_correction":  "typed gate + RG vdep + leakage correction (strip color/shape, re-probe)",
    "ita08_groundback":  "typed gate + RG vdep + anchor label groundback penalty (IoU < 0.35 → −0.20)",
    "ita08_full":        "typed gate + RG vdep + correction + groundback (maximum quality stack)",
}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="Run ita08: typed frontier gate + RG vdep check + leakage correction + groundback."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita08_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita07_report.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=6,
                    help="Concurrent Gemini workers.")
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
    ap.add_argument("--frontier-word-f1-floor", type=float, default=-1.0)
    ap.add_argument("--rg-vdep-model", default="openai:gpt-5.3-codex",
                    help="Model for RG text-only vision-dependence probe (format: provider:model_name).")
    ap.add_argument("--rg-vdep-workers", type=int, default=4,
                    help="Concurrent workers for RG vdep API calls.")
    ap.add_argument("--groundback-iou-threshold", type=float, default=0.35,
                    help="IoU threshold below which anchor groundback applies 0.20 score penalty.")
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview")
    ap.add_argument("--image-dependence-workers", type=int, default=6)
    ap.add_argument("--skip-image-dependence", action="store_true")
    ap.add_argument(
        "--prior-cache-intermediate-dir",
        default=_ITA07_QUALITY_BUNDLE_INTERMEDIATE,
        help="Intermediate dir from ita07_quality_bundle for verified-tuple cache reuse.",
    )
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
    """Diagnostics specific to ita08: RG text-leakage indicators."""
    rg_rows = [r for r in rows if str(r.get("question_type") or "") == "REVERSE_GROUND"]
    corrected = sum(1 for r in rg_rows if r.get("rg_correction_applied"))
    groundback_failed = sum(1 for r in rg_rows if r.get("groundback_failed"))
    # Count anchor labels with color+shape tokens (pre-correction leaky candidates)
    color_tokens = frozenset({"red", "blue", "green", "brown", "white", "black", "gray", "grey",
                               "yellow", "orange", "purple", "pink", "silver", "gold"})
    leaky_label_rows = sum(
        1 for r in rg_rows
        if any(t in str(r.get("anchor_label") or "").lower().split() for t in color_tokens)
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
        "# SGOCR Mixed ITA08 Sweep",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{bundle_id}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Live report: [{bundle_id}_report.md]({report_path})",
        f"- Log: [run.log]({log_path})",
        "",
        "## Goal",
        "",
        "**Fix RG text-leakage via typed frontier gate + cross-model RG vdep check.**",
        "",
        "Root cause from ita07: frontier gate self-selects easy-RG rows (Gemini gets right",
        "with image = same rows it gets right text-only). Two targeted fixes:",
        "1. **Typed frontier gate**: apply only to DR/YN/TP/AP — bypass for RG entirely.",
        "2. **RG vdep check**: single `gpt-5.3-codex` text-only call per RG row.",
        "3. **Leakage correction** (correction variants): strip color/shape tokens, re-probe.",
        "4. **Anchor groundback** (groundback variants): Qwen re-localizes anchor by label;",
        "   IoU < 0.35 → 0.20 score penalty (ambiguous labels, not hard reject).",
        "",
        "Cache: `cache_level=verified` (reuse ita07_quality_bundle OCR + Qwen + verified_tuples).",
        f"Workers: {perf['workers']}. RG vdep model: `{perf['rg_vdep_model']}`.",
        "",
        "## Engineering Changes",
        "",
        "| Location | Change |",
        "|---|---|",
        "| `semantic_dev40_tuning.py` | 6 new fields: `inline_frontier_gate_question_types`, `rg_vdep_check_enabled`, `rg_vdep_model`, `rg_leakage_correction_enabled`, `anchor_label_groundback_enabled`, `anchor_label_groundback_iou_threshold` |",
        "| `full_pipeline_dev40.py` | Typed frontier gate (per-row question_type check) |",
        "| `full_pipeline_dev40.py` | RG vdep check block after vision_dependence_gate |",
        "| `full_pipeline_dev40.py` | Anchor label groundback block after degenerate_anchor_filter |",
        "| `dev200_eval.py` | `_call_openai_text_only_eval()` — text-only OpenAI probe (no image) |",
        "| `dev200_eval.py` | `apply_rg_vdep_check()` — RG vdep gate with optional leakage correction |",
        "| `dev200_eval.py` | `_strip_rg_leakage_tokens()` — strip color/shape tokens from RG questions |",
        "| `qwen_anchor_vllm.py` | `GROUNDBACK_QWEN_PROMPT` — re-localize anchor by label text |",
        "| `qwen_anchor_vllm.py` | `groundback_check_many()` — batch groundback IoU checks |",
        "",
        "## Sweep Parameters",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Frontier gate question types | `DR,YN,TP,AP` (RG bypassed) |",
        f"| RG vdep model | `{perf['rg_vdep_model']}` |",
        f"| Groundback IoU threshold | `{perf['groundback_iou_threshold']}` |",
        f"| Qwen degenerate threshold | `{perf['qwen_degenerate_threshold']}` |",
        f"| Spatial min centroid offset | `{perf['spatial_min_centroid_offset']}` |",
        f"| Qwen batch size | `{perf['qwen_batch_size']}` |",
        f"| Semantic workers | `{perf['workers']}` |",
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
        "## RG Leakage Diagnostics",
        "",
        "| Variant | Total RG | Corrected RG | Groundback-failed | Leaky-label candidates |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        rg = result.get("rg_leakage_stats") or {}
        if not rg:
            lines.append(f"| `{name}` | — | — | — | — |")
            continue
        lines.append(
            f"| `{name}` |"
            f" {rg.get('total_rg', 0)} |"
            f" {rg.get('corrected_rg', 0)} |"
            f" {rg.get('groundback_failed_rg', 0)} |"
            f" {rg.get('leaky_label_rg_candidates', 0)} |"
        )

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
                f"- Groundback failed (score −0.20): {rg.get('groundback_failed_rg', 0)}",
                f"- Leaky-label candidates (color token in anchor_label): {rg.get('leaky_label_rg_candidates', 0)}",
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
        "# Mixed ITA08 Sweep",
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

    # All ITA08 variants inherit ita07_quality_bundle base:
    # structural fixes + frontier gate (now typed: DR/YN/TP/AP only) + qwen_degenerate_anchor_filter.
    # RG is no longer gated by the frontier gate — instead gets a cross-model text-only probe.
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
        # Centroid offset and mechanical color
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
        # ita07_quality_bundle base flags (carried through all ita08 variants)
        "SGOCR_INLINE_FRONTIER_GATE_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED": "1",
        # ita08: typed gate — bypass RG from frontier gate
        "SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": _TYPED_GATE_QT,
        # ita08: RG vdep check enabled on all variants
        "SGOCR_RG_VDEP_CHECK_ENABLED": "1",
        "SGOCR_RG_VDEP_MODEL": str(args.rg_vdep_model),
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": str(float(args.groundback_iou_threshold)),
    }

    correction_flags = {
        "SGOCR_RG_LEAKAGE_CORRECTION_ENABLED": "1",
    }
    groundback_flags = {
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
    }

    variants = {
        "ita08_typed_gate": {
            "env": {**common_env},
            "note": VARIANT_DESCRIPTIONS["ita08_typed_gate"],
        },
        "ita08_correction": {
            "env": {**common_env, **correction_flags},
            "note": VARIANT_DESCRIPTIONS["ita08_correction"],
        },
        "ita08_groundback": {
            "env": {**common_env, **groundback_flags},
            "note": VARIANT_DESCRIPTIONS["ita08_groundback"],
        },
        "ita08_full": {
            "env": {**common_env, **correction_flags, **groundback_flags},
            "note": VARIANT_DESCRIPTIONS["ita08_full"],
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
                "frontier_word_f1_floor": float(args.frontier_word_f1_floor),
                "rg_vdep_model": str(args.rg_vdep_model),
                "rg_vdep_workers": int(args.rg_vdep_workers),
                "groundback_iou_threshold": float(args.groundback_iou_threshold),
                "image_dependence_model": str(args.image_dependence_model),
            },
            "nemotron_diagnostic": nemotron_frontend_diagnostic(),
            "results": {name: {"status": "pending", "note": spec["note"]} for name, spec in variants.items()},
        }
    _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

    prior_cache_dir = Path(args.prior_cache_intermediate_dir) if args.prior_cache_intermediate_dir else None

    for name, spec in variants.items():
        if payload["results"].get(name, {}).get("status") in {"ok", "cached"}:
            print(f"[skip] {name} already completed — skipping", flush=True)
            continue
        payload["results"][name] = {"status": "running", "note": spec["note"]}
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

        out_dir = root_final / f"{args.bundle_id}_{name}"
        intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}"

        # Verified-tuple cache reuse is only safe for variants that do not change the
        # anchor stage. Groundback runs inside the anchor stage, so those variants must
        # fall back to OCR-level cache reuse and rebuild anchors/verified tuples.
        if prior_cache_dir and prior_cache_dir.exists():
            if "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED" in spec["env"]:
                variant_cache_level = "ocr"
            else:
                variant_cache_level = "verified"
            variant_cache_dir = prior_cache_dir
        else:
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
                    idep_dir = out_dir / "evals" / "image_dependence_ita08"
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
