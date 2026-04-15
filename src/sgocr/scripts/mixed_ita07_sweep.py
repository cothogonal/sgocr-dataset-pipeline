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

# ita07 goal:
#   Apply structural fixes from the ita04-06 post-mortem and test their downstream impact.
#
#   Root causes fixed in this sweep (all baked into common_env):
#     1. STRUCTURAL_FALLBACK_QWEN_PROMPT: replaced "blue rectangular bar" with neutral structural
#        examples ('bar chart segment', 'pie chart wedge', 'legend panel'). The biased example
#        was causing Qwen to output "blue rectangular bar" for ~53% of ChartQA anchors, dominating
#        the answer distribution and making ANCHOR_PROPERTY questions circular.
#     2. is_degenerate_inventory threshold: raised from 0.75 → 0.92. Previously, images with
#        concentrated valid labels (e.g. airport with 3 planes → 100% concentration) were
#        incorrectly triggering the structural fallback and losing the "plane" anchor.
#        The new threshold plus a _VALID_CONCENTRATED_LABELS exemption set (plane, car, person…)
#        prevents this regression.
#     3. Upstream ANCHOR_PROPERTY::anchor_shape filter: dropped from build_question_candidates
#        when anchor_label contains shape tokens (rectangular, circular…). Prevents circular
#        questions like "What shape is the blue rectangular bar?" (answer = "rectangular").
#     4. Upstream REVERSE_GROUND structural filter (SGOCR_RG_STRUCTURAL_ANCHOR_FILTER_ENABLED):
#        drops RG candidates whose anchor_label contains both a color and a structural shape token
#        (e.g. "blue rectangular bar"). These labels make RG questions text-leaky because
#        text-only models can exploit chart-knowledge priors to answer spatial questions.
#
#   Cache strategy:
#     All fixes affect the Qwen stage (structural fallback prompt + degenerate threshold change
#     → different fallback triggering). Using cache_level="ocr" to reuse OCR/text-detection
#     results from ita06_control while re-running all Qwen + Gemini stages.
#     Workers reduced from 8 → 4 to avoid Gemini 429 rate limits.
#
#   Variants (4 runs, no control — all share the structural fixes):
#     ita07_reform_base     — structural fixes only; establishes new baseline quality
#     ita07_frontier        — structural fixes + frontier_gate; quality lane post-fix
#     ita07_rg_guarded      — structural fixes + frontier_gate + RG structural filter
#     ita07_quality_bundle  — structural fixes + frontier_gate + RG filter + anchor_filter
#
#   Hypotheses:
#     ita07_reform_base:    "blue rectangular bar" concentration drops dramatically.
#                           Expect 0-5 "rectangular" answers (vs 21 in ita06_control).
#                           "plane" labels should reappear on natural images.
#                           vision_nec should improve with cleaner anchor vocabulary.
#     ita07_frontier:       frontier gate now acts on a cleaner candidate pool.
#                           Expect text_leaky to drop vs ita06_frontier_strict (38%).
#                           sweep_score should remain ≥15 while text_leaky approaches <30%.
#     ita07_rg_guarded:     RG text_only acc should drop toward img_acc (positive vision_delta).
#                           Small yield cost (RG count drops), but quality improves for RG.
#     ita07_quality_bundle: Maximum quality stack. Tests if anchor_filter is now additive
#                           after the structural fallback fix removes the dominant generator
#                           of non-degenerate-but-low-quality labels.
#
#   Key comparison axes:
#     ita07_reform_base vs ita06_control       → impact of structural fixes alone
#     ita07_frontier vs ita06_frontier_strict  → frontier gate improvement with clean anchors
#     ita07_rg_guarded vs ita07_frontier       → RG filter's effect on RG text_leakage
#     ita07_quality_bundle vs ita07_rg_guarded → additive value of anchor_filter post-fix

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "160_sgocr_mixed_ita07_launch_2026-04-13.md"

_ITA06_CONTROL_INTERMEDIATE = (
    "/home/wdree/percy/vqafromscratch/data/ocr_spatial_qa/intermediate/mixed_dev150"
    "/sgocr_mixed_ita06_20260413_040517_ita06_control"
)

VARIANT_ORDER = [
    "ita07_reform_base",
    "ita07_frontier",
    "ita07_rg_guarded",
    "ita07_quality_bundle",
]

VARIANT_DESCRIPTIONS = {
    "ita07_reform_base":    "structural fixes only — new baseline (no gate, no RG filter)",
    "ita07_frontier":       "structural fixes + frontier_gate — quality lane post-fix",
    "ita07_rg_guarded":     "structural fixes + frontier_gate + RG structural anchor filter",
    "ita07_quality_bundle": "structural fixes + frontier_gate + RG filter + anchor_filter — max quality stack",
}

# All variants change the Qwen stage (new degenerate threshold + fallback prompt).
# cache_level="ocr" for all: reuse text-detection, re-run Qwen and everything downstream.
_QWEN_MODIFYING: set[str] = set(VARIANT_ORDER)


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="Run ita07: structural fixes + frontier gate + RG filter variants."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita07_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita06_report.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=6,
                    help="Concurrent Gemini workers.")
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--qwen-batch-size", type=int, default=4)
    ap.add_argument("--qwen-gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--qwen-max-model-len", type=int, default=2048)
    ap.add_argument("--qwen-pass-count", type=int, default=1)
    ap.add_argument("--qwen-temperature", type=float, default=0.0)
    ap.add_argument("--qwen-consensus-iou", type=float, default=0.55)
    ap.add_argument("--qwen-min-support", type=int, default=1)
    ap.add_argument("--group-min-instances", type=int, default=3)
    ap.add_argument("--cheap-proxy-reject-score", type=int, default=6)
    ap.add_argument("--qwen-degenerate-threshold", type=float, default=0.92,
                    help="Raised from 0.75: valid concentrated-label images no longer trigger structural fallback.")
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.15)
    ap.add_argument("--frontier-word-f1-floor", type=float, default=-1.0,
                    help="Frontier gate word-F1 floor. -1.0 = strict binary gate (default).")
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview")
    ap.add_argument("--image-dependence-workers", type=int, default=6,
                    help="Workers for image-dependence eval.")
    ap.add_argument("--skip-image-dependence", action="store_true")
    ap.add_argument(
        "--prior-cache-intermediate-dir",
        default=_ITA06_CONTROL_INTERMEDIATE,
        help="Intermediate dir from ita06_control for OCR cache reuse. "
             "All ita07 variants use cache_level=ocr (re-run Qwen with new prompts/threshold).",
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


def _anchor_label_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Extra diagnostics for anchor label quality — key for ita07 structural fix validation."""
    from collections import Counter
    labels = Counter(str(row.get("anchor_label") or "").lower() for row in rows)
    rect_rows = sum(ct for lab, ct in labels.items() if "rectangular" in lab or "rectangle" in lab)
    blue_rect_rows = labels.get("blue rectangular bar", 0)
    plane_rows = sum(ct for lab, ct in labels.items() if "plane" in lab or "aircraft" in lab)
    scene_rows = sum(ct for lab, ct in labels.items() if "scene" in lab)
    top5 = labels.most_common(5)
    return {
        "rectangular_label_rows": rect_rows,
        "blue_rect_bar_rows": blue_rect_rows,
        "plane_rows": plane_rows,
        "scene_element_rows": scene_rows,
        "top5_anchor_labels": top5,
    }


def _answer_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Track 'rectangular' answer concentration — the circular-AP diagnostic."""
    from collections import Counter
    answers = Counter(str(row.get("answer") or "").lower() for row in rows)
    rect_ans = sum(ct for ans, ct in answers.items() if "rectangular" in ans)
    return {
        "rectangular_answer_rows": rect_ans,
        "top5_answers": answers.most_common(5),
    }


def _render_live_doc(payload: dict[str, Any]) -> str:
    perf = payload["performance_profile"]
    bundle_id = payload["bundle_id"]
    report_path = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150" / f"{bundle_id}_report.md"
    log_path = REPO_ROOT / "logs" / bundle_id / "run.log"
    results = payload["results"]

    lines = [
        "# SGOCR Mixed ITA07 Sweep",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{bundle_id}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Live report: [{bundle_id}_report.md]({report_path})",
        f"- Log: [run.log]({log_path})",
        "",
        "## Goal",
        "",
        "**Apply structural anchor fixes + test two quality axes (frontier gate, RG filter).**",
        "",
        "Three fixes baked into all variants:",
        "1. **Structural fallback prompt**: removed `'blue rectangular bar'` example → neutral structural",
        "   examples (`'bar chart segment'`, `'pie chart wedge'`, `'legend panel'`).",
        "2. **Degenerate inventory threshold**: raised 0.75 → 0.92, with `_VALID_CONCENTRATED_LABELS`",
        "   exemption (`plane`, `car`, `person`, …). Prevents airport/crowd images from triggering fallback.",
        "3. **Upstream ANCHOR_PROPERTY::anchor_shape filter**: dropped when anchor_label contains",
        "   shape token (e.g. `'rectangular'`). Eliminates circular questions.",
        "",
        "New tunable axis in ita07:",
        "- **REVERSE_GROUND structural anchor filter** (`SGOCR_RG_STRUCTURAL_ANCHOR_FILTER_ENABLED`):",
        "  drops RG candidates whose anchor_label contains color + structural shape token.",
        "  Targets the `RG text_only > img_acc` inversion seen in ita06 (text_only=0.71-0.93 for RG).",
        "",
        "Cache: `cache_level=ocr` (reuse OCR from ita06_control, re-run Qwen + Gemini).",
        "Workers: 4 (down from 8 to avoid Gemini 429 rate limits).",
        "",
        "## Engineering Changes",
        "",
        "| Location | Change |",
        "|---|---|",
        "| `qwen_anchor_vllm.py` `STRUCTURAL_FALLBACK_QWEN_PROMPT` | Replace `'blue rectangular bar'` with neutral structural examples |",
        "| `qwen_anchor_vllm.py` `is_degenerate_inventory` | Raise default threshold 0.75→0.92; add `_VALID_CONCENTRATED_LABELS` exemption |",
        "| `dev40_complete.py` `build_question_candidates` | Skip `anchor_shape` AP candidate when anchor_label has shape token |",
        "| `dev40_complete.py` `build_question_candidates` | New `SGOCR_RG_STRUCTURAL_ANCHOR_FILTER_ENABLED` flag: drop RG for color+shape anchor labels |",
        "| `semantic_dev40_tuning.py` | New `rg_structural_anchor_filter_enabled` field; threshold default updated to 0.92 |",
        "",
        "## Sweep Parameters",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Qwen degenerate threshold | `{perf['qwen_degenerate_threshold']}` (raised from 0.75) |",
        f"| Spatial min centroid offset | `{perf['spatial_min_centroid_offset']}` |",
        f"| Frontier word-F1 floor | `{perf['frontier_word_f1_floor']}` |",
        f"| Group min instances | `{perf['group_min_instances']}` |",
        f"| Qwen batch size | `{perf['qwen_batch_size']}` |",
        f"| Semantic workers | `{perf['workers']}` (reduced from 8) |",
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
        "## Structural Fix Diagnostics",
        "",
        "| Variant | Rect-label rows | blue-rect-bar rows | Plane rows | Rect answers | Top-5 anchors |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        al = result.get("anchor_label_stats") or {}
        ans = result.get("answer_stats") or {}
        if not al:
            lines.append(f"| `{name}` | — | — | — | — | — |")
            continue
        top5 = ", ".join(f"`{lab}`({ct})" for lab, ct in (al.get("top5_anchor_labels") or [])[:3])
        lines.append(
            f"| `{name}` |"
            f" {al.get('rectangular_label_rows', 0)} |"
            f" {al.get('blue_rect_bar_rows', 0)} |"
            f" {al.get('plane_rows', 0)} |"
            f" {ans.get('rectangular_answer_rows', 0)} |"
            f" {top5} |"
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
        al = result.get("anchor_label_stats") or {}
        ans = result.get("answer_stats") or {}
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
        if al:
            lines += [
                "**Structural fix diagnostics:**",
                "",
                f"- Rectangular-label rows: {al.get('rectangular_label_rows', 0)} (target: < 10 vs ita06 ~94)",
                f"- `blue rectangular bar` rows: {al.get('blue_rect_bar_rows', 0)} (target: < 5 vs ita06 ~61)",
                f"- Plane rows: {al.get('plane_rows', 0)} (target: restored vs ita06 = 0)",
                f"- Rectangular answers: {ans.get('rectangular_answer_rows', 0)} (target: < 3 vs ita06 = 21)",
                f"- Top-5 anchor labels: {al.get('top5_anchor_labels', [])}",
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
            for mode_key, mode_label in [("image_accuracy", "Image+Q"), ("text_only_accuracy", "Text-only"), ("vision_delta", "Δ")]:
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
        "# Mixed ITA07 Sweep",
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

    # Structural fixes baked into all variants.
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
        # ita02_sceneaware flags (carried through all iterations)
        "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
        "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
        "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
        "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "0",
        # ita03_structural_fallback flags
        "SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED": "1",
        # Raised from 0.75 → 0.92 to prevent valid-concentrated-label images triggering fallback
        "SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD": str(float(args.qwen_degenerate_threshold)),
        # Centroid offset and mechanical color
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
        # ita04_anti_ocr winner flag
        "SGOCR_QWEN_ANTI_OCR_PROMPT_ENABLED": "1",
    }

    frontier_strict_flags = {
        "SGOCR_INLINE_FRONTIER_GATE_ENABLED": "1",
    }
    rg_filter_flags = {
        "SGOCR_RG_STRUCTURAL_ANCHOR_FILTER_ENABLED": "1",
    }
    anchor_filter_flags = {
        "SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED": "1",
    }

    variants = {
        "ita07_reform_base": {
            "env": {**common_env},
            "note": VARIANT_DESCRIPTIONS["ita07_reform_base"],
        },
        "ita07_frontier": {
            "env": {**common_env, **frontier_strict_flags},
            "note": VARIANT_DESCRIPTIONS["ita07_frontier"],
        },
        "ita07_rg_guarded": {
            "env": {**common_env, **frontier_strict_flags, **rg_filter_flags},
            "note": VARIANT_DESCRIPTIONS["ita07_rg_guarded"],
        },
        "ita07_quality_bundle": {
            "env": {**common_env, **frontier_strict_flags, **rg_filter_flags, **anchor_filter_flags},
            "note": VARIANT_DESCRIPTIONS["ita07_quality_bundle"],
        },
    }

    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    live_doc = LIVE_DOC_DIR / LIVE_DOC_NAME

    now = datetime.now().isoformat(timespec="seconds")
    existing_payload = _load_json(report_json)
    if existing_payload:
        # Resume mode: preserve completed variant results, update timestamp only.
        payload = existing_payload
        payload["updated_at"] = now
        # Ensure any new variants are seeded as pending.
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

        # Cache strategy: all ita07 variants change the Qwen stage (new degenerate threshold +
        # fallback prompt). Use cache_level="ocr" to reuse text-detection but re-run Qwen.
        # _QWEN_MODIFYING contains all variants, so all use cache_level="ocr".
        if prior_cache_dir and prior_cache_dir.exists():
            variant_cache_level = "ocr" if name in _QWEN_MODIFYING else "verified"
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
            result["anchor_label_stats"] = _anchor_label_stats(rows)
            result["answer_stats"] = _answer_stats(rows)

            if not args.skip_image_dependence and rows:
                payload["results"][name] = {**result, "status": "image_dependence"}
                _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)
                try:
                    idep_dir = out_dir / "evals" / "image_dependence_ita07"
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
