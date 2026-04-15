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

# ita05 goal:
#   Baseline: ita04_anti_ocr (ita04 winner — best yield + lowest text-leakage).
#   anti_ocr is now baked into common_env; all variants share the ita04 Qwen cache.
#
#   Engineering change in this sweep:
#     SGOCR_INLINE_FRONTIER_GATE_WORD_F1_FLOOR  — softer frontier gate for REVERSE_GROUND.
#     Default=-1.0 (use binary soft_correct gate). When set to e.g. 0.3, also accept RG rows
#     where word-F1 >= 0.3 even if soft_correct is False. This allows partially-correct
#     spatial descriptions through the gate, recovering yield lost by the strict binary gate.
#
#   Key ita04 findings:
#     • anti_ocr wins on yield (387 vs 322 for control) AND text-leakage (28.4% vs 34.5%)
#     • frontier_gate wins on quality (sweep_score=15.0, vision_nec=37.3%) but costs ~30% yield
#     • anti_ocr + frontier_gate was never tested in ita04 — largest unexplored gap
#     • gate_bundle (frontier+vision_dep together) = 0 rows; the two hard gates exclude everything
#     • RG text-only accuracy = 0.75 in control (high leakage); anti_ocr drops it slightly
#
#   Variants:
#     ita05_control          — anti_ocr baseline (= ita04_anti_ocr config, reference point)
#     ita05_frontier_strict  — anti_ocr + frontier_gate binary (untested in ita04; key combo)
#     ita05_frontier_lenient — anti_ocr + frontier_gate with word_f1_floor=0.3 (new lenient gate)
#     ita05_anchor_bundle    — anti_ocr + anchor_filter (two complementary upstream fixes)
#     ita05_quality_bundle   — anti_ocr + anchor_filter + frontier_strict (high-quality combo)
#
#   Cache strategy:
#     All variants share anti_ocr as the Qwen baseline. The default prior-cache-intermediate-dir
#     points to ita04_anti_ocr, so all variants use cache_level="verified" (reuse Nemotron+Qwen,
#     re-run Gemini teacher generation only). No _QWEN_MODIFYING variants — all reuse Qwen.
#
#   NOTE: ita04_gate_bundle = 0 rows showed that frontier_gate + vision_dep_gate together
#   eliminate everything. ita05 avoids stacking both hard gates. Instead, the lenient frontier
#   gate (word_f1_floor) is designed to leave more rows for potential downstream filtering.

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "158_sgocr_mixed_ita05_launch_2026-04-13.md"

_ITA04_ANTI_OCR_INTERMEDIATE = (
    "/home/wdree/percy/vqafromscratch/data/ocr_spatial_qa/intermediate/mixed_dev150"
    "/sgocr_mixed_ita04_20260412_222128_ita04_anti_ocr"
)

VARIANT_ORDER = [
    "ita05_control",
    "ita05_frontier_strict",
    "ita05_frontier_lenient",
    "ita05_anchor_bundle",
    "ita05_quality_bundle",
]

VARIANT_DESCRIPTIONS = {
    "ita05_control":          "anti_ocr baseline — ita04 winner baked into common_env (reference)",
    "ita05_frontier_strict":  "anti_ocr + frontier_gate binary — key untested combo from ita04",
    "ita05_frontier_lenient": "anti_ocr + frontier_gate lenient (word_f1_floor=0.3) — new softer gate for RG rows",
    "ita05_anchor_bundle":    "anti_ocr + anchor_filter — two complementary upstream quality fixes",
    "ita05_quality_bundle":   "anti_ocr + anchor_filter + frontier_strict — high-quality full combo",
}

# All variants share the anti_ocr Qwen cache from ita04_anti_ocr; none change the Qwen prompt.
_QWEN_MODIFYING: set[str] = set()


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="Run ita05: anti_ocr as baseline + lenient frontier gate + anchor bundle combos."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita05_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita04_report.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=8)
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
    ap.add_argument("--qwen-degenerate-threshold", type=float, default=0.75)
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.15)
    ap.add_argument("--frontier-word-f1-floor", type=float, default=0.3,
                    help="word_f1 floor for the lenient frontier gate variant (ita05_frontier_lenient).")
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview")
    ap.add_argument("--image-dependence-workers", type=int, default=8)
    ap.add_argument("--skip-image-dependence", action="store_true")
    ap.add_argument(
        "--prior-cache-intermediate-dir",
        default=_ITA04_ANTI_OCR_INTERMEDIATE,
        help="Intermediate dir from ita04_anti_ocr for cache reuse. "
             "All ita05 variants are anti_ocr-based, so all use cache_level=verified.",
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


def _render_live_doc(payload: dict[str, Any]) -> str:
    perf = payload["performance_profile"]
    bundle_id = payload["bundle_id"]
    report_path = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150" / f"{bundle_id}_report.md"
    log_path = REPO_ROOT / "logs" / bundle_id / "run.log"
    results = payload["results"]

    lines = [
        "# SGOCR Mixed ITA05 Sweep",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{bundle_id}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Live report: [{bundle_id}_report.md]({report_path})",
        f"- Log: [run.log]({log_path})",
        "",
        "## Goal",
        "",
        "**Anti-OCR as new baseline + lenient frontier gate**: bake ita04's winning `anti_ocr` config",
        "into common_env, then test untested combos and the new softer frontier gate.",
        "",
        "Baseline is `ita04_anti_ocr` (387 rows, sweep=11.9966, text_leaky=28.4% — ita04 winner).",
        "",
        "Three axes tested:",
        "1. **anti_ocr + frontier_gate** (`ita05_frontier_strict`): the key untested combo from ita04.",
        "   Hypothesis: anti_ocr headroom (+65 rows) absorbs frontier_gate's ~30% yield cost,",
        "   yielding quality near ita04_frontier_gate but with more rows.",
        "2. **Lenient frontier gate** (`ita05_frontier_lenient`): new `word_f1_floor=0.3` flag.",
        "   For REVERSE_GROUND only, accept rows where frontier model gets ≥30% of words right",
        "   even if soft_correct=False. Addresses RG-specific yield loss from strict gate.",
        "3. **Anchor bundle** (`ita05_anchor_bundle`, `ita05_quality_bundle`): anti_ocr + anchor_filter.",
        "   Two independent upstream fixes that should complement each other.",
        "",
        "**Engineering change**: `SGOCR_INLINE_FRONTIER_GATE_WORD_F1_FLOOR` added to tuning.",
        "Default=-1.0 (binary gate unchanged). When ≥0.0, RG rows with word_f1 ≥ floor also pass.",
        "Motivated by: (a) gate_bundle=0 rows in ita04 — binary gate too strict for stacking;",
        "(b) RG text-only acc=0.75 suggests strict gate may be rejecting valid but spatially-phrased rows.",
        "",
        "**Cache efficiency**: all variants reuse Nemotron+Qwen from ita04_anti_ocr intermediate dir",
        "(cache_level=verified). Only Gemini teacher generation and post-gen filtering re-run.",
        "",
        "## New Flag",
        "",
        "| Flag | Description |",
        "|---|---|",
        "| `SGOCR_INLINE_FRONTIER_GATE_WORD_F1_FLOOR` | Lenient frontier gate: also accept rows where word-F1 ≥ floor (REVERSE_GROUND only; -1.0 = disabled) |",
        "",
        "## Sweep Parameters",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Qwen degenerate threshold | `{perf['qwen_degenerate_threshold']}` |",
        f"| Spatial min centroid offset | `{perf['spatial_min_centroid_offset']}` |",
        f"| Frontier word-F1 floor | `{perf['frontier_word_f1_floor']}` |",
        f"| Group min instances | `{perf['group_min_instances']}` |",
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
        "# Mixed ITA05 Sweep",
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

    # anti_ocr is the new baseline — baked into common_env for all variants.
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
        # ita02_sceneaware flags (kept through ita03/ita04/ita05)
        "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
        "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
        "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
        "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "0",
        # ita03_structural_fallback flags
        "SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD": str(float(args.qwen_degenerate_threshold)),
        # Centroid offset and mechanical color (kept from ita03)
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
        # ita04_anti_ocr winner flag — now baked into the baseline for all ita05 variants
        "SGOCR_QWEN_ANTI_OCR_PROMPT_ENABLED": "1",
    }

    frontier_strict_flags = {
        "SGOCR_INLINE_FRONTIER_GATE_ENABLED": "1",
        # word_f1_floor=-1.0 means standard binary gate (default; no change needed)
    }
    frontier_lenient_flags = {
        "SGOCR_INLINE_FRONTIER_GATE_ENABLED": "1",
        "SGOCR_INLINE_FRONTIER_GATE_WORD_F1_FLOOR": str(float(args.frontier_word_f1_floor)),
    }
    anchor_filter_flags = {
        "SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED": "1",
    }

    variants = {
        "ita05_control": {
            "env": {**common_env},
            "note": VARIANT_DESCRIPTIONS["ita05_control"],
        },
        "ita05_frontier_strict": {
            "env": {**common_env, **frontier_strict_flags},
            "note": VARIANT_DESCRIPTIONS["ita05_frontier_strict"],
        },
        "ita05_frontier_lenient": {
            "env": {**common_env, **frontier_lenient_flags},
            "note": VARIANT_DESCRIPTIONS["ita05_frontier_lenient"],
        },
        "ita05_anchor_bundle": {
            "env": {**common_env, **anchor_filter_flags},
            "note": VARIANT_DESCRIPTIONS["ita05_anchor_bundle"],
        },
        "ita05_quality_bundle": {
            "env": {**common_env, **anchor_filter_flags, **frontier_strict_flags},
            "note": VARIANT_DESCRIPTIONS["ita05_quality_bundle"],
        },
    }

    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    live_doc = LIVE_DOC_DIR / LIVE_DOC_NAME

    now = datetime.now().isoformat(timespec="seconds")
    payload: dict[str, Any] = {
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
        payload["results"][name] = {"status": "running", "note": spec["note"]}
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

        out_dir = root_final / f"{args.bundle_id}_{name}"
        intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}"

        # Cache strategy:
        # All ita05 variants share anti_ocr Qwen output. The prior_cache_dir points to
        # ita04_anti_ocr's intermediate dir. No variants modify the Qwen prompt relative
        # to the baseline, so all use cache_level="verified" (reuse Nem+Qwen, re-run Gemini).
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

            if not args.skip_image_dependence and rows:
                payload["results"][name] = {**result, "status": "image_dependence"}
                _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)
                try:
                    idep_dir = out_dir / "evals" / "image_dependence_ita05"
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
