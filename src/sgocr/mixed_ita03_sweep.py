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

# ita03 goal:
#   Baseline: ita02_sceneaware (best ita02 config — local-suppression + grouped reads).
#   Test each ita03 improvement in isolation, then bundle the promising ones.
#
#   Variants:
#     ita03_control            — ita02_sceneaware baseline (no new flags)
#     ita03_structural_fallback — Qwen degenerate-detection + shape/region prompt (ChartQA fix)
#     ita03_second_pass        — second Qwen pass for low-yield images (natural scene yield)
#     ita03_leakage            — answer-leakage filter (word-count & color-in-label)
#     ita03_centroid15         — spatial gate: suppress "specific location" for anchors within 15% of center
#     ita03_diversity          — per-image anchor diversity score bonus (0.05)
#     ita03_mechanical_color   — deterministic color answer from anchor_label
#     ita03_qwen_yield         — structural fallback + second pass together (Qwen yield combo)
#     ita03_quality_bundle     — leakage + centroid + diversity + mechanical color (quality combo)
#     ita03_all                — all new ita03 features together
#
#   NOTE: cheap proxy threshold stays at 6 (was never triggered in ita02 on mix150,
#   so zero effect; lower threshold is a separate experiment for a future sweep).

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "156_sgocr_mixed_ita03_launch_2026-04-12.md"

VARIANT_ORDER = [
    "ita03_control",
    "ita03_structural_fallback",
    "ita03_second_pass",
    "ita03_leakage",
    "ita03_centroid15",
    "ita03_diversity",
    "ita03_mechanical_color",
    "ita03_qwen_yield",
    "ita03_quality_bundle",
    "ita03_all",
]

VARIANT_DESCRIPTIONS = {
    "ita03_control":            "ita02_sceneaware baseline — no new features",
    "ita03_structural_fallback":"Qwen degenerate-detection + shape/region fallback prompt (targets ChartQA dropout)",
    "ita03_second_pass":        "Second Qwen pass for images with <N anchor detections",
    "ita03_leakage":            "Answer-leakage filter (word-count quoted text; color-in-anchor-label)",
    "ita03_centroid15":         "Spatial gate: suppress specific-location annotation for near-center anchors (15% offset)",
    "ita03_diversity":          "Per-image anchor diversity quality bonus (0.05) in candidate selection",
    "ita03_mechanical_color":   "Deterministic color extraction from anchor_label for ANCHOR_PROPERTY QAs",
    "ita03_qwen_yield":         "Structural fallback + second pass — both Qwen yield improvements",
    "ita03_quality_bundle":     "Leakage + centroid + diversity + mechanical color — all quality improvements",
    "ita03_all":                "All ita03 features together",
}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(description="Run ita03: quality + yield improvements on top of ita02_sceneaware baseline.")
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita03_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita02_report.json")
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
    ap.add_argument("--qwen-min-anchor-detections", type=int, default=3)
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.15)
    ap.add_argument("--anchor-diversity-bonus", type=float, default=0.05)
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview",
                    help="Gemini model to use for image-dependence eval (text-only vs. image+text comparison).")
    ap.add_argument("--image-dependence-workers", type=int, default=8)
    ap.add_argument("--skip-image-dependence", action="store_true",
                    help="Skip image-dependence eval (useful for testing).")
    ap.add_argument(
        "--prior-cache-intermediate-dir",
        default=str(
            OCR_SPATIAL_QA_INTERMEDIATE_ROOT
            / "mixed_dev150"
            / "sgocr_mixed_ita03_20260412_140645_ita03_control"
        ),
        help="Intermediate dir from a prior run to reuse for GPU-heavy stages. "
             "Variants that don't change Qwen settings will use cache_level=verified (no GPU). "
             "Variants that change Qwen settings will use cache_level=ocr (skips Nemotron only).",
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
    """Count accepted QAs per source dataset (chartqa / textocr / coco_text)."""
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


def _type_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        qt = str((row.get("tags") or {}).get("question_type") or "unknown").upper()
        counts[qt] += 1
    return dict(counts)


def _render_live_doc(payload: dict[str, Any]) -> str:
    perf = payload["performance_profile"]
    bundle_id = payload["bundle_id"]
    report_path = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150" / f"{bundle_id}_report.md"
    log_path = REPO_ROOT / "logs" / bundle_id / "run.log"
    results = payload["results"]

    lines = [
        f"# SGOCR Mixed ITA03 Sweep",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{bundle_id}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Live report: [{bundle_id}_report.md]({report_path})",
        f"- Log: [run.log]({log_path})",
        "",
        "## Goal",
        "",
        "Baseline is `ita02_sceneaware` (local-suppression + grouped reads, best ita02 config).",
        "Test each ita03 improvement in isolation, then bundle the promising ones.",
        "",
        "Key expected wins:",
        "- **Structural fallback** (`ita03_structural_fallback`): fixes 100% ChartQA dropout from ita01/02 by detecting",
        "  degenerate Qwen inventories and re-running with a shape/region-focused prompt.",
        "- **Second-pass yield** (`ita03_second_pass`): recovers natural-scene images with fewer than N anchor",
        "  detections by running a second Qwen pass at temp+0.25.",
        "- **Leakage filter** (`ita03_leakage`): removes questions where the answer is derivable from the question",
        "  text alone (word-count when text is quoted; color already in anchor_label).",
        "- **Centroid gate** (`ita03_centroid15`): suppresses spatial annotations for anchors within 15% of the",
        "  image center — these are almost never meaningfully 'off-center' in a spatial sense.",
        "",
        "## Engineering Changes",
        "",
        "New tuning flags added across `semantic_dev40_tuning.py`, `dev40_complete.py`,",
        "`qwen_anchor_vllm.py`, and `full_pipeline_dev40.py`:",
        "",
        "| Flag | Description |",
        "|---|---|",
        "| `SGOCR_ANSWER_LEAKAGE_FILTER_ENABLED` | Drop QAs where the expected answer is derivable from the question text alone |",
        "| `SGOCR_SPATIAL_MIN_CENTROID_OFFSET` | Fraction of image dimensions from center below which spatial annotation is suppressed |",
        "| `SGOCR_PER_IMAGE_ANCHOR_DIVERSITY_BONUS` | Quality bonus added when a candidate uses an anchor not yet seen in that image |",
        "| `SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED` | Extract color adjective from anchor_label as the deterministic expected answer for anchor_color QAs |",
        "| `SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED` | Detect degenerate Qwen inventories and re-run with a shape/region prompt |",
        "| `SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD` | Fraction of rows sharing a label that triggers the structural fallback (default 0.75) |",
        "| `SGOCR_QWEN_MIN_ANCHOR_DETECTIONS_PER_IMAGE` | Images below this detection count get a second Qwen pass at higher temp |",
        "",
        "## Sweep Parameters",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Qwen degenerate threshold | `{perf['qwen_degenerate_threshold']}` |",
        f"| Qwen min anchor detections (second-pass trigger) | `{perf['qwen_min_anchor_detections']}` |",
        f"| Spatial min centroid offset | `{perf['spatial_min_centroid_offset']}` |",
        f"| Anchor diversity bonus | `{perf['anchor_diversity_bonus']}` |",
        f"| Group min instances | `{perf['group_min_instances']}` |",
        f"| Qwen batch size | `{perf['qwen_batch_size']}` |",
        f"| Semantic workers | `{perf['workers']}` |",
        "",
        "## New Evaluation: Image-Dependence Score",
        "",
        "After each variant, Gemini Flash is called twice per accepted QA row:",
        "1. With image + question → `image_soft`",
        "2. With question text only → `text_only_soft`",
        "",
        "Δ = `image_soft − text_only_soft` per row. Aggregates:",
        "- **`vision_delta_mean`**: mean Δ across all rows (positive = dataset requires vision)",
        "- **`vision_necessary_rate`**: fraction of rows where Δ > 0 (model needed the image)",
        "- **`text_leaky_rate`**: fraction correct without the image (pure leakage signal)",
        "",
        "> **Future work (not implemented):** naïve model baseline — run a much weaker model",
        "> (e.g. Gemini Haiku) on the same questions. If cheap model ≈ frontier, questions are",
        "> too easy. The gap between frontier and cheap model is a proxy for dataset challenge ceiling.",
        "",
        "> **Future work (not implemented):** LLM spatial judge for REVERSE_GROUND — instead of",
        "> F1 ≥ 0.5 word overlap, use an LLM to score whether the spatial description is factually",
        "> accurate given the image and anchor bounding box.",
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
        if status in {"running", "answer_benchmark", "image_dependence"}:
            desc = VARIANT_DESCRIPTIONS.get(name, "")
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
            f"| Metric | Value |",
            f"|---|---|",
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
                f"| Mode | All | DR | RG | YN | TP | AP |",
                f"|---|---|---|---|---|---|---|",
            ]
            for mode_key, mode_label in [("image_accuracy", "Image+Q"), ("text_only_accuracy", "Text-only"), ("vision_delta", "Δ")]:
                by_type = idep.get("by_type") or {}
                row_parts = [f"| {mode_label} |", f" {idep.get(mode_key, 0.0):.4f} |"]
                for qt in ["DIRECT_READ", "REVERSE_GROUND", "YES_NO", "TEXT_PROPERTY", "ANCHOR_PROPERTY"]:
                    val = by_type.get(qt, {}).get(mode_key.replace("image_accuracy", "image_acc").replace("text_only_accuracy", "text_only_acc").replace("vision_delta", "vision_delta"), 0.0)
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
                lines.append(f"Top answers: {', '.join(f'\"{ans}\" ({cnt})' for ans, cnt in top10[:5])}")
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
    perf = payload["performance_profile"]
    lines = [
        "# Mixed ITA03 Sweep",
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
        # ita02_sceneaware baseline flags
        "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
        "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
        "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
        "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "0",
    }

    structural_fallback_flags = {
        "SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD": str(float(args.qwen_degenerate_threshold)),
    }
    second_pass_flags = {
        "SGOCR_QWEN_MIN_ANCHOR_DETECTIONS_PER_IMAGE": str(int(args.qwen_min_anchor_detections)),
    }
    leakage_flags = {
        "SGOCR_ANSWER_LEAKAGE_FILTER_ENABLED": "1",
    }
    centroid_flags = {
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
    }
    diversity_flags = {
        "SGOCR_PER_IMAGE_ANCHOR_DIVERSITY_BONUS": str(float(args.anchor_diversity_bonus)),
    }
    mechanical_color_flags = {
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
    }

    variants = {
        "ita03_control": {
            "env": {**common_env},
            "note": VARIANT_DESCRIPTIONS["ita03_control"],
        },
        "ita03_structural_fallback": {
            "env": {**common_env, **structural_fallback_flags},
            "note": VARIANT_DESCRIPTIONS["ita03_structural_fallback"],
        },
        "ita03_second_pass": {
            "env": {**common_env, **second_pass_flags},
            "note": VARIANT_DESCRIPTIONS["ita03_second_pass"],
        },
        "ita03_leakage": {
            "env": {**common_env, **leakage_flags},
            "note": VARIANT_DESCRIPTIONS["ita03_leakage"],
        },
        "ita03_centroid15": {
            "env": {**common_env, **centroid_flags},
            "note": VARIANT_DESCRIPTIONS["ita03_centroid15"],
        },
        "ita03_diversity": {
            "env": {**common_env, **diversity_flags},
            "note": VARIANT_DESCRIPTIONS["ita03_diversity"],
        },
        "ita03_mechanical_color": {
            "env": {**common_env, **mechanical_color_flags},
            "note": VARIANT_DESCRIPTIONS["ita03_mechanical_color"],
        },
        "ita03_qwen_yield": {
            "env": {**common_env, **structural_fallback_flags, **second_pass_flags},
            "note": VARIANT_DESCRIPTIONS["ita03_qwen_yield"],
        },
        "ita03_quality_bundle": {
            "env": {**common_env, **leakage_flags, **centroid_flags, **diversity_flags, **mechanical_color_flags},
            "note": VARIANT_DESCRIPTIONS["ita03_quality_bundle"],
        },
        "ita03_all": {
            "env": {
                **common_env,
                **structural_fallback_flags,
                **second_pass_flags,
                **leakage_flags,
                **centroid_flags,
                **diversity_flags,
                **mechanical_color_flags,
            },
            "note": VARIANT_DESCRIPTIONS["ita03_all"],
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
            "qwen_min_anchor_detections": int(args.qwen_min_anchor_detections),
            "spatial_min_centroid_offset": float(args.spatial_min_centroid_offset),
            "anchor_diversity_bonus": float(args.anchor_diversity_bonus),
            "image_dependence_model": str(args.image_dependence_model),
        },
        "nemotron_diagnostic": nemotron_frontend_diagnostic(),
        "results": {name: {"status": "pending", "note": spec["note"]} for name, spec in variants.items()},
    }
    _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

    # Variants that modify the Qwen anchor inventory (structural fallback / second pass)
    # need the OCR layer re-used but must re-run Qwen.  All others can skip GPU entirely
    # by reusing the full verified cache from the prior (or current-run) control intermediate.
    _QWEN_MODIFYING = {"ita03_structural_fallback", "ita03_second_pass", "ita03_qwen_yield", "ita03_all"}
    prior_cache_dir = Path(args.prior_cache_intermediate_dir) if args.prior_cache_intermediate_dir else None

    for name, spec in variants.items():
        payload["results"][name] = {"status": "running", "note": spec["note"]}
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

        out_dir = root_final / f"{args.bundle_id}_{name}"
        intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}"

        # Decide cache strategy for this variant
        if prior_cache_dir and prior_cache_dir.exists():
            if name in _QWEN_MODIFYING:
                variant_cache_dir = prior_cache_dir
                variant_cache_level = "ocr"   # skip Nemotron, re-run Qwen with new settings
            else:
                variant_cache_dir = prior_cache_dir
                variant_cache_level = "verified"  # skip GPU entirely
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
                    idep_dir = out_dir / "evals" / "image_dependence_ita03"
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
