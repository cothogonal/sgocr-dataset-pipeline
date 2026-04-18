from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .dev200_eval import run_image_dependence_eval
from .mixed_ocr_frontend_canary import _build_variant_cmd, _q01_baseline
from ..bootstrap import write_json
from ..dual_anchor import (
    RescueSelectionConfig,
    build_subset_ocr_cache,
    drain_ollama_model,
    load_json,
    load_jsonl,
    merge_intermediate_by_image,
    select_rescue_image_ids,
    summarize_rescue_selection,
    write_source_subset,
)
from ..nemotron_frontend import nemotron_frontend_diagnostic
from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from ..run_quality import compute_anchor_coverage, compute_answer_distribution, compute_run_quality


LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "177_sgocr_mixed_ita22_launch_2026-04-17.md"
_ITA15_T8_INTERMEDIATE = (
    OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150" /
    "sgocr_mixed_ita15_20260415_204008_ita15_t8"
)
_TYPED_GATE_QT = "DIRECT_READ,YES_NO,TEXT_PROPERTY,ANCHOR_PROPERTY"

VARIANT_ORDER = [
    "precision_r48_t6",
    "precision_r72_t6",
    "apheavy_r72_t6",
    "balanced_r72_t6_yesno1",
]

VARIANT_DESCRIPTIONS: dict[str, str] = {
    "precision_r48_t6": (
        "Primary target=6 precision pass, then DAM01 Qwen rescue on the top 48 low-diversity images. "
        "Keeps YES/NO disabled, RG capped at 1 with structural + vdep guards, and TP capped to one visual-only item."
    ),
    "precision_r72_t6": (
        "Same precision-first logic as the control variant, but rescue the top 72 weak-coverage images to see whether "
        "broader rescue coverage can recover AP/RG/TP diversity without dragging vdep."
    ),
    "apheavy_r72_t6": (
        "AP-heavy recovery: rescue the top 72 AP-deficient images with stronger anchor-property / answer-entropy weights "
        "and a stronger final property/diversity bonus, while keeping the same precision gates."
    ),
    "balanced_r72_t6_yesno1": (
        "Balanced recovery with the same rescue breadth as precision_r72_t6, but allow at most one positive YES/NO "
        "question per image in the final pass to test limited type recovery without reopening the high-leak negative path."
    ),
}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="Run ita22: precision-first target-6 primary pass + dual-anchor rescue for mixed_dev150."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita22_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita21_20260417_060334_report.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-tags-per-image", type=int, default=14)
    ap.add_argument("--grounding-threshold", type=float, default=0.28)
    ap.add_argument("--group-min-instances", type=int, default=3)
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.20)
    ap.add_argument("--primary-ocr-cache-dir", default=str(_ITA15_T8_INTERMEDIATE))
    ap.add_argument("--gemma-model", default="gemma4:e4b-it-q4_K_M")
    ap.add_argument("--gemma-base-url", default="http://localhost:11434")
    ap.add_argument("--gemma-num-ctx", type=int, default=4096)
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    ap.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--qwen-batch-size", type=int, default=3)
    ap.add_argument("--qwen-max-model-len", type=int, default=2048)
    ap.add_argument("--switch-min-free-mib", type=int, default=10000)
    ap.add_argument("--switch-timeout-seconds", type=int, default=45)
    ap.add_argument("--rg-vdep-model", default="gemini:gemini-3-pro-preview")
    ap.add_argument("--primary-target-per-image", type=int, default=6)
    ap.add_argument("--direct-read-selection-bonus", type=float, default=0.05)
    ap.add_argument("--dr-generic-anchor-penalty", type=float, default=0.18)
    ap.add_argument("--dr-same-anchor-repeat-penalty", type=float, default=0.10)
    ap.add_argument("--property-candidate-selection-bonus", type=float, default=0.30)
    ap.add_argument("--per-image-anchor-diversity-bonus", type=float, default=0.15)
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview")
    ap.add_argument("--image-dependence-workers", type=int, default=4)
    ap.add_argument("--skip-image-dependence", action="store_true")
    ap.add_argument("--variant", action="append", dest="variants", default=[])
    return ap.parse_args()


def _run_variant_streaming(
    *,
    source_dir: Path,
    out_dir: Path,
    intermediate_dir: Path,
    cache_intermediate_dir: Path | None,
    cache_level: str,
    model: str,
    workers: int,
    max_side: int,
    device: str,
    env_overrides: dict[str, str],
    cli_overrides: dict[str, str],
) -> dict[str, Any]:
    if (out_dir / "summary.json").exists() and (out_dir / "ocr_qa_dataset.jsonl").exists():
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        rows = load_jsonl(out_dir / "ocr_qa_dataset.jsonl")
        metrics = compute_run_quality(summary, rows)
        return {"status": "cached", "metrics": metrics, "summary": summary}

    cmd, env = _build_variant_cmd(
        source_dir=source_dir,
        out_dir=out_dir,
        intermediate_dir=intermediate_dir,
        cache_intermediate_dir=cache_intermediate_dir,
        cache_level=cache_level,
        model=model,
        workers=workers,
        max_side=max_side,
        device=device,
        env_overrides=env_overrides,
        cli_overrides=cli_overrides,
    )

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _drain(pipe: Any, buf: io.StringIO, dest: Any) -> None:
        for line in pipe:
            dest.write(line)
            dest.flush()
            buf.write(line)
        pipe.close()

    t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_buf, sys.stdout), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_buf, sys.stderr), daemon=True)
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()

    payload = {
        "cmd": [str(part) for part in cmd],
        "returncode": int(proc.returncode),
        "stdout_tail": stdout_buf.getvalue()[-4000:],
        "stderr_tail": stderr_buf.getvalue()[-4000:],
    }
    if proc.returncode != 0:
        return {"status": "failed", "process": payload}
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rows = load_jsonl(out_dir / "ocr_qa_dataset.jsonl")
    metrics = compute_run_quality(summary, rows)
    return {"status": "ok", "process": payload, "metrics": metrics, "summary": summary}


def _source_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"chartqa": 0, "textocr": 0, "coco_text": 0, "other": 0}
    for row in rows:
        image_id = str(row.get("image_id") or "")
        lowered = image_id.lower()
        if "chartqa" in lowered:
            counts["chartqa"] += 1
        elif "textocr" in lowered:
            counts["textocr"] += 1
        elif "coco" in lowered:
            counts["coco_text"] += 1
        else:
            counts["other"] += 1
    return counts


def _ita22_score(
    *,
    metrics: dict[str, Any],
    answer_distribution: dict[str, Any],
    anchor_coverage: dict[str, Any],
    image_dependence: dict[str, Any] | None,
) -> float:
    type_div = float(metrics.get("type_diversity") or 0.0)
    num_types = min(float(metrics.get("num_types_with_coverage") or 0.0) / 5.0, 1.0)
    answer_entropy = min(float(answer_distribution.get("answer_entropy") or 0.0) / 4.8, 1.0)
    top1_penalty = float(answer_distribution.get("top1_concentration") or 0.0)
    anchor_div = min(float(anchor_coverage.get("mean_unique_anchors") or 0.0) / 2.0, 1.0)
    vdep = float((image_dependence or {}).get("vision_necessary_rate") or 0.0)
    leak = float((image_dependence or {}).get("text_leaky_rate") or 0.0)
    score = (
        0.18 * type_div
        + 0.10 * num_types
        + 0.16 * answer_entropy
        + 0.10 * anchor_div
        + 0.30 * vdep
        - 0.26 * leak
        - 0.06 * top1_penalty
    )
    return round(score * 100.0, 2)


def _enrich_completed_result(
    *,
    out_dir: Path,
    result: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rows = load_jsonl(out_dir / "ocr_qa_dataset.jsonl")
    result["source_breakdown"] = _source_breakdown(rows)
    result["answer_distribution"] = compute_answer_distribution(rows)
    result["anchor_coverage"] = compute_anchor_coverage(rows)
    if not args.skip_image_dependence and rows:
        idep_dir = out_dir / "evals" / "image_dependence_ita22"
        result["image_dependence"] = run_image_dependence_eval(
            experiment_dir=out_dir,
            model=str(args.image_dependence_model),
            out_dir=idep_dir,
            workers=int(args.image_dependence_workers),
        )
    result["ita22_score"] = _ita22_score(
        metrics=result.get("metrics") or {},
        answer_distribution=result.get("answer_distribution") or {},
        anchor_coverage=result.get("anchor_coverage") or {},
        image_dependence=result.get("image_dependence"),
    )
    return result


def _render_report(payload: dict[str, Any]) -> str:
    report_path = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150" / f"{payload['bundle_id']}_report.md"
    log_path = REPO_ROOT / "logs" / payload["bundle_id"] / "run.log"
    primary = payload.get("primary_run") or {}
    lines = [
        f"# {payload['bundle_id']}",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Live report: [{payload['bundle_id']}_report.md]({report_path})",
        f"- Log: [run.log]({log_path})",
        "",
        "## Goal",
        "",
        "**ita22: keep vdep gains from stricter filtering, cut text leakage further, and recover type diversity via rescue instead of overfilling weak DIRECT_READ rows.**",
        "",
        "Pipeline changes exercised here:",
        "- REVERSE_GROUND is no longer a free branch: structural-anchor filter + semantic vdep check + color-scrubbed anchor phrases + cap=1.",
        "- TEXT_PROPERTY stays alive, but only as visual TP with one-per-image cap, location-first wording, and a high-prior answer reject on generic anchors.",
        "- DIRECT_READ no longer gets a large unconditional selector advantage; the primary pass runs at target=6 and rescue images are chosen from type-deficient outputs.",
        "",
        "## Primary Precision Pass",
        "",
        f"- Anchor backend: `gemma4_ollama:{payload['performance_profile']['gemma_model']}`",
        f"- RG vdep model: `{payload['performance_profile']['rg_vdep_model']}`",
        f"- Primary target per image: `{payload['performance_profile']['primary_target_per_image']}`",
        "",
        "| Status | Phase | Out dir | Accepted | Types | Type div | Ans H | VNec | Leak |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| `{str(primary.get('status') or 'pending')}` |"
            f" `{str(primary.get('phase') or '—')}` |"
            f" `{str(primary.get('out_dir') or '—')}` |"
            f" {int(((primary.get('metrics') or {}).get('accepted_qas') or 0))} |"
            f" {int(((primary.get('metrics') or {}).get('num_types_with_coverage') or 0))} |"
            f" {float(((primary.get('metrics') or {}).get('type_diversity') or 0.0)):.3f} |"
            f" {float(((primary.get('answer_distribution') or {}).get('answer_entropy') or 0.0)):.3f} |"
            f" {float(((primary.get('image_dependence') or {}).get('vision_necessary_rate') or 0.0))*100:.1f}% |"
            f" {float(((primary.get('image_dependence') or {}).get('text_leaky_rate') or 0.0))*100:.1f}% |"
        ),
        "",
        "## Variant Summary",
        "",
        "| Variant | Status | Rescue | Yes/No | Target | Types | Type div | Ans H | Top1 | VNec | Leak | DR | RG | YN | TP | AP | Score |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANT_ORDER:
        result = payload["results"].get(name) or {}
        status = result.get("status", "pending")
        if status in {"pending", "running", "image_dependence"}:
            lines.append(
                f"| `{name}` | `{status}` | {int((result.get('rescue_summary') or {}).get('selected_images') or 0)} |"
                f" {int(result.get('max_yesno_per_image') or 0)} | {int(result.get('target_per_image') or 0)} |"
                " — | — | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        if status == "failed":
            lines.append(
                f"| `{name}` | `failed` | {int((result.get('rescue_summary') or {}).get('selected_images') or 0)} |"
                f" {int(result.get('max_yesno_per_image') or 0)} | {int(result.get('target_per_image') or 0)} |"
                " — | — | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        metrics = result.get("metrics") or {}
        answer_distribution = result.get("answer_distribution") or {}
        image_dependence = result.get("image_dependence") or {}
        qtypes = metrics.get("question_types") or {}
        lines.append(
            f"| `{name}` | `{str(status)}` | {int((result.get('rescue_summary') or {}).get('selected_images') or 0)} |"
            f" {int(result.get('max_yesno_per_image') or 0)} | {int(result.get('target_per_image') or 0)} |"
            f" {int(metrics.get('num_types_with_coverage') or 0)} |"
            f" {float(metrics.get('type_diversity') or 0.0):.3f} |"
            f" {float(answer_distribution.get('answer_entropy') or 0.0):.3f} |"
            f" {float(answer_distribution.get('top1_concentration') or 0.0):.3f} |"
            f" {float(image_dependence.get('vision_necessary_rate') or 0.0)*100:.1f}% |"
            f" {float(image_dependence.get('text_leaky_rate') or 0.0)*100:.1f}% |"
            f" {int(qtypes.get('DIRECT_READ') or 0)} |"
            f" {int(qtypes.get('REVERSE_GROUND') or 0)} |"
            f" {int(qtypes.get('YES_NO') or 0)} |"
            f" {int(qtypes.get('TEXT_PROPERTY') or 0)} |"
            f" {int(qtypes.get('ANCHOR_PROPERTY') or 0)} |"
            f" {float(result.get('ita22_score') or 0.0):.2f} |"
        )
    lines += [
        "",
        "## Variant Notes",
        "",
    ]
    for name in VARIANT_ORDER:
        lines.append(f"- `{name}`: {VARIANT_DESCRIPTIONS[name]}")
    return "\n".join(lines) + "\n"


def _write_all(*, payload: dict[str, Any], report_json: Path, report_md: Path, live_doc: Path) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(_render_report(payload), encoding="utf-8")
    live_doc.write_text(_render_report(payload), encoding="utf-8")


def _common_env(args: argparse.Namespace) -> dict[str, str]:
    baseline_env, _ = _q01_baseline()
    return {
        **baseline_env,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
        "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "florence",
        "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "independent_raw",
        "SGOCR_ANCHOR_RELABEL_MODE": "none",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
        "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
        "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "basic",
        "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
        "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
        "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
        "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "0",
        "SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD": "0.92",
        "SGOCR_QWEN_ITA15_PROMPT_ENABLED": "1",
        "SGOCR_SAM3_REFINE_MODE": "none",
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED": "1",
        "SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": _TYPED_GATE_QT,
        "SGOCR_RG_STRUCTURAL_ANCHOR_FILTER_ENABLED": "1",
        "SGOCR_RG_LEAKY_LABEL_HARD_REJECT_ENABLED": "1",
        "SGOCR_RG_VDEP_CHECK_ENABLED": "1",
        "SGOCR_RG_VDEP_MODEL": str(args.rg_vdep_model),
        "SGOCR_RG_SCRUB_COLOR_ANCHOR_PHRASES_ENABLED": "1",
        "SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": "0.0",
        "SGOCR_RG_PER_IMAGE_HARD_CAP": "1",
        "SGOCR_TP_VISUAL_ONLY_ENABLED": "1",
        "SGOCR_TP_PER_IMAGE_HARD_CAP": "1",
        "SGOCR_TP_VISUAL_AVOID_TEXT_REFERENCE_WITH_SPECIFIC_LOCATION_ENABLED": "1",
        "SGOCR_TP_VISUAL_HIGH_PRIOR_ANSWER_FILTER_ENABLED": "1",
        "SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": str(float(args.property_candidate_selection_bonus)),
        "SGOCR_DIRECT_READ_SELECTION_BONUS": str(float(args.direct_read_selection_bonus)),
        "SGOCR_DR_GENERIC_ANCHOR_PENALTY": str(float(args.dr_generic_anchor_penalty)),
        "SGOCR_DR_SAME_ANCHOR_REPEAT_PENALTY": str(float(args.dr_same_anchor_repeat_penalty)),
        "SGOCR_PER_IMAGE_ANCHOR_DIVERSITY_BONUS": str(float(args.per_image_anchor_diversity_bonus)),
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "1",
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY": "0.18",
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
        "SGOCR_ANCHOR_PROPERTY_SPECIFIC_THRESHOLD": "3",
        "SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED": "1",
        "SGOCR_MAX_NEGATIVE_YESNO_PER_IMAGE": "0",
        "SGOCR_MAX_YESNO_PER_IMAGE": "0",
    }


def _build_variants(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    _, baseline_cli = _q01_baseline()
    common_cli = {
        **baseline_cli,
        "--grounding-threshold": str(float(args.grounding_threshold)),
        "--max-tags-per-image": str(int(args.max_tags_per_image)),
    }
    common_env = _common_env(args)
    primary_env = {
        **common_env,
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "gemma4_ollama",
        "SGOCR_GEMMA_OLLAMA_MODEL": str(args.gemma_model),
        "SGOCR_GEMMA_OLLAMA_BASE_URL": str(args.gemma_base_url),
        "SGOCR_GEMMA_OLLAMA_NUM_CTX": str(int(args.gemma_num_ctx)),
        "SGOCR_GEMMA_ITA16_PROMPT_VARIANT": "antidoc",
    }
    rescue_qwen_base = {
        **common_env,
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_MODEL": str(args.qwen_model),
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": str(float(args.qwen_gpu_memory_utilization)),
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": str(int(args.qwen_batch_size)),
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": str(int(args.qwen_max_model_len)),
        "SGOCR_QWEN_DAM01_PROMPT_ENABLED": "1",
    }
    primary_cli = {**common_cli, "--target-per-image": str(int(args.primary_target_per_image))}
    precision_final_env = dict(primary_env)
    apheavy_final_env = {
        **primary_env,
        "SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": "0.48",
        "SGOCR_PER_IMAGE_ANCHOR_DIVERSITY_BONUS": "0.20",
    }
    yesno1_final_env = {
        **primary_env,
        "SGOCR_MAX_YESNO_PER_IMAGE": "1",
    }
    return {
        "primary_gemma": {
            "env": primary_env,
            "cli": primary_cli,
        },
        "precision_r48_t6": {
            "note": VARIANT_DESCRIPTIONS["precision_r48_t6"],
            "rescue_config": RescueSelectionConfig(
                max_images=48,
                min_types_per_image=3,
                missing_anchor_property_weight=3.0,
                missing_reverse_ground_weight=0.9,
                missing_yesno_weight=0.1,
                missing_text_property_weight=0.9,
            ),
            "rescue_env": rescue_qwen_base,
            "rescue_cli": primary_cli,
            "final_env": precision_final_env,
            "final_cli": {**common_cli, "--target-per-image": "6"},
        },
        "precision_r72_t6": {
            "note": VARIANT_DESCRIPTIONS["precision_r72_t6"],
            "rescue_config": RescueSelectionConfig(
                max_images=72,
                min_types_per_image=3,
                missing_anchor_property_weight=3.0,
                missing_reverse_ground_weight=0.9,
                missing_yesno_weight=0.1,
                missing_text_property_weight=0.9,
            ),
            "rescue_env": rescue_qwen_base,
            "rescue_cli": primary_cli,
            "final_env": precision_final_env,
            "final_cli": {**common_cli, "--target-per-image": "6"},
        },
        "apheavy_r72_t6": {
            "note": VARIANT_DESCRIPTIONS["apheavy_r72_t6"],
            "rescue_config": RescueSelectionConfig(
                max_images=72,
                min_types_per_image=3,
                missing_anchor_property_weight=3.5,
                missing_reverse_ground_weight=0.8,
                missing_yesno_weight=0.1,
                missing_text_property_weight=1.0,
                target_answer_entropy=1.40,
                low_answer_entropy_weight=0.75,
            ),
            "rescue_env": rescue_qwen_base,
            "rescue_cli": primary_cli,
            "final_env": apheavy_final_env,
            "final_cli": {**common_cli, "--target-per-image": "6"},
        },
        "balanced_r72_t6_yesno1": {
            "note": VARIANT_DESCRIPTIONS["balanced_r72_t6_yesno1"],
            "rescue_config": RescueSelectionConfig(
                max_images=72,
                min_types_per_image=4,
                missing_anchor_property_weight=2.8,
                missing_reverse_ground_weight=0.8,
                missing_yesno_weight=0.6,
                missing_text_property_weight=0.8,
            ),
            "rescue_env": rescue_qwen_base,
            "rescue_cli": primary_cli,
            "final_env": yesno1_final_env,
            "final_cli": {**common_cli, "--target-per-image": "6"},
        },
    }


def main() -> None:
    args = parse_args()
    if str(args.model) != "gemini-2.5-flash":
        raise SystemExit("ita22 is fixed to teacher=model gemini-2.5-flash.")

    variants = _build_variants(args)
    active_variant_names = VARIANT_ORDER if not args.variants else [name for name in VARIANT_ORDER if name in set(args.variants)]
    if not active_variant_names:
        raise SystemExit("No ita22 variants selected.")
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150"
    source_dir = root_final / args.source_name
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    LIVE_DOC_DIR.mkdir(parents=True, exist_ok=True)
    live_doc = LIVE_DOC_DIR / LIVE_DOC_NAME

    now = datetime.now().isoformat(timespec="seconds")
    existing_payload = load_json(report_json)
    if existing_payload:
        payload = existing_payload
        payload["updated_at"] = now
        for name in active_variant_names:
            payload["results"].setdefault(name, {"status": "pending", "note": variants[name]["note"]})
    else:
        payload = {
            "bundle_id": args.bundle_id,
            "source_dir": str(source_dir),
            "source_manifest": source_manifest,
            "reference_report_json": str(root_final / args.reference_report_json),
            "reference_report": load_json(root_final / args.reference_report_json),
            "created_at": now,
            "updated_at": now,
            "performance_profile": {
                "workers": int(args.workers),
                "gemma_model": str(args.gemma_model),
                "gemma_num_ctx": int(args.gemma_num_ctx),
                "qwen_model": str(args.qwen_model),
                "qwen_gpu_memory_utilization": float(args.qwen_gpu_memory_utilization),
                "qwen_batch_size": int(args.qwen_batch_size),
                "qwen_max_model_len": int(args.qwen_max_model_len),
                "switch_min_free_mib": int(args.switch_min_free_mib),
                "switch_timeout_seconds": int(args.switch_timeout_seconds),
                "group_min_instances": int(args.group_min_instances),
                "spatial_min_centroid_offset": float(args.spatial_min_centroid_offset),
                "rg_vdep_model": str(args.rg_vdep_model),
                "primary_target_per_image": int(args.primary_target_per_image),
                "direct_read_selection_bonus": float(args.direct_read_selection_bonus),
                "dr_generic_anchor_penalty": float(args.dr_generic_anchor_penalty),
                "dr_same_anchor_repeat_penalty": float(args.dr_same_anchor_repeat_penalty),
                "property_candidate_selection_bonus": float(args.property_candidate_selection_bonus),
                "per_image_anchor_diversity_bonus": float(args.per_image_anchor_diversity_bonus),
                "image_dependence_model": str(args.image_dependence_model),
                "teacher_model": str(args.model),
            },
            "nemotron_diagnostic": nemotron_frontend_diagnostic(),
            "primary_run": {"status": "pending"},
            "results": {name: {"status": "pending", "note": variants[name]["note"]} for name in active_variant_names},
        }
    _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

    primary_out_dir = root_final / f"{args.bundle_id}_primary_gemma"
    primary_intermediate_dir = root_intermediate / f"{args.bundle_id}_primary_gemma"
    primary_cache_dir = Path(str(args.primary_ocr_cache_dir)) if str(args.primary_ocr_cache_dir).strip() else None
    primary_cache_level = "ocr" if primary_cache_dir and primary_cache_dir.exists() else "none"
    if payload.get("primary_run", {}).get("status") not in {"ok", "cached"}:
        payload["primary_run"] = {"status": "running", "phase": "primary_gemma"}
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)
        primary_result = _run_variant_streaming(
            source_dir=source_dir,
            out_dir=primary_out_dir,
            intermediate_dir=primary_intermediate_dir,
            cache_intermediate_dir=primary_cache_dir if primary_cache_level != "none" else None,
            cache_level=primary_cache_level,
            model=str(args.model),
            workers=int(args.workers),
            max_side=int(args.max_side),
            device=str(args.device),
            env_overrides=variants["primary_gemma"]["env"],
            cli_overrides=variants["primary_gemma"]["cli"],
        )
        primary_result["out_dir"] = str(primary_out_dir)
        primary_result["intermediate_dir"] = str(primary_intermediate_dir)
        primary_result["phase"] = "done"
        if primary_result.get("status") in {"ok", "cached"}:
            primary_result = _enrich_completed_result(out_dir=primary_out_dir, result=primary_result, args=args)
        payload["primary_run"] = primary_result
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

    primary_rows = load_jsonl(primary_out_dir / "ocr_qa_dataset.jsonl")
    primary_verified_tuples = load_jsonl(primary_intermediate_dir / "verified_tuples.jsonl")
    if not primary_rows or not primary_verified_tuples:
        raise SystemExit("ita22 primary Gemma run did not produce final rows and verified tuples.")

    for name in active_variant_names:
        if payload["results"].get(name, {}).get("status") in {"ok", "cached"}:
            print(f"[skip] {name} already completed — skipping", flush=True)
            continue
        spec = variants[name]
        max_yesno = int(spec["final_env"].get("SGOCR_MAX_YESNO_PER_IMAGE", "0"))
        payload["results"][name] = {
            "status": "running",
            "phase": "rescue_select",
            "note": spec["note"],
            "target_per_image": int(spec["final_cli"]["--target-per-image"]),
            "max_yesno_per_image": max_yesno,
        }
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

        rescue_ids, rescue_metadata = select_rescue_image_ids(
            final_rows=primary_rows,
            verified_tuples=primary_verified_tuples,
            config=spec["rescue_config"],
        )
        rescue_id_set = set(rescue_ids)
        rescue_summary = summarize_rescue_selection(rescue_metadata)
        subset_source_dir = root_final / f"{args.bundle_id}_{name}_rescue_source"
        write_source_subset(
            source_dir=source_dir,
            out_dir=subset_source_dir,
            image_ids=rescue_id_set,
            role="ita22 rescue subset",
            note=f"{name}: rescue top {len(rescue_ids)} type-deficient images from primary precision pass",
        )
        write_json(
            subset_source_dir / "ita22_selection.json",
            {
                "variant": name,
                "config": asdict(spec["rescue_config"]),
                "summary": rescue_summary,
                "images": rescue_metadata,
            },
        )
        payload["results"][name].update(
            {
                "phase": "subset_cache",
                "rescue_summary": rescue_summary,
                "rescue_config": asdict(spec["rescue_config"]),
                "rescue_image_ids": rescue_ids,
            }
        )
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

        subset_cache_dir = root_intermediate / f"{args.bundle_id}_{name}_rescue_ocr_cache"
        build_subset_ocr_cache(
            source_intermediate_dir=primary_intermediate_dir,
            out_dir=subset_cache_dir,
            image_ids=rescue_id_set,
        )

        rescue_out_dir = root_final / f"{args.bundle_id}_{name}_rescue_qwen"
        rescue_intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}_rescue_qwen"
        if rescue_ids:
            payload["results"][name]["phase"] = "qwen_rescue"
            _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)
            drain_ollama_model(
                str(args.gemma_model),
                min_free_mib=int(args.switch_min_free_mib),
                timeout_s=float(args.switch_timeout_seconds),
            )
            rescue_result = _run_variant_streaming(
                source_dir=subset_source_dir,
                out_dir=rescue_out_dir,
                intermediate_dir=rescue_intermediate_dir,
                cache_intermediate_dir=subset_cache_dir,
                cache_level="ocr",
                model=str(args.model),
                workers=int(args.workers),
                max_side=int(args.max_side),
                device=str(args.device),
                env_overrides=spec["rescue_env"],
                cli_overrides=spec["rescue_cli"],
            )
        else:
            rescue_result = {"status": "skipped", "note": "no rescue images selected"}

        if rescue_ids and rescue_result.get("status") not in {"ok", "cached"}:
            payload["results"][name] = {
                "status": "failed",
                "phase": "qwen_rescue",
                "note": spec["note"],
                "rescue_out_dir": str(rescue_out_dir),
                "rescue_image_ids": rescue_ids,
                "rescue_summary": rescue_summary,
                "rescue_config": asdict(spec["rescue_config"]),
                "target_per_image": int(spec["final_cli"]["--target-per-image"]),
                "max_yesno_per_image": max_yesno,
                "rescue_process": rescue_result.get("process"),
                "rescue_error": rescue_result.get("note") or "qwen rescue phase failed",
            }
            _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)
            continue

        merged_intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}_merged_verified"
        if merged_intermediate_dir.exists():
            shutil.rmtree(merged_intermediate_dir)
        payload["results"][name].update(
            {
                "phase": "merge_verified",
                "rescue_result_status": str(rescue_result.get("status") or "skipped"),
                "rescue_process": rescue_result.get("process"),
                "rescue_out_dir": str(rescue_out_dir),
                "rescue_intermediate_dir": str(rescue_intermediate_dir),
            }
        )
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)
        if rescue_ids and rescue_result.get("status") in {"ok", "cached"}:
            merge_intermediate_by_image(
                primary_intermediate_dir=primary_intermediate_dir,
                rescue_intermediate_dir=rescue_intermediate_dir,
                out_dir=merged_intermediate_dir,
                rescue_image_ids=rescue_id_set,
            )
        else:
            merge_intermediate_by_image(
                primary_intermediate_dir=primary_intermediate_dir,
                rescue_intermediate_dir=primary_intermediate_dir,
                out_dir=merged_intermediate_dir,
                rescue_image_ids=set(),
            )

        final_out_dir = root_final / f"{args.bundle_id}_{name}"
        final_intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}"
        payload["results"][name]["phase"] = "final_teacher"
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)
        final_result = _run_variant_streaming(
            source_dir=source_dir,
            out_dir=final_out_dir,
            intermediate_dir=final_intermediate_dir,
            cache_intermediate_dir=merged_intermediate_dir,
            cache_level="verified",
            model=str(args.model),
            workers=int(args.workers),
            max_side=int(args.max_side),
            device=str(args.device),
            env_overrides=spec["final_env"],
            cli_overrides=spec["final_cli"],
        )
        final_result["out_dir"] = str(final_out_dir)
        final_result["intermediate_dir"] = str(final_intermediate_dir)
        final_result["merged_intermediate_dir"] = str(merged_intermediate_dir)
        final_result["rescue_out_dir"] = str(rescue_out_dir)
        final_result["rescue_image_ids"] = rescue_ids
        final_result["rescue_summary"] = rescue_summary
        final_result["rescue_config"] = asdict(spec["rescue_config"])
        final_result["target_per_image"] = int(spec["final_cli"]["--target-per-image"])
        final_result["max_yesno_per_image"] = max_yesno
        final_result["note"] = spec["note"]
        final_result["phase"] = "done"
        if final_result.get("status") in {"ok", "cached"}:
            final_result = _enrich_completed_result(out_dir=final_out_dir, result=final_result, args=args)

        payload["results"][name] = final_result
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

    print(str(live_doc))


if __name__ == "__main__":
    main()
