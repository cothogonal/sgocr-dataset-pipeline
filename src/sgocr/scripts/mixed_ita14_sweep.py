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

# ita14 goal:
#   Stem from ita13_propbonus_offset20 (v0 winner). Two changes:
#
#   1. Qwen no-text-ref prompt (SGOCR_QWEN_NO_TEXT_REF_PROMPT_ENABLED=1):
#      Replaces the anti-OCR suffix with a stronger guardrail that forbids any
#      reference to or paraphrase of text visible on/inside the anchor, and adds
#      a disambiguation instruction for same-type anchor labels (e.g. "blue sign"
#      vs "red sign"). Targets the residual text-leakage path where anchor labels
#      describe visible text non-verbatim.
#
#   2. Target + RG cap relaxation:
#      target_per_image 5→8, rg_per_image_hard_cap 2→3.
#      Data shows rank-6 candidate quality (mean 0.415) is nearly indistinguishable
#      from rank-5 (mean 0.439); gap at the boundary is typically 0.001–0.020.
#      74/127 images have pool > 5 so the higher cap is binding for most images.
#
#   Cache strategy: OCR cache from ita13_propbonus_offset20 intermediate
#   (reuses text detections, re-runs Qwen with new prompt).

VARIANT_ORDER = ["ita14_no_text_ref"]

VARIANT_DESCRIPTIONS: dict[str, str] = {
    "ita14_no_text_ref": (
        "ita13_propbonus_offset20 base + no_text_ref prompt + target=8 + rg_cap=3"
    ),
}

# OCR cache: reuse text_detections from ita13_propbonus_offset20
_ITA13_PO_INTERMEDIATE = (
    str(OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150" /
        "sgocr_mixed_ita13_20260414_174736_ita13_propbonus_offset20")
)

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "168_sgocr_mixed_ita14_launch_2026-04-15.md"


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
    rg_rows = [r for r in rows if str(r.get("question_type") or "") == "REVERSE_GROUND"]
    corrected = sum(1 for r in rg_rows if r.get("rg_correction_applied"))
    groundback_failed = sum(1 for r in rg_rows if r.get("groundback_failed"))
    return {
        "rg_count": len(rg_rows),
        "rg_corrected": corrected,
        "rg_groundback_failed": groundback_failed,
    }


def _render_sweep_report(payload: dict[str, Any]) -> str:
    bundle_id = payload["bundle_id"]
    perf = payload.get("performance_profile", {})
    results = payload.get("results", {})
    report_path = str(OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150" / f"{bundle_id}_report.md")
    log_path = str(REPO_ROOT / "logs" / bundle_id / "run.log")

    lines = [
        f"# {bundle_id}",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{bundle_id}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Live report: [{bundle_id}_report.md]({report_path})",
        f"- Log: [run.log]({log_path})",
        "",
        "## Goal",
        "",
        "**Stem from ita13_propbonus_offset20 with stronger Qwen text-ref guardrail + relaxed selection caps.**",
        "",
        "Changes from ita13_propbonus_offset20:",
        "- `SGOCR_QWEN_NO_TEXT_REF_PROMPT_ENABLED=1`: forbids any text reference in anchor labels,",
        "  adds disambiguation instruction for same-type anchors.",
        "- `target_per_image=8` (was 5): rank-6 quality gap is only ~0.02 mean; 74/127 images have pool > 5.",
        "- `rg_per_image_hard_cap=3` (was 2): allows one more RG per image.",
        "",
        "## Engineering Changes",
        "",
        "| Location | Change |",
        "|---|---|",
        "| `qwen_anchor_vllm.py` | Added `_NO_TEXT_REF_SUFFIX`, `INDEPENDENT_QWEN_INVENTORY_PROMPT_NO_TEXT_REF`, `STRUCTURAL_FALLBACK_QWEN_PROMPT_NO_TEXT_REF` |",
        "| `semantic_dev40_tuning.py` | Added `qwen_no_text_ref_prompt_enabled` + `SGOCR_QWEN_NO_TEXT_REF_PROMPT_ENABLED` env-var |",
        "| `full_pipeline_dev40.py` | Wired no_text_ref prompt (takes priority over anti_ocr) in inventory + structural fallback paths |",
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

    return "\n".join(lines) + "\n"


def _render_live_doc(payload: dict[str, Any]) -> str:
    return _render_sweep_report(payload)


def _write_all(*, payload: dict[str, Any], report_json: Path, report_md: Path, live_doc: Path) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(_render_sweep_report(payload), encoding="utf-8")
    live_doc.write_text(_render_live_doc(payload), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="ita14: no_text_ref Qwen prompt + target=8 + rg_cap=3, stemming from ita13_propbonus_offset20."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita14_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita13_20260414_174736_report.json")
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
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.20)  # offset20 value
    ap.add_argument("--rg-candidate-oversample-boost", type=float, default=1.5)
    ap.add_argument("--rg-per-image-hard-cap", type=int, default=3)        # ita14: was 2
    ap.add_argument("--target-per-image", type=int, default=8)             # ita14: was 5
    ap.add_argument("--property-candidate-selection-bonus", type=float, default=0.5)
    ap.add_argument("--frontier-word-f1-floor", type=float, default=-1.0)
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview")
    ap.add_argument("--image-dependence-workers", type=int, default=4)
    ap.add_argument("--skip-image-dependence", action="store_true")
    return ap.parse_args()


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
        "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
        "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
        "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
        "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "0",
        "SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD": str(float(args.qwen_degenerate_threshold)),
        # ita14: no_text_ref replaces anti_ocr
        "SGOCR_QWEN_NO_TEXT_REF_PROMPT_ENABLED": "1",
        # SAM3 not installed in this environment
        "SGOCR_SAM3_REFINE_MODE": "none",
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED": "1",
        "SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": _TYPED_GATE_QT,
        "SGOCR_RG_LEAKY_LABEL_HARD_REJECT_ENABLED": "1",
        "SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": str(float(args.rg_candidate_oversample_boost)),
        "SGOCR_RG_PER_IMAGE_HARD_CAP": str(int(args.rg_per_image_hard_cap)),
        "SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": str(float(args.property_candidate_selection_bonus)),
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
    }

    common_cli = {
        **baseline_cli,
        "--target-per-image": str(int(args.target_per_image)),
    }

    _ita13_po_cache = Path(_ITA13_PO_INTERMEDIATE)

    variants: dict[str, dict[str, Any]] = {
        "ita14_no_text_ref": {
            "env": dict(common_env),
            "cli": dict(common_cli),
            "note": VARIANT_DESCRIPTIONS["ita14_no_text_ref"],
            "cache_dir": _ita13_po_cache,
            "cache_level": "ocr",   # reuse text_detections, re-run Qwen with new prompt
        },
    }

    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    LIVE_DOC_DIR.mkdir(parents=True, exist_ok=True)
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
                "rg_candidate_oversample_boost": float(args.rg_candidate_oversample_boost),
                "rg_per_image_hard_cap": int(args.rg_per_image_hard_cap),
                "target_per_image": int(args.target_per_image),
                "property_candidate_selection_bonus": float(args.property_candidate_selection_bonus),
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
            cli_overrides=spec["cli"],
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
                    idep_dir = out_dir / "evals" / "image_dependence_ita14"
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
