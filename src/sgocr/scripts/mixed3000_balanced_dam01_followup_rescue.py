from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from ..bootstrap import write_json
from ..dual_anchor import (
    RescueImageScore,
    build_subset_ocr_cache,
    drain_ollama_model,
    load_json,
    load_jsonl,
    merge_intermediate_by_image,
    score_rescue_images,
    summarize_rescue_selection,
    write_source_subset,
)
from ..paths import LOGS_ROOT, OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from .mixed3000_balanced_dam01_run import _enrich_completed_result, _run_variant_streaming
from .mixed_dam01_sweep import VARIANT_DESCRIPTIONS, _build_variants


TARGET_VARIANT = "balanced_dam01_r48"


def parse_args() -> argparse.Namespace:
    default_base_bundle = "sgocr_mixed3000_balanced_dam01_20260417_173000"
    default_suffix = "followup_next912"
    ap = argparse.ArgumentParser(
        description="Run an additive post-primary rescue follow-up from a completed mixed3000 balanced_dam01 bundle."
    )
    ap.add_argument("--base-bundle-id", default=default_base_bundle)
    ap.add_argument("--bundle-id", default=f"{default_base_bundle}_{default_suffix}")
    ap.add_argument("--variant", default=TARGET_VARIANT, choices=[TARGET_VARIANT])
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-tags-per-image", type=int, default=14)
    ap.add_argument("--grounding-threshold", type=float, default=0.28)
    ap.add_argument("--group-min-instances", type=int, default=3)
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.20)
    ap.add_argument("--rg-candidate-oversample-boost", type=float, default=1.5)
    ap.add_argument("--gemma-model", default="gemma4:e4b-it-q4_K_M")
    ap.add_argument("--gemma-base-url", default="http://localhost:11434")
    ap.add_argument("--gemma-num-ctx", type=int, default=4096)
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    ap.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--qwen-batch-size", type=int, default=3)
    ap.add_argument("--qwen-max-model-len", type=int, default=2048)
    ap.add_argument("--switch-min-free-mib", type=int, default=10000)
    ap.add_argument("--switch-timeout-seconds", type=int, default=45)
    ap.add_argument("--rank-offset", type=int, default=0)
    ap.add_argument("--max-images", type=int, default=912)
    ap.add_argument("--exclude-existing-rescue", action="store_true", default=True)
    ap.add_argument("--include-existing-rescue", action="store_false", dest="exclude_existing_rescue")
    ap.add_argument("--prepare-only", action="store_true")
    return ap.parse_args()


def _load_manifest_image_ids(path: Path) -> list[str]:
    payload = load_json(path) or {}
    if isinstance(payload.get("items"), list):
        return [
            str(item.get("image_id") or "").strip()
            for item in payload["items"]
            if str(item.get("image_id") or "").strip()
        ]
    return []


def _score_metadata(
    *,
    scored_rows: list[RescueImageScore],
    selected_ids: set[str],
    excluded_existing_ids: set[str],
) -> list[dict[str, Any]]:
    existing_selected = set()
    rows: list[dict[str, Any]] = []
    eligible_rank = 0
    for global_rank, row in enumerate(scored_rows, start=1):
        image_id = str(row.image_id)
        excluded_existing = image_id in excluded_existing_ids
        eligible_for_window = row.score > 0.0 and not excluded_existing
        if eligible_for_window:
            eligible_rank += 1
        selected_for_followup = image_id in selected_ids
        rows.append(
            {
                "image_id": image_id,
                "global_rank": global_rank,
                "eligible_rank": eligible_rank if eligible_for_window else None,
                "score": row.score,
                "final_rows": row.final_rows,
                "final_type_count": row.final_type_count,
                "final_type_counts": row.final_type_counts,
                "answer_entropy": row.answer_entropy,
                "unique_anchor_labels": row.unique_anchor_labels,
                "generic_anchor_ratio": row.generic_anchor_ratio,
                "missing_types": row.missing_types,
                "excluded_existing_rescue": excluded_existing,
                "selected_for_followup": selected_for_followup,
            }
        )
        if excluded_existing:
            existing_selected.add(image_id)
    return rows


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['bundle_id']}",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Base bundle: `{payload['base_bundle_id']}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Status: `{payload['status']}`",
        f"- Report json: `{payload['report_json']}`",
        f"- Log dir: `{payload['log_dir']}`",
        "",
        "## Purpose",
        "",
        "Run a rescue-only follow-up from the saved primary Gemma artifacts, excluding the already rescued base tranche by default.",
        "",
        "## Selection",
        "",
        f"- Variant: `{payload['variant']}`",
        f"- Rank offset: `{payload['rank_offset']}`",
        f"- Requested additional rescue images: `{payload['max_images']}`",
        f"- Exclude existing base rescue tranche: `{payload['exclude_existing_rescue']}`",
        f"- Existing base rescue images found: `{payload['existing_rescue_count']}`",
        f"- Selected follow-up rescue images: `{payload['selected_rescue_count']}`",
        f"- Selection json: `{payload['selection_json']}`",
        "",
        "## Outputs",
        "",
        f"- Follow-up rescue source: `{payload['rescue_source_dir']}`",
        f"- Follow-up rescue out dir: `{payload['rescue_out_dir']}`",
        f"- Follow-up final out dir: `{payload['final_out_dir']}`",
        f"- Merged verified cache: `{payload['merged_intermediate_dir']}`",
        "",
        "## Notes",
        "",
        f"- Base variant description: {VARIANT_DESCRIPTIONS[payload['variant']]}",
        "- This follow-up reuses the saved primary OCR and verified artifacts. It does not rerun Nemotron or primary Gemma.",
        "- When the base `r48` rescue exists, the follow-up final cache includes both the original rescue tranche and the new follow-up tranche.",
    ]
    return "\n".join(lines) + "\n"


def _write_status(payload: dict[str, Any], report_json: Path, report_md: Path) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(_render_report(payload), encoding="utf-8")


def _ensure_base_primary_ready(primary_out_dir: Path, primary_intermediate_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary_rows = load_jsonl(primary_out_dir / "ocr_qa_dataset.jsonl")
    primary_verified_tuples = load_jsonl(primary_intermediate_dir / "verified_tuples.jsonl")
    if not primary_rows or not primary_verified_tuples:
        raise SystemExit(
            "Base primary Gemma artifacts are not ready yet. Wait for the current bundle to finish primary_gemma first."
        )
    return primary_rows, primary_verified_tuples


def main() -> None:
    args = parse_args()
    if str(args.model) != "gemini-2.5-flash":
        raise SystemExit("This follow-up lane is fixed to teacher=model gemini-2.5-flash.")

    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev3000"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev3000"
    root_final.mkdir(parents=True, exist_ok=True)
    root_intermediate.mkdir(parents=True, exist_ok=True)

    base_bundle_id = str(args.base_bundle_id)
    bundle_id = str(args.bundle_id)
    report_json = root_final / f"{bundle_id}_report.json"
    report_md = root_final / f"{bundle_id}_report.md"
    log_dir = LOGS_ROOT / bundle_id
    log_dir.mkdir(parents=True, exist_ok=True)

    base_report_json = root_final / f"{base_bundle_id}_report.json"
    base_payload = load_json(base_report_json)
    if not base_payload:
        raise SystemExit(f"Missing base bundle report: {base_report_json}")

    source_dir = Path(str(base_payload.get("source_dir") or ""))
    if not source_dir.exists():
        raise SystemExit(f"Missing source_dir from base bundle: {source_dir}")

    primary_out_dir = Path(
        str(((base_payload.get("primary_run") or {}).get("out_dir")) or (root_final / f"{base_bundle_id}_primary_gemma"))
    )
    primary_intermediate_dir = Path(
        str(((base_payload.get("primary_run") or {}).get("intermediate_dir")) or (root_intermediate / f"{base_bundle_id}_primary_gemma"))
    )
    primary_rows, primary_verified_tuples = _ensure_base_primary_ready(primary_out_dir, primary_intermediate_dir)

    existing_rescue_source_dir = root_final / f"{base_bundle_id}_{args.variant}_rescue_source"
    existing_rescue_intermediate_dir = root_intermediate / f"{base_bundle_id}_{args.variant}_rescue_qwen"
    existing_rescue_ids = (
        set(_load_manifest_image_ids(existing_rescue_source_dir / "manifest.json"))
        if bool(args.exclude_existing_rescue) and (existing_rescue_source_dir / "manifest.json").exists()
        else set()
    )

    variants = _build_variants(args)
    variant_spec = variants[str(args.variant)]
    scored_rows = score_rescue_images(
        final_rows=primary_rows,
        verified_tuples=primary_verified_tuples,
        config=variant_spec["rescue_config"],
    )
    eligible_rows = [
        row for row in scored_rows
        if float(row.score) > 0.0 and str(row.image_id) not in existing_rescue_ids
    ]
    selected_rows = eligible_rows[int(args.rank_offset) : int(args.rank_offset) + int(args.max_images)]
    selected_ids = [str(row.image_id) for row in selected_rows]
    selected_id_set = set(selected_ids)
    selection_metadata = _score_metadata(
        scored_rows=scored_rows,
        selected_ids=selected_id_set,
        excluded_existing_ids=existing_rescue_ids,
    )
    selected_metadata = [row for row in selection_metadata if bool(row.get("selected_for_followup"))]
    selection_summary = summarize_rescue_selection(
        [
            {
                **row,
                "selected_for_rescue": bool(row.get("selected_for_followup")),
            }
            for row in selected_metadata
        ]
    )

    rescue_source_dir = root_final / f"{bundle_id}_rescue_source"
    selection_json = rescue_source_dir / "followup_selection.json"
    rescue_out_dir = root_final / f"{bundle_id}_rescue_qwen"
    rescue_intermediate_dir = root_intermediate / f"{bundle_id}_rescue_qwen"
    subset_cache_dir = root_intermediate / f"{bundle_id}_rescue_ocr_cache"
    base_merged_intermediate_dir = root_intermediate / f"{bundle_id}_base_merged_verified"
    merged_intermediate_dir = root_intermediate / f"{bundle_id}_merged_verified"
    final_out_dir = root_final / bundle_id
    final_intermediate_dir = root_intermediate / bundle_id

    payload = load_json(report_json) or {
        "bundle_id": bundle_id,
        "base_bundle_id": base_bundle_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "prepared" if args.prepare_only else "running",
        "variant": str(args.variant),
        "rank_offset": int(args.rank_offset),
        "max_images": int(args.max_images),
        "exclude_existing_rescue": bool(args.exclude_existing_rescue),
        "existing_rescue_count": len(existing_rescue_ids),
        "selected_rescue_count": len(selected_ids),
        "source_dir": str(source_dir),
        "primary_out_dir": str(primary_out_dir),
        "primary_intermediate_dir": str(primary_intermediate_dir),
        "rescue_source_dir": str(rescue_source_dir),
        "rescue_out_dir": str(rescue_out_dir),
        "merged_intermediate_dir": str(merged_intermediate_dir),
        "final_out_dir": str(final_out_dir),
        "selection_json": str(selection_json),
        "report_json": str(report_json),
        "log_dir": str(log_dir),
        "selection_summary": selection_summary,
        "phase": "selection",
    }
    payload["status"] = "prepared" if args.prepare_only else str(payload.get("status") or "running")
    payload["existing_rescue_count"] = len(existing_rescue_ids)
    payload["selected_rescue_count"] = len(selected_ids)
    payload["selection_summary"] = selection_summary
    _write_status(payload, report_json, report_md)

    rescue_source_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        selection_json,
        {
            "bundle_id": bundle_id,
            "base_bundle_id": base_bundle_id,
            "variant": str(args.variant),
            "rank_offset": int(args.rank_offset),
            "max_images": int(args.max_images),
            "exclude_existing_rescue": bool(args.exclude_existing_rescue),
            "existing_rescue_image_ids": sorted(existing_rescue_ids),
            "selected_rescue_image_ids": selected_ids,
            "summary": selection_summary,
            "ranking": selection_metadata,
        },
    )

    if args.prepare_only:
        return
    if not selected_ids:
        raise SystemExit("No follow-up rescue images were selected.")

    write_source_subset(
        source_dir=source_dir,
        out_dir=rescue_source_dir,
        image_ids=selected_id_set,
        role="balanced_dam01 follow-up rescue subset",
        note=(
            f"{args.variant}: follow-up rescue after excluding {len(existing_rescue_ids)} existing rescue images; "
            f"taking {len(selected_ids)} images starting at offset {int(args.rank_offset)} in the remaining deficit ranking"
        ),
    )
    build_subset_ocr_cache(
        source_intermediate_dir=primary_intermediate_dir,
        out_dir=subset_cache_dir,
        image_ids=selected_id_set,
    )

    payload["phase"] = "qwen_rescue"
    _write_status(payload, report_json, report_md)
    drain_ollama_model(
        str(args.gemma_model),
        min_free_mib=int(args.switch_min_free_mib),
        timeout_s=float(args.switch_timeout_seconds),
    )
    rescue_result = _run_variant_streaming(
        source_dir=rescue_source_dir,
        out_dir=rescue_out_dir,
        intermediate_dir=rescue_intermediate_dir,
        cache_intermediate_dir=subset_cache_dir,
        cache_level="ocr",
        model=str(args.model),
        workers=int(args.workers),
        max_side=int(args.max_side),
        device=str(args.device),
        env_overrides=variant_spec["rescue_env"],
        cli_overrides=variant_spec["rescue_cli"],
    )
    rescue_result["out_dir"] = str(rescue_out_dir)
    rescue_result["intermediate_dir"] = str(rescue_intermediate_dir)
    payload["rescue_result"] = rescue_result
    _write_status(payload, report_json, report_md)
    if rescue_result.get("status") not in {"ok", "cached"}:
        payload["status"] = "failed"
        _write_status(payload, report_json, report_md)
        return

    if base_merged_intermediate_dir.exists():
        shutil.rmtree(base_merged_intermediate_dir)
    if existing_rescue_ids and (existing_rescue_intermediate_dir / "verified_tuples.jsonl").exists():
        merge_intermediate_by_image(
            primary_intermediate_dir=primary_intermediate_dir,
            rescue_intermediate_dir=existing_rescue_intermediate_dir,
            out_dir=base_merged_intermediate_dir,
            rescue_image_ids=existing_rescue_ids,
        )
    else:
        merge_intermediate_by_image(
            primary_intermediate_dir=primary_intermediate_dir,
            rescue_intermediate_dir=primary_intermediate_dir,
            out_dir=base_merged_intermediate_dir,
            rescue_image_ids=set(),
        )

    if merged_intermediate_dir.exists():
        shutil.rmtree(merged_intermediate_dir)
    merge_intermediate_by_image(
        primary_intermediate_dir=base_merged_intermediate_dir,
        rescue_intermediate_dir=rescue_intermediate_dir,
        out_dir=merged_intermediate_dir,
        rescue_image_ids=selected_id_set,
    )

    payload["phase"] = "final_teacher"
    _write_status(payload, report_json, report_md)
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
        env_overrides=variant_spec["final_env"],
        cli_overrides=variant_spec["final_cli"],
    )
    final_result["out_dir"] = str(final_out_dir)
    final_result["intermediate_dir"] = str(final_intermediate_dir)
    final_result["merged_intermediate_dir"] = str(merged_intermediate_dir)
    final_result["note"] = (
        f"{VARIANT_DESCRIPTIONS[str(args.variant)]} Follow-up rescue adds {len(selected_ids)} more low-coverage images."
    )
    if final_result.get("status") in {"ok", "cached"}:
        final_result = _enrich_completed_result(final_out_dir, final_result)
    payload["final_result"] = final_result
    payload["status"] = "ok" if final_result.get("status") in {"ok", "cached"} else str(final_result.get("status") or "failed")
    payload["phase"] = "done"
    _write_status(payload, report_json, report_md)


if __name__ == "__main__":
    main()
