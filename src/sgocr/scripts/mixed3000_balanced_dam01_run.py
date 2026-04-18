from __future__ import annotations

import argparse
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..bootstrap import write_json, write_jsonl
from ..dual_anchor import (
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
from ..paths import LOGS_ROOT, OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT, SRC_ROOT
from ..run_quality import compute_anchor_coverage, compute_answer_distribution, compute_run_quality
from .production_config import VARIANT_DESCRIPTIONS, build_variant_cmd, build_variants


DEFAULT_PRIOR_SOURCE_MANIFEST = (
    OCR_SPATIAL_QA_FINAL_ROOT
    / "mixed_dev150"
    / "chartqa50_textocr50_cocotext50_source_20260410_110952"
    / "manifest.json"
)
TARGET_VARIANT = "balanced_dam01_r48"


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="Prepare or run a resumable 3k-image balanced_dam01_r48 production lane without evals."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed3000_balanced_dam01_{stamp}")
    ap.add_argument("--source-name", default=f"chartqa1000_textocr1000_cocotext1000_source_{stamp}")
    ap.add_argument("--chartqa-count", type=int, default=1000)
    ap.add_argument("--textocr-count", type=int, default=1000)
    ap.add_argument("--coco-text-count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-manifest", action="append", dest="exclude_manifests", default=[])
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-tags-per-image", type=int, default=14)
    ap.add_argument("--grounding-threshold", type=float, default=0.28)
    ap.add_argument("--group-min-instances", type=int, default=3)
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.20)
    ap.add_argument("--rg-candidate-oversample-boost", type=float, default=1.5)
    ap.add_argument("--primary-ocr-cache-dir", default="")
    ap.add_argument("--gemma-model", default="gemma4:e4b-it-q4_K_M")
    ap.add_argument("--gemma-base-url", default="http://localhost:11434")
    ap.add_argument("--gemma-num-ctx", type=int, default=4096)
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    ap.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--qwen-batch-size", type=int, default=3)
    ap.add_argument("--qwen-max-model-len", type=int, default=2048)
    ap.add_argument("--switch-min-free-mib", type=int, default=10000)
    ap.add_argument("--switch-timeout-seconds", type=int, default=45)
    return ap.parse_args()


def _sample_chartqa_candidates() -> list[dict[str, str]]:
    chartqa_root = REPO_ROOT / "data" / "vm_ssl" / "raw" / "chartqa" / "images"
    out: list[dict[str, str]] = []
    for path in sorted(chartqa_root.glob("*/*.png")):
        image_id = f"chartqa:shared:{path.stem}"
        out.append(
            {
                "image_id": image_id,
                "image_path": str(path.relative_to(REPO_ROOT)),
                "dataset_source": "chartqa_train",
            }
        )
    return out


def _sample_coco_text_candidates() -> list[dict[str, str]]:
    coco_root = REPO_ROOT / "data" / "vm_ssl" / "raw" / "coco_text_materialized" / "train"
    out: list[dict[str, str]] = []
    for path in sorted(coco_root.glob("COCO_train2014_*.jpg")):
        numeric = str(int(path.stem.split("_")[-1]))
        image_id = f"coco_text:train:{numeric}"
        out.append(
            {
                "image_id": image_id,
                "image_path": str(path.relative_to(REPO_ROOT)),
                "dataset_source": "coco_text_train",
            }
        )
    return out


def _sample_textocr_train_candidates() -> list[dict[str, str]]:
    raw_json = REPO_ROOT / "data" / "vm_ssl" / "raw" / "textocr_full" / "TextOCR_0.1_train.json"
    images_root = REPO_ROOT / "data" / "vm_ssl" / "raw" / "textocr_trainval"
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for image_id, meta in payload.get("imgs", {}).items():
        filename = Path(str(meta.get("file_name") or "")).name
        if not filename:
            continue
        image_path = images_root / filename
        if not image_path.exists():
            continue
        out.append(
            {
                "image_id": f"textocr:train:{image_id}",
                "image_path": str(image_path.relative_to(REPO_ROOT)),
                "dataset_source": "textocr_train",
            }
        )
    return sorted(out, key=lambda row: row["image_id"])


def _coerce_image_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = load_jsonl(path)
        return rows, {"name": path.name, "kind": "jsonl"}

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("items"), list):
        rows = [
            {
                "image_id": str(item.get("image_id") or ""),
                "dataset_source": str(item.get("dataset_source") or ""),
                "origin": str(payload.get("name") or path.name),
            }
            for item in payload["items"]
            if str(item.get("image_id") or "")
        ]
        return rows, {"name": str(payload.get("name") or path.name), "kind": "source_manifest"}
    if isinstance(payload.get("image_ids"), list):
        rows = [{"image_id": str(image_id), "origin": str(payload.get("name") or path.name)} for image_id in payload["image_ids"] if str(image_id)]
        return rows, {"name": str(payload.get("name") or path.name), "kind": "exclude_manifest"}
    image_ids_path = payload.get("image_ids_path")
    if image_ids_path:
        nested = Path(str(image_ids_path))
        if not nested.is_absolute():
            nested = (path.parent / nested).resolve()
        rows = load_jsonl(nested)
        return rows, {"name": str(payload.get("name") or path.name), "kind": "exclude_manifest", "image_ids_path": str(nested)}
    raise ValueError(f"Unsupported exclude manifest format: {path}")


def _load_excluded_images(paths: list[Path]) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    excluded_ids: set[str] = set()
    manifests: list[dict[str, Any]] = []
    rows_out: list[dict[str, Any]] = []
    for manifest_path in paths:
        rows, meta = _coerce_image_rows(manifest_path)
        count_before = len(excluded_ids)
        for row in rows:
            image_id = str(row.get("image_id") or "")
            if not image_id:
                continue
            excluded_ids.add(image_id)
            rows_out.append(
                {
                    "image_id": image_id,
                    "dataset_source": str(row.get("dataset_source") or ""),
                    "origin_manifest": str(manifest_path),
                    "origin_name": str(meta.get("name") or manifest_path.name),
                }
            )
        manifests.append(
            {
                "path": str(manifest_path),
                "name": str(meta.get("name") or manifest_path.name),
                "kind": str(meta.get("kind") or "unknown"),
                "image_ids_added": len(excluded_ids) - count_before,
            }
        )
    dedup_rows = {}
    for row in rows_out:
        dedup_rows.setdefault(str(row["image_id"]), row)
    return excluded_ids, manifests, list(dedup_rows.values())


def _select_rows(
    *,
    candidates: list[dict[str, str]],
    count: int,
    seed: int,
    excluded_ids: set[str],
) -> tuple[list[dict[str, str]], int, int]:
    total = len(candidates)
    available = [row for row in candidates if str(row["image_id"]) not in excluded_ids]
    if len(available) < count:
        raise SystemExit(
            f"Not enough available images after exclusion: need {count}, have {len(available)}"
        )
    rng = random.Random(seed)
    selected = rng.sample(available, count)
    return sorted(selected, key=lambda row: row["image_id"]), total, total - len(available)


def build_source_universe_with_exclusions(
    *,
    out_dir: Path,
    chartqa_count: int,
    textocr_count: int,
    coco_text_count: int,
    seed: int,
    exclude_manifest_paths: list[Path],
) -> dict[str, Any]:
    excluded_ids, exclude_manifests, excluded_rows = _load_excluded_images(exclude_manifest_paths)

    chartqa_rows, chartqa_total, chartqa_excluded = _select_rows(
        candidates=_sample_chartqa_candidates(),
        count=chartqa_count,
        seed=seed + 101,
        excluded_ids=excluded_ids,
    )
    textocr_rows, textocr_total, textocr_excluded = _select_rows(
        candidates=_sample_textocr_train_candidates(),
        count=textocr_count,
        seed=seed + 202,
        excluded_ids=excluded_ids,
    )
    coco_rows, coco_total, coco_excluded = _select_rows(
        candidates=_sample_coco_text_candidates(),
        count=coco_text_count,
        seed=seed + 303,
        excluded_ids=excluded_ids,
    )

    rows = sorted(
        chartqa_rows + textocr_rows + coco_rows,
        key=lambda row: (row["dataset_source"], row["image_id"]),
    )
    raw_rows = [
        {
            "image_id": row["image_id"],
            "image_path": row["image_path"],
            "tuple": {
                "image_id": row["image_id"],
                "image_path": row["image_path"],
                "dataset_source": row["dataset_source"],
            },
        }
        for row in rows
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "raw_results.jsonl", raw_rows)
    write_jsonl(out_dir / "bootstrap_tuples.jsonl", [])
    write_jsonl(out_dir / "excluded_image_ids.jsonl", excluded_rows)

    current_source_rows = [
        {
            "image_id": row["image_id"],
            "dataset_source": row["dataset_source"],
            "origin_manifest": str(out_dir / "manifest.json"),
            "origin_name": out_dir.name,
        }
        for row in rows
    ]
    next_exclude_rows = {str(row["image_id"]): dict(row) for row in excluded_rows}
    for row in current_source_rows:
        next_exclude_rows[str(row["image_id"])] = row
    write_jsonl(out_dir / "next_run_exclude_image_ids.jsonl", list(next_exclude_rows.values()))

    manifest = {
        "name": out_dir.name,
        "image_count": len(rows),
        "seed": int(seed),
        "source_counts": {
            "chartqa_train": int(chartqa_count),
            "textocr_train": int(textocr_count),
            "coco_text_train": int(coco_text_count),
        },
        "pool_counts": {
            "chartqa_train_total": int(chartqa_total),
            "chartqa_train_excluded": int(chartqa_excluded),
            "textocr_train_total": int(textocr_total),
            "textocr_train_excluded": int(textocr_excluded),
            "coco_text_train_total": int(coco_total),
            "coco_text_train_excluded": int(coco_excluded),
        },
        "selection_policy": (
            "uniform random sample per source dataset on the current local train-image pool, "
            "excluding images listed in the configured prior processed-image manifests"
        ),
        "exclude_manifests": exclude_manifests,
        "items": rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(
        out_dir / "next_run_exclude_manifest.json",
        {
            "name": f"{out_dir.name}_next_run_exclude",
            "bundle_source_name": out_dir.name,
            "image_ids_count": len(next_exclude_rows),
            "image_ids_path": str(out_dir / "next_run_exclude_image_ids.jsonl"),
            "source_manifest_path": str(out_dir / "manifest.json"),
            "parent_exclude_manifests": exclude_manifests,
            "selection_policy": "union of all prior processed-image manifests plus the current 3k source universe",
        },
    )
    notes = "\n".join(
        [
            f"# {out_dir.name}",
            "",
            "- role: `source universe for 3k balanced_dam01_r48 production run`",
            f"- chartqa_train images: `{chartqa_count}`",
            f"- textocr_train images: `{textocr_count}`",
            f"- coco_text_train images: `{coco_text_count}`",
            f"- seed: `{seed}`",
            f"- prior excluded image ids: `{len(excluded_ids)}`",
            f"- next-run exclude manifest: `{out_dir / 'next_run_exclude_manifest.json'}`",
        ]
    ) + "\n"
    (out_dir / "notes.md").write_text(notes, encoding="utf-8")
    return manifest


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

    cmd, env = build_variant_cmd(
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
        image_id = str(row.get("image_id") or "").lower()
        if "chartqa" in image_id:
            counts["chartqa"] += 1
        elif "textocr" in image_id:
            counts["textocr"] += 1
        elif "coco" in image_id:
            counts["coco_text"] += 1
        else:
            counts["other"] += 1
    return counts


def _enrich_completed_result(out_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    rows = load_jsonl(out_dir / "ocr_qa_dataset.jsonl")
    result["source_breakdown"] = _source_breakdown(rows)
    result["answer_distribution"] = compute_answer_distribution(rows)
    result["anchor_coverage"] = compute_anchor_coverage(rows)
    result["artifact_counts"] = {
        "accepted_rows": len(rows),
        "rejected_rows": len(load_jsonl(out_dir / "rejected_dataset.jsonl")),
        "raw_rows": len(load_jsonl(out_dir / "raw_results.jsonl")),
    }
    return result


def _prepare_primary_resume_ocr_cache(
    *,
    source_dir: Path,
    partial_intermediate_dir: Path,
    out_dir: Path,
) -> Path | None:
    required = (
        "runtime_models.json",
        "text_nodes.jsonl",
        "text_detections.jsonl",
    )
    if not all((partial_intermediate_dir / filename).exists() for filename in required):
        return None

    source_manifest = load_json(source_dir / "manifest.json") or {}
    image_ids = {
        str(item.get("image_id") or "").strip()
        for item in list(source_manifest.get("items") or [])
        if str(item.get("image_id") or "").strip()
    }
    if not image_ids:
        image_ids = {
            str((row.get("tuple") or {}).get("image_id") or row.get("image_id") or "").strip()
            for row in load_jsonl(source_dir / "raw_results.jsonl")
            if str((row.get("tuple") or {}).get("image_id") or row.get("image_id") or "").strip()
        }
    if not image_ids:
        return None

    build_subset_ocr_cache(
        source_intermediate_dir=partial_intermediate_dir,
        out_dir=out_dir,
        image_ids=image_ids,
    )
    return out_dir


def _render_report(payload: dict[str, Any]) -> str:
    primary = payload.get("primary_run") or {}
    result = payload.get("result") or {}
    lines = [
        f"# {payload['bundle_id']}",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Status: `{payload['status']}`",
        f"- Live report: [{Path(payload['report_path']).name}]({payload['report_path']})",
        f"- Log dir: `{payload['log_dir']}`",
        "",
        "## Goal",
        "",
        "Prepare and run a fresh `3000`-image additive build using only the core parts of the current "
        "`balanced_dam01_r48` champion pipeline: primary Gemma pass, targeted Qwen rescue subset, and final "
        "Gemini teacher pass. No eval stages are included in this bundle.",
        "",
        "## Source Universe",
        "",
        f"- Source dir: `{payload['source_dir']}`",
        f"- Source manifest: `{payload['source_manifest_path']}`",
        f"- Total images: `{int((payload.get('source_manifest') or {}).get('image_count') or 0)}`",
        f"- Prior processed-image manifests: `{len(payload.get('exclude_manifests') or [])}`",
        f"- Next-run exclude manifest: `{payload['next_run_exclude_manifest_path']}`",
        f"- Next-run exclude ids: `{payload['next_run_exclude_image_ids_path']}`",
        "",
        "## Reproduction",
        "",
        f"- Command: `{payload['launch_command']}`",
        "",
        "## Phase Status",
        "",
        "| Phase | Status | Out dir | Accepted | Rejected |",
        "|---|---|---|---:|---:|",
        (
            f"| `primary_gemma` | `{str(primary.get('status') or 'pending')}` |"
            f" `{str(primary.get('out_dir') or '—')}` |"
            f" {int(((primary.get('artifact_counts') or {}).get('accepted_rows') or 0))} |"
            f" {int(((primary.get('artifact_counts') or {}).get('rejected_rows') or 0))} |"
        ),
        (
            f"| `{TARGET_VARIANT}` | `{str(result.get('status') or 'pending')}` |"
            f" `{str(result.get('out_dir') or '—')}` |"
            f" {int(((result.get('artifact_counts') or {}).get('accepted_rows') or 0))} |"
            f" {int(((result.get('artifact_counts') or {}).get('rejected_rows') or 0))} |"
        ),
        "",
        "## Notes",
        "",
        f"- Variant recipe: {VARIANT_DESCRIPTIONS[TARGET_VARIANT]}",
        "- `rejected_dataset.jsonl` is expected in every completed dataset output dir for later rescue attempts.",
        "- The bundle is resumable at major phase boundaries because source, subset, merged-cache, and final outputs are all persisted under stable bundle-specific paths.",
    ]
    return "\n".join(lines) + "\n"


def _write_status(payload: dict[str, Any], report_json: Path, report_md: Path) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(_render_report(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if str(args.model) != "gemini-2.5-flash":
        raise SystemExit("This production lane is fixed to teacher=model gemini-2.5-flash.")

    bundle_id = str(args.bundle_id)
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev3000"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev3000"
    root_final.mkdir(parents=True, exist_ok=True)
    root_intermediate.mkdir(parents=True, exist_ok=True)

    log_dir = LOGS_ROOT / bundle_id
    log_dir.mkdir(parents=True, exist_ok=True)

    source_dir = root_final / str(args.source_name)
    report_json = root_final / f"{bundle_id}_report.json"
    report_md = root_final / f"{bundle_id}_report.md"

    existing_payload = load_json(report_json)
    exclude_manifest_paths = [Path(p).resolve() for p in args.exclude_manifests]
    if not exclude_manifest_paths and DEFAULT_PRIOR_SOURCE_MANIFEST.exists():
        exclude_manifest_paths = [DEFAULT_PRIOR_SOURCE_MANIFEST.resolve()]

    if not source_dir.exists():
        source_manifest = build_source_universe_with_exclusions(
            out_dir=source_dir,
            chartqa_count=int(args.chartqa_count),
            textocr_count=int(args.textocr_count),
            coco_text_count=int(args.coco_text_count),
            seed=int(args.seed),
            exclude_manifest_paths=exclude_manifest_paths,
        )
    else:
        source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))

    variants = build_variants(args)
    variant_spec = variants[TARGET_VARIANT]
    primary_spec = variants["primary_gemma"]

    if existing_payload:
        payload = existing_payload
        payload["status"] = "prepared" if args.prepare_only else str(payload.get("status") or "running")
    else:
        payload = {
            "bundle_id": bundle_id,
            "status": "prepared" if args.prepare_only else "launching",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "report_path": str(report_md),
            "log_dir": str(log_dir),
            "source_dir": str(source_dir),
            "source_manifest_path": str(source_dir / "manifest.json"),
            "source_manifest": source_manifest,
            "exclude_manifests": [str(path) for path in exclude_manifest_paths],
            "next_run_exclude_manifest_path": str(source_dir / "next_run_exclude_manifest.json"),
            "next_run_exclude_image_ids_path": str(source_dir / "next_run_exclude_image_ids.jsonl"),
            "launch_command": (
                f"cd {REPO_ROOT} && PYTHONPATH={SRC_ROOT} "
                f"{sys.executable} -m sgocr.scripts.mixed3000_balanced_dam01_run "
                f"--bundle-id {bundle_id} --source-name {source_dir.name}"
            ),
            "nemotron_diagnostic": nemotron_frontend_diagnostic(),
            "performance_profile": {
                "workers": int(args.workers),
                "max_side": int(args.max_side),
                "device": str(args.device),
                "gemma_model": str(args.gemma_model),
                "gemma_num_ctx": int(args.gemma_num_ctx),
                "qwen_model": str(args.qwen_model),
                "qwen_gpu_memory_utilization": float(args.qwen_gpu_memory_utilization),
                "qwen_batch_size": int(args.qwen_batch_size),
                "qwen_max_model_len": int(args.qwen_max_model_len),
                "switch_min_free_mib": int(args.switch_min_free_mib),
                "switch_timeout_seconds": int(args.switch_timeout_seconds),
                "teacher_model": str(args.model),
                "variant": TARGET_VARIANT,
                "no_evals": True,
            },
            "primary_run": {"status": "pending", "note": "primary Gemma stage"},
            "result": {"status": "pending", "note": variant_spec["note"]},
        }
    _write_status(payload, report_json, report_md)

    if args.prepare_only:
        return

    payload["status"] = "running"
    _write_status(payload, report_json, report_md)

    primary_out_dir = root_final / f"{bundle_id}_primary_gemma"
    primary_intermediate_dir = root_intermediate / f"{bundle_id}_primary_gemma"
    primary_cache_dir = Path(str(args.primary_ocr_cache_dir)).resolve() if str(args.primary_ocr_cache_dir).strip() else None
    primary_cache_level = "ocr" if primary_cache_dir and primary_cache_dir.exists() else "none"
    if primary_cache_level == "none" and not (primary_out_dir / "summary.json").exists():
        inferred_primary_cache_dir = _prepare_primary_resume_ocr_cache(
            source_dir=source_dir,
            partial_intermediate_dir=primary_intermediate_dir,
            out_dir=root_intermediate / f"{bundle_id}_primary_gemma_resume_ocr_cache",
        )
        if inferred_primary_cache_dir is not None:
            primary_cache_dir = inferred_primary_cache_dir
            primary_cache_level = "ocr"

    if payload.get("primary_run", {}).get("status") not in {"ok", "cached"}:
        payload["primary_run"] = {"status": "running", "phase": "primary_gemma"}
        _write_status(payload, report_json, report_md)
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
            env_overrides=primary_spec["env"],
            cli_overrides=primary_spec["cli"],
        )
        primary_result["out_dir"] = str(primary_out_dir)
        primary_result["intermediate_dir"] = str(primary_intermediate_dir)
        primary_result["phase"] = "done"
        if primary_result.get("status") in {"ok", "cached"}:
            primary_result = _enrich_completed_result(primary_out_dir, primary_result)
        payload["primary_run"] = primary_result
        _write_status(payload, report_json, report_md)

    primary_rows = load_jsonl(primary_out_dir / "ocr_qa_dataset.jsonl")
    primary_verified_tuples = load_jsonl(primary_intermediate_dir / "verified_tuples.jsonl")
    if not primary_rows or not primary_verified_tuples:
        raise SystemExit("Primary Gemma phase did not produce final rows and verified tuples.")

    if payload.get("result", {}).get("status") in {"ok", "cached"}:
        payload["status"] = "ok"
        _write_status(payload, report_json, report_md)
        return

    payload["result"] = {
        "status": "running",
        "phase": "rescue_select",
        "note": variant_spec["note"],
        "qwen_prompt_mode": variant_spec["qwen_prompt_mode"],
        "target_per_image": int(variant_spec["final_cli"]["--target-per-image"]),
    }
    _write_status(payload, report_json, report_md)

    rescue_ids, rescue_metadata = select_rescue_image_ids(
        final_rows=primary_rows,
        verified_tuples=primary_verified_tuples,
        config=variant_spec["rescue_config"],
    )
    rescue_id_set = set(rescue_ids)
    rescue_summary = summarize_rescue_selection(rescue_metadata)
    subset_source_dir = root_final / f"{bundle_id}_{TARGET_VARIANT}_rescue_source"
    write_source_subset(
        source_dir=source_dir,
        out_dir=subset_source_dir,
        image_ids=rescue_id_set,
        role="balanced_dam01_r48 rescue subset",
        note=f"{TARGET_VARIANT}: rescue top {len(rescue_ids)} images from primary deficits",
    )
    write_json(
        subset_source_dir / "balanced_dam01_r48_selection.json",
        {
            "variant": TARGET_VARIANT,
            "summary": rescue_summary,
            "images": rescue_metadata,
        },
    )
    subset_cache_dir = root_intermediate / f"{bundle_id}_{TARGET_VARIANT}_rescue_ocr_cache"
    build_subset_ocr_cache(
        source_intermediate_dir=primary_intermediate_dir,
        out_dir=subset_cache_dir,
        image_ids=rescue_id_set,
    )

    rescue_out_dir = root_final / f"{bundle_id}_{TARGET_VARIANT}_rescue_qwen"
    rescue_intermediate_dir = root_intermediate / f"{bundle_id}_{TARGET_VARIANT}_rescue_qwen"
    if rescue_ids:
        payload["result"].update(
            {
                "phase": "qwen_rescue",
                "rescue_summary": rescue_summary,
                "rescue_image_ids": rescue_ids,
                "rescue_source_dir": str(subset_source_dir),
                "subset_cache_dir": str(subset_cache_dir),
            }
        )
        _write_status(payload, report_json, report_md)
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
            env_overrides=variant_spec["rescue_env"],
            cli_overrides=variant_spec["rescue_cli"],
        )
    else:
        rescue_result = {"status": "skipped", "note": "no rescue images selected"}

    if rescue_ids and rescue_result.get("status") not in {"ok", "cached"}:
        payload["result"] = {
            "status": "failed",
            "phase": "qwen_rescue",
            "note": variant_spec["note"],
            "rescue_image_ids": rescue_ids,
            "rescue_summary": rescue_summary,
            "rescue_process": rescue_result.get("process"),
            "rescue_error": rescue_result.get("note") or "qwen rescue phase failed",
            "rescue_source_dir": str(subset_source_dir),
            "subset_cache_dir": str(subset_cache_dir),
            "rescue_out_dir": str(rescue_out_dir),
        }
        payload["status"] = "failed"
        _write_status(payload, report_json, report_md)
        return

    merged_intermediate_dir = root_intermediate / f"{bundle_id}_{TARGET_VARIANT}_merged_verified"
    if merged_intermediate_dir.exists():
        shutil.rmtree(merged_intermediate_dir)
    payload["result"].update(
        {
            "phase": "merge_verified",
            "rescue_result_status": str(rescue_result.get("status") or "skipped"),
            "rescue_out_dir": str(rescue_out_dir),
            "rescue_intermediate_dir": str(rescue_intermediate_dir),
            "rescue_summary": rescue_summary,
            "rescue_image_ids": rescue_ids,
        }
    )
    _write_status(payload, report_json, report_md)
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

    final_out_dir = root_final / f"{bundle_id}_{TARGET_VARIANT}"
    final_intermediate_dir = root_intermediate / f"{bundle_id}_{TARGET_VARIANT}"
    payload["result"]["phase"] = "final_teacher"
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
    final_result["phase"] = "done"
    final_result["note"] = variant_spec["note"]
    final_result["qwen_prompt_mode"] = variant_spec["qwen_prompt_mode"]
    final_result["target_per_image"] = int(variant_spec["final_cli"]["--target-per-image"])
    if final_result.get("status") in {"ok", "cached"}:
        final_result = _enrich_completed_result(final_out_dir, final_result)

    payload["result"] = final_result
    payload["status"] = "ok" if final_result.get("status") in {"ok", "cached"} else str(final_result.get("status") or "failed")
    _write_status(payload, report_json, report_md)


if __name__ == "__main__":
    main()
