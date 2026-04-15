from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..bootstrap import write_json
from .mixed_ocr_frontend_canary import _q01_baseline, build_source_universe
from ..nemotron_frontend import nemotron_frontend_diagnostic
from ..paths import LOGS_ROOT, OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from ..run_quality import compute_run_quality


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(description="Run a single 3k-image mix build using the open-selected color+sibling frontier.")
    ap.add_argument("--bundle-id", default=f"sgocr_mixed3000_color_sibling_{stamp}")
    ap.add_argument("--source-name", default=f"chartqa1000_textocr1000_cocotext1000_source_{stamp}")
    ap.add_argument("--chartqa-count", type=int, default=1000)
    ap.add_argument("--textocr-count", type=int, default=1000)
    ap.add_argument("--coco-text-count", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--qwen-batch-size", type=int, default=4)
    ap.add_argument("--qwen-gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--qwen-max-model-len", type=int, default=2048)
    ap.add_argument("--progress-interval-sec", type=int, default=60)
    ap.add_argument("--reuse-source-dir", default="")
    ap.add_argument("--cache-intermediate-dir", default="")
    ap.add_argument("--cache-level", default="none", choices=["none", "ocr", "verified"])
    return ap.parse_args()


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_stage(*, out_dir: Path, intermediate_dir: Path, cache_intermediate_dir: Path | None = None) -> tuple[str, dict[str, int]]:
    counts = {
        "text_detections": _count_jsonl(intermediate_dir / "text_detections.jsonl"),
        "text_nodes": _count_jsonl(intermediate_dir / "text_nodes.jsonl"),
        "resolvable_nodes": _count_jsonl(intermediate_dir / "text_nodes_resolvable.jsonl"),
        "anchor_tags": _count_jsonl(intermediate_dir / "anchor_tags.jsonl"),
        "grounded_anchors": _count_jsonl(intermediate_dir / "grounded_anchors.jsonl"),
        "verified_tuples": _count_jsonl(intermediate_dir / "verified_tuples.jsonl"),
        "candidate_tuples": _count_jsonl(intermediate_dir / "candidate_tuples.jsonl"),
        "selected_tuples": _count_jsonl(intermediate_dir / "selected_tuples.jsonl"),
        "raw_results": _count_jsonl(out_dir / "raw_results.jsonl"),
        "final_rows": _count_jsonl(out_dir / "ocr_qa_dataset.jsonl"),
    }
    if cache_intermediate_dir is not None:
        cache_counts = {
            "text_detections": _count_jsonl(cache_intermediate_dir / "text_detections.jsonl"),
            "text_nodes": _count_jsonl(cache_intermediate_dir / "text_nodes.jsonl"),
            "resolvable_nodes": _count_jsonl(cache_intermediate_dir / "text_nodes_resolvable.jsonl"),
            "anchor_tags": _count_jsonl(cache_intermediate_dir / "anchor_tags.jsonl"),
            "grounded_anchors": _count_jsonl(cache_intermediate_dir / "grounded_anchors.jsonl"),
            "verified_tuples": _count_jsonl(cache_intermediate_dir / "verified_tuples.jsonl"),
        }
        for key, value in cache_counts.items():
            counts[key] = max(int(counts.get(key, 0)), int(value))
    if counts["final_rows"] > 0:
        stage = "final_dataset"
    elif counts["raw_results"] > 0:
        stage = "teacher_or_verify"
    elif counts["selected_tuples"] > 0:
        stage = "teacher_generation"
    elif counts["candidate_tuples"] > 0:
        stage = "candidate_build"
    elif counts["verified_tuples"] > 0:
        stage = "selection"
    elif counts["grounded_anchors"] > 0:
        stage = "verified_tuple_build"
    elif counts["anchor_tags"] > 0:
        stage = "anchor_grounding"
    elif counts["resolvable_nodes"] > 0:
        stage = "anchor_tag_discovery"
    elif counts["text_nodes"] > 0:
        stage = "resolvability"
    elif counts["text_detections"] > 0:
        stage = "ocr_consensus"
    else:
        stage = "startup"
    return stage, counts


def _render_report(payload: dict[str, Any]) -> str:
    counts = payload.get("progress_counts") or {}
    lines = [
        "# Mixed3000 Color-Sibling Run",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Status: `{payload['status']}`",
        "",
        "## Paths",
        "",
        f"- Source universe: `{payload['source_dir']}`",
        f"- Dataset experiment: `{payload['out_dir']}`",
        f"- Intermediate dir: `{payload['intermediate_dir']}`",
        f"- Progress log: `{payload['progress_log']}`",
        f"- Build log: `{payload['build_log']}`",
        "",
        "## Configuration",
        "",
        "- Variant: `open_selected_color_sibling_v0`",
        "- OCR frontend: `nemotron_v2`",
        "- local tag discovery: `qwen3_vl_vllm` open/color-specific",
        "- anchor grounding: `qwen3_vl_vllm` selected-tags",
        "- relabel: `none`",
        f"- semantic workers: `{payload['workers']}`",
        f"- Qwen batch size: `{payload['qwen_batch_size']}`",
        f"- Qwen GPU memory utilization: `{payload['qwen_gpu_mem_util']}`",
        f"- Qwen max model len: `{payload['qwen_max_model_len']}`",
        "",
        "## Source Mix",
        "",
        f"- chartqa_train: `{payload['source_manifest']['source_counts']['chartqa_train']}`",
        f"- textocr_train: `{payload['source_manifest']['source_counts']['textocr_train']}`",
        f"- coco_text_train: `{payload['source_manifest']['source_counts']['coco_text_train']}`",
        f"- total images: `{payload['source_manifest']['image_count']}`",
        "",
        "## Live Progress",
        "",
        f"- Current stage: `{payload.get('current_stage', 'pending')}`",
        f"- Last progress update: `{payload.get('last_progress_at', 'pending')}`",
        "",
        "| Artifact | Rows |",
        "|---|---:|",
        f"| `text_detections` | {counts.get('text_detections', 0)} |",
        f"| `text_nodes` | {counts.get('text_nodes', 0)} |",
        f"| `resolvable_nodes` | {counts.get('resolvable_nodes', 0)} |",
        f"| `anchor_tags` | {counts.get('anchor_tags', 0)} |",
        f"| `grounded_anchors` | {counts.get('grounded_anchors', 0)} |",
        f"| `verified_tuples` | {counts.get('verified_tuples', 0)} |",
        f"| `candidate_tuples` | {counts.get('candidate_tuples', 0)} |",
        f"| `selected_tuples` | {counts.get('selected_tuples', 0)} |",
        f"| `raw_results` | {counts.get('raw_results', 0)} |",
        f"| `final_rows` | {counts.get('final_rows', 0)} |",
    ]
    if payload.get("status") == "ok":
        metrics = payload.get("metrics") or {}
        lines.extend(
            [
                "",
                "## Final Metrics",
                "",
                f"- accepted rows: `{metrics.get('accepted_qas')}`",
                f"- images with rows: `{metrics.get('images_with_final_rows')}`",
                f"- inline frontier mean: `{metrics.get('inline_frontier_mean')}`",
                f"- sweep score: `{metrics.get('sweep_score')}`",
            ]
        )
    if payload.get("status") == "failed":
        lines.extend(
            [
                "",
                "## Failure",
                "",
                "```text",
                str(payload.get("failure_tail") or ""),
                "```",
            ]
        )
    return "\n".join(lines) + "\n"


def _write_status(report_json: Path, report_md: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(_render_report(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    bundle_id = str(args.bundle_id)
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev3000"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev3000"
    bundle_log_dir = LOGS_ROOT / bundle_id
    bundle_log_dir.mkdir(parents=True, exist_ok=True)

    source_dir = Path(args.reuse_source_dir) if str(args.reuse_source_dir).strip() else (root_final / str(args.source_name))
    out_dir = root_final / f"{bundle_id}_open_selected_color_sibling_v0"
    intermediate_dir = root_intermediate / f"{bundle_id}_open_selected_color_sibling_v0"
    report_json = root_final / f"{bundle_id}_report.json"
    report_md = root_final / f"{bundle_id}_report.md"
    build_log = bundle_log_dir / "build.log"
    progress_log = bundle_log_dir / "progress.log"

    root_final.mkdir(parents=True, exist_ok=True)
    root_intermediate.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        build_source_universe(
            out_dir=source_dir,
            chartqa_count=int(args.chartqa_count),
            textocr_count=int(args.textocr_count),
            coco_text_count=int(args.coco_text_count),
            seed=int(args.seed),
        )
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))

    baseline_env, baseline_cli = _q01_baseline()
    env_overrides = {
        **baseline_env,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
        "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "open",
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "selected_tags",
        "SGOCR_ANCHOR_RELABEL_MODE": "none",
        "SGOCR_QWEN_ANCHOR_MODEL": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": f"{float(args.qwen_gpu_mem_util):.2f}",
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": str(int(args.qwen_batch_size)),
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": str(int(args.qwen_max_model_len)),
        "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
        "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
        "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    }

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "sgocr" / "src")
    env["SGOCR_PROGRESS_LOG_PATH"] = str(progress_log)
    env.update(env_overrides)
    cmd = [
        sys.executable,
        "-m",
        "sgocr.dev200_harness",
        "build-dev40-semantic",
        "--source-experiment-dir",
        str(source_dir),
        "--out-dir",
        str(out_dir),
        "--intermediate-dir",
        str(intermediate_dir),
        "--model",
        str(args.model),
        "--device",
        str(args.device),
        "--workers",
        str(int(args.workers)),
        "--max-side",
        str(int(args.max_side)),
        "--cache-level",
        str(args.cache_level),
    ]
    if str(args.cache_intermediate_dir).strip():
        cmd.extend(["--cache-intermediate-dir", str(Path(args.cache_intermediate_dir))])
    for key, value in baseline_cli.items():
        cmd.extend([key, value])

    payload: dict[str, Any] = {
        "bundle_id": bundle_id,
        "status": "launching",
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "intermediate_dir": str(intermediate_dir),
        "build_log": str(build_log),
        "progress_log": str(progress_log),
        "workers": int(args.workers),
        "qwen_batch_size": int(args.qwen_batch_size),
        "qwen_gpu_mem_util": float(args.qwen_gpu_mem_util),
        "qwen_max_model_len": int(args.qwen_max_model_len),
        "cache_level": str(args.cache_level),
        "cache_intermediate_dir": str(Path(args.cache_intermediate_dir)) if str(args.cache_intermediate_dir).strip() else "",
        "source_manifest": source_manifest,
        "nemotron_diagnostic": nemotron_frontend_diagnostic(),
        "cmd": [str(x) for x in cmd],
        "current_stage": "startup",
        "last_progress_at": None,
        "progress_counts": {},
    }
    _write_status(report_json, report_md, payload)

    with build_log.open("w", encoding="utf-8") as log_handle, progress_log.open("a", encoding="utf-8") as prog_handle:
        log_handle.write("CMD: " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        log_handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )

        stop_event = threading.Event()

        def monitor() -> None:
            while not stop_event.is_set():
                stage, counts = _detect_stage(
                    out_dir=out_dir,
                    intermediate_dir=intermediate_dir,
                    cache_intermediate_dir=(Path(args.cache_intermediate_dir) if str(args.cache_intermediate_dir).strip() else None),
                )
                stamp = datetime.now().isoformat(timespec="seconds")
                payload["status"] = "running"
                payload["current_stage"] = stage
                payload["last_progress_at"] = stamp
                payload["progress_counts"] = counts
                _write_status(report_json, report_md, payload)
                prog_handle.write(f"[{stamp}] stage={stage} counts={json.dumps(counts, sort_keys=True)}\n")
                prog_handle.flush()
                for _ in range(max(1, int(args.progress_interval_sec))):
                    if stop_event.is_set():
                        break
                    time.sleep(1)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        ret = proc.wait()
        stop_event.set()
        thread.join(timeout=5)

    stage, counts = _detect_stage(
        out_dir=out_dir,
        intermediate_dir=intermediate_dir,
        cache_intermediate_dir=(Path(args.cache_intermediate_dir) if str(args.cache_intermediate_dir).strip() else None),
    )
    payload["current_stage"] = stage
    payload["last_progress_at"] = datetime.now().isoformat(timespec="seconds")
    payload["progress_counts"] = counts

    if ret != 0:
        payload["status"] = "failed"
        tail = build_log.read_text(encoding="utf-8", errors="ignore")[-4000:]
        payload["failure_tail"] = tail
        _write_status(report_json, report_md, payload)
        raise SystemExit(ret)

    summary = _read_json(out_dir / "summary.json") or {}
    rows_path = out_dir / "ocr_qa_dataset.jsonl"
    rows = []
    if rows_path.exists():
        with rows_path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    metrics = compute_run_quality(summary, rows)
    payload["status"] = "ok"
    payload["summary"] = summary
    payload["metrics"] = metrics
    _write_status(report_json, report_md, payload)
    print(str(report_md))


if __name__ == "__main__":
    main()
