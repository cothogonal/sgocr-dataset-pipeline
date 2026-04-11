from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .bootstrap import load_textocr_bootstrap_candidates, select_dev_subset, write_json, write_jsonl
from .dev200_eval import SweepRun, prepare_bridge_ft_eval
from .paths import LOGS_ROOT, OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from .run_quality import compute_run_quality
from .secrets import GEMINI, OPENAI, missing_secret_env_vars
from .semantic_dev200_p_sweep import build_specs as build_p_specs
from .semantic_dev40_sweep_v2 import append_timeline, now_stamp, write_text


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the current p02 SGOCR pipeline on 1k images and launch full eval.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev1000_p02_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--source-name", default="dev1000_source_universe_v1")
    ap.add_argument("--raw-json", default="data/vm_ssl/raw/textocr_full/TextOCR_0.1_val.json")
    ap.add_argument("--images-root", default="data/vm_ssl/raw/textocr_trainval")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--frontier-workers", type=int, default=8)
    ap.add_argument("--ft-max-steps", type=int, default=1000)
    ap.add_argument("--ft-batch-size", type=int, default=64)
    ap.add_argument("--ft-grad-accum", type=int, default=2)
    ap.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    return ap.parse_args()


def _p02_spec():
    for spec in build_p_specs("dev1000_full_eval"):
        if spec.name == "p02_o04_softcap012":
            return spec
    raise RuntimeError("Could not find p02_o04_softcap012 spec")


def _run_and_log(*, cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("CMD: " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed with exit code {ret}: {' '.join(cmd)}")


def _spawn_and_log(*, cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    handle.write("CMD: " + " ".join(shlex.quote(part) for part in cmd) + "\n")
    handle.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    setattr(proc, "_codex_log_handle", handle)
    return proc


def _close_spawn_log(proc: subprocess.Popen[str]) -> None:
    handle = getattr(proc, "_codex_log_handle", None)
    if handle is not None:
        handle.close()


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# SGOCR Dev1000 P02 Full Eval",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Updated: `{now_stamp()}`",
        f"- Status: `{payload['status']}`",
        "",
        "## Paths",
        "",
        f"- Source universe: `{payload['source_experiment_dir']}`",
        f"- Dataset experiment: `{payload['experiment_dir']}`",
        f"- Timeline: `{payload['timeline_path']}`",
        f"- Frontier benchmark dir: `{payload['frontier_benchmark_dir']}`",
        f"- Bridge FT eval dir: `{payload['bridge_ft_dir']}`",
        f"- TextOCR eval summary: `{payload['textocr_eval_summary']}`",
        "",
        "## Pipeline",
        "",
        "- Dataset build: current `p02_o04_softcap012` SGOCR semantic pipeline on `1000` TextOCR-val images.",
        "- Dataset eval: frontier benchmark on `gemini-3-flash-preview` and `gpt-5.3-codex`, plus agreement analysis.",
        "- Model eval: `1000`-step bridge fine-tune from clean `9k` champion checkpoint, then `TextOCR` val eval with `text_exact` scoring.",
        "",
        "## Current Snapshot",
        "",
        f"- Accepted rows: `{payload.get('accepted_rows', 'pending')}`",
        f"- Precision-first score: `{payload.get('sweep_score', 'pending')}`",
        f"- Bridge FT run id: `{payload['bridge_ft_run_id']}`",
        f"- Bridge checkpoint: `{payload.get('bridge_checkpoint', 'pending')}`",
    ]
    return "\n".join(lines) + "\n"


def _latest_checkpoint(run_id: str, max_steps: int) -> Path:
    exact = REPO_ROOT / "logs" / run_id / f"step_{int(max_steps)}.tar"
    if exact.exists():
        return exact
    candidates = sorted((REPO_ROOT / "logs" / run_id).glob("step_*.tar"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found for run id {run_id}")
    return candidates[-1]


def _build_source_universe(*, out_dir: Path, raw_json_path: Path, images_root: Path, limit: int, seed: int) -> None:
    candidates = load_textocr_bootstrap_candidates(raw_json_path, images_root)
    selected = select_dev_subset(candidates, limit=limit, seed=seed)
    manifest = {
        "name": out_dir.name,
        "source": raw_json_path.name,
        "image_count": len(selected),
        "seed": int(seed),
        "selection_policy": "one unique region-grounded bootstrap tuple per image, stratified by density/area/region",
        "items": [row.to_dict() for row in selected],
    }
    raw_rows = [
        {
            "image_id": row.image_id,
            "image_path": row.image_path,
            "tuple": {
                "image_id": row.image_id,
                "image_path": row.image_path,
                "dataset_source": "textocr_val",
            },
        }
        for row in selected
    ]
    notes = "\n".join(
        [
            f"# {out_dir.name}",
            "",
            "- bootstrap source: `TextOCR_0.1_val`",
            "- role: `source universe for semantic SGOCR pipeline`",
            f"- selected images: `{len(selected)}`",
            f"- seed: `{seed}`",
        ]
    ) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "bootstrap_tuples.jsonl", [row.to_dict() for row in selected])
    write_jsonl(out_dir / "raw_results.jsonl", raw_rows)
    write_text(out_dir / "notes.md", notes)


def main() -> None:
    args = parse_args()
    missing = missing_secret_env_vars([GEMINI, OPENAI])
    if missing:
        raise SystemExit(f"Missing required env var(s): {', '.join(missing)}")

    bundle_id = str(args.bundle_id)
    bundle_dir = LOGS_ROOT / bundle_id
    timeline_path = bundle_dir / "timeline.log"
    report_path = bundle_dir / "report.md"
    manifest_path = bundle_dir / "manifest.json"

    source_experiment_dir = OCR_SPATIAL_QA_FINAL_ROOT / "dev1000" / str(args.source_name)
    experiment_name = f"{bundle_id}_p02_dev1000"
    experiment_dir = OCR_SPATIAL_QA_FINAL_ROOT / "dev1000" / experiment_name
    intermediate_dir = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev1000" / experiment_name
    frontier_benchmark_dir = experiment_dir / "evals" / "frontier_benchmark_full"
    textocr_eval_summary = experiment_dir / "evals" / "bridge_ft_1k" / "textocr_eval" / "summary.json"
    bridge_ft_run_id = f"{experiment_name}_bridgeft1k"

    payload: dict[str, Any] = {
        "bundle_id": bundle_id,
        "status": "initializing",
        "source_experiment_dir": str(source_experiment_dir),
        "experiment_dir": str(experiment_dir),
        "timeline_path": str(timeline_path),
        "frontier_benchmark_dir": str(frontier_benchmark_dir),
        "bridge_ft_dir": str(experiment_dir / "evals" / "bridge_ft_1k"),
        "textocr_eval_summary": str(textocr_eval_summary),
        "bridge_ft_run_id": bridge_ft_run_id,
    }
    write_json(manifest_path, payload)
    write_text(report_path, _render_report(payload))

    append_timeline(timeline_path, f"BUNDLE {bundle_id} start limit={int(args.limit)} source={source_experiment_dir.name}")
    append_timeline(timeline_path, f"SOURCE build start raw_json={args.raw_json} images_root={args.images_root}")
    _build_source_universe(
        out_dir=source_experiment_dir,
        raw_json_path=Path(args.raw_json),
        images_root=Path(args.images_root),
        limit=int(args.limit),
        seed=int(args.seed),
    )
    append_timeline(timeline_path, f"SOURCE build done dir={source_experiment_dir}")

    spec = _p02_spec()
    semantic_env = os.environ.copy()
    semantic_env.update(spec.env)
    semantic_cmd = [
        sys.executable,
        "-m",
        "sgocr.dev200_harness",
        "build-dev40-semantic",
        "--source-experiment-dir",
        str(source_experiment_dir),
        "--out-dir",
        str(experiment_dir),
        "--intermediate-dir",
        str(intermediate_dir),
        "--model",
        str(args.model),
        "--workers",
        str(int(args.workers)),
        "--max-side",
        str(int(args.max_side)),
        "--target-per-image",
        spec.cli.get("--target-per-image", "5"),
        "--max-detections",
        spec.cli.get("--max-detections", "128"),
        "--grounding-threshold",
        spec.cli.get("--grounding-threshold", "0.28"),
        "--max-tags-per-image",
        spec.cli.get("--max-tags-per-image", "14"),
        "--cache-level",
        "none",
    ]
    append_timeline(timeline_path, "DATASET build start pipeline=p02_o04_softcap012")
    _run_and_log(cmd=semantic_cmd, log_path=bundle_dir / "dataset_build.log", env=semantic_env)
    append_timeline(timeline_path, f"DATASET build done experiment={experiment_dir}")

    summary = json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))
    dataset_rows = [json.loads(line) for line in (experiment_dir / "ocr_qa_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = compute_run_quality(summary, dataset_rows)
    payload["accepted_rows"] = int(metrics["accepted_qas"])
    payload["sweep_score"] = round(float(metrics["sweep_score"]), 4)
    payload["status"] = "dataset_built"
    write_json(manifest_path, payload)
    write_text(report_path, _render_report(payload))

    run = SweepRun(
        bundle_id=bundle_id,
        name="p02_dev1000",
        experiment_dir=experiment_dir,
        intermediate_dir=intermediate_dir,
    )
    bridge_ft_dir = prepare_bridge_ft_eval(
        run,
        max_steps=int(args.ft_max_steps),
        batch_size=int(args.ft_batch_size),
        grad_accum_steps=int(args.ft_grad_accum),
    )
    payload["bridge_ft_dir"] = str(bridge_ft_dir)
    write_json(manifest_path, payload)
    write_text(report_path, _render_report(payload))

    train_script = bridge_ft_dir / "run_bridge_ft_1k.sh"
    train_env = os.environ.copy()
    train_env["RUN_ID"] = bridge_ft_run_id
    append_timeline(timeline_path, f"TRAIN launch run_id={bridge_ft_run_id}")
    train_proc = _spawn_and_log(
        cmd=["bash", str(train_script)],
        log_path=bundle_dir / "bridge_ft_1k.log",
        env=train_env,
    )

    try:
        frontier_cmd = [
            sys.executable,
            "-m",
            "sgocr.dev200_eval",
            "run-frontier-benchmark",
            "--experiment-dir",
            str(experiment_dir),
            "--model",
            "openai:gpt-5.3-codex",
            "--model",
            "gemini:gemini-3-flash-preview",
            "--out-dir",
            str(frontier_benchmark_dir),
            "--workers",
            str(int(args.frontier_workers)),
        ]
        append_timeline(timeline_path, "FRONTIER benchmark start models=gpt-5.3-codex,gemini-3-flash-preview")
        _run_and_log(cmd=frontier_cmd, log_path=bundle_dir / "frontier_benchmark.log", env=os.environ.copy())
        append_timeline(timeline_path, f"FRONTIER benchmark done dir={frontier_benchmark_dir}")

        agreement_cmd = [
            sys.executable,
            "-m",
            "sgocr.dev200_eval",
            "compute-frontier-agreement",
            "--benchmark-dir",
            str(frontier_benchmark_dir),
        ]
        append_timeline(timeline_path, "FRONTIER agreement start")
        _run_and_log(cmd=agreement_cmd, log_path=bundle_dir / "frontier_agreement.log", env=os.environ.copy())
        append_timeline(timeline_path, "FRONTIER agreement done")

        train_ret = train_proc.wait()
        if train_ret != 0:
            raise RuntimeError(f"Bridge fine-tune failed with exit code {train_ret}")
        append_timeline(timeline_path, f"TRAIN complete run_id={bridge_ft_run_id}")
    finally:
        _close_spawn_log(train_proc)

    checkpoint_path = _latest_checkpoint(bridge_ft_run_id, int(args.ft_max_steps))
    payload["bridge_checkpoint"] = str(checkpoint_path)
    payload["status"] = "ft_complete"
    write_json(manifest_path, payload)
    write_text(report_path, _render_report(payload))

    textocr_eval_dir = bridge_ft_dir / "textocr_eval"
    textocr_eval_dir.mkdir(parents=True, exist_ok=True)
    textocr_cmd = [
        sys.executable,
        "tasks/mm_bridge/scripts/mm_dualvm_eval.py",
        "--checkpoint",
        str(checkpoint_path),
        "--device",
        str(args.device),
        "--batch_size",
        "96",
        "--num_workers",
        "2",
        "--prefetch_factor",
        "2",
        "--textocr_annotations_root",
        "data/vm_ssl/raw/textocr_full",
        "--textocr_images_root",
        "data/vm_ssl/raw/textocr_trainval",
        "--eval_split",
        "textocr_val",
        "--scorer",
        "text_exact",
        "--output_json",
        str(textocr_eval_summary),
    ]
    append_timeline(timeline_path, "TEXTOCR eval start")
    textocr_env = os.environ.copy()
    textocr_env["PYTHONPATH"] = str(REPO_ROOT)
    _run_and_log(cmd=textocr_cmd, log_path=bundle_dir / "textocr_eval.log", env=textocr_env)
    append_timeline(timeline_path, f"TEXTOCR eval done summary={textocr_eval_summary}")

    payload["status"] = "complete"
    write_json(manifest_path, payload)
    write_text(report_path, _render_report(payload))
    append_timeline(timeline_path, f"BUNDLE {bundle_id} complete")


if __name__ == "__main__":
    main()
