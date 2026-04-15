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

from ..bootstrap import load_textocr_bootstrap_candidates, select_dev_subset, write_json, write_jsonl
from ..paths import LOGS_ROOT, OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from ..run_quality import compute_run_quality
from .semantic_dev200_p_sweep import build_specs as build_p_specs
from .semantic_dev40_sweep_v2 import append_timeline, now_stamp, write_text


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(description="Build the current p02 SGOCR pipeline on 5k fresh images.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev5000_p02_build_{stamp}")
    ap.add_argument("--source-name", default=f"dev5000_source_universe_{stamp}")
    ap.add_argument("--raw-json", default="data/vm_ssl/raw/textocr_full/TextOCR_0.1_val.json")
    ap.add_argument("--images-root", default="data/vm_ssl/raw/textocr_trainval")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--gemini-api-mode", default="sync", choices=["sync", "batch"])
    return ap.parse_args()


def _p02_spec():
    for spec in build_p_specs("dev5000_build"):
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


def _make_zero_soft_target(target_len: int = 196) -> list[float]:
    return [0.0] * int(target_len)


def _build_pointing_row(row: dict[str, Any]) -> dict[str, Any]:
    tags = dict(row.get("tags") or {})
    grounding = dict(row.get("grounding") or {})
    image_id = str(row.get("image_id") or "")
    question_id = str(row.get("sample_id") or row.get("ann_id") or image_id)
    image_path_value = str(row.get("image_path") or "")
    image_path = str((REPO_ROOT / image_path_value).resolve()) if image_path_value and not os.path.isabs(image_path_value) else image_path_value
    bbox_xyxy = grounding.get("text_bbox_xyxy")
    if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
        text_bbox = grounding.get("text_bbox") or row.get("text_bbox")
        if isinstance(text_bbox, list) and len(text_bbox) == 4:
            x, y, w, h = [float(v) for v in text_bbox]
            bbox_xyxy = [x, y, x + w, y + h]
        else:
            bbox_xyxy = None
    image_source = tags.get("image_source") or row.get("dataset_source") or ""
    source_dataset = "sgocr_dev5000"
    return {
        "id": question_id,
        "question_id": question_id,
        "image_id": image_id,
        "image_path": image_path,
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "answers": [row.get("answer", "")],
        "canonical_answer": row.get("answer", ""),
        "has_vqa_target": True,
        "has_grounding_target": False,
        "soft_target": _make_zero_soft_target(),
        "bbox_xyxy": bbox_xyxy,
        "split": "sgocr_dev5000_train",
        "source_dataset": source_dataset,
        "dataset_name": source_dataset,
        "mixture_name": f"{source_dataset}::{row.get('dataset_source', 'unknown')}",
        "metadata": {
            "source_dataset": source_dataset,
            "question_type": tags.get("question_type", row.get("question_type", "")),
            "answer_type": tags.get("answer_type", ""),
            "difficulty": tags.get("difficulty", ""),
            "ambiguity_level": tags.get("ambiguity_level", ""),
            "image_source": image_source,
            "sample_id": row.get("sample_id", ""),
            "anchor_label": row.get("anchor_label", ""),
            "textocr_image_id": image_id if str(image_source).startswith("textocr") else "",
        },
        "image": {
            "id": image_id,
            "path": image_path,
            "width": row.get("image_width"),
            "height": row.get("image_height"),
        },
    }


def _prepare_pointing_export(experiment_dir: Path, *, max_steps: int = 9000) -> dict[str, Any]:
    eval_root = experiment_dir / "evals" / "pointing_train_prep"
    eval_root.mkdir(parents=True, exist_ok=True)
    dataset_rows = [json.loads(line) for line in (experiment_dir / "ocr_qa_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    pointing_rows = [_build_pointing_row(row) for row in dataset_rows]
    pointing_index_path = eval_root / "sgocr_pointing_train_index.jsonl"
    write_jsonl(pointing_index_path, pointing_rows)
    manifest = {
        "experiment_dir": str(experiment_dir),
        "dataset_rows": len(dataset_rows),
        "pointing_index_path": str(pointing_index_path),
        "recommended_train_plan": {
            "max_steps": int(max_steps),
            "pointing_mix_ratio": 0.10,
            "use_grounding_loss": True,
            "grounding_loss_weight": 0.0,
            "answer_kd_weight": 0.3,
        },
    }
    write_json(eval_root / "manifest.json", manifest)
    return manifest


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# SGOCR Dev5000 P02 Build",
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
        f"- Pointing export: `{payload['pointing_index_path']}`",
        "",
        "## Pipeline",
        "",
        "- Dataset build: current `p02_o04_softcap012` SGOCR semantic pipeline on `5000` fresh TextOCR-val images.",
        "- No frontier benchmark in this stage.",
        f"- Gemini API mode requested: `{payload['gemini_api_mode']}`",
        "",
        "## Current Snapshot",
        "",
        f"- Accepted rows: `{payload.get('accepted_rows', 'pending')}`",
        f"- Precision-first score: `{payload.get('sweep_score', 'pending')}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    bundle_id = str(args.bundle_id)
    bundle_dir = LOGS_ROOT / bundle_id
    timeline_path = bundle_dir / "timeline.log"
    report_path = bundle_dir / "report.md"
    manifest_path = bundle_dir / "manifest.json"

    source_experiment_dir = OCR_SPATIAL_QA_FINAL_ROOT / "dev5000" / str(args.source_name)
    experiment_name = f"{bundle_id}_p02_dev5000"
    experiment_dir = OCR_SPATIAL_QA_FINAL_ROOT / "dev5000" / experiment_name
    intermediate_dir = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev5000" / experiment_name
    pointing_dir = experiment_dir / "evals" / "pointing_train_prep"

    payload: dict[str, Any] = {
        "bundle_id": bundle_id,
        "status": "initializing",
        "source_experiment_dir": str(source_experiment_dir),
        "experiment_dir": str(experiment_dir),
        "timeline_path": str(timeline_path),
        "pointing_index_path": str(pointing_dir / "sgocr_pointing_train_index.jsonl"),
        "gemini_api_mode": str(args.gemini_api_mode),
    }
    bundle_dir.mkdir(parents=True, exist_ok=True)
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
    semantic_env["SGOCR_GEMINI_API_MODE"] = str(args.gemini_api_mode)
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
        "--device",
        str(args.device),
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
    pointing_manifest = _prepare_pointing_export(experiment_dir)
    append_timeline(timeline_path, f"POINTING export done rows={pointing_manifest['dataset_rows']}")

    payload["accepted_rows"] = int(metrics["accepted_qas"])
    payload["sweep_score"] = round(float(metrics["sweep_score"]), 4)
    payload["status"] = "complete"
    write_json(manifest_path, payload)
    write_text(report_path, _render_report(payload))
    append_timeline(timeline_path, f"BUNDLE {bundle_id} complete")


if __name__ == "__main__":
    main()
