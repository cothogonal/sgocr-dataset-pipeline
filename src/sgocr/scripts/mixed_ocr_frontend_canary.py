from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..bootstrap import write_json, write_jsonl
from ..nemotron_frontend import nemotron_frontend_diagnostic
from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from ..run_quality import compute_run_quality
from .semantic_dev200_q_sweep import build_specs as build_q_specs


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(description="Run mixed OCR-front-end SGOCR canaries on chartqa/textocr/coco_text train images.")
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_frontend_canary_{stamp}")
    ap.add_argument("--source-name", default=f"chartqa50_textocr50_cocotext50_source_{stamp}")
    ap.add_argument("--chartqa-count", type=int, default=50)
    ap.add_argument("--textocr-count", type=int, default=50)
    ap.add_argument("--coco-text-count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def _q01_baseline() -> tuple[dict[str, str], dict[str, str]]:
    spec = build_q_specs("mixed_frontend_canary")[0]
    return dict(spec.env), dict(spec.cli)


def _sample_chartqa(count: int, seed: int) -> list[dict[str, str]]:
    chartqa_root = REPO_ROOT / "data" / "vm_ssl" / "raw" / "chartqa" / "images"
    candidates = sorted(chartqa_root.glob("*/*.png"))
    rng = random.Random(seed)
    selected = rng.sample(candidates, count)
    out = []
    for path in sorted(selected):
        image_id = f"chartqa:shared:{path.stem}"
        out.append(
            {
                "image_id": image_id,
                "image_path": str(path.relative_to(REPO_ROOT)),
                "dataset_source": "chartqa_train",
            }
        )
    return out


def _sample_coco_text(count: int, seed: int) -> list[dict[str, str]]:
    coco_root = REPO_ROOT / "data" / "vm_ssl" / "raw" / "coco_text_materialized" / "train"
    candidates = sorted(coco_root.glob("COCO_train2014_*.jpg"))
    rng = random.Random(seed)
    selected = rng.sample(candidates, count)
    out = []
    for path in sorted(selected):
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


def _sample_textocr_train(count: int, seed: int) -> list[dict[str, str]]:
    raw_json = REPO_ROOT / "data" / "vm_ssl" / "raw" / "textocr_full" / "TextOCR_0.1_train.json"
    images_root = REPO_ROOT / "data" / "vm_ssl" / "raw" / "textocr_trainval"
    payload = json.loads(raw_json.read_text(encoding="utf-8"))
    candidates: list[dict[str, str]] = []
    for image_id, meta in payload.get("imgs", {}).items():
        filename = Path(str(meta.get("file_name") or "")).name
        if not filename:
            continue
        image_path = images_root / filename
        if not image_path.exists():
            continue
        candidates.append(
            {
                "image_id": f"textocr:train:{image_id}",
                "image_path": str(image_path.relative_to(REPO_ROOT)),
                "dataset_source": "textocr_train",
            }
        )
    rng = random.Random(seed)
    selected = rng.sample(candidates, count)
    return sorted(selected, key=lambda row: row["image_id"])


def build_source_universe(*, out_dir: Path, chartqa_count: int, textocr_count: int, coco_text_count: int, seed: int) -> dict[str, Any]:
    rows = []
    rows.extend(_sample_chartqa(chartqa_count, seed + 101))
    rows.extend(_sample_textocr_train(textocr_count, seed + 202))
    rows.extend(_sample_coco_text(coco_text_count, seed + 303))
    rows = sorted(rows, key=lambda row: (row["dataset_source"], row["image_id"]))
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
    manifest = {
        "name": out_dir.name,
        "image_count": len(rows),
        "seed": seed,
        "source_counts": {
            "chartqa_train": chartqa_count,
            "textocr_train": textocr_count,
            "coco_text_train": coco_text_count,
        },
        "selection_policy": "uniform random sample per source dataset on the current local train-image pool",
        "items": rows,
    }
    notes = "\n".join(
        [
            f"# {out_dir.name}",
            "",
            "- role: `source universe for mixed OCR frontend canary`",
            f"- chartqa_train images: `{chartqa_count}`",
            f"- textocr_train images: `{textocr_count}`",
            f"- coco_text_train images: `{coco_text_count}`",
            f"- seed: `{seed}`",
        ]
    ) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "raw_results.jsonl", raw_rows)
    write_jsonl(out_dir / "bootstrap_tuples.jsonl", [])
    (out_dir / "notes.md").write_text(notes, encoding="utf-8")
    return manifest


def _build_variant_cmd(
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
) -> tuple[list[str], dict[str, str]]:
    """Build the subprocess cmd and env for a variant run. Shared by _run_variant and streaming runners."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "sgocr" / "src")
    env.update(env_overrides)
    cmd = [
        sys.executable,
        "-m",
        "sgocr.scripts.dev200_harness",
        "build-dev40-semantic",
        "--source-experiment-dir",
        str(source_dir),
        "--out-dir",
        str(out_dir),
        "--intermediate-dir",
        str(intermediate_dir),
        "--model",
        model,
        "--device",
        device,
        "--workers",
        str(workers),
        "--max-side",
        str(max_side),
        "--cache-level",
        cache_level,
    ]
    if cache_intermediate_dir:
        cmd.extend(["--cache-intermediate-dir", str(cache_intermediate_dir)])
    for key, value in cli_overrides.items():
        cmd.extend([key, value])
    return cmd, env


def _run_variant(
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
        rows = [json.loads(line) for line in (out_dir / "ocr_qa_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
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
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    payload = {
        "cmd": [str(part) for part in cmd],
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        return {"status": "failed", "process": payload}
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out_dir / "ocr_qa_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = compute_run_quality(summary, rows)
    return {"status": "ok", "process": payload, "metrics": metrics, "summary": summary}


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Mixed OCR Frontend Canary",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Source universe: `{payload['source_dir']}`",
        f"- Image counts: chartqa=`{payload['source_manifest']['source_counts']['chartqa_train']}`, textocr=`{payload['source_manifest']['source_counts']['textocr_train']}`, coco_text=`{payload['source_manifest']['source_counts']['coco_text_train']}`",
        "",
        "## Variants",
        "",
        "- `cur`: frozen `p02_o04_softcap012` pipeline",
        "- `cur+loose`: same pipeline with `SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES=0.85`, `SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES=0.40`",
        "- `nemo`: same downstream pipeline with `SGOCR_OCR_FRONTEND=nemotron_v2`",
        "",
        "## Nemotron Diagnostic",
        "",
        "```json",
        json.dumps(payload["nemotron_diagnostic"], indent=2, sort_keys=True),
        "```",
        "",
        "## Results",
        "",
        "| Variant | Status | Accepted | Images w/ rows | Inline mean | Sweep | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key in ("cur", "cur_loose", "nemo"):
        result = payload["results"].get(key) or {}
        status = result.get("status", "missing")
        if status not in {"ok", "cached"}:
            note = ""
            process = result.get("process") or {}
            if process:
                note = (process.get("stderr_tail") or process.get("stdout_tail") or "").replace("\n", " ")[:180]
            lines.append(f"| `{key}` | {status} | - | - | - | - | {note} |")
            continue
        metrics = result["metrics"]
        lines.append(
            f"| `{key}` | ok | {metrics['accepted_qas']} | {metrics['images_with_final_rows']} | "
            f"{metrics['inline_frontier_mean']:.4f} | {metrics['sweep_score']:.4f} | |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150"
    source_dir = root_final / args.source_name

    source_manifest = build_source_universe(
        out_dir=source_dir,
        chartqa_count=int(args.chartqa_count),
        textocr_count=int(args.textocr_count),
        coco_text_count=int(args.coco_text_count),
        seed=int(args.seed),
    )

    baseline_env, baseline_cli = _q01_baseline()
    results: dict[str, Any] = {}
    variants = {
        "cur": {
            "env": dict(baseline_env),
            "cli": dict(baseline_cli),
            "cache_intermediate_dir": None,
            "cache_level": "none",
        },
        "cur_loose": {
            "env": {
                **baseline_env,
                "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "0.85",
                "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.40",
            },
            "cli": dict(baseline_cli),
            "cache_intermediate_dir": root_intermediate / f"{args.bundle_id}_cur",
            "cache_level": "ocr",
        },
        "nemo": {
            "env": {
                **baseline_env,
                "SGOCR_OCR_FRONTEND": "nemotron_v2",
            },
            "cli": dict(baseline_cli),
            "cache_intermediate_dir": None,
            "cache_level": "none",
        },
    }
    for name, spec in variants.items():
        out_dir = root_final / f"{args.bundle_id}_{name}"
        intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}"
        results[name] = _run_variant(
            source_dir=source_dir,
            out_dir=out_dir,
            intermediate_dir=intermediate_dir,
            cache_intermediate_dir=spec["cache_intermediate_dir"],
            cache_level=spec["cache_level"],
            model=str(args.model),
            workers=int(args.workers),
            max_side=int(args.max_side),
            device=str(args.device),
            env_overrides=spec["env"],
            cli_overrides=spec["cli"],
        )
        results[name]["out_dir"] = str(out_dir)
        results[name]["intermediate_dir"] = str(intermediate_dir)

    payload = {
        "bundle_id": args.bundle_id,
        "source_dir": str(source_dir),
        "source_manifest": source_manifest,
        "results": results,
        "nemotron_diagnostic": nemotron_frontend_diagnostic(),
    }
    report_path = root_final / f"{args.bundle_id}_report.md"
    write_json(root_final / f"{args.bundle_id}_report.json", payload)
    report_path.write_text(render_report(payload), encoding="utf-8")
    print(str(report_path))


if __name__ == "__main__":
    main()
