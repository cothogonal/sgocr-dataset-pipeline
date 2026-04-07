from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests
from PIL import Image

from .bootstrap import write_json, write_jsonl
from .paths import LOGS_ROOT, OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT, repo_relative
from .secrets import GEMINI, OPENAI, get_secret
from train.vqa_data import normalize_text_answer, normalize_vqa_answer


DEFAULT_BUNDLE_ID = "sgocr_dev200_20260406_200346"
DEFAULT_BRIDGE_CHAMPION = (
    REPO_ROOT / "logs" / "mmtier0_vm_v1_20260330_001641_winner_ftclean_bridge_v2" / "step_9000.tar"
)
DEFAULT_MODEL_SPECS = [
    "openai:gpt-5.3-codex",
    "gemini:gemini-3-flash-preview",
    "gemini:gemini-3-pro-preview",
]

GEMINI_MODEL_ALIASES = {
    "gemini-3.1-flash": "gemini-3-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3.1-pro": "gemini-3-pro-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
}


@dataclass(frozen=True)
class SweepRun:
    bundle_id: str
    name: str
    experiment_dir: Path
    intermediate_dir: Path


@dataclass(frozen=True)
class BenchmarkModelSpec:
    provider: str
    model: str
    requested_model: str | None = None

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _normalize_answer_by_type(answer: str, answer_type: str) -> str:
    kind = str(answer_type or "").strip().lower()
    if kind in {"yes", "no", "number", "text_string", "attribute", "spatial_phrase"}:
        if kind in {"yes", "no", "number"}:
            return normalize_vqa_answer(answer)
        return normalize_text_answer(answer)
    # Default to text normalization for SGOCR rows; punctuation/case matter less than literal match.
    return normalize_text_answer(answer)


def discover_bundle_runs(bundle_id: str) -> list[SweepRun]:
    final_root = OCR_SPATIAL_QA_FINAL_ROOT / "dev200"
    interm_root = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200"
    runs: list[SweepRun] = []
    for experiment_dir in sorted(final_root.glob(f"{bundle_id}_*")):
        if not experiment_dir.is_dir():
            continue
        summary_path = experiment_dir / "summary.json"
        dataset_path = experiment_dir / "ocr_qa_dataset.jsonl"
        if not summary_path.exists() or not dataset_path.exists():
            continue
        name = experiment_dir.name.removeprefix(f"{bundle_id}_")
        runs.append(
            SweepRun(
                bundle_id=bundle_id,
                name=name,
                experiment_dir=experiment_dir,
                intermediate_dir=interm_root / experiment_dir.name,
            )
        )
    return runs


def _make_zero_soft_target(target_len: int = 196) -> list[float]:
    return [0.0] * int(target_len)


def _build_pointing_row(row: dict[str, Any]) -> dict[str, Any]:
    tags = dict(row.get("tags") or {})
    grounding = dict(row.get("grounding") or {})
    image_id = str(row.get("image_id") or "")
    question_id = str(row.get("sample_id") or row.get("ann_id") or image_id)
    bbox_xyxy = grounding.get("text_bbox_xyxy")
    if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
        text_bbox = grounding.get("text_bbox") or row.get("text_bbox")
        if isinstance(text_bbox, list) and len(text_bbox) == 4:
            x, y, w, h = [float(v) for v in text_bbox]
            bbox_xyxy = [x, y, x + w, y + h]
        else:
            bbox_xyxy = None
    return {
        "id": question_id,
        "question_id": question_id,
        "image_id": image_id,
        "image_path": row.get("image_path"),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "answers": [row.get("answer", "")],
        "canonical_answer": row.get("answer", ""),
        "has_vqa_target": True,
        "has_grounding_target": False,
        "soft_target": _make_zero_soft_target(),
        "bbox_xyxy": bbox_xyxy,
        "split": "sgocr_dev200_train",
        "source_dataset": "sgocr_dev200",
        "dataset_name": "sgocr_dev200",
        "mixture_name": f"sgocr_dev200::{row.get('dataset_source', 'unknown')}",
        "metadata": {
            "source_dataset": "sgocr_dev200",
            "question_type": tags.get("question_type", row.get("question_type", "")),
            "answer_type": tags.get("answer_type", ""),
            "difficulty": tags.get("difficulty", ""),
            "ambiguity_level": tags.get("ambiguity_level", ""),
            "image_source": tags.get("image_source", row.get("dataset_source", "")),
            "sample_id": row.get("sample_id", ""),
            "anchor_label": row.get("anchor_label", ""),
        },
        "image": {
            "id": image_id,
            "path": row.get("image_path"),
            "width": row.get("image_width"),
            "height": row.get("image_height"),
        },
    }


def prepare_bridge_ft_eval(run: SweepRun, *, max_steps: int = 1000, batch_size: int = 64, grad_accum_steps: int = 2) -> Path:
    eval_root = run.experiment_dir / "evals" / "bridge_ft_1k"
    eval_root.mkdir(parents=True, exist_ok=True)
    dataset_rows = _load_jsonl(run.experiment_dir / "ocr_qa_dataset.jsonl")
    pointing_rows = [_build_pointing_row(row) for row in dataset_rows]
    pointing_index_path = eval_root / "sgocr_pointing_train_index.jsonl"
    write_jsonl(pointing_index_path, pointing_rows)

    mix_config = {
        "recommended_sampling_weights": {
            "sgocr_dev200": 1.0,
        }
    }
    write_json(eval_root / "mix_config.json", mix_config)

    manifest = {
        "bundle_id": run.bundle_id,
        "run_name": run.name,
        "experiment_dir": str(run.experiment_dir),
        "dataset_rows": len(dataset_rows),
        "pointing_index_path": str(pointing_index_path),
        "bridge_checkpoint": str(DEFAULT_BRIDGE_CHAMPION),
        "train_plan": {
            "max_steps": int(max_steps),
            "batch_size": int(batch_size),
            "grad_accum_steps": int(grad_accum_steps),
            "pointing_mix_ratio": 0.25,
            "use_grounding_loss": True,
            "grounding_loss_weight": 0.0,
            "answer_kd_weight": 0.3,
        },
    }
    write_json(eval_root / "manifest.json", manifest)

    run_script = f"""#!/usr/bin/env bash
set -euo pipefail

cd {str(REPO_ROOT)!r}

RUN_ID="${{RUN_ID:-{run.experiment_dir.name}_bridgeft1k}}"
POINTING_INDEX_PATH="{str(pointing_index_path)}"
CHAMP_CKPT="{str(DEFAULT_BRIDGE_CHAMPION)}"

./bin/runmm_v1.sh "${{RUN_ID}}" \
  --init_from_mm_checkpoint "${{CHAMP_CKPT}}" \
  --vision_model siglip2_b16 \
  --vision_checkpoint logs/hf_vision/openclip_siglip2_b16_webli \
  --lm_checkpoint logs/lm_final/step_45000.tar \
  --tokenizer_path logs/mix_bpe_16k/tokenizer.pt \
  --seed 42 \
  --batch_size {int(batch_size)} \
  --grad_accum_steps {int(grad_accum_steps)} \
  --eval_batch_size 96 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --max_steps {int(max_steps)} \
  --manual_max_steps \
  --eval_every 0 \
  --eval_batches 0 \
  --final_eval_batches 0 \
  --skip_final_eval \
  --ckpt_every {int(max_steps)} \
  --freeze_mode bridge_plus_top_lm \
  --use_grounding_loss \
  --grounding_loss_weight 0.0 \
  --pointing_index_path "${{POINTING_INDEX_PATH}}" \
  --pointing_mix_ratio 0.25 \
  --answer_kd_labels_path data/distillation/qwen25vl3b_vqav2_train_v1 \
  --answer_kd_weight 0.3 \
  --answer_kd_temp 4.0 \
  --min_train_steps_per_s 0
"""
    script_path = eval_root / "run_bridge_ft_1k.sh"
    script_path.write_text(run_script, encoding="utf-8")
    script_path.chmod(0o755)
    return eval_root


def prepare_bundle_evals(bundle_id: str) -> dict[str, Any]:
    runs = discover_bundle_runs(bundle_id)
    prepared: list[dict[str, Any]] = []
    for run in runs:
        bridge_dir = prepare_bridge_ft_eval(run)
        prepared.append(
            {
                "run_name": run.name,
                "experiment_dir": str(run.experiment_dir),
                "bridge_ft_dir": str(bridge_dir),
            }
        )
    payload = {
        "bundle_id": bundle_id,
        "runs_discovered": len(runs),
        "prepared_runs": prepared,
    }
    out_path = LOGS_ROOT / bundle_id / "eval_prep_summary.json"
    write_json(out_path, payload)
    return payload


def _encode_image(path: Path, *, max_side: int = 1024, quality: int = 90) -> tuple[str, int, int]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1.0, float(max_side) / float(max(width, height)))
        if scale < 1.0:
            img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
        return base64.b64encode(buf.getvalue()).decode("ascii"), img.size[0], img.size[1]


def _benchmark_prompt(row: dict[str, Any]) -> str:
    tags = dict(row.get("tags") or {})
    question_type = str(tags.get("question_type") or row.get("question_type") or "").strip()
    return (
        "Answer the visual question using the image.\n"
        "Give only the shortest literal answer.\n"
        "Do not explain.\n"
        "Do not add punctuation unless it is part of the answer.\n"
        f"Question type: {question_type or 'unknown'}\n"
        f"Question: {row.get('question', '')}"
    )


def _call_openai_responses(model: str, row: dict[str, Any], *, timeout_s: int = 120) -> dict[str, Any]:
    image_b64, width, height = _encode_image(Path(row["image_path"]))
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _benchmark_prompt(row)},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ],
            }
        ],
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 32,
    }
    resp = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {get_secret(OPENAI)}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()
    answer = str(payload.get("output_text") or "").strip()
    if not answer:
        parts: list[str] = []
        for item in payload.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    parts.append(str(text))
        answer = " ".join(parts).strip()
    return {
        "provider": "openai",
        "model": model,
        "answer": answer,
        "usage": payload.get("usage", {}),
        "image_width": width,
        "image_height": height,
    }


def _gemini_generation_config(model: str) -> dict[str, Any]:
    lower = model.lower()
    if "gemini-3" in lower:
        level = "minimal" if "flash" in lower else "low"
        return {
            "temperature": 0.0,
            "thinkingConfig": {
                "thinkingLevel": level,
            },
            "maxOutputTokens": 32,
        }
    if "gemini-2.5" in lower and "flash" in lower:
        return {
            "temperature": 0.0,
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
            "maxOutputTokens": 32,
        }
    return {
        "temperature": 0.0,
        "maxOutputTokens": 32,
    }


def _call_gemini_eval(model: str, row: dict[str, Any], *, timeout_s: int = 120) -> dict[str, Any]:
    image_b64, width, height = _encode_image(Path(row["image_path"]))
    body = {
        "contents": [
            {
                "parts": [
                    {"text": _benchmark_prompt(row)},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                ]
            }
        ],
        "generationConfig": _gemini_generation_config(model),
    }
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "x-goog-api-key": get_secret(GEMINI),
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()
    parts: list[str] = []
    for cand in payload.get("candidates", []):
        content = cand.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                parts.append(str(text))
    return {
        "provider": "gemini",
        "model": model,
        "answer": " ".join(parts).strip(),
        "usage": payload.get("usageMetadata", {}),
        "image_width": width,
        "image_height": height,
    }


def _parse_model_spec(text: str) -> BenchmarkModelSpec:
    provider, model = str(text).split(":", 1)
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in {"openai", "gemini"}:
        raise ValueError(f"Unsupported provider in model spec: {text}")
    if not model:
        raise ValueError(f"Missing model name in spec: {text}")
    requested = model
    if provider == "gemini":
        model = GEMINI_MODEL_ALIASES.get(model, model)
    return BenchmarkModelSpec(provider=provider, model=model, requested_model=requested)


def _call_model(spec: BenchmarkModelSpec, row: dict[str, Any]) -> dict[str, Any]:
    if spec.provider == "openai":
        return _call_openai_responses(spec.model, row)
    if spec.provider == "gemini":
        return _call_gemini_eval(spec.model, row)
    raise ValueError(f"Unsupported provider: {spec.provider}")


def _score_prediction(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    tags = dict(row.get("tags") or {})
    gold = str(row.get("answer") or "")
    answer_type = str(tags.get("answer_type") or "")
    pred_norm = _normalize_answer_by_type(prediction, answer_type)
    gold_norm = _normalize_answer_by_type(gold, answer_type)
    return {
        "prediction_raw": prediction,
        "prediction_norm": pred_norm,
        "gold_norm": gold_norm,
        "exact_correct": bool(pred_norm == gold_norm),
    }


def run_frontier_benchmark(
    *,
    experiment_dir: Path,
    model_specs: list[str],
    limit: int = 0,
    only_question_types: list[str] | None = None,
    out_dir: Path | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    rows = _load_jsonl(experiment_dir / "ocr_qa_dataset.jsonl")
    if only_question_types:
        wanted = {str(x).strip().upper() for x in only_question_types if str(x).strip()}
        rows = [row for row in rows if str((row.get("tags") or {}).get("question_type", "")).upper() in wanted]
    if limit > 0:
        rows = rows[: int(limit)]
    parsed_specs = [_parse_model_spec(text) for text in model_specs]
    out_dir = out_dir or (experiment_dir / "evals" / "frontier_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in parsed_specs:
        per_model: list[dict[str, Any]] = []
        errors = 0

        def run_one(row: dict[str, Any]) -> dict[str, Any]:
            try:
                result = _call_model(spec, row)
                scored = _score_prediction(row, result["answer"])
                return {
                    "sample_id": row.get("sample_id"),
                    "image_id": row.get("image_id"),
                    "question": row.get("question"),
                    "gold_answer": row.get("answer"),
                    "question_type": (row.get("tags") or {}).get("question_type"),
                    "difficulty": (row.get("tags") or {}).get("difficulty"),
                    "ambiguity_level": (row.get("tags") or {}).get("ambiguity_level"),
                    "answer_type": (row.get("tags") or {}).get("answer_type"),
                    "provider": spec.provider,
                    "model": spec.model,
                    "requested_model": spec.requested_model or spec.model,
                    "usage": result.get("usage", {}),
                    **scored,
                }
            except Exception as exc:
                return {
                    "sample_id": row.get("sample_id"),
                    "image_id": row.get("image_id"),
                    "question": row.get("question"),
                    "gold_answer": row.get("answer"),
                    "question_type": (row.get("tags") or {}).get("question_type"),
                    "difficulty": (row.get("tags") or {}).get("difficulty"),
                    "ambiguity_level": (row.get("tags") or {}).get("ambiguity_level"),
                    "answer_type": (row.get("tags") or {}).get("answer_type"),
                    "provider": spec.provider,
                    "model": spec.model,
                    "requested_model": spec.requested_model or spec.model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "prediction_raw": "",
                    "prediction_norm": "",
                    "gold_norm": _normalize_answer_by_type(str(row.get("answer") or ""), str((row.get("tags") or {}).get("answer_type") or "")),
                    "exact_correct": False,
                }
        if int(workers) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
                futures = [ex.submit(run_one, row) for row in rows]
                for fut in concurrent.futures.as_completed(futures):
                    per_model.append(fut.result())
        else:
            per_model = [run_one(row) for row in rows]
        per_model.sort(key=lambda row: str(row.get("sample_id") or ""))
        errors = sum(1 for row in per_model if row.get("error_type"))
        prediction_rows.extend(per_model)
        valid = [row for row in per_model if not row.get("error_type")]
        accuracy = (
            sum(1 for row in valid if bool(row.get("exact_correct"))) / len(valid)
            if valid
            else 0.0
        )
        summaries.append(
            {
                "provider": spec.provider,
                "model": spec.model,
                "requested_model": spec.requested_model or spec.model,
                "rows": len(per_model),
                "valid_rows": len(valid),
                "errors": errors,
                "exact_accuracy": accuracy,
            }
        )
        write_jsonl(out_dir / "predictions.jsonl", prediction_rows)
        write_json(
            out_dir / "summary.json",
            {
                "experiment_dir": str(experiment_dir),
                "rows": len(rows),
                "models": summaries,
            },
        )
    write_jsonl(out_dir / "predictions.jsonl", prediction_rows)
    summary = {
        "experiment_dir": str(experiment_dir),
        "rows": len(rows),
        "models": summaries,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def compute_frontier_agreement(*, benchmark_dir: Path) -> dict[str, Any]:
    rows = _load_jsonl(benchmark_dir / "predictions.jsonl")
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sample.setdefault(str(row.get("sample_id") or ""), []).append(row)
    pair_counts: dict[tuple[str, str], dict[str, int]] = {}
    sample_rows: list[dict[str, Any]] = []
    high_level_groups: dict[tuple[str, str], list[int]] = {}
    for sample_id, items in by_sample.items():
        items = [row for row in items if not row.get("error_type")]
        if len(items) < 2:
            continue
        model_keys = [f"{row['provider']}:{row['model']}" for row in items]
        preds = [str(row.get("prediction_norm") or "") for row in items]
        unanimous = len(set(preds)) == 1
        sample_rows.append(
            {
                "sample_id": sample_id,
                "question_type": items[0].get("question_type"),
                "difficulty": items[0].get("difficulty"),
                "ambiguity_level": items[0].get("ambiguity_level"),
                "models_present": model_keys,
                "predictions": preds,
                "unanimous_exact_agreement": unanimous,
            }
        )
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = f"{items[i]['provider']}:{items[i]['model']}"
                b = f"{items[j]['provider']}:{items[j]['model']}"
                key = tuple(sorted((a, b)))
                bucket = pair_counts.setdefault(key, {"matches": 0, "total": 0})
                bucket["total"] += 1
                if preds[i] == preds[j]:
                    bucket["matches"] += 1
        group_key = (
            str(items[0].get("question_type") or "unknown"),
            str(items[0].get("difficulty") or "unknown"),
        )
        high_level_groups.setdefault(group_key, []).append(1 if unanimous else 0)
    pairwise = [
        {
            "model_a": key[0],
            "model_b": key[1],
            "matches": stats["matches"],
            "total": stats["total"],
            "agreement_rate": (stats["matches"] / stats["total"]) if stats["total"] else 0.0,
        }
        for key, stats in sorted(pair_counts.items())
    ]
    by_group = [
        {
            "question_type": key[0],
            "difficulty": key[1],
            "rows": len(vals),
            "unanimous_rate": statistics.mean(vals) if vals else 0.0,
        }
        for key, vals in sorted(high_level_groups.items())
    ]
    summary = {
        "rows": len(sample_rows),
        "pairwise": pairwise,
        "overall_unanimous_rate": (
            sum(1 for row in sample_rows if row["unanimous_exact_agreement"]) / len(sample_rows)
            if sample_rows
            else 0.0
        ),
        "by_question_type_difficulty": by_group,
    }
    write_jsonl(benchmark_dir / "agreement_by_sample.jsonl", sample_rows)
    write_json(benchmark_dir / "agreement_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare and run dev200 eval lanes.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_prepare = sub.add_parser("prepare-bundle-evals")
    ap_prepare.add_argument("--bundle-id", default=DEFAULT_BUNDLE_ID)

    ap_bench = sub.add_parser("run-frontier-benchmark")
    ap_bench.add_argument("--experiment-dir", required=True)
    ap_bench.add_argument("--model", action="append", dest="models", default=[])
    ap_bench.add_argument("--limit", type=int, default=0)
    ap_bench.add_argument("--question-type", action="append", dest="question_types", default=[])
    ap_bench.add_argument("--out-dir", default="")
    ap_bench.add_argument("--workers", type=int, default=1)

    ap_agree = sub.add_parser("compute-frontier-agreement")
    ap_agree.add_argument("--benchmark-dir", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "prepare-bundle-evals":
        payload = prepare_bundle_evals(str(args.bundle_id))
        print(json.dumps(payload, indent=2))
        return
    if args.cmd == "run-frontier-benchmark":
        models = list(args.models or DEFAULT_MODEL_SPECS)
        payload = run_frontier_benchmark(
            experiment_dir=Path(args.experiment_dir),
            model_specs=models,
            limit=int(args.limit),
            only_question_types=list(args.question_types or []),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            workers=int(args.workers),
        )
        print(json.dumps(payload, indent=2))
        return
    if args.cmd == "compute-frontier-agreement":
        payload = compute_frontier_agreement(benchmark_dir=Path(args.benchmark_dir))
        print(json.dumps(payload, indent=2))
        return
    raise SystemExit(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
