from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import re
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import time

import requests
from collections import Counter
from PIL import Image

from ..bootstrap import write_json, write_jsonl
from ..paths import LOGS_ROOT, OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT, repo_relative
from ..secrets import GEMINI, OPENAI, get_secret
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
            "path": image_path,
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

    if question_type == "REVERSE_GROUND":
        anchor_label = str(tags.get("anchor_label") or "")
        anchor_hint = f" (anchor object: {anchor_label})" if anchor_label else ""
        return (
            "Answer the visual question using the image.\n"
            "This is a location question — answer with a SHORT plain-text location phrase only.\n"
            "Do NOT return coordinates, JSON, point data, arrays, or any structured format.\n"
            "Do NOT say 'I cannot determine' or refuse — give the most visually plausible answer.\n"
            "Answer length: 1 to 6 words. No explanation.\n"
            f"Example: Q: 'Where is the text \"EXIT\"?' → A: 'on the green sign'\n"
            f"Question type: REVERSE_GROUND{anchor_hint}\n"
            f"Question: {row.get('question', '')}"
        )

    return (
        "Answer the visual question using the image.\n"
        "Give only the shortest literal answer.\n"
        "Do not explain.\n"
        "Do not add punctuation unless it is part of the answer.\n"
        f"Question type: {question_type or 'unknown'}\n"
        f"Question: {row.get('question', '')}"
    )


def _ambiguity_prompt(row: dict[str, Any]) -> str:
    tags = dict(row.get("tags") or {})
    question_type = str(tags.get("question_type") or row.get("question_type") or "").strip().upper()
    return (
        "You are judging whether a visual question-answer tuple is ambiguous relative to the image.\n"
        "Decide whether the tuple is UNDERDETERMINED: would a careful annotator reasonably think that more than one text region, "
        "anchor object, or location could satisfy this question-answer pair?\n"
        "\n"
        "Judge ambiguity, NOT factual correctness. Assume the proposed answer is intended; decide whether the wording still leaves "
        "multiple plausible referents in the image.\n"
        "\n"
        "Mark ambiguous=true when any of these hold:\n"
        "- the question/answer could refer to multiple text regions or multiple similar anchors\n"
        "- the anchor phrase is too generic (for example sign, wall, document, box, label) for this image\n"
        "- repeated same-type objects need extra color/part/relative-position detail that is missing\n"
        "- localization is repetitive or vague and does not uniquely isolate one text target\n"
        "\n"
        "Mark ambiguous=false when the question-answer pair clearly isolates one intended text target, even if the image contains other text.\n"
        "\n"
        "Return JSON only with this exact schema:\n"
        "{\"ambiguous\": true, \"confidence\": 0.0, \"reason\": \"short phrase\"}\n"
        "\n"
        f"Question type: {question_type or 'UNKNOWN'}\n"
        f"Question: {row.get('question', '')}\n"
        f"Proposed answer: {row.get('answer', '')}"
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_ambiguity_response(text: str) -> dict[str, Any]:
    payload = _extract_json_object(text)
    if not isinstance(payload, dict):
        lowered = str(text or "").strip().lower()
        if "ambiguous" in lowered and "false" in lowered:
            return {"ambiguous": False, "confidence": 0.0, "reason": "freeform_false", "parse_ok": False}
        if "ambiguous" in lowered and "true" in lowered:
            return {"ambiguous": True, "confidence": 0.0, "reason": "freeform_true", "parse_ok": False}
        if lowered.startswith("yes"):
            return {"ambiguous": True, "confidence": 0.0, "reason": "freeform_yes", "parse_ok": False}
        if lowered.startswith("no"):
            return {"ambiguous": False, "confidence": 0.0, "reason": "freeform_no", "parse_ok": False}
        raise ValueError(f"Could not parse ambiguity JSON from: {text[:200]}")
    ambiguous = bool(payload.get("ambiguous"))
    confidence_raw = payload.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(payload.get("reason") or "").strip()
    return {
        "ambiguous": ambiguous,
        "confidence": confidence,
        "reason": reason,
        "parse_ok": True,
    }


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


def _call_openai_text_only_eval(model: str, row: dict[str, Any], *, timeout_s: int = 120) -> dict[str, Any]:
    """Text-only eval using OpenAI API — no image. Adversarial RG vision-dependence probe.

    Mirrors _call_openai_responses but omits the image from the input content. Used to
    check whether an RG question can be answered from text alone, identifying text-leaky rows.
    """
    # reasoning models (codex, o-series) accept the `reasoning` param; non-reasoning models don't.
    _is_reasoning_model = any(tok in model for tok in ("codex", "o1", "o3", "o4", "-o-"))
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _benchmark_prompt(row)},
                ],
            }
        ],
        "max_output_tokens": 32,
    }
    if _is_reasoning_model:
        body["reasoning"] = {"effort": "low"}
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
        "mode": "text_only",
    }


def _call_openai_ambiguity_eval(model: str, row: dict[str, Any], *, timeout_s: int = 120) -> dict[str, Any]:
    image_b64, width, height = _encode_image(Path(row["image_path"]))
    body = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": _ambiguity_prompt(row)},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                ],
            }
        ],
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
        "max_output_tokens": 120,
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
    parsed = _parse_ambiguity_response(answer)
    return {
        "provider": "openai",
        "model": model,
        "raw_response": answer,
        "usage": payload.get("usage", {}),
        "image_width": width,
        "image_height": height,
        **parsed,
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


def _gemini_post_with_backoff(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout_s: int,
    max_retries: int = 6,
    base_delay: float = 2.0,
) -> requests.Response:
    """POST to Gemini with exponential backoff on 429 / 5xx responses."""
    for attempt in range(max_retries + 1):
        resp = requests.post(url, headers=headers, json=body, timeout=timeout_s)
        if resp.status_code not in {429, 500, 502, 503, 504} or attempt == max_retries:
            resp.raise_for_status()
            return resp
        delay = base_delay * (2 ** attempt)
        # Honour Retry-After header when present
        retry_after = resp.headers.get("Retry-After")
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                pass
        time.sleep(delay)
    resp.raise_for_status()  # unreachable, satisfies type checker
    return resp


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
    resp = _gemini_post_with_backoff(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "x-goog-api-key": get_secret(GEMINI),
            "Content-Type": "application/json",
        },
        body=body,
        timeout_s=timeout_s,
    )
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


def _call_gemini_ambiguity_eval(model: str, row: dict[str, Any], *, timeout_s: int = 120) -> dict[str, Any]:
    image_b64, width, height = _encode_image(Path(row["image_path"]))
    body = {
        "contents": [
            {
                "parts": [
                    {"text": _ambiguity_prompt(row)},
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                ]
            }
        ],
        "generationConfig": {
            **_gemini_generation_config(model),
            "maxOutputTokens": 120,
        },
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
    answer = " ".join(parts).strip()
    parsed = _parse_ambiguity_response(answer)
    return {
        "provider": "gemini",
        "model": model,
        "raw_response": answer,
        "usage": payload.get("usageMetadata", {}),
        "image_width": width,
        "image_height": height,
        **parsed,
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


def _call_gemini_text_only_eval(model: str, row: dict[str, Any], *, timeout_s: int = 120) -> dict[str, Any]:
    """Same as _call_gemini_eval but sends only the question text — no image.

    Used to measure image-dependence: Δ = score(with_image) - score(text_only).
    A high Δ means the question genuinely requires vision.
    """
    body = {
        "contents": [
            {
                "parts": [
                    {"text": _benchmark_prompt(row)},
                ]
            }
        ],
        "generationConfig": _gemini_generation_config(model),
    }
    resp = _gemini_post_with_backoff(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "x-goog-api-key": get_secret(GEMINI),
            "Content-Type": "application/json",
        },
        body=body,
        timeout_s=timeout_s,
    )
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
        "mode": "text_only",
    }


def run_image_dependence_eval(
    *,
    experiment_dir: Path,
    model: str = "gemini-3-flash-preview",
    dataset_path: Path | None = None,
    limit: int = 0,
    out_dir: Path | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Run Gemini Flash twice per row (with image and without image) to measure vision necessity.

    For each row:
      - image_soft: soft_correct with image
      - text_only_soft: soft_correct with question text only
      - vision_delta: image_soft - text_only_soft
      - vision_necessary: image_soft == 1 and text_only_soft == 0

    Aggregate stats indicate what fraction of QAs genuinely required the image.
    """
    model = GEMINI_MODEL_ALIASES.get(model, model)
    dataset_path = dataset_path or (experiment_dir / "ocr_qa_dataset.jsonl")
    rows = _load_jsonl(dataset_path)
    if limit > 0:
        rows = rows[: int(limit)]
    out_dir = out_dir or (experiment_dir / "evals" / "image_dependence")
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        try:
            image_result = _call_gemini_eval(model, row)
            image_scored = _score_prediction(row, image_result["answer"])
        except Exception as exc:
            image_scored = {"soft_correct": False, "error": str(exc)}
        try:
            text_result = _call_gemini_text_only_eval(model, row)
            text_scored = _score_prediction(row, text_result["answer"])
        except Exception as exc:
            text_scored = {"soft_correct": False, "error": str(exc)}

        image_soft = int(bool(image_scored.get("soft_correct")))
        text_only_soft = int(bool(text_scored.get("soft_correct")))
        vision_delta = image_soft - text_only_soft
        return {
            "sample_id": row.get("sample_id"),
            "image_id": row.get("image_id"),
            "question_type": (row.get("tags") or {}).get("question_type"),
            "answer_type": (row.get("tags") or {}).get("answer_type"),
            "image_soft": image_soft,
            "text_only_soft": text_only_soft,
            "vision_delta": vision_delta,
            "vision_necessary": vision_delta > 0,
            "text_leaky": text_only_soft == 1,
            "image_prediction": image_result.get("answer") if "error" not in image_scored else "",
            "text_prediction": text_result.get("answer") if "error" not in text_scored else "",
            "gold_answer": row.get("answer"),
        }

    if int(workers) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futures = [ex.submit(run_one, row) for row in rows]
            result_rows = [fut.result() for fut in concurrent.futures.as_completed(futures)]
    else:
        result_rows = [run_one(row) for row in rows]
    result_rows.sort(key=lambda r: str(r.get("sample_id") or ""))

    n = len(result_rows) or 1
    image_acc = sum(r["image_soft"] for r in result_rows) / n
    text_acc = sum(r["text_only_soft"] for r in result_rows) / n
    vision_delta_mean = sum(r["vision_delta"] for r in result_rows) / n
    vision_necessary_rate = sum(1 for r in result_rows if r["vision_necessary"]) / n
    text_leaky_rate = sum(1 for r in result_rows if r["text_leaky"]) / n

    def _by_type(qtype: str) -> dict[str, float]:
        subset = [r for r in result_rows if str(r.get("question_type") or "").upper() == qtype]
        nn = len(subset) or 1
        return {
            "n": len(subset),
            "image_acc": round(sum(r["image_soft"] for r in subset) / nn, 4),
            "text_only_acc": round(sum(r["text_only_soft"] for r in subset) / nn, 4),
            "vision_delta": round(sum(r["vision_delta"] for r in subset) / nn, 4),
            "vision_necessary_rate": round(sum(1 for r in subset if r["vision_necessary"]) / nn, 4),
            "text_leaky_rate": round(sum(1 for r in subset if r["text_leaky"]) / nn, 4),
        }

    summary = {
        "experiment_dir": str(experiment_dir),
        "dataset_path": str(dataset_path),
        "model": model,
        "rows": len(result_rows),
        "image_accuracy": round(image_acc, 4),
        "text_only_accuracy": round(text_acc, 4),
        "vision_delta_mean": round(vision_delta_mean, 4),
        "vision_necessary_rate": round(vision_necessary_rate, 4),
        "text_leaky_rate": round(text_leaky_rate, 4),
        "by_type": {
            "DIRECT_READ": _by_type("DIRECT_READ"),
            "REVERSE_GROUND": _by_type("REVERSE_GROUND"),
            "YES_NO": _by_type("YES_NO"),
            "TEXT_PROPERTY": _by_type("TEXT_PROPERTY"),
            "ANCHOR_PROPERTY": _by_type("ANCHOR_PROPERTY"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "rows.jsonl", result_rows)
    return summary


def apply_vision_dependence_gate(
    rows: list[dict[str, Any]],
    *,
    model: str = "gemini-3-flash-preview",
    workers: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter accepted rows by vision-dependence: reject any row where text-only eval is sufficient.

    Runs one Gemini Flash call (text-only, no image) per row. If the model gets the answer
    correct from question text alone, the row is rejected — it doesn't require the image.
    This is a verification gate rather than a heuristic filter: it empirically tests the
    property we care about.

    Returns (accepted_rows, gate_stats).
    """
    model = GEMINI_MODEL_ALIASES.get(model, model)

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        try:
            text_result = _call_gemini_text_only_eval(model, row)
            text_scored = _score_prediction(row, text_result["answer"])
            text_leaky = bool(text_scored.get("soft_correct"))
        except Exception:
            text_leaky = False  # on error, conservatively accept the row
        return {"sample_id": row.get("sample_id"), "text_leaky": text_leaky}

    if int(workers) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
            results_list = list(ex.map(run_one, rows))
    else:
        results_list = [run_one(row) for row in rows]

    leak_by_id = {r["sample_id"]: r["text_leaky"] for r in results_list}
    accepted = [row for row in rows if not leak_by_id.get(row.get("sample_id"), False)]
    rejected_count = len(rows) - len(accepted)
    leaky_rate = round(rejected_count / max(len(rows), 1), 4)
    gate_stats = {
        "model": model,
        "total": len(rows),
        "accepted": len(accepted),
        "rejected": rejected_count,
        "text_leaky_rate": leaky_rate,
    }
    return accepted, gate_stats


# Color and shape tokens used to detect leakable surface artifacts in RG question text.
# Defined locally (not imported from dev40_complete) to avoid circular import.
_RG_COLOR_TOKENS: frozenset[str] = frozenset({
    "red", "blue", "green", "brown", "white", "black", "gray", "grey",
    "yellow", "orange", "purple", "pink", "silver", "gold",
})
_RG_SHAPE_TOKENS: frozenset[str] = frozenset({
    "rectangular", "circular", "square", "oval", "round",
    "triangular", "hexagonal", "cylindrical", "spherical",
    "wedge", "segment", "emblem", "badge", "bar",
})


def _strip_rg_leakage_tokens(question: str) -> str:
    """Strip standalone color and shape tokens from an RG question string.

    Returns the cleaned question, or the original string unchanged if nothing was stripped.
    Only strips whole-word matches — partial matches inside longer words are not affected.
    Trailing punctuation on each word is stripped before comparison.
    """
    words = question.split()
    cleaned = [
        w for w in words
        if w.lower().rstrip(".,?!") not in _RG_COLOR_TOKENS
        and w.lower().rstrip(".,?!") not in _RG_SHAPE_TOKENS
    ]
    if len(cleaned) == len(words):
        return question
    return " ".join(cleaned)


def apply_rg_vdep_check(
    rows: list[dict[str, Any]],
    *,
    model: str,
    workers: int = 4,
    correction_enabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Vision-dependence gate for REVERSE_GROUND rows using a single cross-model text-only call.

    Non-RG rows pass through unconditionally. For each RG row, runs one text-only call
    via the provider specified in `model` (format: 'provider:model_name', e.g.
    'openai:gpt-5.3-codex'). If the model answers correctly without the image, the row is
    text-leaky and rejected. API errors are treated as conservative keep (don't reject on
    eval failure).

    When correction_enabled=True, rejected rows undergo a correction pass: color and shape
    tokens are stripped from the question and the vdep check is re-run once. Rows that pass
    after correction are included in the output with rg_correction_applied=True.

    Returns (accepted_rows, gate_stats).
    """
    if ":" in model:
        provider, model_name = model.split(":", 1)
    else:
        provider, model_name = "openai", model

    rg_rows = [row for row in rows if str(row.get("question_type") or "") == "REVERSE_GROUND"]
    non_rg_rows = [row for row in rows if str(row.get("question_type") or "") != "REVERSE_GROUND"]

    stats: dict[str, Any] = {
        "model": model,
        "total_rg": len(rg_rows),
        "accepted_rg": 0,
        "rejected_rg": 0,
        "text_leaky_rg_rate": 0.0,
        "corrected_rg": 0,
    }

    if not rg_rows:
        return rows, stats

    def _call_text_only(row: dict[str, Any]) -> dict[str, Any]:
        if provider == "openai":
            return _call_openai_text_only_eval(model_name, row)
        return _call_gemini_text_only_eval(model_name, row)

    def _is_leaky(row: dict[str, Any]) -> bool:
        try:
            result = _call_text_only(row)
            scored = _score_prediction(row, result["answer"])
            return bool(scored.get("soft_correct"))
        except Exception:
            return False  # conservative keep on API error

    if int(workers) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
            leaky_flags = list(ex.map(_is_leaky, rg_rows))
    else:
        leaky_flags = [_is_leaky(r) for r in rg_rows]

    accepted_rg: list[dict[str, Any]] = []
    rejected_rg: list[dict[str, Any]] = []
    for row, leaky in zip(rg_rows, leaky_flags):
        if leaky:
            rejected_rg.append(row)
        else:
            accepted_rg.append(row)

    corrected_count = 0
    if correction_enabled and rejected_rg:
        correctable: list[dict[str, Any]] = []
        for row in rejected_rg:
            q = str(row.get("question") or "")
            corrected_q = _strip_rg_leakage_tokens(q)
            if corrected_q != q and corrected_q.strip():
                correctable.append({**row, "question": corrected_q})

        if correctable:
            if int(workers) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
                    correction_leaky = list(ex.map(_is_leaky, correctable))
            else:
                correction_leaky = [_is_leaky(r) for r in correctable]

            for row, still_leaky in zip(correctable, correction_leaky):
                if not still_leaky:
                    row["rg_correction_applied"] = True
                    accepted_rg.append(row)
                    corrected_count += 1

    total_accepted_rg = len(accepted_rg)
    total_rejected_rg = len(rg_rows) - total_accepted_rg
    stats["accepted_rg"] = total_accepted_rg
    stats["rejected_rg"] = total_rejected_rg
    stats["text_leaky_rg_rate"] = round(total_rejected_rg / max(len(rg_rows), 1), 4)
    stats["corrected_rg"] = corrected_count

    return non_rg_rows + accepted_rg, stats


def _call_model_ambiguity(spec: BenchmarkModelSpec, row: dict[str, Any]) -> dict[str, Any]:
    if spec.provider == "openai":
        return _call_openai_ambiguity_eval(spec.model, row)
    if spec.provider == "gemini":
        return _call_gemini_ambiguity_eval(spec.model, row)
    raise ValueError(f"Unsupported provider: {spec.provider}")


def _strip_accents(s: str) -> str:
    """Normalize to NFKC (fullwidth/ligature decomposition) then NFD and remove combining accent marks."""
    s = unicodedata.normalize("NFKC", s)
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


_RG_STOPWORDS: frozenset[str] = frozenset({
    # English function words
    "a", "an", "the", "in", "on", "at", "of", "to", "is", "it", "be", "as",
    "and", "or", "for", "with", "by", "from", "this", "that", "are", "was",
    "not", "but", "if", "its", "into", "do", "so",
    # SGOCR gold-template words (appear in nearly every gold answer, so useless as discriminators)
    "area", "image", "text",
    # Common spatial connectors that appear in any location description
    "near", "around",
})


def _word_f1(gold_norm: str, pred_norm: str) -> float:
    """Raw unigram word-level F1, accent-insensitive."""
    g = _strip_accents(gold_norm).split()
    p = _strip_accents(pred_norm).split()
    if not g or not p:
        return 0.0
    common = sum((Counter(g) & Counter(p)).values())
    precision = common / len(p)
    recall = common / len(g)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _content_word_f1(gold_norm: str, pred_norm: str) -> float:
    """Content-word F1 for REVERSE_GROUND: strips stopwords and template words before scoring.

    Raw word-F1 has false positives because spatial descriptions share many function words
    ("on", "the", "in", "of", "area", "image"). This version measures overlap on semantically
    meaningful words (object labels, directional terms: 'upper', 'lower', 'left', 'right',
    'center', 'top', 'bottom', anchor nouns, etc.).
    """
    def content_words(s: str) -> list[str]:
        return [w for w in s.split() if w not in _RG_STOPWORDS and len(w) > 1]

    g = content_words(gold_norm)
    p = content_words(pred_norm)
    if not g or not p:
        return 0.0
    common = sum((Counter(g) & Counter(p)).values())
    precision = common / len(p)
    recall = common / len(g)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _partial_correct_direct_read(gold_norm: str, pred_norm: str) -> bool:
    """Partial credit for DIRECT_READ: the model read correct text but in more or less context.

    Accepts when:
    - Gold word-set ⊆ pred word-set (model read a superset — full caption vs. fragment).
    - Pred word-set ⊆ gold word-set AND pred is short ≤ 3 words (partial read).
    - Plural tolerance: trailing-s stems are compared (RAILHAWK ≈ RAILHAWKS).
    - Word-join tolerance: concatenated pred tokens match a single gold token, or vice versa
      (handles OCR merge artifacts: "EXTRALITE" ≈ "EXTRA LITE", "INGOD" ≈ "IN GOD").
    - Substring matching for short gold fragments (<3 chars after stripping).
    - Trailing OCR punctuation artifacts stripped from gold before comparison.
    """
    # Strip trailing OCR artifacts (e.g. "REFRIGERATORS]" → "REFRIGERATORS", "en," → "en")
    gold_clean = gold_norm.rstrip("][(.,;:!?)'\"").strip()
    if not gold_clean:
        return False
    # Accent-insensitive comparison throughout
    g = _strip_accents(gold_clean)
    p = _strip_accents(pred_norm)
    # Short gold fragments: substring containment instead of hard False
    # e.g. gold="en", pred="en cada" → True
    if len(gold_clean) < 3:
        return bool(g) and g in p
    g_words = g.split()
    p_words = p.split()
    if not g_words or not p_words:
        return False

    def _stem(w: str) -> str:
        return w[:-1] if len(w) > 3 and w.endswith("s") else w

    g_set = {_stem(w) for w in g_words}
    p_set = {_stem(w) for w in p_words}

    # Model read a superset of what was expected.
    if g_set.issubset(p_set):
        return True
    # Model gave a short sub-fragment of the gold (partial read).
    if len(p_words) <= 3 and p_set.issubset(g_set):
        return True
    # Word-join: OCR merged multi-word text into one token, or model split a merged token.
    # Only apply when the discrepancy is a single concatenated token vs. 2+ tokens.
    if len(g_words) == 1 and len(g) >= 4 and len(p_words) >= 2:
        if g == "".join(p_words):
            return True
    if len(p_words) == 1 and len(p) >= 4 and len(g_words) >= 2:
        if p == "".join(g_words):
            return True
    return False


def _score_prediction(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    tags = dict(row.get("tags") or {})
    question_type = str(tags.get("question_type") or row.get("question_type") or "").strip().upper()
    gold = str(row.get("answer") or "")
    answer_type = str(tags.get("answer_type") or "")
    pred_norm = _normalize_answer_by_type(prediction, answer_type)
    gold_norm = _normalize_answer_by_type(gold, answer_type)
    exact = bool(pred_norm == gold_norm)
    # Also accept accent-equivalent exact matches (e.g. "CORAZON" == "CORAZÓN" after stripping)
    if not exact:
        exact = bool(_strip_accents(pred_norm) == _strip_accents(gold_norm))

    # DIRECT_READ: partial credit when the model reads more/less context around the target text.
    partial = False
    if question_type == "DIRECT_READ" and not exact:
        partial = _partial_correct_direct_read(gold_norm, pred_norm)

    # REVERSE_GROUND: word-level F1 ≥ 0.5 counts as semantically correct.
    # Exact match is useless here — gold encodes our template format, frontier models use natural
    # language to describe the same location.
    word_f1_val = 0.0
    semantic = False
    if question_type == "REVERSE_GROUND":
        # Use content-word F1 (strips stopwords + template words) to avoid false positives
        # from shared function words like "on", "the", "in", "of", "area", "image".
        word_f1_val = _content_word_f1(gold_norm, pred_norm)
        semantic = word_f1_val >= 0.5

    # soft_correct: the most lenient correct signal per question type.
    soft = exact or partial or semantic

    return {
        "prediction_raw": prediction,
        "prediction_norm": pred_norm,
        "gold_norm": gold_norm,
        "exact_correct": exact,
        "partial_correct": partial,
        "semantic_correct": semantic,
        "word_f1": round(word_f1_val, 4),
        "soft_correct": soft,
    }


def run_frontier_benchmark(
    *,
    experiment_dir: Path,
    model_specs: list[str],
    dataset_path: Path | None = None,
    limit: int = 0,
    only_question_types: list[str] | None = None,
    out_dir: Path | None = None,
    workers: int = 1,
    write_index: bool = True,
) -> dict[str, Any]:
    dataset_path = dataset_path or (experiment_dir / "ocr_qa_dataset.jsonl")
    rows = _load_jsonl(dataset_path)
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
        n = len(valid) or 1
        exact_acc = sum(1 for r in valid if r.get("exact_correct")) / n
        # partial_correct: DIRECT_READ superset/subset credit (valid for all types, 0 outside DR)
        partial_acc = sum(1 for r in valid if r.get("exact_correct") or r.get("partial_correct")) / n
        # semantic_correct: REVERSE_GROUND word-F1 ≥ 0.5 credit
        semantic_acc = sum(1 for r in valid if r.get("exact_correct") or r.get("semantic_correct")) / n
        # soft_correct: best per-type signal combined
        soft_acc = sum(1 for r in valid if r.get("soft_correct")) / n

        def _acc_by_type(qtype: str) -> dict[str, float]:
            subset = [r for r in valid if str(r.get("question_type") or "").upper() == qtype]
            nn = len(subset) or 1
            return {
                "n": len(subset),
                "exact": round(sum(1 for r in subset if r.get("exact_correct")) / nn, 4),
                "soft": round(sum(1 for r in subset if r.get("soft_correct")) / nn, 4),
            }

        summaries.append(
            {
                "provider": spec.provider,
                "model": spec.model,
                "requested_model": spec.requested_model or spec.model,
                "rows": len(per_model),
                "valid_rows": len(valid),
                "errors": errors,
                "exact_accuracy": round(exact_acc, 4),
                "partial_accuracy": round(partial_acc, 4),
                "semantic_accuracy": round(semantic_acc, 4),
                "soft_accuracy": round(soft_acc, 4),
                "by_type": {
                    "DIRECT_READ": _acc_by_type("DIRECT_READ"),
                    "REVERSE_GROUND": _acc_by_type("REVERSE_GROUND"),
                    "YES_NO": _acc_by_type("YES_NO"),
                    "TEXT_PROPERTY": _acc_by_type("TEXT_PROPERTY"),
                    "ANCHOR_PROPERTY": _acc_by_type("ANCHOR_PROPERTY"),
                },
            }
        )
        write_jsonl(out_dir / "predictions.jsonl", prediction_rows)
        write_json(
            out_dir / "summary.json",
            {
                "experiment_dir": str(experiment_dir),
                "dataset_path": str(dataset_path),
                "rows": len(rows),
                "models": summaries,
            },
        )
    write_jsonl(out_dir / "predictions.jsonl", prediction_rows)
    summary = {
        "experiment_dir": str(experiment_dir),
        "dataset_path": str(dataset_path),
        "rows": len(rows),
        "models": summaries,
    }
    write_json(out_dir / "summary.json", summary)
    # Write a per-sample eval index back to the experiment dir so the review app can display
    # frontier model predictions alongside each QA pair without re-joining at load time.
    if write_index:
        _write_frontier_eval_index(experiment_dir, prediction_rows)
    return summary


def run_frontier_ambiguity_eval(
    *,
    experiment_dir: Path,
    model_specs: list[str],
    dataset_path: Path | None = None,
    limit: int = 0,
    only_question_types: list[str] | None = None,
    out_dir: Path | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    dataset_path = dataset_path or (experiment_dir / "ocr_qa_dataset.jsonl")
    rows = _load_jsonl(dataset_path)
    if only_question_types:
        wanted = {str(x).strip().upper() for x in only_question_types if str(x).strip()}
        rows = [row for row in rows if str((row.get("tags") or {}).get("question_type", "")).upper() in wanted]
    if limit > 0:
        rows = rows[: int(limit)]
    parsed_specs = [_parse_model_spec(text) for text in model_specs]
    out_dir = out_dir or (experiment_dir / "evals" / "frontier_ambiguity")
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for spec in parsed_specs:
        per_model: list[dict[str, Any]] = []

        def run_one(row: dict[str, Any]) -> dict[str, Any]:
            try:
                result = _call_model_ambiguity(spec, row)
                return {
                    "sample_id": row.get("sample_id"),
                    "image_id": row.get("image_id"),
                    "question": row.get("question"),
                    "proposed_answer": row.get("answer"),
                    "question_type": (row.get("tags") or {}).get("question_type"),
                    "difficulty": (row.get("tags") or {}).get("difficulty"),
                    "ambiguity_level": (row.get("tags") or {}).get("ambiguity_level"),
                    "provider": spec.provider,
                    "model": spec.model,
                    "requested_model": spec.requested_model or spec.model,
                    "usage": result.get("usage", {}),
                    "ambiguous": bool(result.get("ambiguous")),
                    "confidence": float(result.get("confidence") or 0.0),
                    "reason": str(result.get("reason") or ""),
                    "parse_ok": bool(result.get("parse_ok")),
                    "raw_response": str(result.get("raw_response") or ""),
                }
            except Exception as exc:
                return {
                    "sample_id": row.get("sample_id"),
                    "image_id": row.get("image_id"),
                    "question": row.get("question"),
                    "proposed_answer": row.get("answer"),
                    "question_type": (row.get("tags") or {}).get("question_type"),
                    "difficulty": (row.get("tags") or {}).get("difficulty"),
                    "ambiguity_level": (row.get("tags") or {}).get("ambiguity_level"),
                    "provider": spec.provider,
                    "model": spec.model,
                    "requested_model": spec.requested_model or spec.model,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "ambiguous": False,
                    "confidence": 0.0,
                    "reason": "",
                    "parse_ok": False,
                    "raw_response": "",
                }

        if int(workers) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
                futures = [ex.submit(run_one, row) for row in rows]
                for fut in concurrent.futures.as_completed(futures):
                    per_model.append(fut.result())
        else:
            per_model = [run_one(row) for row in rows]
        per_model.sort(key=lambda row: str(row.get("sample_id") or ""))
        valid = [row for row in per_model if not row.get("error_type")]
        n = len(valid) or 1

        def _rate_by_type(qtype: str) -> dict[str, float]:
            subset = [r for r in valid if str(r.get("question_type") or "").upper() == qtype]
            nn = len(subset) or 1
            return {
                "n": len(subset),
                "ambiguous_rate": round(sum(1 for r in subset if r.get("ambiguous")) / nn, 4),
                "mean_confidence": round(sum(float(r.get("confidence") or 0.0) for r in subset) / nn, 4),
            }

        prediction_rows.extend(per_model)
        summaries.append(
            {
                "provider": spec.provider,
                "model": spec.model,
                "requested_model": spec.requested_model or spec.model,
                "rows": len(per_model),
                "valid_rows": len(valid),
                "errors": sum(1 for row in per_model if row.get("error_type")),
                "ambiguous_rate": round(sum(1 for r in valid if r.get("ambiguous")) / n, 4),
                "mean_confidence": round(sum(float(r.get("confidence") or 0.0) for r in valid) / n, 4),
                "parse_ok_rate": round(sum(1 for r in valid if r.get("parse_ok")) / n, 4),
                "by_type": {
                    "DIRECT_READ": _rate_by_type("DIRECT_READ"),
                    "REVERSE_GROUND": _rate_by_type("REVERSE_GROUND"),
                    "YES_NO": _rate_by_type("YES_NO"),
                    "TEXT_PROPERTY": _rate_by_type("TEXT_PROPERTY"),
                    "ANCHOR_PROPERTY": _rate_by_type("ANCHOR_PROPERTY"),
                },
            }
        )
        write_jsonl(out_dir / "ambiguity_predictions.jsonl", prediction_rows)
        write_json(
            out_dir / "ambiguity_summary.json",
            {
                "experiment_dir": str(experiment_dir),
                "dataset_path": str(dataset_path),
                "rows": len(rows),
                "models": summaries,
            },
        )
    summary = {
        "experiment_dir": str(experiment_dir),
        "dataset_path": str(dataset_path),
        "rows": len(rows),
        "models": summaries,
    }
    write_jsonl(out_dir / "ambiguity_predictions.jsonl", prediction_rows)
    write_json(out_dir / "ambiguity_summary.json", summary)
    return summary


def _write_frontier_eval_index(experiment_dir: Path, prediction_rows: list[dict[str, Any]]) -> None:
    """Merge all frontier predictions into experiment_dir/frontier_evals_index.jsonl.

    Each row in the output file has the format:
      {"sample_id": "...", "evals": [{"model": "...", "prediction": "...", ...}, ...]}

    Existing entries for different models are preserved so that running multiple eval
    passes accumulates results rather than overwriting them.
    """
    index_path = experiment_dir / "frontier_evals_index.jsonl"
    # Load existing index
    by_sample: dict[str, dict[str, Any]] = {}
    if index_path.exists():
        for row in _iter_jsonl(index_path):
            sid = str(row.get("sample_id") or "")
            if sid:
                by_sample[sid] = row
    # Merge new predictions
    for row in prediction_rows:
        sid = str(row.get("sample_id") or "")
        if not sid or row.get("error_type"):
            continue
        entry = by_sample.setdefault(sid, {"sample_id": sid, "evals": []})
        existing_evals: list[dict[str, Any]] = entry.get("evals") or []  # type: ignore[assignment]
        model_key = str(row.get("requested_model") or row.get("model") or "")
        # Replace entry for same model, append for new models
        existing_evals = [e for e in existing_evals if str(e.get("model") or "") != model_key]
        existing_evals.append({
            "model": model_key,
            "provider": row.get("provider"),
            "prediction": row.get("prediction_raw") or "",
            "prediction_norm": row.get("prediction_norm") or "",
            "exact_correct": bool(row.get("exact_correct")),
            "partial_correct": bool(row.get("partial_correct")),
            "semantic_correct": bool(row.get("semantic_correct")),
            "soft_correct": bool(row.get("soft_correct")),
            "word_f1": row.get("word_f1"),
        })
        entry["evals"] = existing_evals
    write_jsonl(index_path, list(by_sample.values()))


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


def compute_frontier_ambiguity_agreement(*, benchmark_dir: Path) -> dict[str, Any]:
    rows = _load_jsonl(benchmark_dir / "ambiguity_predictions.jsonl")
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
        labels = [bool(row.get("ambiguous")) for row in items]
        unanimous = len(set(labels)) == 1
        sample_rows.append(
            {
                "sample_id": sample_id,
                "question_type": items[0].get("question_type"),
                "difficulty": items[0].get("difficulty"),
                "ambiguity_level": items[0].get("ambiguity_level"),
                "models_present": model_keys,
                "ambiguous_predictions": labels,
                "unanimous_ambiguity_agreement": unanimous,
            }
        )
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = f"{items[i]['provider']}:{items[i]['model']}"
                b = f"{items[j]['provider']}:{items[j]['model']}"
                key = tuple(sorted((a, b)))
                bucket = pair_counts.setdefault(key, {"matches": 0, "total": 0})
                bucket["total"] += 1
                if labels[i] == labels[j]:
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
            sum(1 for row in sample_rows if row["unanimous_ambiguity_agreement"]) / len(sample_rows)
            if sample_rows
            else 0.0
        ),
        "by_question_type_difficulty": by_group,
    }
    write_jsonl(benchmark_dir / "ambiguity_agreement_by_sample.jsonl", sample_rows)
    write_json(benchmark_dir / "ambiguity_agreement_summary.json", summary)
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

    ap_amb = sub.add_parser("run-frontier-ambiguity-eval")
    ap_amb.add_argument("--experiment-dir", required=True)
    ap_amb.add_argument("--model", action="append", dest="models", default=[])
    ap_amb.add_argument("--limit", type=int, default=0)
    ap_amb.add_argument("--question-type", action="append", dest="question_types", default=[])
    ap_amb.add_argument("--out-dir", default="")
    ap_amb.add_argument("--workers", type=int, default=1)

    ap_amb_agree = sub.add_parser("compute-frontier-ambiguity-agreement")
    ap_amb_agree.add_argument("--benchmark-dir", required=True)
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
    if args.cmd == "run-frontier-ambiguity-eval":
        models = list(args.models or DEFAULT_MODEL_SPECS)
        payload = run_frontier_ambiguity_eval(
            experiment_dir=Path(args.experiment_dir),
            model_specs=models,
            limit=int(args.limit),
            only_question_types=list(args.question_types or []),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            workers=int(args.workers),
        )
        print(json.dumps(payload, indent=2))
        return
    if args.cmd == "compute-frontier-ambiguity-agreement":
        payload = compute_frontier_ambiguity_agreement(benchmark_dir=Path(args.benchmark_dir))
        print(json.dumps(payload, indent=2))
        return
    raise SystemExit(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
