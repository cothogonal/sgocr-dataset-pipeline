from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .semantic_grounding import sanitize_anchor_label


OFFICIAL_QWEN_BBOX_PROMPT = (
    'Locate every instance that belongs to the following categories: "{categories}". '
    'Report bbox coordinates in JSON format like this: {{"bbox_2d": [x1, y1, x2, y2], "label": "category"}}.'
)

OPEN_QWEN_LOCAL_DISCOVERY_PROMPT = (
    'Locate the visible objects, surfaces, or object parts in this crop that could naturally anchor nearby text. '
    'Report bbox coordinates in JSON format like this: {{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}.'
)


def _lazy_import_vllm() -> tuple[Any, Any, Any, Any]:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    return process_vision_info, AutoProcessor, LLM, SamplingParams


def _strip_code_fences(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _iter_json_candidates(text: str) -> list[Any]:
    cleaned = _strip_code_fences(text)
    if not cleaned:
        return []
    for candidate in (cleaned, cleaned[cleaned.find("[") :] if "[" in cleaned else "", cleaned[cleaned.find("{") :] if "{" in cleaned else ""):
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except Exception:
            pass
    matches = re.findall(r"\{[^{}]*\"bbox_2d\"[^{}]*\}", cleaned, flags=re.DOTALL)
    out: list[Any] = []
    for match in matches:
        try:
            out.append(json.loads(match))
        except Exception:
            continue
    return out


def _scale_relative_bbox(bbox: list[float], *, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    abs_box = [
        max(0.0, min(round(x1 * width / 1000.0, 2), float(width - 1))),
        max(0.0, min(round(y1 * height / 1000.0, 2), float(height - 1))),
        max(0.0, min(round(x2 * width / 1000.0, 2), float(width - 1))),
        max(0.0, min(round(y2 * height / 1000.0, 2), float(height - 1))),
    ]
    if abs_box[2] < abs_box[0]:
        abs_box[0], abs_box[2] = abs_box[2], abs_box[0]
    if abs_box[3] < abs_box[1]:
        abs_box[1], abs_box[3] = abs_box[3], abs_box[1]
    return abs_box


@dataclass(frozen=True)
class QwenAnchorRequest:
    image_id: str
    categories: list[str]
    image_path: str | None = None
    image_obj: Image.Image | None = None
    prompt_text: str | None = None
    enforce_allowed_labels: bool = True


def normalize_qwen_description(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels: list[str] = []
    seen: set[str] = set()
    regions: list[dict[str, Any]] = []
    raw_text = ""
    for row in rows:
        label = str(row.get("raw_label") or row.get("label") or "").strip()
        if label and label not in seen:
            seen.add(label)
            labels.append(label)
        box = row.get("box")
        if isinstance(box, list) and len(box) >= 4 and label:
            regions.append(
                {
                    "label": label,
                    "box": [round(float(value), 2) for value in box[:4]],
                }
            )
        if not raw_text:
            raw_text = str(row.get("raw_text") or "").strip()
    caption = "; ".join(labels[:6]) if labels else raw_text
    return {
        "caption": caption,
        "labels": labels,
        "regions": regions[:12],
        "raw_text": raw_text,
    }


def _normalize_to_allowed_label(label: str, allowed_categories: list[str]) -> str | None:
    raw = str(label or "").strip()
    if not raw:
        return None
    allowed_clean = [str(cat).strip() for cat in allowed_categories if str(cat).strip()]
    allowed_set = set(allowed_clean)
    if raw in allowed_set:
        return raw
    normalized = sanitize_anchor_label(raw)
    if normalized and normalized in allowed_set:
        return normalized
    return None


class QwenAnchorGrounderVLLM:
    def __init__(
        self,
        *,
        model_name: str,
        gpu_memory_utilization: float = 0.90,
        batch_size: int = 2,
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 9800 * 32 * 32,
        max_model_len: int = 2048,
    ) -> None:
        process_vision_info, AutoProcessor, LLM, SamplingParams = _lazy_import_vllm()
        self._process_vision_info = process_vision_info
        self._processor = AutoProcessor.from_pretrained(model_name)
        self._llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=float(gpu_memory_utilization),
            max_model_len=int(max_model_len),
            enforce_eager=False,
            tensor_parallel_size=1,
            seed=0,
        )
        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=768,
            top_k=-1,
            stop_token_ids=[],
        )
        self._batch_size = max(1, int(batch_size))
        self._min_pixels = int(min_pixels)
        self._max_pixels = int(max_pixels)

    def _prepare_inputs_for_vllm(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages,
            image_patch_size=self._processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data: dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

    def detect_many(self, requests: list[QwenAnchorRequest]) -> dict[str, list[dict[str, Any]]]:
        if not requests:
            return {}
        inputs: list[dict[str, Any]] = []
        image_sizes: dict[str, tuple[int, int]] = {}
        for req in requests:
            image_ref: Any
            if req.image_obj is not None:
                image = req.image_obj.convert("RGB")
                width, height = image.size
                image_ref = image
            elif req.image_path:
                image_path = Path(req.image_path)
                with Image.open(image_path) as image:
                    width, height = image.size
                image_ref = str(image_path)
            else:
                raise ValueError("QwenAnchorRequest requires image_path or image_obj")
            image_sizes[req.image_id] = (width, height)
            categories = ", ".join(dict.fromkeys(cat for cat in req.categories if str(cat).strip()))
            prompt = req.prompt_text or OFFICIAL_QWEN_BBOX_PROMPT.format(categories=categories)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_ref,
                            "min_pixels": self._min_pixels,
                            "max_pixels": self._max_pixels,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            inputs.append(self._prepare_inputs_for_vllm(messages))
        out: dict[str, list[dict[str, Any]]] = {}
        for start in range(0, len(inputs), self._batch_size):
            batch_requests = requests[start : start + self._batch_size]
            batch_inputs = inputs[start : start + self._batch_size]
            outputs = self._llm.generate(batch_inputs, sampling_params=self._sampling_params)
            for req, output in zip(batch_requests, outputs):
                width, height = image_sizes[req.image_id]
                generated_text = output.outputs[0].text if output.outputs else ""
                rows: list[dict[str, Any]] = []
                for item in _iter_json_candidates(generated_text):
                    bbox = item.get("bbox_2d")
                    raw_label = str(item.get("label") or "").strip()
                    label = _normalize_to_allowed_label(raw_label, req.categories) if req.enforce_allowed_labels else raw_label
                    if not isinstance(bbox, list) or len(bbox) < 4 or not label:
                        continue
                    rows.append(
                        {
                            "label": label,
                            "raw_label": raw_label,
                            "box": _scale_relative_bbox(bbox, width=width, height=height),
                            "score": 0.62,
                            "source": "qwen3_vl_vllm",
                            "prompt_text": prompt,
                            "raw_text": generated_text,
                        }
                    )
                out[req.image_id] = rows
        return out

    def close(self) -> None:
        del self._llm
