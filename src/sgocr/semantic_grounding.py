from __future__ import annotations

import base64
import io
import json
import re
from contextlib import nullcontext
from typing import Any

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModelForZeroShotObjectDetection, AutoProcessor

from .gemini_batch import GeminiBatchRequest, batch_generate_json, image_part_from_payload
from .semantic_dev40_tuning import load_semantic_dev40_tuning
from .secrets import GEMINI, get_secret

from .pipeline.anchor_analysis import *  # noqa: F401,F403
from .pipeline.anchor_analysis import (
    _dedupe_labels,
    is_generic_anchor_label,
    sanitize_anchor_label,
)


def _normalize_florence_description(raw_text: str, parsed: Any, *, task: str) -> dict[str, Any]:
    payload = parsed.get(task) if isinstance(parsed, dict) and task in parsed else parsed
    labels: list[str] = []
    regions: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        raw_labels = payload.get("labels") or payload.get("texts") or payload.get("phrases") or []
        raw_boxes = payload.get("bboxes") or payload.get("boxes") or []
        for index, label in enumerate(raw_labels):
            text = str(label or "").strip()
            if not text:
                continue
            labels.append(text)
            if index < len(raw_boxes):
                box = raw_boxes[index]
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    regions.append(
                        {
                            "label": text,
                            "box": [round(float(value), 2) for value in box[:4]],
                        }
                    )
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                text = str(item.get("label") or item.get("text") or "").strip()
                if not text:
                    continue
                labels.append(text)
                box = item.get("bbox") or item.get("box")
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    regions.append(
                        {
                            "label": text,
                            "box": [round(float(value), 2) for value in box[:4]],
                        }
                    )
            else:
                text = str(item or "").strip()
                if text:
                    labels.append(text)
    deduped = _dedupe_labels(labels)
    caption = "; ".join(deduped[:6]) if deduped else str(raw_text or "").strip()
    return {
        "caption": caption,
        "labels": deduped,
        "regions": regions[:12],
        "raw_text": str(raw_text or "").strip(),
    }


class FlorenceTagger:
    def __init__(self, *, model_name: str = "microsoft/Florence-2-large", device: str = "cpu") -> None:
        self.device = device
        self.model_name = model_name
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        if device.startswith("cuda"):
            model_kwargs["dtype"] = self.dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device).eval()
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.use_cache = False

    def describe_batch(
        self,
        images: list[Image.Image],
        *,
        task: str = "<DENSE_REGION_CAPTION>",
        max_new_tokens: int = 128,
        batch_size: int = 2,
    ) -> list[dict[str, Any]]:
        if not images:
            return []
        out: list[dict[str, Any]] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            prompts = [task] * len(batch)
            inputs = self.processor(text=prompts, images=batch, return_tensors="pt", padding=True)
            inputs = {
                key: (value.to(self.device, dtype=self.dtype) if key == "pixel_values" else value.to(self.device))
                for key, value in inputs.items()
            }
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=3,
                    use_cache=False,
                )
            generated = self.processor.batch_decode(outputs, skip_special_tokens=False)
            for image, raw_text in zip(batch, generated):
                try:
                    parsed = self.processor.post_process_generation(raw_text, task=task, image_size=(image.width, image.height))
                except Exception:
                    parsed = None
                out.append(_normalize_florence_description(raw_text, parsed, task=task))
        return out

    def close(self) -> None:
        del self.model
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _encode_pil_image(image: Image.Image, *, max_side: int = 640, quality: int = 85) -> dict[str, Any]:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(1.0, float(max_side) / float(max(width, height)))
    if scale < 1.0:
        image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "width": image.size[0],
        "height": image.size[1],
    }


def _extract_gemini_response_text(payload: dict[str, Any]) -> str:
    for candidate in payload.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                return str(text)
    raise RuntimeError("Gemini anchor relabel response did not contain text")


class GeminiAnchorRelabeler:
    def __init__(self, *, model_name: str = "gemini-2.5-flash", max_side: int = 640) -> None:
        self.model_name = model_name
        self.max_side = max_side

    def relabel(
        self,
        image_crop: Image.Image,
        *,
        current_label: str,
        candidate_labels: list[str],
        example_texts: list[str],
        relation: str,
        timeout_s: int = 120,
    ) -> dict[str, Any]:
        payload = _encode_pil_image(image_crop, max_side=self.max_side)
        prompt = self._build_prompt(
            current_label=current_label,
            candidate_labels=candidate_labels,
            example_texts=example_texts,
            relation=relation,
        )
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": payload["mime_type"],
                                "data": payload["data"],
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseJsonSchema": self._response_schema(),
            },
        }
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
            headers={
                "x-goog-api-key": get_secret(GEMINI),
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout_s,
        )
        response.raise_for_status()
        payload_json = response.json()
        raw_text = _extract_gemini_response_text(payload_json)
        normalized = self._normalize_relabel_response(
            raw_text=raw_text,
            usage=payload_json.get("usageMetadata", {}),
            current_label=current_label,
            candidate_labels=candidate_labels,
        )
        if not self._should_retry_generic(normalized, current_label=current_label):
            return normalized
        retry_prompt = self._build_prompt(
            current_label=current_label,
            candidate_labels=candidate_labels,
            example_texts=example_texts,
            relation=relation,
            anti_generic_retry=True,
        )
        retry_body = {
            "contents": [
                {
                    "parts": [
                        {"text": retry_prompt},
                        {
                            "inline_data": {
                                "mime_type": payload["mime_type"],
                                "data": payload["data"],
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
                "responseJsonSchema": self._response_schema(),
            },
        }
        retry_response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
            headers={
                "x-goog-api-key": get_secret(GEMINI),
                "Content-Type": "application/json",
            },
            json=retry_body,
            timeout=timeout_s,
        )
        retry_response.raise_for_status()
        retry_payload_json = retry_response.json()
        retry_raw_text = _extract_gemini_response_text(retry_payload_json)
        retried = self._normalize_relabel_response(
            raw_text=retry_raw_text,
            usage=retry_payload_json.get("usageMetadata", {}),
            current_label=current_label,
            candidate_labels=candidate_labels,
        )
        return retried if not self._should_retry_generic(retried, current_label=current_label) else normalized

    def relabel_many(
        self,
        items: list[dict[str, Any]],
        *,
        chunk_size: int,
        poll_interval_s: int,
        timeout_s: int,
    ) -> dict[str, dict[str, Any]]:
        requests_by_key: list[GeminiBatchRequest] = []
        for item in items:
            payload = _encode_pil_image(item["image_crop"], max_side=self.max_side)
            requests_by_key.append(
                GeminiBatchRequest(
                    key=str(item["key"]),
                    contents=[
                        self._build_prompt(
                            current_label=str(item["current_label"]),
                            candidate_labels=list(item["candidate_labels"]),
                            example_texts=list(item["example_texts"]),
                            relation=str(item["relation"]),
                        ),
                        image_part_from_payload(payload),
                    ],
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                        "response_json_schema": self._response_schema(),
                    },
                    metadata={"request_kind": "anchor_relabel"},
                )
            )
        results = batch_generate_json(
            model=self.model_name,
            requests=requests_by_key,
            display_name_prefix="sgocr-anchor-relabel",
            chunk_size=int(chunk_size),
            poll_interval_s=int(poll_interval_s),
            timeout_s=int(timeout_s),
        )
        normalized: dict[str, dict[str, Any]] = {}
        retry_items: list[dict[str, Any]] = []
        for item in items:
            key = str(item["key"])
            outcome = results.get(key)
            if outcome is None or outcome.error:
                normalized[key] = {
                    "label": str(item["current_label"]),
                    "alternates": [str(item["current_label"])],
                    "usage": {},
                    "raw_text": f"relabel_failed: {(outcome.error if outcome else 'missing batch response')}",
                }
                continue
            try:
                normalized[key] = self._normalize_relabel_response(
                    raw_text=outcome.raw_text,
                    usage=outcome.usage,
                    current_label=str(item["current_label"]),
                    candidate_labels=list(item["candidate_labels"]),
                )
                if self._should_retry_generic(normalized[key], current_label=str(item["current_label"])):
                    retry_items.append(item)
            except Exception as exc:
                normalized[key] = {
                    "label": str(item["current_label"]),
                    "alternates": [str(item["current_label"])],
                    "usage": {},
                    "raw_text": f"relabel_failed: {exc}",
                }
        if retry_items:
            retry_requests: list[GeminiBatchRequest] = []
            for item in retry_items:
                payload = _encode_pil_image(item["image_crop"], max_side=self.max_side)
                retry_requests.append(
                    GeminiBatchRequest(
                        key=str(item["key"]),
                        contents=[
                            self._build_prompt(
                                current_label=str(item["current_label"]),
                                candidate_labels=list(item["candidate_labels"]),
                                example_texts=list(item["example_texts"]),
                                relation=str(item["relation"]),
                                anti_generic_retry=True,
                            ),
                            image_part_from_payload(payload),
                        ],
                        generation_config={
                            "temperature": 0.1,
                            "response_mime_type": "application/json",
                            "response_json_schema": self._response_schema(),
                        },
                        metadata={"request_kind": "anchor_relabel_retry"},
                    )
                )
            retry_results = batch_generate_json(
                model=self.model_name,
                requests=retry_requests,
                display_name_prefix="sgocr-anchor-relabel-retry",
                chunk_size=int(chunk_size),
                poll_interval_s=int(poll_interval_s),
                timeout_s=int(timeout_s),
            )
            for item in retry_items:
                key = str(item["key"])
                outcome = retry_results.get(key)
                if outcome is None or outcome.error:
                    continue
                try:
                    retried = self._normalize_relabel_response(
                        raw_text=outcome.raw_text,
                        usage=outcome.usage,
                        current_label=str(item["current_label"]),
                        candidate_labels=list(item["candidate_labels"]),
                    )
                    if not self._should_retry_generic(retried, current_label=str(item["current_label"])):
                        normalized[key] = retried
                except Exception:
                    continue
        return normalized

    def _build_prompt(
        self,
        *,
        current_label: str,
        candidate_labels: list[str],
        example_texts: list[str],
        relation: str,
        anti_generic_retry: bool = False,
    ) -> str:
        tuning = load_semantic_dev40_tuning()
        anti_generic = tuning.anchor_relabel_generic_mode == "anti_generic" or anti_generic_retry
        lines = [
            "You are naming the best visible referring expression for the object, object-part, or surface that the text belongs to in an image crop.",
            "Return a short lowercase label of 1 to 5 words.",
            "Use only visible object language.",
            "Include a visible color adjective when it is clear and stable, for example `red can`, `blue jersey`, `white airplane tail`, or `silver car door`.",
            "Prefer a specific visible object class over a generic support surface.",
            "If the text belongs to a visible object part, prefer the part name, such as `car door`, `airplane tail`, `helmet`, `bus side`, `storefront window`, or `jersey chest`.",
            "Do not use world knowledge, sports terminology, professions, subject-matter interpretations, or editorial descriptions.",
            "Do not mention the text content itself in the label.",
            "You may propose a better literal label than the candidate list if the crop clearly shows it.",
        ]
        if anti_generic:
            lines.extend(
                [
                    "Avoid generic labels such as `sign`, `wall`, `board`, `panel`, `display`, `surface`, `object`, `area`, `sign wall`, or `display panel` unless there is truly no more specific visible object.",
                    "If a person, vehicle, clothing item, container, storefront element, device, or object part is visible, name that instead of a generic text surface.",
                    "Good labels: `red airplane tail`, `blue baseball helmet`, `white bus side`, `silver car door`, `green bottle`, `black jersey chest`.",
                ]
            )
        else:
            lines.extend(
                [
                    "If the crop is a printed sheet, page, poster, flyer, brochure, book cover, chart, or document-like surface, prefer those visible surface labels over generic scene nouns.",
                    "If the crop is a bottle or can, prefer `bottle` or `can` over building parts like `door` or `window`.",
                    "If the crop is a book or cover, prefer `book cover`.",
                ]
            )
        if anti_generic_retry:
            lines.append("Your previous answer was too generic. Retry with a more specific visible object or object-part label.")
        lines.extend(
            [
                f'current_label: "{current_label}"',
                f"candidate_labels: {json.dumps(candidate_labels[:8], ensure_ascii=False)}",
                f"example_texts: {json.dumps(example_texts[:6], ensure_ascii=False)}",
                f'relation_to_text: "{relation}"',
                "Respond as JSON with keys `label` and `alternates`.",
            ]
        )
        return "\n".join(lines) + "\n"

    def _response_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "alternates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 4,
                },
            },
            "required": ["label", "alternates"],
            "additionalProperties": False,
        }

    def _normalize_relabel_response(
        self,
        *,
        raw_text: str,
        usage: dict[str, Any],
        current_label: str,
        candidate_labels: list[str],
    ) -> dict[str, Any]:
        parsed = json.loads(raw_text)
        label = sanitize_anchor_label(str(parsed.get("label") or ""))
        alternates = []
        for item in [parsed.get("label"), *(parsed.get("alternates") or []), current_label, *candidate_labels]:
            cleaned = sanitize_anchor_label(str(item or ""))
            if cleaned and cleaned not in alternates:
                alternates.append(cleaned)
        if not label:
            fallback_label = sanitize_anchor_label(current_label)
            label = fallback_label if fallback_label else (alternates[0] if alternates else None)
        return {
            "label": label or sanitize_anchor_label(current_label) or current_label,
            "alternates": alternates,
            "usage": usage,
            "raw_text": raw_text,
        }

    def _should_retry_generic(self, normalized: dict[str, Any], *, current_label: str) -> bool:
        tuning = load_semantic_dev40_tuning()
        if tuning.anchor_relabel_generic_mode != "anti_generic":
            return False
        label = str(normalized.get("label") or "")
        if not is_generic_anchor_label(label):
            return False
        return True


class GroundingDinoGrounder:
    def __init__(self, *, model_name: str = "IDEA-Research/grounding-dino-base", device: str = "cpu") -> None:
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_name).to(device).eval()

    def detect(self, image: Image.Image, tag: str, *, threshold: float = 0.30, text_threshold: float = 0.20) -> list[dict[str, Any]]:
        prompt = f"{tag}."
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        processed = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = processed["boxes"]
        scores = processed["scores"]
        results = []
        for box, score in zip(boxes, scores):
            results.append(
                {
                    "label": tag,
                    "box": [round(float(value), 2) for value in box.tolist()],
                    "score": round(float(score), 6),
                    "source": "grounding_dino",
                }
            )
        return results

    def close(self) -> None:
        del self.model
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


class Sam3Refiner:
    def __init__(self, *, device: str = "cpu", confidence_threshold: float = 0.45) -> None:
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self._state: dict[str, Any] | None = None
        self._current_image_size: tuple[int, int] | None = None
        self._prompt_cache: dict[tuple[str, float], list[dict[str, Any]]] = {}
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as exc:  # pragma: no cover - exercised only when optional dep is missing
            raise RuntimeError(f"SAM3 unavailable: {exc}") from exc

        autocast_ctx = self._autocast_context()
        with autocast_ctx:
            self.model = build_sam3_image_model(device=device)
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=self.confidence_threshold)

    def _autocast_context(self) -> Any:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def set_image(self, image: Image.Image) -> None:
        self._prompt_cache = {}
        self._current_image_size = image.size
        with self._autocast_context():
            self._state = self.processor.set_image(image)
        self.processor.reset_all_prompts(self._state)

    def detect(self, prompt: str, *, threshold: float | None = None, max_results: int = 4) -> list[dict[str, Any]]:
        if self._state is None or self._current_image_size is None:
            raise ValueError("set_image must be called before detect")
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            return []
        thresh = float(self.confidence_threshold if threshold is None else threshold)
        cache_key = (prompt_text.lower(), thresh)
        if cache_key in self._prompt_cache:
            return [dict(row) for row in self._prompt_cache[cache_key][:max_results]]

        self.processor.reset_all_prompts(self._state)
        self.processor.confidence_threshold = thresh
        with self._autocast_context():
            self._state = self.processor.set_text_prompt(prompt_text, self._state)

        rows: list[dict[str, Any]] = []
        boxes = self._state.get("boxes")
        scores = self._state.get("scores")
        if boxes is None or scores is None:
            self._prompt_cache[cache_key] = []
            self.processor.reset_all_prompts(self._state)
            return []
        for box, score in zip(boxes, scores):
            rows.append(
                {
                    "label": prompt_text,
                    "box": [round(float(value), 2) for value in box.tolist()],
                    "score": round(float(score), 6),
                    "source": "sam3",
                    "sam3_prompt": prompt_text,
                }
            )
        rows = sorted(rows, key=lambda row: (-float(row.get("score") or 0.0), row["label"]))
        self._prompt_cache[cache_key] = rows
        self.processor.reset_all_prompts(self._state)
        return [dict(row) for row in rows[:max_results]]

    def close(self) -> None:
        del self.processor
        del self.model
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
