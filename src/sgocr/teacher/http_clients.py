from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from ..secrets import GEMINI, OPENAI, get_secret
from .prompts import PromptVariant, render_prompt, response_schema


@dataclass(frozen=True)
class ImagePayload:
    mime_type: str
    image_b64: str
    width: int
    height: int


def encode_image(path: Path, *, max_side: int = 768, quality: int = 85) -> ImagePayload:
    with Image.open(path) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1.0, float(max_side) / float(max(width, height)))
        if scale < 1.0:
            img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality), optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return ImagePayload(mime_type="image/jpeg", image_b64=encoded, width=img.size[0], height=img.size[1])


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    for cand in payload.get("candidates", []):
        content = cand.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                return str(text)
    raise RuntimeError("Gemini response did not contain text")


def call_gemini(model: str, tuple_row: dict[str, Any], image_payload: ImagePayload, variant: PromptVariant, *, timeout_s: int = 120) -> dict[str, Any]:
    prompt = render_prompt(tuple_row, variant)
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": image_payload.mime_type,
                            "data": image_payload.image_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseJsonSchema": response_schema(variant.question_count),
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
    raw_text = _extract_gemini_text(payload)
    return {
        "provider": "gemini",
        "model": model,
        "raw_text": raw_text,
        "parsed": json.loads(raw_text),
        "usage": payload.get("usageMetadata", {}),
    }


def call_openai_chat(model: str, tuple_row: dict[str, Any], image_payload: ImagePayload, variant: PromptVariant, *, timeout_s: int = 120) -> dict[str, Any]:
    prompt = render_prompt(tuple_row, variant)
    data_uri = f"data:{image_payload.mime_type};base64,{image_payload.image_b64}"
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "sgocr_qa_items",
                "strict": True,
                "schema": response_schema(variant.question_count),
            },
        },
        "temperature": 0.2,
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {get_secret(OPENAI)}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()
    raw_text = str(payload["choices"][0]["message"]["content"])
    return {
        "provider": "openai",
        "model": model,
        "raw_text": raw_text,
        "parsed": json.loads(raw_text),
        "usage": payload.get("usage", {}),
    }
