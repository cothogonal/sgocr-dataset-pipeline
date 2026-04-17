from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .qwen_anchor_vllm import (
    QwenAnchorRequest,
    _StageProgressLogger,
    normalize_independent_anchor_label,
)


# ---------------------------------------------------------------------------
# Gemma4-adapted inventory prompt.
# Based on the ita15 no_text_ref extension, adapted for general VLMs:
#   - Coordinates requested as normalized [0, 1] floats (more natural for non-Qwen models)
#   - "with" compound label convention dropped (Qwen-specific training artifact)
#   - Instructions reworded for instruction-following generalists
# ---------------------------------------------------------------------------

GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15 = (
    "Locate every visible object or surface in this image that has text on or near it, or that "
    "could naturally anchor nearby text. Skip large featureless backgrounds with no nearby text.\n"
    "Rules for labels:\n"
    "- Use only visual descriptors: object type, shape, color, material, or structural role.\n"
    "- Never quote, paraphrase, or reference any text visible in the image.\n"
    "- REQUIRED: always use the word 'with' between the subject and its color/descriptor. "
    "CORRECT: 'person with red jersey', 'sign with blue background', 'keyboard with black keys'. "
    "WRONG: 'person red jersey', 'blue background sign', 'keyboard black keys'.\n"
    "- When the same object type appears multiple times, distinguish with color or position "
    "(e.g. 'sign with blue background' vs 'sign with red background', 'left panel' vs 'right panel').\n"
    "For photos with people: label by role and appearance using 'with' "
    "(e.g. 'baseball player with red jersey', 'referee with black uniform') — not by jersey numbers or text.\n"
    "For charts and infographics: draw one bounding box per logical region — do NOT draw many small "
    "boxes for repeated elements of the same type. Prefer broad unified regions: 'bar chart plot area', "
    "'y-axis region', 'legend area', 'chart title'. Skip copyright notices, watermarks, and decorative "
    "side panels.\n"
    "Return a JSON array. Each element must have exactly two keys: "
    '"bbox_2d" (normalized [x1, y1, x2, y2] floats in [0, 1]) and "label" (string). '
    "Example: "
    '[{"bbox_2d": [0.10, 0.05, 0.45, 0.30], "label": "bar chart plot area"}, '
    '{"bbox_2d": [0.60, 0.10, 0.90, 0.85], "label": "person with blue jacket"}, ...]'
)

# ITA16 anti-doc variant: same as ita15 base but explicitly forbids the
# 'document/page/paper' mislabel that Gemma generates for any flat surface
# with text on it.  Replaces generic label with physical-object description.
GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA16_ANTIDOC = (
    GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15.rstrip("]")
    + ", ...]"  # keep the example intact
).replace(
    "For charts and infographics:",
    "Never label a region as 'document', 'page', or 'paper' — instead describe "
    "the physical object: 'white paper sheet', 'printed flyer', 'spiral notebook', "
    "'manila folder', 'open book', 'binder'.\n"
    "For charts and infographics:",
)

# ITA16 compact variant: minimal prompt testing whether brevity beats rules.
# Hypothesis: Gemma over-thinks the complex multi-rule prompt and produces
# more consistent anchors when given a short, clear instruction.
GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA16_COMPACT = (
    "Find every object or surface in this image that has text on or near it. "
    "Return one bounding box per distinct object — do not repeat the same box. "
    "Label each with a specific color+type (e.g. 'blue billboard', 'red fire hydrant', "
    "'white monitor', 'person with orange vest'). "
    "Avoid generic labels like 'document', 'image', 'area', 'background', 'surface', 'object'. "
    "For charts: use region labels like 'bar chart area', 'y-axis region', 'legend area'. "
    "Never copy, paraphrase, or reference visible text in a label.\n"
    'Return JSON: [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}]'
)

GEMMA_STRUCTURAL_FALLBACK_PROMPT_ITA15 = (
    "Locate every distinct bounded region in this chart or document that contains or is "
    "immediately adjacent to text: bars, segments, cells, buttons, badges, axis areas, titles.\n"
    "Rules for labels:\n"
    "- Draw one bounding box per logical region. Do NOT draw many small boxes for repeated elements "
    "of the same type — for example, draw one 'y-axis region' covering all axis labels, not separate "
    "boxes for each tick mark.\n"
    "- Label by shape and structural role with a color when useful "
    "(e.g. 'blue bar chart segment', 'red pie wedge', 'gray legend area', 'axis label region').\n"
    "- Always use 'with' to attach a color: 'bar with blue fill', NOT 'blue fill bar'.\n"
    "- Never reference visible text. Distinguish same-type regions by color or position.\n"
    "- Skip copyright notices, watermarks, and decorative side panels.\n"
    "Return a JSON array. Each element must have exactly two keys: "
    '"bbox_2d" (normalized [x1, y1, x2, y2] floats in [0, 1]) and "label" (string). '
    "Example: "
    '[{"bbox_2d": [0.05, 0.10, 0.20, 0.90], "label": "y-axis region"}, '
    '{"bbox_2d": [0.25, 0.50, 0.40, 0.90], "label": "bar with blue fill"}, ...]'
)


# ---------------------------------------------------------------------------
# Coordinate parsing helpers
# ---------------------------------------------------------------------------

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
    for candidate in (
        cleaned,
        cleaned[cleaned.find("["):] if "[" in cleaned else "",
        cleaned[cleaned.find("{"):] if "{" in cleaned else "",
    ):
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


def _scale_bbox(bbox: list[float], *, width: int, height: int) -> list[float]:
    """Convert bbox to absolute pixels.

    Accepts both:
      - normalized 0-1 floats (Gemma4 native output when prompted correctly)
      - 0-1000 Qwen-style integers (fallback if the model ignores the prompt spec)

    Detection: if all four values are in [0, 1] → treat as normalized.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        # Already normalized to [0, 1]
        ax1 = max(0.0, min(round(x1 * width, 2), float(width - 1)))
        ay1 = max(0.0, min(round(y1 * height, 2), float(height - 1)))
        ax2 = max(0.0, min(round(x2 * width, 2), float(width - 1)))
        ay2 = max(0.0, min(round(y2 * height, 2), float(height - 1)))
    else:
        # Qwen-style 0-1000 space
        ax1 = max(0.0, min(round(x1 * width / 1000.0, 2), float(width - 1)))
        ay1 = max(0.0, min(round(y1 * height / 1000.0, 2), float(height - 1)))
        ax2 = max(0.0, min(round(x2 * width / 1000.0, 2), float(width - 1)))
        ay2 = max(0.0, min(round(y2 * height / 1000.0, 2), float(height - 1)))
    if ax2 < ax1:
        ax1, ax2 = ax2, ax1
    if ay2 < ay1:
        ay1, ay2 = ay2, ay1
    return [ax1, ay1, ax2, ay2]


# ---------------------------------------------------------------------------
# Ollama HTTP client (vision chat)
# ---------------------------------------------------------------------------

def _encode_image_b64(path: str | None, image_obj: Image.Image | None) -> tuple[str, int, int]:
    """Return (base64_jpeg, width, height)."""
    if image_obj is not None:
        img = image_obj.convert("RGB")
    elif path is not None:
        with Image.open(path) as _img:
            img = _img.convert("RGB")
    else:
        raise ValueError("Need image_path or image_obj")
    width, height = img.size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), width, height


def _ollama_chat(
    *,
    model: str,
    prompt: str,
    image_b64: str,
    base_url: str,
    num_ctx: int,
    timeout_s: int,
) -> str:
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }
        ],
        "stream": False,
        "options": {"num_ctx": num_ctx, "temperature": 0.0},
    }
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json=body,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    return str((data.get("message") or {}).get("content") or "")


# ---------------------------------------------------------------------------
# GemmaOllamaAnchorGrounder
# ---------------------------------------------------------------------------

class GemmaOllamaAnchorGrounder:
    """Anchor inventory grounder backed by a Gemma4 model served via Ollama.

    Implements the same detect_many() interface as QwenAnchorGrounderVLLM so it can
    be dropped in as an alternative backend in full_pipeline_dev40.py.
    """

    def __init__(
        self,
        *,
        model: str = "gemma4:e4b-it-q4_K_M",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        timeout_s: int = 120,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._num_ctx = num_ctx
        self._timeout_s = timeout_s

    def detect_many(
        self,
        requests_list: list[QwenAnchorRequest],
        *,
        progress_logger: _StageProgressLogger | None = None,
        sampling_temperature: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        if not requests_list:
            return {}
        owns_progress = False
        if progress_logger is None:
            progress_logger = _StageProgressLogger("gemma_detect_many", len(requests_list))
            progress_logger.start()
            owns_progress = True

        out: dict[str, list[dict[str, Any]]] = {}
        for idx, req in enumerate(requests_list, start=1):
            image_b64, width, height = _encode_image_b64(req.image_path, req.image_obj)
            prompt = req.prompt_text or GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15
            try:
                raw_text = _ollama_chat(
                    model=self._model,
                    prompt=prompt,
                    image_b64=image_b64,
                    base_url=self._base_url,
                    num_ctx=self._num_ctx,
                    timeout_s=self._timeout_s,
                )
            except Exception as exc:
                print(
                    f"[gemma_detect_many] ollama error image_id={req.image_id} err={exc}",
                    flush=True,
                )
                out[req.image_id] = []
                progress_logger.tick(idx, extra=f"image_id={req.image_id} rows=0 err={type(exc).__name__}")
                continue

            rows: list[dict[str, Any]] = []
            for item in _iter_json_candidates(raw_text):
                bbox = item.get("bbox_2d")
                raw_label = str(item.get("label") or "").strip()
                if not isinstance(bbox, list) or len(bbox) < 4 or not raw_label:
                    continue
                label = normalize_independent_anchor_label(raw_label) or raw_label
                scaled = _scale_bbox(bbox, width=width, height=height)
                rows.append(
                    {
                        "label": label,
                        "raw_label": raw_label,
                        "box": scaled,
                        "score": 0.58,
                        "source": "gemma4_ollama",
                        "prompt_text": prompt,
                        "raw_text": raw_text,
                    }
                )
            out[req.image_id] = rows
            progress_logger.tick(idx, extra=f"image_id={req.image_id} rows={len(rows)}")

        if owns_progress:
            progress_logger.finish(extra=f"images={len(out)}")
        return out

    def groundback_check_many(
        self,
        image_anchors_by_image: dict[str, list[dict[str, Any]]],
        *,
        image_path_by_image: dict[str, str] | None = None,
    ) -> dict[str, list[tuple[int, float]]]:
        """No-op groundback stub — Gemma4 doesn't support Qwen-style bbox relocation.

        Returns an empty dict so the caller applies no groundback penalties.
        The pipeline treats missing entries as un-checked anchors (no penalty, no filter).
        """
        return {}

    def close(self) -> None:
        pass  # no resources to release
