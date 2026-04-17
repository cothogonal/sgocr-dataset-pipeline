from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .semantic_grounding import (
    QWEN_INDEPENDENT_RAW_INVENTORY_CATEGORY_TEXT,
    normalize_independent_anchor_label,
    sanitize_anchor_label,
)


OFFICIAL_QWEN_BBOX_PROMPT = (
    'Locate every instance that belongs to the following categories: "{categories}". '
    'Report bbox coordinates in JSON format like this: {{"bbox_2d": [x1, y1, x2, y2], "label": "category"}}.'
)

OPEN_QWEN_LOCAL_DISCOVERY_PROMPT = (
    'Locate the visible objects, surfaces, or object parts in this crop that could naturally anchor nearby text. '
    'Report bbox coordinates in JSON format like this: {{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}.'
)

OPEN_QWEN_LOCAL_DISCOVERY_PROMPT_COLOR_SPECIFIC = (
    'Locate the visible objects, surfaces, or object parts in this crop that could naturally anchor nearby text. '
    'Prefer specific visible object or object-part labels over generic text surfaces. '
    'Include a visible color adjective when it is clear and stable, for example "red jersey", "blue sign", "silver car door", or "white airplane tail". '
    'Avoid generic labels like "sign wall", "display panel", or "object" when a more specific visible object exists. '
    'Report bbox coordinates in JSON format like this: {{"bbox_2d": [x1, y1, x2, y2], "label": "object"}}.'
)

INDEPENDENT_QWEN_INVENTORY_PROMPT = OFFICIAL_QWEN_BBOX_PROMPT.format(
    categories=QWEN_INDEPENDENT_RAW_INVENTORY_CATEGORY_TEXT
)

# Structural fallback prompt: used when the standard inventory returns degenerate results
# (e.g. "all visible objects" for chart/infographic images that lack identifiable scene objects).
# Focuses on bounded visual regions and shapes rather than named scene objects, so it works
# for charts, documents, diagrams, and any image where object vocabulary is inappropriate.
STRUCTURAL_FALLBACK_QWEN_PROMPT = (
    "Locate the distinct bounded visual regions or shapes in this image that contain or are "
    "immediately adjacent to text. Focus on shapes with clear edges: bars, segments, panels, "
    "plates, buttons, emblems, signs, or labeled surface areas. Prefer specific visible structural "
    "labels based on shape and context, for example 'bar chart segment', 'pie chart wedge', "
    "'legend panel', 'table cell', or 'circular badge'. Do not use the actual color as the "
    "primary label component — describe the shape or region type first. "
    "Avoid generic labels like 'all visible objects' or 'image area'. "
    'Report bbox coordinates in JSON format like this: {{"bbox_2d": [x1, y1, x2, y2], "label": "shape"}}.'
)

GROUNDBACK_QWEN_PROMPT = (
    'Locate the single element described as: "{label}". '
    'If present, report its bbox in JSON format like this: {{"bbox_2d": [x1, y1, x2, y2], "label": "element"}}. '
    'Report only one bbox for the best match.'
)

_DEGENERATE_LABELS = frozenset({
    "all visible objects",
    "all objects",
    "unknown",
    "all visible text",
    "image",
    "scene",
    "",
})

# Anti-OCR suffix: appended to inventory prompts to discourage using visible text content as labels.
# Without this, Qwen sometimes labels anchors with the text they contain (e.g. "DAN BROWN BREWING CO")
# rather than the object type (e.g. "brewery sign"). This causes REVERSE_GROUND vision leakage.
_ANTI_OCR_SUFFIX = (
    " Label each detected region with a concise visual object or shape type — "
    "never copy verbatim text content visible inside the region as the label."
)

INDEPENDENT_QWEN_INVENTORY_PROMPT_ANTI_OCR = INDEPENDENT_QWEN_INVENTORY_PROMPT.rstrip(".") + "." + _ANTI_OCR_SUFFIX
STRUCTURAL_FALLBACK_QWEN_PROMPT_ANTI_OCR = STRUCTURAL_FALLBACK_QWEN_PROMPT.rstrip(".") + _ANTI_OCR_SUFFIX

# No-text-ref suffix: stronger than anti-OCR — forbids any reference to text content (not just
# verbatim copies) and requires disambiguation when multiple anchors share the same label type.
# Use this when anchor labels are leaking text content through paraphrase or partial quotes.
_NO_TEXT_REF_SUFFIX = (
    " Label each detected region using only visual descriptors: object type, shape, color, material, "
    "or structural role. Never reference, quote, or paraphrase any text visible on or inside the region. "
    "When multiple instances of the same object type appear, distinguish them with a color or position "
    "qualifier, for example 'blue sign' vs 'red sign', or 'left panel' vs 'right panel'."
)

INDEPENDENT_QWEN_INVENTORY_PROMPT_NO_TEXT_REF = INDEPENDENT_QWEN_INVENTORY_PROMPT.rstrip(".") + "." + _NO_TEXT_REF_SUFFIX
STRUCTURAL_FALLBACK_QWEN_PROMPT_NO_TEXT_REF = STRUCTURAL_FALLBACK_QWEN_PROMPT.rstrip(".") + _NO_TEXT_REF_SUFFIX

# ITA15 prompt: self-contained rewrite for Qwen3-VL independent inventory.
# Does NOT extend the no_text_ref category-list format — that format triggers
# Qwen repetition loops when extended beyond ~185 tokens.  Instead this is a
# standalone instruction-following prompt that avoids the category list entirely.
#
# Improvements over the original compressed draft:
#   - Stronger "with" requirement (REQUIRED keyword + CORRECT/WRONG examples)
#   - Chart consolidation: one box per logical region, not per tick mark
#   - Size preference and chart fluff suppression retained
INDEPENDENT_QWEN_INVENTORY_PROMPT_ITA15 = (
    "Locate every visible object or surface that has text on or near it, or that could serve as a "
    "spatial anchor for nearby text. Skip large featureless backgrounds that have no text nearby.\n"
    "REQUIRED: always use the word 'with' between a subject and its color or descriptor. "
    "CORRECT: 'player with red jersey', 'sign with blue background'. "
    "WRONG: 'player red jersey', 'blue background sign'.\n"
    "Use visual descriptors only — never quote, paraphrase, or reference any text visible in the image. "
    "Distinguish same-type objects with color or position (e.g. 'sign with blue background' vs 'sign with red background').\n"
    "For photos with people: label by role and appearance "
    "(e.g. 'baseball player with red jersey', 'referee with black uniform') — not by jersey text or numbers.\n"
    "For charts and infographics: draw one bounding box per logical region — do NOT draw many small "
    "boxes for repeated elements of the same type. Prefer broad regions: 'bar chart plot area', "
    "'y-axis region', 'legend area'. Skip copyright notices, watermarks, and decorative side panels.\n"
    'Report as JSON: {{"bbox_2d": [x1, y1, x2, y2], "label": "description"}}'
)

# ITA15 structural fallback: chart/doc version of the above.
STRUCTURAL_FALLBACK_QWEN_PROMPT_ITA15 = (
    "Locate distinct bounded regions in this chart or document that contain or adjoin text: "
    "bars, segments, cells, buttons, badges, axis areas, titles.\n"
    "Draw one bounding box per logical region — do NOT draw many small boxes for repeated elements "
    "of the same type (e.g. draw one 'y-axis region', not 9 separate axis-tick boxes).\n"
    "REQUIRED: use 'with' to attach a color: 'bar with blue fill', NOT 'blue fill bar'.\n"
    "Label by shape and structural role with color when useful "
    "(e.g. 'bar with blue fill', 'gray legend area', 'axis label region'). "
    "Never quote, paraphrase, or reference visible text. Distinguish same-type regions by color or position. "
    "Skip copyright notices, watermarks, and decorative side panels.\n"
    'Report as JSON: {{"bbox_2d": [x1, y1, x2, y2], "label": "description"}}'
)

# Per-anchor degenerate label set: labels that are too generic or abstract to anchor a useful QA.
# These are individual-anchor checks (not whole-image inventory checks like _DEGENERATE_LABELS).
_DEGENERATE_ANCHOR_LABELS: frozenset[str] = frozenset({
    "object part", "surface", "area", "region", "part", "section",
    "texture", "background", "element", "item", "thing", "object",
    "entity", "feature", "detail", "structure", "view", "content",
    # Mislabels that arise when Qwen sees text/numbers on clothing or equipment
    "document", "printed page", "page", "text document", "printed document",
    # Pure background single-word labels
    "sky", "ceiling", "pavement", "sidewalk",
}) | _DEGENERATE_LABELS

# Words that, when they appear as the trailing token of a multi-word compound label,
# indicate it is a pure background region with no text-anchoring value.
# e.g. "light blue sky", "overcast sky", "white ceiling"
_BACKGROUND_TRAILING_WORDS: frozenset[str] = frozenset({"sky", "ceiling", "pavement", "sidewalk"})


def is_degenerate_anchor_label(label: str) -> bool:
    """Return True if a single anchor label is too generic to anchor a useful QA.

    Catches:
    - Exact-match generic labels: 'object part', 'surface', 'area', etc.
    - Compound labels ending in pure-background terms: 'light blue sky', 'white ceiling'
    """
    norm = str(label or "").strip().lower()
    if len(norm) <= 2:
        return True
    if norm in _DEGENERATE_ANCHOR_LABELS:
        return True
    words = norm.split()
    if len(words) >= 2 and words[-1] in _BACKGROUND_TRAILING_WORDS:
        return True
    return False


# Words that, when they appear as the trailing token of a multi-word anchor label,
# indicate the label is describing text content rather than a visual object.
# Examples: "x-axis region date labels" → labels; "footer area text" → text
# These arise from Gemma4 appending the text role of a region to its structural label.
_TEXT_REF_TRAILING_WORDS: frozenset[str] = frozenset({
    "labels", "caption", "footnote", "footnotes", "watermark", "text",
})

# Substrings that indicate the label references visible text content rather than a
# visual object regardless of position. "printed text", "visible text", etc.
_TEXT_CONTENT_SUBSTRINGS: tuple[str, ...] = ("printed text", "visible text", "written text")

# Prefixes that indicate the anchor is fundamentally a text-content descriptor
# rather than a visual object. e.g. "text block black text", "text area small print"
_TEXT_CONTENT_PREFIXES: tuple[str, ...] = ("text block", "text area")


def is_text_ref_anchor_label(label: str) -> bool:
    """Return True if an anchor label references visible text content rather than a visual object.

    Catches compound labels where a model appends the text role of a region:
      - "x-axis region date labels"  (trailing 'labels')
      - "footer area text"           (trailing 'text')
      - "text block black text"      (prefix 'text block')
      - "text area small print"      (prefix 'text area')
      - "stack rectangular plaques printed text"  (substring 'printed text')

    These labels leak text content through the anchor name, undermining image-dependence.
    Intended to be used as an additional gate alongside is_degenerate_anchor_label().
    """
    norm = str(label or "").strip().lower()
    if not norm:
        return False
    words = norm.split()
    # trailing text-role word in a multi-word label
    if len(words) >= 2 and words[-1] in _TEXT_REF_TRAILING_WORDS:
        return True
    # label contains a text-content substring
    for sub in _TEXT_CONTENT_SUBSTRINGS:
        if sub in norm:
            return True
    # label is primarily a text-content descriptor (prefix check)
    for prefix in _TEXT_CONTENT_PREFIXES:
        if norm.startswith(prefix):
            return True
    return False


def is_ocr_text_label(label: str) -> bool:
    """Return True if the anchor label looks like OCR-copied text rather than a visual object type.

    Heuristics:
    - Mostly uppercase tokens with spaces/punctuation (e.g. "DAN BROWN BREWING COMPANY")
    - Multi-word title-case proper noun strings that are unusually long (e.g. "Springfield City Hall")
    """
    s = str(label or "").strip()
    if len(s) < 5:
        return False
    tokens = s.split()
    if len(tokens) < 2:
        return False
    # All-caps tokens dominate: e.g. "DAN BROWN BREWING CO" → 4/4 uppercase tokens
    upper_count = sum(1 for t in tokens if t.replace("&", "").replace("'", "").replace(".", "").isupper() and t.isascii())
    if upper_count >= max(2, len(tokens) - 1):
        return True
    # Multi-word title-case phrase that's unusually long (≥3 words, avg token len ≥ 5)
    if len(tokens) >= 3 and all(t[0].isupper() for t in tokens if t[0].isalpha()):
        avg_len = sum(len(t) for t in tokens) / len(tokens)
        if avg_len >= 5:
            return True
    return False


# Real object labels that should never trigger structural fallback even when concentrated.
# A scene with 4 planes is not degenerate — it's a coherent airport image.
_VALID_CONCENTRATED_LABELS: frozenset[str] = frozenset({
    "plane", "airplane", "aircraft", "jet", "helicopter",
    "car", "truck", "bus", "van", "vehicle", "motorcycle", "bicycle",
    "person", "people", "man", "woman", "child",
    "building", "house", "tower", "bridge",
    "tree", "bush", "grass",
    "bottle", "can", "cup", "glass",
    "sign", "board", "poster", "banner",
    "table", "chair", "desk", "shelf",
    "screen", "monitor", "display",
    "book", "box", "bag",
    "door", "window",
    "boat", "ship",
    "dog", "cat", "bird",
})


def is_degenerate_inventory(rows: list[dict[str, Any]], *, threshold: float = 0.92) -> bool:
    """Return True if the inventory result for one image is degenerate.

    Degenerate means: most labels are the same catch-all phrase (e.g. 'all visible objects'),
    OR the inventory is empty, OR a single label covers more than `threshold` fraction of rows
    AND that label is not a valid real-world object (plane, car, person, etc.).

    The valid-label exemption prevents airport images (all planes) or crowd images (all people)
    from being flagged as degenerate and triggering the structural fallback unnecessarily.
    """
    if not rows:
        return True
    from collections import Counter
    label_counts: Counter[str] = Counter()
    for row in rows:
        raw = str(row.get("label") or row.get("raw_label") or "").strip().lower()
        label_counts[raw] += 1
    total = len(rows)
    most_common_label, most_common_count = label_counts.most_common(1)[0]
    if most_common_label in _DEGENERATE_LABELS:
        return True
    if most_common_label in _VALID_CONCENTRATED_LABELS:
        return False
    return (most_common_count / total) >= threshold


class _StageProgressLogger:
    def __init__(self, stage_name: str, total: int) -> None:
        self.stage_name = str(stage_name)
        self.total = max(0, int(total))
        self._start = time.time()
        self._last_fraction = -1
        self._path = Path(os.environ["SGOCR_PROGRESS_LOG_PATH"]).expanduser() if os.environ.get("SGOCR_PROGRESS_LOG_PATH") else None

    def _emit(self, message: str) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        line = f"[{stamp}] [stage:{self.stage_name}] {message}"
        print(line, flush=True)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def start(self) -> None:
        self._emit(f"start total={self.total}")

    def tick(self, completed: int, *, extra: str = "") -> None:
        if self.total <= 0:
            return
        completed = max(0, int(completed))
        fraction = min(100, int((completed * 100) / self.total))
        if completed < self.total and fraction <= self._last_fraction:
            return
        if completed < self.total and fraction < 1:
            return
        self._last_fraction = fraction
        elapsed = max(time.time() - self._start, 1e-6)
        rate = completed / elapsed if completed > 0 else 0.0
        suffix = f" {extra}" if extra else ""
        self._emit(
            f"progress completed={completed}/{self.total} pct={fraction}% elapsed_s={elapsed:.1f} rate_per_s={rate:.2f}{suffix}"
        )

    def finish(self, *, extra: str = "") -> None:
        elapsed = max(time.time() - self._start, 1e-6)
        suffix = f" {extra}" if extra else ""
        self._emit(f"finish completed={self.total}/{self.total} pct=100% elapsed_s={elapsed:.1f}{suffix}")


def _wait_for_gpu_memory(min_free_mib: int = 10_000, poll_interval_s: int = 30, timeout_s: int = 600) -> None:
    """Block until nvidia-smi reports >= min_free_mib MiB free on GPU 0.

    Ollama releases VRAM within ~5 minutes of `ollama stop`. This loop avoids an
    immediate OOM from vLLM startup racing a still-loaded Ollama model.
    """
    deadline = time.time() + timeout_s
    while True:
        try:
            raw = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True,
                timeout=10,
            )
            free_mib = int(raw.strip().splitlines()[0].strip())
        except Exception:
            return  # no nvidia-smi → not a GPU machine, proceed
        if free_mib >= min_free_mib:
            print(f"[vllm_startup] GPU free memory: {free_mib} MiB — OK", flush=True)
            return
        remaining = max(0, int(deadline - time.time()))
        print(
            f"[vllm_startup] GPU free memory: {free_mib} MiB < {min_free_mib} MiB required. "
            f"Waiting {poll_interval_s}s (timeout in {remaining}s) — is Ollama still holding VRAM? "
            f"Run: ollama stop gemma4:e4b-it-q4_K_M",
            flush=True,
        )
        if time.time() >= deadline:
            print(f"[vllm_startup] WARNING: GPU memory still low after {timeout_s}s — proceeding anyway", flush=True)
            return
        time.sleep(poll_interval_s)


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
    normalize_open_labels: bool = False


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


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a[:4]]
    bx1, by1, bx2, by2 = [float(value) for value in box_b[:4]]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    area_b = max((bx2 - bx1) * (by2 - by1), 1e-6)
    return inter_area / max(area_a + area_b - inter_area, 1e-6)


def merge_qwen_inventory_passes(
    pass_results: list[dict[str, list[dict[str, Any]]]],
    *,
    iou_threshold: float = 0.55,
    min_support: int = 1,
) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {}
    image_ids = sorted({image_id for rows_by_image in pass_results for image_id in rows_by_image.keys()})
    for image_id in image_ids:
        clusters: list[list[dict[str, Any]]] = []
        all_rows: list[dict[str, Any]] = []
        for pass_index, rows_by_image in enumerate(pass_results):
            for row in rows_by_image.get(image_id, []):
                candidate = dict(row)
                candidate["_pass_index"] = pass_index
                candidate["label"] = (
                    normalize_independent_anchor_label(str(candidate.get("raw_label") or candidate.get("label") or ""))
                    or str(candidate.get("label") or "").strip().lower()
                )
                if not str(candidate.get("label") or "").strip():
                    continue
                all_rows.append(candidate)
        all_rows.sort(key=lambda row: (-float(row.get("score") or 0.0), row.get("label") or ""))
        for row in all_rows:
            placed = False
            for cluster in clusters:
                exemplar = cluster[0]
                if str(exemplar.get("label") or "") != str(row.get("label") or ""):
                    continue
                if _bbox_iou(list(exemplar.get("box") or []), list(row.get("box") or [])) >= iou_threshold:
                    cluster.append(row)
                    placed = True
                    break
            if not placed:
                clusters.append([row])
        image_rows: list[dict[str, Any]] = []
        for cluster in clusters:
            pass_support = len({int(row.get("_pass_index") or 0) for row in cluster})
            if pass_support < max(1, int(min_support)):
                continue
            representative = max(cluster, key=lambda row: (float(row.get("score") or 0.0), str(row.get("raw_label") or "")))
            alternate_labels = sorted({str(row.get("raw_label") or row.get("label") or "").strip() for row in cluster if str(row.get("raw_label") or row.get("label") or "").strip()})
            merged_row = {
                key: value
                for key, value in representative.items()
                if not str(key).startswith("_")
            }
            merged_row["score"] = round(float(merged_row.get("score") or 0.0) + 0.03 * max(0, pass_support - 1), 4)
            merged_row["source"] = "qwen3_vl_vllm_inventory"
            merged_row["pass_support_count"] = pass_support
            merged_row["alternate_labels"] = alternate_labels
            image_rows.append(merged_row)
        merged[image_id] = image_rows
    return merged


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
        _wait_for_gpu_memory()
        process_vision_info, AutoProcessor, LLM, SamplingParams = _lazy_import_vllm()
        self._process_vision_info = process_vision_info
        self._SamplingParams = SamplingParams
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

    def detect_many(
        self,
        requests: list[QwenAnchorRequest],
        *,
        progress_logger: _StageProgressLogger | None = None,
        sampling_temperature: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        if not requests:
            return {}
        owns_progress = False
        if progress_logger is None:
            progress_logger = _StageProgressLogger("qwen_detect_many", len(requests))
            progress_logger.start()
            owns_progress = True
        inputs: list[dict[str, Any]] = []
        image_sizes: dict[str, tuple[int, int]] = {}
        request_prompts: dict[str, str] = {}
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
            request_prompts[req.image_id] = prompt
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
        completed = 0
        sampling_params = self._sampling_params
        if sampling_temperature is not None and float(sampling_temperature) != float(self._sampling_params.temperature):
            sampling_params = self._SamplingParams(
                temperature=float(sampling_temperature),
                max_tokens=768,
                top_k=-1,
                stop_token_ids=[],
            )
        for start in range(0, len(inputs), self._batch_size):
            batch_requests = requests[start : start + self._batch_size]
            batch_inputs = inputs[start : start + self._batch_size]
            outputs = self._llm.generate(batch_inputs, sampling_params=sampling_params)
            for req, output in zip(batch_requests, outputs):
                width, height = image_sizes[req.image_id]
                generated_text = output.outputs[0].text if output.outputs else ""
                rows: list[dict[str, Any]] = []
                for item in _iter_json_candidates(generated_text):
                    bbox = item.get("bbox_2d")
                    raw_label = str(item.get("label") or "").strip()
                    if req.enforce_allowed_labels:
                        label = _normalize_to_allowed_label(raw_label, req.categories)
                    elif req.normalize_open_labels:
                        label = normalize_independent_anchor_label(raw_label)
                    else:
                        label = raw_label
                    if not isinstance(bbox, list) or len(bbox) < 4 or not label:
                        continue
                    rows.append(
                        {
                            "label": label,
                            "raw_label": raw_label,
                            "box": _scale_relative_bbox(bbox, width=width, height=height),
                            "score": 0.62,
                            "source": "qwen3_vl_vllm",
                            "prompt_text": request_prompts[req.image_id],
                            "raw_text": generated_text,
                        }
                    )
                out[req.image_id] = rows
                completed += 1
                progress_logger.tick(completed, extra=f"image_id={req.image_id} rows={len(rows)}")
        if owns_progress:
            progress_logger.finish(extra=f"images={len(out)}")
        return out

    def groundback_check_many(
        self,
        image_anchors_by_image: dict[str, list[dict[str, Any]]],
        *,
        image_path_by_image: dict[str, str],
    ) -> dict[str, list[tuple[int, float]]]:
        """Re-ground each anchor label to verify it uniquely identifies the element.

        For every anchor in image_anchors_by_image (for images present in image_path_by_image),
        runs a single Qwen inference with only the label text as the query. The returned bbox
        is compared to the original anchor bbox via IoU.

        Returns dict[image_id, list[(anchor_idx, iou)]]. Anchors with no Qwen output or where
        Qwen returns a bbox that doesn't match the original get iou=0.0 (treated as failed by
        the caller). Images missing from image_path_by_image are skipped silently.
        """
        # Build a flat list of (image_id, anchor_idx, anchor) for all anchors to check.
        work: list[tuple[str, int, dict[str, Any]]] = []
        for image_id, anchors in image_anchors_by_image.items():
            if image_id not in image_path_by_image:
                continue
            for idx, anchor in enumerate(anchors):
                label = str(anchor.get("label") or "").strip()
                if label:
                    work.append((image_id, idx, anchor))

        if not work:
            return {}

        # Each groundback query gets a unique synthetic image_id so results can be matched back.
        qwen_requests = [
            QwenAnchorRequest(
                image_id=f"{image_id}::gb::{anchor_idx}",
                image_path=image_path_by_image[image_id],
                categories=[],
                prompt_text=GROUNDBACK_QWEN_PROMPT.format(label=anchor["label"]),
                enforce_allowed_labels=False,
                normalize_open_labels=False,
            )
            for image_id, anchor_idx, anchor in work
        ]

        raw_out = self.detect_many(qwen_requests)

        results: dict[str, list[tuple[int, float]]] = {}
        for image_id, anchor_idx, anchor in work:
            key = f"{image_id}::gb::{anchor_idx}"
            rows = raw_out.get(key, [])
            original_box = list(anchor.get("box") or [])
            if not rows or len(original_box) < 4:
                iou = 0.0
            else:
                best_iou = max(_bbox_iou(original_box, list(row.get("box") or [])) for row in rows)
                iou = round(best_iou, 4)
            results.setdefault(image_id, []).append((anchor_idx, iou))

        return results

    def close(self) -> None:
        del self._llm
