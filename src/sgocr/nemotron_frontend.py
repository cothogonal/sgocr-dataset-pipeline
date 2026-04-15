from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .bootstrap import normalize_answer, region_key_for_bbox
from .bootstrap_kd import bbox_xyxy_to_xywh, compute_resolvability
from .paths import REPO_ROOT


DEFAULT_NEMOTRON_SRC = Path("/tmp/nemotron-ocr-v2/nemotron-ocr/src")
DEFAULT_NEMOTRON_MIN_CONFIDENCE = 0.65
DEFAULT_NEMOTRON_BATCH_SIZE = 8
DEFAULT_NEMOTRON_RECOGNIZER_CHUNK = 256
DEFAULT_NEMOTRON_RELATIONAL_CHUNK = 256
DEFAULT_NEMOTRON_INFER_LENGTH = 1024


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


def _nemotron_src_root() -> Path:
    raw = os.environ.get("SGOCR_NEMOTRON_SRC", "").strip()
    if raw:
        return Path(raw)
    return DEFAULT_NEMOTRON_SRC


def _ensure_nemotron_src_on_path() -> Path:
    src_root = _nemotron_src_root().resolve()
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    return src_root


def _load_nemotron_pipeline() -> Any:
    src_root = _ensure_nemotron_src_on_path()
    if not src_root.exists():
        raise RuntimeError(
            f"Nemotron OCR source tree is missing at {src_root}. "
            "Set SGOCR_NEMOTRON_SRC to a valid checkout of nvidia/nemotron-ocr-v2."
        )
    try:
        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2
    except Exception as exc:  # pragma: no cover - host/toolchain dependent
        raise RuntimeError(
            "Nemotron OCR v2 is unavailable on this host. "
            "The NVIDIA package depends on compiled CUDA/C++ ops (nemotron_ocr_cpp), "
            "and this environment does not currently provide a working build/import path. "
            f"Import failed from {src_root}: {exc}"
        ) from exc
    return NemotronOCRV2


def nemotron_frontend_diagnostic() -> dict[str, Any]:
    src_root = _nemotron_src_root().resolve()
    try:
        pipeline_cls = _load_nemotron_pipeline()
    except Exception as exc:
        return {
            "available": False,
            "src_root": str(src_root),
            "error": str(exc),
        }
    return {
        "available": True,
        "src_root": str(src_root),
        "pipeline_class": f"{pipeline_cls.__module__}.{pipeline_cls.__name__}",
    }


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _chunked[T](items: list[T], size: int) -> list[list[T]]:
    return [items[idx : idx + size] for idx in range(0, len(items), max(1, size))]


def _resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    repo_path = (REPO_ROOT / path).resolve()
    if repo_path.exists():
        return repo_path
    return repo_path


def run_nemotron_ocr_stage(
    *,
    image_specs: list[dict[str, str]],
    image_source_map: dict[str, str],
    max_detections: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    NemotronOCRV2 = _load_nemotron_pipeline()
    clarity_floor = _float_env("SGOCR_NEMOTRON_MIN_CONFIDENCE", DEFAULT_NEMOTRON_MIN_CONFIDENCE)
    detector_batch_size = _int_env("SGOCR_NEMOTRON_DETECTOR_BATCH_SIZE", DEFAULT_NEMOTRON_BATCH_SIZE)
    recognizer_chunk_size = _int_env("SGOCR_NEMOTRON_RECOGNIZER_CHUNK_SIZE", DEFAULT_NEMOTRON_RECOGNIZER_CHUNK)
    relational_chunk_size = _int_env("SGOCR_NEMOTRON_RELATIONAL_CHUNK_SIZE", DEFAULT_NEMOTRON_RELATIONAL_CHUNK)
    infer_length = _int_env("SGOCR_NEMOTRON_INFER_LENGTH", DEFAULT_NEMOTRON_INFER_LENGTH)
    include_invalid = bool(int(os.environ.get("SGOCR_NEMOTRON_INCLUDE_INVALID", "0") or "0"))
    merge_level = os.environ.get("SGOCR_NEMOTRON_MERGE_LEVEL", "word").strip() or "word"

    ocr = NemotronOCRV2(
        lang="en",
        detector_max_batch_size=detector_batch_size,
        recognizer_chunk_size=recognizer_chunk_size,
        relational_chunk_size=relational_chunk_size,
        infer_length=infer_length,
        skip_relational=True,
    )

    detections_by_image: dict[str, list[dict[str, Any]]] = {}
    detection_rows: list[dict[str, Any]] = []
    text_nodes: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    per_image_box_counts: list[int] = []
    progress = _StageProgressLogger("nemotron_ocr", len(image_specs))
    progress.start()

    image_dims: dict[str, tuple[int, int]] = {}
    for spec in image_specs:
        with Image.open(_resolve_image_path(spec["image_path"])).convert("RGB") as image:
            image_dims[spec["image_id"]] = (int(image.width), int(image.height))

    processed_images = 0
    for batch_specs in _chunked(image_specs, detector_batch_size):
        batch_paths = [str(_resolve_image_path(spec["image_path"])) for spec in batch_specs]
        batch_predictions = ocr(batch_paths, merge_level=merge_level, include_invalid=include_invalid)
        if not isinstance(batch_predictions, list):
            raise RuntimeError(f"Unexpected Nemotron batch return type: {type(batch_predictions)!r}")
        if len(batch_predictions) != len(batch_specs):
            raise RuntimeError(
                f"Nemotron returned {len(batch_predictions)} prediction sets for {len(batch_specs)} images."
            )

        for spec, predictions in zip(batch_specs, batch_predictions):
            image_width, image_height = image_dims[spec["image_id"]]
            rows_for_image: list[dict[str, Any]] = []
            source = image_source_map.get(spec["image_id"], "textocr_train")
            source_counts[source] = source_counts.get(source, 0) + 1
            for idx, pred in enumerate(list(predictions or [])[:max_detections], start=1):
                left = float(pred.get("left") or 0.0)
                right = float(pred.get("right") or 0.0)
                upper = float(pred.get("upper") or 0.0)
                lower = float(pred.get("lower") or 0.0)
                x1 = max(0.0, min(left, right) * image_width)
                x2 = min(float(image_width), max(left, right) * image_width)
                y1 = max(0.0, min(lower, upper) * image_height)
                y2 = min(float(image_height), max(lower, upper) * image_height)
                if x2 <= x1 or y2 <= y1:
                    continue
                raw_text = str(pred.get("text") or "").strip()
                if not raw_text:
                    continue
                node_id = f"nemotron_{idx:03d}"
                polygon = [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                    round(x1, 2),
                    round(y2, 2),
                ]
                bbox = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                bbox_xywh = bbox_xyxy_to_xywh(bbox)
                confidence = float(pred.get("confidence") or 0.0)
                detection_row = {
                    "image_id": spec["image_id"],
                    "image_path": spec["image_path"],
                    "image_width": int(image_width),
                    "image_height": int(image_height),
                    "bbox": bbox,
                    "polygon": polygon,
                    "detection_confidence": round(confidence, 6),
                    "detector_source": "nemotron_v2",
                    "detector_sources": ["nemotron_v2"],
                    "node_id": node_id,
                }
                rows_for_image.append(detection_row)
                detection_rows.append(detection_row)

                resolvability = compute_resolvability(polygon, int(image_width), int(image_height))
                resolvability["clarity_confidence"] = confidence
                resolvability["clarity_mode"] = "nemotron_confidence"
                resolvability["passes"] = bool(confidence >= clarity_floor)
                text_nodes.append(
                    {
                        "image_id": spec["image_id"],
                        "node_id": node_id,
                        "image_path": spec["image_path"],
                        "image_width": int(image_width),
                        "image_height": int(image_height),
                        "text": raw_text,
                        "text_normalized": normalize_answer(raw_text),
                        "polygon": polygon,
                        "bbox": bbox,
                        "bbox_xywh": [round(float(value), 2) for value in bbox_xywh],
                        "confidence": round(confidence, 6),
                        "consensus_tier": "nemotron_v2",
                        "model_votes": [{"model_name": "nemotron_v2", "text": raw_text, "confidence": confidence}],
                        "detection_confidence": round(confidence, 6),
                        "detector_source": "nemotron_v2",
                        "detector_sources": ["nemotron_v2"],
                        "region_key": region_key_for_bbox(bbox_xywh, int(image_width), int(image_height)),
                        "resolvable": bool(resolvability["passes"]),
                        "resolvability": resolvability,
                        "source_dataset": source,
                    }
                )
            detections_by_image[spec["image_id"]] = rows_for_image
            per_image_box_counts.append(len(rows_for_image))
            processed_images += 1
            progress.tick(
                processed_images,
                extra=f"image_id={spec['image_id']} detections={len(rows_for_image)} total_text_nodes={len(text_nodes)}",
            )

    detection_summary = {
        "images": len(image_specs),
        "detected_boxes": len(detection_rows),
        "mean_boxes_per_image": (len(detection_rows) / len(image_specs)) if image_specs else 0.0,
        "median_boxes_per_image": float(sorted(per_image_box_counts)[len(per_image_box_counts) // 2]) if per_image_box_counts else 0.0,
        "frontend": "nemotron_v2",
        "lang": "en",
        "merge_level": merge_level,
        "detector_max_batch_size": detector_batch_size,
        "recognizer_chunk_size": recognizer_chunk_size,
        "relational_chunk_size": relational_chunk_size,
        "skip_relational": True,
        "clarity_confidence_floor": clarity_floor,
    }
    consensus_stats = {
        "total_candidates": len(detection_rows),
        "accepted_nodes": len(text_nodes),
        "dropped_nodes": 0,
        "by_source": source_counts,
        "consensus_tier_counts": {"nemotron_v2": len(text_nodes)},
        "drop_reasons": {},
        "string_length_bins": {},
        "frontend": "nemotron_v2",
        "clarity_confidence_floor": clarity_floor,
    }
    progress.finish(extra=f"detections={len(detection_rows)} text_nodes={len(text_nodes)}")
    return detections_by_image, detection_rows, detection_summary, text_nodes, consensus_stats
