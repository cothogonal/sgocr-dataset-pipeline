from __future__ import annotations

import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from lib.infra.io import load_jsonl

from .bootstrap import is_valid_text, normalize_answer, region_key_for_bbox
from .bootstrap_kd import bbox_xywh_to_xyxy, bbox_xyxy_to_xywh, compute_resolvability, overlap_fraction
from .dev40_complete import union_bbox
from .consensus import OCRVote, choose_consensus
from .ocr_runtime import CraftDetector, PARSeqRecognizer, PaddleOCRDetector, PaddleOCRRecognizer, TrOCRRecognizer, bbox_to_polygon, crop_with_padding
from .ollama_anchor import (
    GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15,
    GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA16_ANTIDOC,
    GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA16_COMPACT,
    GEMMA_STRUCTURAL_FALLBACK_PROMPT_ITA15,
    GemmaOllamaAnchorGrounder,
)
from .qwen_anchor_vllm import (
    INDEPENDENT_QWEN_INVENTORY_PROMPT,
    INDEPENDENT_QWEN_INVENTORY_PROMPT_ANTI_OCR,
    INDEPENDENT_QWEN_INVENTORY_PROMPT_DAM01,
    INDEPENDENT_QWEN_INVENTORY_PROMPT_NO_TEXT_REF,
    INDEPENDENT_QWEN_INVENTORY_PROMPT_ITA15,
    OPEN_QWEN_LOCAL_DISCOVERY_PROMPT,
    OPEN_QWEN_LOCAL_DISCOVERY_PROMPT_COLOR_SPECIFIC,
    QwenAnchorGrounderVLLM,
    QwenAnchorRequest,
    is_degenerate_anchor_label,
    is_ocr_text_label,
    is_text_ref_anchor_label,
    merge_qwen_inventory_passes,
    normalize_qwen_description,
)
from .semantic_dev40_tuning import load_semantic_dev40_tuning
from .semantic_grounding import (
    FlorenceTagger,
    FALLBACK_TAGS,
    GeminiAnchorRelabeler,
    GroundingDinoGrounder,
    QWEN_GLOBAL_INVENTORY_CATEGORIES,
    QWEN_LOCAL_DISCOVERY_CATEGORIES,
    Sam3Refiner,
    SAFE_FALLBACK_TAGS,
    GENERIC_TEXT_ANCHORS,
    annotate_anchor_candidate_support,
    anchor_local_location_metadata,
    anchor_candidate_viable,
    anchor_relevance,
    categorize_anchor,
    consolidate_anchor_candidates,
    expand_grounding_tags_for_node,
    expand_box,
    normalize_independent_anchor_label,
    remap_region_box_to_image,
    sanitize_anchor_label,
    sanitize_anchor_tags,
    relation_between_text_and_anchor,
)
from .tuple_builder import (
    _anchor_area_fraction,
    _bbox_iou,
    _cluster_relabel_groups,
    _degenerate_consensus_text,
    _length_bucket,
    _merge_detection_candidates,
    dedupe_anchor_rows,
    dedupe_preserve_order,
    preferred_anchor_order,
)

SEMANTIC_PROMPT_VARIANT = "semantic_dev40_v8"
RUNTIME_MODELS = {
    "pipeline_logic": SEMANTIC_PROMPT_VARIANT,
    "ocr_frontend": "classic",
    "anchor_tag_discovery_backend": "florence",
    "anchor_candidate_backend": "florence_dino",
    "qwen_anchor_inventory_mode": "selected_tags",
    "qwen_anchor_inventory_pass_count": 1,
    "qwen_anchor_inventory_temperature": 0.0,
    "qwen_anchor_inventory_consensus_iou": 0.55,
    "qwen_anchor_inventory_min_support": 1,
    "detector": "PP-OCRv5_server_det",
    "recognizers": ["parseq", "PP-OCRv5_server_rec", "microsoft/trocr-large-printed"],
    "semantic_tagger": "microsoft/Florence-2-large",
    "grounder": "IDEA-Research/grounding-dino-base",
}


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
        if self.total > 0:
            self._emit(f"start total={self.total}")
        else:
            self._emit("start total=0")

    def tick(self, completed: int, *, extra: str = "") -> None:
        completed = max(0, int(completed))
        if self.total <= 0:
            return
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
        if self.total > 0:
            self._emit(f"finish completed={self.total}/{self.total} pct=100% elapsed_s={elapsed:.1f}{suffix}")
        else:
            self._emit(f"finish elapsed_s={elapsed:.1f}{suffix}")


def _make_stage_progress_logger(stage_name: str, total: int) -> _StageProgressLogger:
    logger = _StageProgressLogger(stage_name, total)
    logger.start()
    return logger


def _ocr_runtime_signature(runtime_models: dict[str, Any]) -> dict[str, Any]:
    return {
        "ocr_frontend": runtime_models.get("ocr_frontend", "classic"),
        "detector": runtime_models.get("detector"),
        "recognizers": runtime_models.get("recognizers"),
    }


def _semantic_runtime_signature(runtime_models: dict[str, Any]) -> dict[str, Any]:
    return {
        "semantic_tagger": runtime_models.get("semantic_tagger"),
        "grounder": runtime_models.get("grounder"),
        "anchor_tag_discovery_backend": runtime_models.get("anchor_tag_discovery_backend", "florence"),
        "anchor_candidate_backend": runtime_models.get("anchor_candidate_backend", "florence_dino"),
        "qwen_anchor_inventory_mode": runtime_models.get("qwen_anchor_inventory_mode", "selected_tags"),
        "qwen_anchor_inventory_pass_count": runtime_models.get("qwen_anchor_inventory_pass_count", 1),
        "qwen_anchor_inventory_temperature": runtime_models.get("qwen_anchor_inventory_temperature", 0.0),
        "qwen_anchor_inventory_consensus_iou": runtime_models.get("qwen_anchor_inventory_consensus_iou", 0.55),
        "qwen_anchor_inventory_min_support": runtime_models.get("qwen_anchor_inventory_min_support", 1),
    }


def _anchor_relabel_model_name(mode: str) -> str | None:
    if mode == "flash":
        return "gemini-2.5-flash"
    if mode == "pro":
        return "gemini-2.5-pro"
    return None


def _sam3_prompt_limit() -> int:
    tuning = load_semantic_dev40_tuning()
    mode_limit = {"none": 0, "top1": 1, "top2": 2, "top3": 3}[tuning.sam3_refine_mode]
    return max(0, min(mode_limit, int(tuning.sam3_topk_prompts)))


def _select_sam3_prompts(
    *,
    seed_candidates: list[dict[str, Any]],
    node_tags: list[str],
    limit: int,
) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()

    def add(label: str) -> None:
        cleaned = sanitize_anchor_label(str(label or ""))
        if not cleaned or cleaned in seen:
            return
        seen.add(cleaned)
        prompts.append(cleaned)

    for row in seed_candidates:
        add(str(row.get("label") or ""))
        for label in row.get("support_labels") or []:
            add(str(label))
        for label in row.get("alternate_labels") or []:
            add(str(label))
        if len(prompts) >= limit:
            return prompts[:limit]

    for label in node_tags:
        add(str(label))
        if len(prompts) >= limit:
            return prompts[:limit]
    return prompts[:limit]


def _should_run_sam3_for_node(
    *,
    seed_candidates: list[dict[str, Any]],
    node: dict[str, Any],
    image_size: tuple[int, int],
) -> bool:
    tuning = load_semantic_dev40_tuning()
    if tuning.sam3_refine_mode == "none":
        return False
    if tuning.sam3_apply_mode != "targeted":
        return True
    if not seed_candidates:
        return True
    top = seed_candidates[0]
    label = str(top.get("label") or "")
    area_frac = _anchor_area_fraction(list(top["box"]), image_size)
    support_count = int(top.get("support_count") or 1)
    cluster_size = int(top.get("cluster_size") or 1)
    conflict_count = int(top.get("label_conflict_count") or 0)
    is_generic = label in GENERIC_TEXT_ANCHORS
    weak_support = support_count <= tuning.sam3_target_support_max
    dense_local_text = int((node.get("kd_metadata") or {}).get("local_text_cluster_size") or 0) >= tuning.sam3_target_cluster_min
    return bool(
        is_generic
        or conflict_count >= 1
        or weak_support
        or cluster_size >= 2
        or area_frac >= tuning.sam3_target_area_start
        or dense_local_text
    )


def load_image_specs(source_experiment_dir: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    raw_path = source_experiment_dir / "raw_results.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen: dict[str, dict[str, str]] = {}
    image_source_map: dict[str, str] = {}
    for row in rows:
        tuple_row = row.get("tuple") or {}
        image_id = str(tuple_row.get("image_id") or row.get("image_id") or "").strip()
        image_path = str(tuple_row.get("image_path") or row.get("image_path") or "").strip()
        if not image_id or not image_path:
            continue
        seen[image_id] = {"image_id": image_id, "image_path": image_path}
        image_source_map[image_id] = str(tuple_row.get("dataset_source") or "textocr_val")
    return [seen[key] for key in sorted(seen)], image_source_map


def run_detection_stage(
    *,
    image_specs: list[dict[str, str]],
    device: str,
    max_detections: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    tuning = load_semantic_dev40_tuning()
    detector = PaddleOCRDetector(
        model_name=str(RUNTIME_MODELS["detector"]),
        device=device,
        box_thresh=tuning.detector_box_thresh,
        unclip_ratio=tuning.detector_unclip_ratio,
    )
    craft_detector = None
    if tuning.detector_mode == "ppocr_craft_ensemble":
        craft_detector = CraftDetector(
            text_threshold=tuning.craft_text_threshold,
            link_threshold=tuning.craft_link_threshold,
            low_text=tuning.craft_low_text,
            long_size=tuning.craft_long_size,
            device=device,
        )
    detections_by_image: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    box_counts: list[int] = []
    source_counts = Counter()
    for spec in image_specs:
        image = Image.open(spec["image_path"]).convert("RGB")
        image_np = np.asarray(image)
        raw_rows: list[dict[str, Any]] = []
        boxes, confidences, seconds, polygons = detector.detect(image_np, max_detections=max_detections)
        for box, confidence, polygon in zip(boxes, confidences, polygons):
            raw_rows.append(
                {
                    "image_id": spec["image_id"],
                    "image_path": spec["image_path"],
                    "image_width": image.width,
                    "image_height": image.height,
                    "bbox": [round(float(value), 2) for value in box],
                    "polygon": [round(float(value), 2) for value in polygon],
                    "detection_confidence": round(float(confidence), 6),
                    "detector_seconds": round(float(seconds), 6),
                    "detector_source": "ppocr",
                    "detector_sources": ["ppocr"],
                }
            )
        if craft_detector is not None:
            craft_boxes, craft_confidences, craft_seconds = craft_detector.detect(image_np, max_detections=max_detections)
            for box, confidence in zip(craft_boxes, craft_confidences):
                raw_rows.append(
                    {
                        "image_id": spec["image_id"],
                        "image_path": spec["image_path"],
                        "image_width": image.width,
                        "image_height": image.height,
                        "bbox": [round(float(value), 2) for value in box],
                        "polygon": [round(float(value), 2) for value in bbox_to_polygon(box)],
                        "detection_confidence": round(float(confidence), 6),
                        "detector_seconds": round(float(craft_seconds), 6),
                        "detector_source": "craft",
                        "detector_sources": ["craft"],
                    }
                )
        merged_rows = _merge_detection_candidates(
            raw_rows,
            merge_overlap=tuning.detector_merge_overlap,
            max_detections=max_detections,
        )
        image_rows = []
        for idx, merged in enumerate(merged_rows, start=1):
            row = {
                "node_id": f"{spec['image_id']}_det_{idx:03d}",
                **merged,
            }
            image_rows.append(row)
            rows.append(row)
            for source in row.get("detector_sources") or [str(row.get("detector_source") or "unknown")]:
                source_counts[str(source)] += 1
        detections_by_image[spec["image_id"]] = image_rows
        box_counts.append(len(image_rows))
    detector.close()
    if craft_detector is not None:
        craft_detector.close()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    summary = {
        "images": len(image_specs),
        "detected_boxes": len(rows),
        "mean_boxes_per_image": statistics.mean(box_counts) if box_counts else 0.0,
        "median_boxes_per_image": statistics.median(box_counts) if box_counts else 0.0,
        "detector_mode": tuning.detector_mode,
        "by_source": dict(source_counts),
    }
    return detections_by_image, rows, summary


def run_recognition_stage(
    *,
    image_specs: list[dict[str, str]],
    detections_by_image: dict[str, list[dict[str, Any]]],
    recognizer: Any,
    model_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in image_specs:
        detections = detections_by_image.get(spec["image_id"], [])
        if not detections:
            continue
        image = Image.open(spec["image_path"]).convert("RGB")
        crops = [crop_with_padding(image, list(row["bbox"])) for row in detections]
        votes: list[OCRVote] = recognizer.recognize(crops)
        for detection, vote in zip(detections, votes):
            rows.append(
                {
                    "image_id": spec["image_id"],
                    "node_id": detection["node_id"],
                    "model_name": model_key,
                    "text": vote.text,
                    "text_normalized": normalize_answer(vote.text),
                    "confidence": round(float(vote.confidence), 6),
                    "rotation": int(vote.rotation),
                    "bbox": detection["bbox"],
                }
            )
    del recognizer
    return rows


def build_consensus_nodes(
    *,
    image_specs: list[dict[str, str]],
    image_source_map: dict[str, str],
    detections_by_image: dict[str, list[dict[str, Any]]],
    reading_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    votes_by_key: dict[tuple[str, str], list[OCRVote]] = defaultdict(list)
    for row in reading_rows:
        votes_by_key[(str(row["image_id"]), str(row["node_id"]))].append(
            OCRVote(
                model_name=str(row["model_name"]),
                text=str(row["text"]),
                confidence=float(row["confidence"]),
                rotation=int(row.get("rotation") or 0),
            )
        )

    text_nodes: list[dict[str, Any]] = []
    by_source = defaultdict(lambda: {"total": 0, "accepted": 0, "dropped": 0})
    tier_counts = Counter()
    drop_reasons = Counter()
    length_bins = Counter()
    for spec in image_specs:
        source = image_source_map.get(spec["image_id"], "textocr_val")
        for detection in detections_by_image.get(spec["image_id"], []):
            by_source[source]["total"] += 1
            decision = choose_consensus(votes_by_key.get((spec["image_id"], detection["node_id"]), []))
            tier_counts[decision.consensus_tier] += 1
            if not decision.accepted:
                by_source[source]["dropped"] += 1
                drop_reasons[decision.failure_reason or "dropped"] += 1
                continue
            if not is_valid_text(decision.text):
                by_source[source]["dropped"] += 1
                drop_reasons["invalid_text"] += 1
                continue
            if _degenerate_consensus_text(decision.text):
                by_source[source]["dropped"] += 1
                drop_reasons["degenerate_repeat_junk"] += 1
                continue
            by_source[source]["accepted"] += 1
            bbox_xywh = bbox_xyxy_to_xywh(list(detection["bbox"]))
            region_key = region_key_for_bbox(bbox_xywh, int(detection["image_width"]), int(detection["image_height"]))
            resolvability = compute_resolvability(list(detection["polygon"]), int(detection["image_width"]), int(detection["image_height"]))
            text_nodes.append(
                {
                    "image_id": spec["image_id"],
                    "node_id": detection["node_id"],
                    "image_path": spec["image_path"],
                    "image_width": int(detection["image_width"]),
                    "image_height": int(detection["image_height"]),
                    "text": decision.text,
                    "text_normalized": decision.text_normalized,
                    "polygon": list(detection["polygon"]),
                    "bbox": [round(float(value), 2) for value in detection["bbox"]],
                    "bbox_xywh": [round(float(value), 2) for value in bbox_xywh],
                    "confidence": round(float(decision.confidence), 6),
                    "consensus_tier": decision.consensus_tier,
                    "model_votes": list(decision.model_votes),
                    "detection_confidence": float(detection["detection_confidence"]),
                    "detector_source": str(detection.get("detector_source") or "ppocr"),
                    "detector_sources": list(detection.get("detector_sources") or [str(detection.get("detector_source") or "ppocr")]),
                    "region_key": region_key,
                    "resolvable": bool(resolvability["passes"]),
                    "resolvability": resolvability,
                    "source_dataset": source,
                }
            )
            length_bins[_length_bucket(decision.text)] += 1

    stats = {
        "total_candidates": sum(bucket["total"] for bucket in by_source.values()),
        "accepted_nodes": len(text_nodes),
        "dropped_nodes": sum(bucket["dropped"] for bucket in by_source.values()),
        "by_source": dict(by_source),
        "consensus_tier_counts": dict(tier_counts),
        "drop_reasons": dict(drop_reasons),
        "string_length_bins": dict(length_bins),
    }
    return text_nodes, stats


def build_resolvability_stats(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [node for node in nodes if node["resolvable"]]
    dropped = [node for node in nodes if not node["resolvable"]]
    by_source = defaultdict(lambda: {"total": 0, "passed": 0, "dropped": 0})
    for node in nodes:
        bucket = by_source[str(node["source_dataset"])]
        bucket["total"] += 1
        if node["resolvable"]:
            bucket["passed"] += 1
        else:
            bucket["dropped"] += 1
    return {
        "total_text_nodes": len(nodes),
        "passing_text_nodes": len(passed),
        "dropped_text_nodes": len(dropped),
        "by_source": dict(by_source),
        "dropped_width_at_224": [round(float(node["resolvability"]["text_px_w"]), 2) for node in dropped],
        "accepted_width_at_224": [round(float(node["resolvability"]["text_px_w"]), 2) for node in passed],
    }


def _gemma_prompt_for_variant(variant: str) -> str:
    if variant == "antidoc":
        return GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA16_ANTIDOC
    if variant == "compact":
        return GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA16_COMPACT
    return GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15


def run_anchor_stage(
    *,
    image_specs: list[dict[str, str]],
    resolvable_nodes: list[dict[str, Any]],
    device: str,
    max_tags_per_image: int,
    grounding_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    tuning = load_semantic_dev40_tuning()
    use_qwen_tag_backend = tuning.anchor_tag_discovery_backend == "qwen3_vl_vllm"
    use_qwen_anchor_backend = tuning.anchor_candidate_backend == "qwen3_vl_vllm"
    use_gemma_anchor_backend = tuning.anchor_candidate_backend == "gemma4_ollama"
    use_independent_qwen_inventory = (use_qwen_anchor_backend or use_gemma_anchor_backend) and tuning.qwen_anchor_inventory_mode == "independent_raw"
    tagger = None if use_qwen_tag_backend or use_independent_qwen_inventory else FlorenceTagger(model_name=str(RUNTIME_MODELS["semantic_tagger"]), device=device)
    grounder = None if (use_qwen_anchor_backend or use_gemma_anchor_backend) else GroundingDinoGrounder(device=device)
    qwen_grounder = (
        QwenAnchorGrounderVLLM(
            model_name=str(tuning.qwen_anchor_model),
            gpu_memory_utilization=float(tuning.qwen_anchor_gpu_memory_utilization),
            batch_size=int(tuning.qwen_anchor_batch_size),
            min_pixels=int(tuning.qwen_anchor_min_pixels),
            max_pixels=int(tuning.qwen_anchor_max_pixels),
            max_model_len=int(tuning.qwen_anchor_max_model_len),
        )
        if use_qwen_anchor_backend or use_qwen_tag_backend
        else GemmaOllamaAnchorGrounder(
            model=str(tuning.gemma_ollama_model),
            base_url=str(tuning.gemma_ollama_base_url),
            num_ctx=int(tuning.gemma_ollama_num_ctx),
        )
        if use_gemma_anchor_backend
        else None
    )
    sam3_refiner: Sam3Refiner | None = None
    sam3_refiner_enabled = _sam3_prompt_limit() > 0
    nodes_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in resolvable_nodes:
        nodes_by_image[str(node["image_id"])].append(node)

    anchor_tag_rows: list[dict[str, Any]] = []
    grounded_anchor_rows: list[dict[str, Any]] = []
    best_anchor_by_node: dict[tuple[str, str], dict[str, Any]] = {}
    image_plans: list[dict[str, Any]] = []
    tag_progress = _make_stage_progress_logger("anchor_tag_discovery", len(image_specs))

    for spec_index, spec in enumerate(image_specs, start=1):
        image_nodes = nodes_by_image.get(spec["image_id"], [])
        if not image_nodes:
            tag_progress.tick(spec_index, extra=f"image_id={spec['image_id']} skipped=no_resolvable_nodes")
            continue
        image = Image.open(spec["image_path"]).convert("RGB")
        context_boxes = [
            expand_box(list(node["bbox"]), image_width=image.width, image_height=image.height, scale=2.5)
            for node in image_nodes
        ]
        if use_independent_qwen_inventory:
            per_node_tags = {str(node["node_id"]): [] for node in image_nodes}
            per_node_local_candidates = {str(node["node_id"]): [] for node in image_nodes}
            for node, context_box in zip(image_nodes, context_boxes):
                anchor_tag_rows.append(
                    {
                        "image_id": spec["image_id"],
                        "node_id": node["node_id"],
                        "caption": "",
                        "raw_labels": [],
                        "semantic_regions": [],
                        "semantic_regions_mapped": [],
                        "discovered_tags": [],
                        "expanded_tags": [],
                        "final_prompt_tags": [],
                        "context_box": context_box,
                        "inventory_mode": "independent_raw",
                    }
                )
            image_plans.append(
                {
                    "spec": spec,
                    "image_nodes": image_nodes,
                    "image_size": (image.width, image.height),
                    "selected_tags": [],
                    "anchor_inventory_categories": [],
                    "per_node_tags": per_node_tags,
                    "per_node_local_candidates": per_node_local_candidates,
                    "independent_inventory_prompt": (
                        _gemma_prompt_for_variant(os.environ.get("SGOCR_GEMMA_ITA16_PROMPT_VARIANT", ""))
                        if use_gemma_anchor_backend and os.environ.get("SGOCR_GEMMA_ITA16_PROMPT_VARIANT", "")
                        else GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15
                        if use_gemma_anchor_backend and tuning.qwen_ita15_prompt_enabled
                        else INDEPENDENT_QWEN_INVENTORY_PROMPT_DAM01
                        if tuning.qwen_dam01_prompt_enabled
                        else INDEPENDENT_QWEN_INVENTORY_PROMPT_ITA15
                        if tuning.qwen_ita15_prompt_enabled
                        else INDEPENDENT_QWEN_INVENTORY_PROMPT_NO_TEXT_REF
                        if tuning.qwen_no_text_ref_prompt_enabled
                        else INDEPENDENT_QWEN_INVENTORY_PROMPT_ANTI_OCR
                        if tuning.qwen_anti_ocr_prompt_enabled
                        else GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15
                        if use_gemma_anchor_backend
                        else INDEPENDENT_QWEN_INVENTORY_PROMPT
                    ),
                }
            )
            tag_progress.tick(
                spec_index,
                extra=f"image_id={spec['image_id']} nodes={len(image_nodes)} skipped=independent_raw",
            )
            continue
        context_crops = [image.crop(tuple(box)) for box in context_boxes]
        if use_qwen_tag_backend:
            local_vocab_mode = tuning.qwen_anchor_tag_discovery_vocab_mode
            qwen_tag_requests = [
                QwenAnchorRequest(
                    image_id=f"{spec['image_id']}::{node['node_id']}",
                    image_obj=crop,
                    categories=list(QWEN_LOCAL_DISCOVERY_CATEGORIES) if local_vocab_mode == "constrained" else [],
                    prompt_text=None
                    if local_vocab_mode == "constrained"
                    else (
                        OPEN_QWEN_LOCAL_DISCOVERY_PROMPT_COLOR_SPECIFIC
                        if tuning.qwen_open_tag_prompt_mode == "color_specific"
                        else OPEN_QWEN_LOCAL_DISCOVERY_PROMPT
                    ),
                    enforce_allowed_labels=local_vocab_mode == "constrained",
                )
                for node, crop in zip(image_nodes, context_crops)
            ]
            qwen_tag_rows = qwen_grounder.detect_many(qwen_tag_requests) if qwen_grounder is not None else {}
            descriptions = [
                normalize_qwen_description(qwen_tag_rows.get(f"{spec['image_id']}::{node['node_id']}", []))
                for node in image_nodes
            ]
        else:
            descriptions = tagger.describe_batch(context_crops)
        per_node_tags: dict[str, list[str]] = {}
        per_node_local_candidates: dict[str, list[dict[str, Any]]] = {}
        image_tag_pool: list[str] = []
        for node, context_box, description in zip(image_nodes, context_boxes, descriptions):
            caption = str(description.get("caption") or "")
            raw_labels = [str(label) for label in (description.get("labels") or []) if str(label).strip()]
            semantic_regions = []
            for region in description.get("regions") or []:
                raw_label = str(region.get("label") or "").strip()
                safe_label = sanitize_anchor_label(raw_label)
                if not safe_label:
                    continue
                semantic_regions.append(
                    {
                        "label": safe_label,
                        "raw_label": raw_label,
                        "box": remap_region_box_to_image(list(region.get("box") or []), context_box),
                        "score": 0.58,
                        "source": "florence_region",
                    }
                )
            tags = sanitize_anchor_tags(caption, *raw_labels, max_tags=6)
            if not tags:
                tags = list(dict.fromkeys(label["label"] for label in semantic_regions))[:3]
            if len(tags) < 2:
                for fallback in SAFE_FALLBACK_TAGS:
                    if fallback not in tags:
                        tags.append(fallback)
                    if len(tags) >= 2:
                        break
            expanded_tags = expand_grounding_tags_for_node(
                base_tags=tags[:6],
                node=node,
                image_nodes=image_nodes,
                image_size=(image.width, image.height),
                mode=tuning.anchor_prompt_expansion_mode,
            )
            combined_tags = list(dict.fromkeys(tags[:6] + expanded_tags))
            per_node_tags[str(node["node_id"])] = combined_tags[:8]
            per_node_local_candidates[str(node["node_id"])] = dedupe_anchor_rows(semantic_regions)
            image_tag_pool.extend(combined_tags[:8])
            anchor_tag_rows.append(
                {
                    "image_id": spec["image_id"],
                    "node_id": node["node_id"],
                    "caption": caption,
                    "raw_labels": raw_labels,
                    "semantic_regions": list(description.get("regions") or []),
                    "semantic_regions_mapped": semantic_regions,
                    "discovered_tags": tags[:6],
                    "expanded_tags": expanded_tags,
                    "final_prompt_tags": combined_tags[:8],
                    "context_box": context_box,
                }
            )

        selected_tags = dedupe_preserve_order(image_tag_pool)
        for fallback in SAFE_FALLBACK_TAGS:
            if fallback not in selected_tags:
                selected_tags.append(fallback)
            if len(selected_tags) >= max_tags_per_image:
                break
        selected_tags = selected_tags[:max_tags_per_image]
        image_plans.append(
            {
                "spec": spec,
                "image_nodes": image_nodes,
                "image_size": (image.width, image.height),
                "selected_tags": selected_tags,
                "anchor_inventory_categories": (
                    list(QWEN_GLOBAL_INVENTORY_CATEGORIES)
                    if use_qwen_anchor_backend and tuning.qwen_anchor_inventory_mode == "global_inventory"
                    else selected_tags
                ),
                "per_node_tags": per_node_tags,
                "per_node_local_candidates": per_node_local_candidates,
            }
        )
        tag_progress.tick(
            spec_index,
            extra=(
                f"image_id={spec['image_id']} nodes={len(image_nodes)} "
                f"selected_tags={len(selected_tags)}"
            ),
        )
    tag_progress.finish(extra=f"image_plans={len(image_plans)} anchor_tag_rows={len(anchor_tag_rows)}")

    image_anchors_by_image: dict[str, list[dict[str, Any]]] = {}
    if qwen_grounder is not None:
        if use_independent_qwen_inventory:
            pass_count = max(1, int(tuning.qwen_anchor_inventory_pass_count))
            ground_progress = _make_stage_progress_logger("anchor_grounding", pass_count)
            qwen_requests = [
                QwenAnchorRequest(
                    image_id=str(plan["spec"]["image_id"]),
                    image_path=str(plan["spec"]["image_path"]),
                    categories=[],
                    prompt_text=str(plan["independent_inventory_prompt"]),
                    enforce_allowed_labels=False,
                    normalize_open_labels=True,
                )
                for plan in image_plans
            ]
            pass_results: list[dict[str, list[dict[str, Any]]]] = []
            for pass_index in range(pass_count):
                raw_grounded = qwen_grounder.detect_many(
                    qwen_requests,
                    sampling_temperature=float(tuning.qwen_anchor_inventory_temperature),
                )
                pass_results.append(raw_grounded)
                raw_rows = sum(len(rows) for rows in raw_grounded.values())
                ground_progress.tick(
                    pass_index + 1,
                    extra=f"pass={pass_index + 1}/{pass_count} images={len(raw_grounded)} raw_rows={raw_rows}",
                )
            merged_grounded = merge_qwen_inventory_passes(
                pass_results,
                iou_threshold=float(tuning.qwen_anchor_inventory_consensus_iou),
                min_support=int(tuning.qwen_anchor_inventory_min_support),
            )
            image_anchors_by_image = {image_id: dedupe_anchor_rows(rows) for image_id, rows in merged_grounded.items()}
            ground_progress.finish(
                extra=f"images={len(image_anchors_by_image)} merged_candidates={sum(len(rows) for rows in image_anchors_by_image.values())}"
            )

            if tuning.qwen_structural_fallback_enabled:
                from .qwen_anchor_vllm import (
                    STRUCTURAL_FALLBACK_QWEN_PROMPT,
                    STRUCTURAL_FALLBACK_QWEN_PROMPT_ANTI_OCR,
                    STRUCTURAL_FALLBACK_QWEN_PROMPT_DAM01,
                    STRUCTURAL_FALLBACK_QWEN_PROMPT_NO_TEXT_REF,
                    STRUCTURAL_FALLBACK_QWEN_PROMPT_ITA15,
                    is_degenerate_inventory,
                )
                _structural_fallback_prompt = (
                    GEMMA_STRUCTURAL_FALLBACK_PROMPT_ITA15
                    if use_gemma_anchor_backend and tuning.qwen_ita15_prompt_enabled
                    else STRUCTURAL_FALLBACK_QWEN_PROMPT_DAM01
                    if tuning.qwen_dam01_prompt_enabled
                    else STRUCTURAL_FALLBACK_QWEN_PROMPT_ITA15
                    if tuning.qwen_ita15_prompt_enabled
                    else STRUCTURAL_FALLBACK_QWEN_PROMPT_NO_TEXT_REF
                    if tuning.qwen_no_text_ref_prompt_enabled
                    else STRUCTURAL_FALLBACK_QWEN_PROMPT_ANTI_OCR
                    if tuning.qwen_anti_ocr_prompt_enabled
                    else GEMMA_STRUCTURAL_FALLBACK_PROMPT_ITA15
                    if use_gemma_anchor_backend
                    else STRUCTURAL_FALLBACK_QWEN_PROMPT
                )
                degenerate_ids = {
                    image_id
                    for image_id, rows in image_anchors_by_image.items()
                    if is_degenerate_inventory(rows, threshold=tuning.qwen_degenerate_label_threshold)
                }
                degenerate_ids |= {
                    req.image_id for req in qwen_requests if req.image_id not in image_anchors_by_image
                }
                if degenerate_ids:
                    fallback_requests = [
                        QwenAnchorRequest(
                            image_id=req.image_id,
                            image_path=req.image_path,
                            categories=[],
                            prompt_text=_structural_fallback_prompt,
                            enforce_allowed_labels=False,
                            normalize_open_labels=True,
                        )
                        for req in qwen_requests
                        if req.image_id in degenerate_ids
                    ]
                    fallback_grounded = qwen_grounder.detect_many(
                        fallback_requests,
                        sampling_temperature=0.10,
                    )
                    for image_id, fallback_rows in fallback_grounded.items():
                        if fallback_rows:
                            existing = image_anchors_by_image.get(image_id, [])
                            image_anchors_by_image[image_id] = dedupe_anchor_rows(existing + fallback_rows)
                    print(
                        f"[structural_fallback] degenerate={len(degenerate_ids)} "
                        f"recovered={sum(1 for iid in degenerate_ids if image_anchors_by_image.get(iid))}",
                        flush=True,
                    )

            if tuning.qwen_min_anchor_detections_per_image > 0:
                low_yield_ids = {
                    image_id
                    for image_id, rows in image_anchors_by_image.items()
                    if len(rows) < tuning.qwen_min_anchor_detections_per_image
                }
                low_yield_ids |= {
                    req.image_id for req in qwen_requests if req.image_id not in image_anchors_by_image
                }
                if low_yield_ids:
                    second_pass_requests = [req for req in qwen_requests if req.image_id in low_yield_ids]
                    second_pass_temp = min(float(tuning.qwen_anchor_inventory_temperature) + 0.25, 1.0)
                    second_pass_grounded = qwen_grounder.detect_many(
                        second_pass_requests,
                        sampling_temperature=second_pass_temp,
                    )
                    for image_id, second_rows in second_pass_grounded.items():
                        combined = merge_qwen_inventory_passes(
                            [
                                {image_id: image_anchors_by_image.get(image_id, [])},
                                {image_id: second_rows},
                            ],
                            iou_threshold=float(tuning.qwen_anchor_inventory_consensus_iou),
                            min_support=1,
                        )
                        image_anchors_by_image[image_id] = dedupe_anchor_rows(combined.get(image_id, []))
                    print(
                        f"[second_pass_low_yield] low_yield={len(low_yield_ids)} "
                        f"temp={second_pass_temp:.2f}",
                        flush=True,
                    )
        else:
            ground_progress = _make_stage_progress_logger("anchor_grounding", len(image_plans))
            qwen_requests = [
                QwenAnchorRequest(
                    image_id=str(plan["spec"]["image_id"]),
                    image_path=str(plan["spec"]["image_path"]),
                    categories=list(plan["anchor_inventory_categories"]),
                )
                for plan in image_plans
            ]
            raw_grounded = qwen_grounder.detect_many(qwen_requests, progress_logger=ground_progress)
            image_anchors_by_image = {image_id: dedupe_anchor_rows(rows) for image_id, rows in raw_grounded.items()}
            ground_progress.finish(extra=f"images={len(image_anchors_by_image)}")
    elif grounder is not None:
        ground_progress = _make_stage_progress_logger("anchor_grounding", len(image_plans))
        for plan in image_plans:
            image = Image.open(plan["spec"]["image_path"]).convert("RGB")
            image_anchors: list[dict[str, Any]] = []
            for tag in plan["selected_tags"]:
                image_anchors.extend(grounder.detect(image, tag, threshold=grounding_threshold))
            image_anchors_by_image[str(plan["spec"]["image_id"])] = dedupe_anchor_rows(image_anchors)
            ground_progress.tick(
                len(image_anchors_by_image),
                extra=f"image_id={plan['spec']['image_id']} candidates={len(image_anchors_by_image[str(plan['spec']['image_id'])])}",
            )
        ground_progress.finish(extra=f"images={len(image_anchors_by_image)}")

    if tuning.qwen_degenerate_anchor_filter_enabled:
        total_filtered = 0
        for image_id in list(image_anchors_by_image.keys()):
            before = image_anchors_by_image[image_id]
            after = [
                row for row in before
                if not is_degenerate_anchor_label(str(row.get("label") or ""))
                and not is_ocr_text_label(str(row.get("label") or ""))
            ]
            total_filtered += len(before) - len(after)
            image_anchors_by_image[image_id] = after
        print(f"[degenerate_anchor_filter] removed={total_filtered} anchors", flush=True)

    if tuning.anchor_text_ref_label_filter_enabled:
        total_filtered = 0
        for image_id in list(image_anchors_by_image.keys()):
            before = image_anchors_by_image[image_id]
            after = [
                row for row in before
                if not is_text_ref_anchor_label(str(row.get("label") or ""))
            ]
            total_filtered += len(before) - len(after)
            image_anchors_by_image[image_id] = after
        print(f"[text_ref_anchor_filter] removed={total_filtered} anchors", flush=True)

    if tuning.anchor_label_groundback_enabled and qwen_grounder is not None and image_anchors_by_image:
        path_by_image_id = {
            str(plan["spec"]["image_id"]): str(plan["spec"]["image_path"])
            for plan in image_plans
        }
        groundback_results = qwen_grounder.groundback_check_many(
            image_anchors_by_image,
            image_path_by_image=path_by_image_id,
        )
        gb_total = gb_failed = 0
        iou_threshold = float(tuning.anchor_label_groundback_iou_threshold)
        for image_id, anchor_results in groundback_results.items():
            anchors = image_anchors_by_image.get(image_id, [])
            for anchor_idx, gb_iou in anchor_results:
                if anchor_idx < len(anchors):
                    anchors[anchor_idx]["groundback_iou"] = gb_iou
                    failed = gb_iou < iou_threshold
                    anchors[anchor_idx]["groundback_failed"] = failed
                    if failed:
                        anchors[anchor_idx]["score"] = max(
                            0.0, float(anchors[anchor_idx].get("score") or 0.0) - 0.20
                        )
                        gb_failed += 1
                    gb_total += 1
        print(
            f"[anchor_groundback] checked={gb_total} failed={gb_failed} iou_threshold={iou_threshold:.2f}",
            flush=True,
        )

    if use_independent_qwen_inventory:
        anchor_tag_rows_by_key = {
            (str(row["image_id"]), str(row["node_id"])): row
            for row in anchor_tag_rows
        }
        for plan in image_plans:
            image_id = str(plan["spec"]["image_id"])
            inventory_rows = image_anchors_by_image.get(image_id, [])
            inventory_tags = dedupe_preserve_order(
                [
                    normalize_independent_anchor_label(str(row.get("raw_label") or row.get("label") or ""))
                    or str(row.get("label") or "").strip().lower()
                    for row in inventory_rows
                    if str(row.get("raw_label") or row.get("label") or "").strip()
                ]
            )[:max_tags_per_image]
            inventory_caption = "; ".join(inventory_tags[:6])
            plan["selected_tags"] = list(inventory_tags)
            for node in plan["image_nodes"]:
                node_id = str(node["node_id"])
                plan["per_node_tags"][node_id] = list(inventory_tags)
                row = anchor_tag_rows_by_key.get((image_id, node_id))
                if row is None:
                    continue
                row["caption"] = inventory_caption
                row["raw_labels"] = list(inventory_tags)
                row["discovered_tags"] = list(inventory_tags)
                row["final_prompt_tags"] = list(inventory_tags)

    if sam3_refiner_enabled:
        sam3_refiner = Sam3Refiner(device=device, confidence_threshold=tuning.sam3_confidence_threshold)

    select_progress = _make_stage_progress_logger("anchor_candidate_selection", len(resolvable_nodes))
    selected_count = 0
    for plan in image_plans:
        spec = plan["spec"]
        image_nodes = plan["image_nodes"]
        image = Image.open(spec["image_path"]).convert("RGB")
        image_anchors = image_anchors_by_image.get(str(spec["image_id"]), [])
        if sam3_refiner is not None:
            sam3_refiner.set_image(image)

        for node in image_nodes:
            candidates = []
            for anchor in plan["per_node_local_candidates"].get(str(node["node_id"]), []):
                if not anchor_candidate_viable(str(anchor["label"]), list(anchor["box"]), list(node["bbox"]), (image.width, image.height)):
                    continue
                relation = relation_between_text_and_anchor(list(node["bbox"]), list(anchor["box"]), (image.width, image.height))
                if relation is None:
                    continue
                relevance = anchor_relevance(list(node["bbox"]), list(node["polygon"]), list(anchor["box"]), (image.width, image.height))
                candidates.append(
                    {
                        **anchor,
                        "relation": relation,
                        "relevance": round(float(relevance), 6),
                        "caption": next((row["caption"] for row in anchor_tag_rows if row["image_id"] == spec["image_id"] and row["node_id"] == node["node_id"]), None),
                        "discovered_tags": plan["per_node_tags"].get(str(node["node_id"]), []),
                        "final_prompt_tags": plan["per_node_tags"].get(str(node["node_id"]), []),
                    }
                )
            for anchor in image_anchors:
                if not anchor_candidate_viable(str(anchor["label"]), list(anchor["box"]), list(node["bbox"]), (image.width, image.height)):
                    continue
                relation = relation_between_text_and_anchor(list(node["bbox"]), list(anchor["box"]), (image.width, image.height))
                if relation is None:
                    continue
                relevance = anchor_relevance(list(node["bbox"]), list(node["polygon"]), list(anchor["box"]), (image.width, image.height))
                candidates.append(
                    {
                        **anchor,
                        "relation": relation,
                        "relevance": round(float(relevance), 6),
                        "caption": next((row["caption"] for row in anchor_tag_rows if row["image_id"] == spec["image_id"] and row["node_id"] == node["node_id"]), None),
                        "discovered_tags": plan["per_node_tags"].get(str(node["node_id"]), []),
                        "final_prompt_tags": plan["per_node_tags"].get(str(node["node_id"]), []),
                    }
                )
            filtered_candidates = [
                row
                for row in candidates
                if anchor_candidate_viable(str(row["label"]), list(row["box"]), list(node["bbox"]), (image.width, image.height))
                and float(row.get("relevance") or 0.0) >= 0.08
            ]
            filtered_candidates = annotate_anchor_candidate_support(filtered_candidates)
            filtered_candidates = consolidate_anchor_candidates(filtered_candidates)
            filtered_candidates = preferred_anchor_order(filtered_candidates, text_box=list(node["bbox"]), image_size=(image.width, image.height))
            if sam3_refiner is not None and _should_run_sam3_for_node(seed_candidates=filtered_candidates, node=node, image_size=(image.width, image.height)):
                sam3_prompts = _select_sam3_prompts(
                    seed_candidates=filtered_candidates,
                    node_tags=plan["per_node_tags"].get(str(node["node_id"]), []),
                    limit=_sam3_prompt_limit(),
                )
                sam3_rows: list[dict[str, Any]] = []
                for prompt in sam3_prompts:
                    for anchor in sam3_refiner.detect(prompt, threshold=tuning.sam3_box_threshold):
                        if not anchor_candidate_viable(str(anchor["label"]), list(anchor["box"]), list(node["bbox"]), (image.width, image.height)):
                            continue
                        relation = relation_between_text_and_anchor(list(node["bbox"]), list(anchor["box"]), (image.width, image.height))
                        if relation is None:
                            continue
                        relevance = anchor_relevance(list(node["bbox"]), list(node["polygon"]), list(anchor["box"]), (image.width, image.height))
                        sam3_rows.append(
                            {
                                **anchor,
                                "relation": relation,
                                "relevance": round(float(relevance), 6),
                                "caption": next((row["caption"] for row in anchor_tag_rows if row["image_id"] == spec["image_id"] and row["node_id"] == node["node_id"]), None),
                                "discovered_tags": plan["per_node_tags"].get(str(node["node_id"]), []),
                                "final_prompt_tags": plan["per_node_tags"].get(str(node["node_id"]), []),
                            }
                        )
                filtered_candidates.extend(sam3_rows)
            filtered_candidates = annotate_anchor_candidate_support(filtered_candidates)
            filtered_candidates = consolidate_anchor_candidates(filtered_candidates)
            filtered_candidates = preferred_anchor_order(filtered_candidates, text_box=list(node["bbox"]), image_size=(image.width, image.height))
            grounded_anchor_rows.append(
                {
                    "image_id": spec["image_id"],
                    "node_id": node["node_id"],
                    "top_candidates": filtered_candidates[:5],
                }
            )
            if filtered_candidates:
                best_anchor_by_node[(spec["image_id"], str(node["node_id"]))] = filtered_candidates[0]
            selected_count += 1
            select_progress.tick(
                selected_count,
                extra=(
                    f"image_id={spec['image_id']} node_id={node['node_id']} "
                    f"top_candidates={len(filtered_candidates[:5])}"
                ),
            )
    select_progress.finish(extra=f"grounded_rows={len(grounded_anchor_rows)} best_anchors={len(best_anchor_by_node)}")
    if sam3_refiner is not None:
        sam3_refiner.close()
    if grounder is not None:
        grounder.close()
    if qwen_grounder is not None:
        qwen_grounder.close()
    if tagger is not None:
        tagger.close()
    return anchor_tag_rows, grounded_anchor_rows, best_anchor_by_node


def _needs_anchor_relabel(tuple_rows: list[dict[str, Any]]) -> bool:
    if not tuple_rows:
        return False
    sample = tuple_rows[0]
    label = str(sample.get("anchor_label") or "")
    support = int((sample.get("semantic_debug") or {}).get("anchor_source_support_count") or 1)
    conflict = max(int((row.get("semantic_debug") or {}).get("anchor_conflict_count") or 0) for row in tuple_rows)
    if label in GENERIC_TEXT_ANCHORS or label in {"wall", "panel"}:
        return True
    if conflict >= 1:
        return True
    if str(sample.get("anchor_category") or "") in {"other", "building"} and support <= 1:
        return True
    if label in {"poster", "sign", "board", "display"} and len(tuple_rows) >= 2:
        return True
    return False


def refine_anchor_labels(
    tuple_rows: list[dict[str, Any]],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    if not tuple_rows:
        return tuple_rows
    tuning = load_semantic_dev40_tuning()
    grouped_rows = _cluster_relabel_groups(tuple_rows)
    relabeler = GeminiAnchorRelabeler(model_name=model_name)
    updated_rows = list(tuple_rows)
    by_id = {id(row): row for row in updated_rows}
    pending_groups: list[dict[str, Any]] = []
    for group_index, group_rows in enumerate(grouped_rows, start=1):
        if not _needs_anchor_relabel(group_rows):
            continue
        anchor_box = list(group_rows[0]["anchor_box"])
        image_path = str(group_rows[0]["image_path"])
        current_label = str(group_rows[0]["anchor_label"])
        relation = str(group_rows[0]["relation"])
        text_boxes = [bbox_xywh_to_xyxy(list(row["text_bbox"])) for row in group_rows]
        union_box = union_bbox(text_boxes + [anchor_box])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            crop_box = expand_box(union_box, image_width=image.width, image_height=image.height, scale=1.18)
            crop = image.crop(tuple(crop_box))
        example_texts = []
        for row in group_rows:
            for text in [row.get("answer"), *(row.get("child_words") or [])]:
                text_value = str(text or "").strip()
                if text_value and text_value not in example_texts:
                    example_texts.append(text_value)
        candidate_labels = [current_label]
        for row in group_rows:
            debug = row.get("semantic_debug") or {}
            for label in [debug.get("raw_anchor_label"), *(debug.get("anchor_support_labels") or []), *(debug.get("final_prompt_tags") or []), *(debug.get("discovered_tags") or [])]:
                label_text = sanitize_anchor_label(str(label or ""))
                if label_text and label_text not in candidate_labels:
                    candidate_labels.append(label_text)
        pending_groups.append(
            {
                "key": f"group_{group_index}",
                "group_rows": group_rows,
                "image_crop": crop,
                "current_label": current_label,
                "candidate_labels": candidate_labels,
                "example_texts": example_texts,
                "relation": relation,
            }
        )

    relabel_results: dict[str, dict[str, Any]] = {}
    if pending_groups and tuning.gemini_api_mode == "batch":
        relabel_results = relabeler.relabel_many(
            pending_groups,
            chunk_size=int(tuning.gemini_batch_chunk_size),
            poll_interval_s=int(tuning.gemini_batch_poll_seconds),
            timeout_s=int(tuning.gemini_batch_timeout_seconds),
        )

    for pending in pending_groups:
        current_label = str(pending["current_label"])
        if tuning.gemini_api_mode == "batch":
            relabel = relabel_results.get(str(pending["key"])) or {
                "label": current_label,
                "alternates": [current_label],
                "usage": {},
                "raw_text": "relabel_failed: missing batch response",
            }
        else:
            try:
                relabel = relabeler.relabel(
                    pending["image_crop"],
                    current_label=current_label,
                    candidate_labels=list(pending["candidate_labels"]),
                    example_texts=list(pending["example_texts"]),
                    relation=str(pending["relation"]),
                )
            except Exception as exc:
                relabel = {
                    "label": current_label,
                    "alternates": [current_label],
                    "usage": {},
                    "raw_text": f"relabel_failed: {exc}",
                }

        new_label = sanitize_anchor_label(str(relabel.get("label") or "")) or current_label
        new_synonyms = []
        for label in [new_label, *(relabel.get("alternates") or []), current_label]:
            cleaned = sanitize_anchor_label(str(label or ""))
            if cleaned and cleaned not in new_synonyms:
                new_synonyms.append(cleaned)
        for row in pending["group_rows"]:
            debug = dict(row.get("semantic_debug") or {})
            debug["anchor_relabel"] = {
                "model": model_name,
                "label": new_label,
                "alternates": new_synonyms,
                "raw_text": relabel.get("raw_text"),
            }
            row["anchor_label"] = new_label
            row["anchor_synonyms"] = list(new_synonyms or [new_label])
            row["anchor_category"] = categorize_anchor(new_label)
            row["anchor_label_source"] = f"semantic_relabel_{model_name}"
            row["semantic_debug"] = debug
            by_id[id(row)] = row
    return list(by_id.values())
