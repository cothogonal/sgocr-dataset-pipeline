from __future__ import annotations

import json
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from .bootstrap import REGION_PHRASES, REGION_SYNONYMS, area_bucket, density_bucket, is_valid_text, normalize_answer, region_key_for_bbox, write_json, write_jsonl
from .bootstrap_kd import bbox_xywh_to_xyxy, bbox_xyxy_to_xywh, compute_resolvability, overlap_fraction
from .consensus import OCRVote, choose_consensus
from .dev40_complete import (
    annotate_inline_frontier,
    build_question_candidates,
    enforce_type_constraints,
    row_to_final_sample,
    run_teacher_batches,
    sample_id_for_candidate,
    select_candidates,
    union_bbox,
)
from .ocr_runtime import CraftDetector, PARSeqRecognizer, PaddleOCRDetector, PaddleOCRRecognizer, TrOCRRecognizer, bbox_to_polygon, crop_with_padding
from .nemotron_frontend import run_nemotron_ocr_stage
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
    collect_semantic_kd_metadata,
    extract_anchor_color,
    expand_grounding_tags_for_node,
    expand_box,
    normalize_independent_anchor_label,
    remap_region_box_to_image,
    sanitize_anchor_label,
    sanitize_anchor_tags,
    local_text_location_metadata,
    relation_between_text_and_anchor,
)
from .run_quality import compute_run_quality

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


def _anchor_area_fraction(box: list[float], image_size: tuple[int, int]) -> float:
    area = max((float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])), 1.0)
    return area / max(float(image_size[0]) * float(image_size[1]), 1.0)


def preferred_anchor_order(candidates: list[dict[str, Any]], *, text_box: list[float], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    tuning = load_semantic_dev40_tuning()
    def adjusted_score(row: dict[str, Any]) -> float:
        label = str(row.get("label") or "")
        area_frac = _anchor_area_fraction(list(row["box"]), image_size)
        generic_penalty = tuning.generic_anchor_penalty if label in GENERIC_TEXT_ANCHORS else 0.0
        oversized_penalty = max(0.0, area_frac - tuning.oversized_anchor_area_start) * 0.35
        local_bonus = tuning.florence_region_bonus if str(row.get("source") or "") == "florence_region" else 0.0
        sam3_bonus = tuning.sam3_relevance_bonus if str(row.get("source") or "") == "sam3" else 0.0
        specificity_bonus = 0.035 if label not in GENERIC_TEXT_ANCHORS else 0.0
        support_bonus = max(0.0, float(row.get("support_count") or 1.0) - 1.0) * tuning.anchor_support_bonus
        source_support_bonus = max(0.0, float(row.get("source_support_count") or 1.0) - 1.0) * tuning.anchor_support_bonus * 0.5
        score = float(row.get("relevance") or 0.0) + local_bonus + sam3_bonus + specificity_bonus + support_bonus + source_support_bonus - generic_penalty - oversized_penalty
        row["selection_score"] = round(score, 6)
        return score

    ordered = sorted(candidates, key=lambda row: (-adjusted_score(row), -float(row.get("relevance") or 0.0), -float(row.get("score") or 0.0), row["label"]))
    return ordered


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    area_b = max((bx2 - bx1) * (by2 - by1), 1e-6)
    return inter_area / max(area_a + area_b - inter_area, 1e-6)


def _merge_detection_candidates(
    rows: list[dict[str, Any]],
    *,
    merge_overlap: float,
    max_detections: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: (-float(row.get("detection_confidence") or 0.0), -max((row["bbox"][2] - row["bbox"][0]) * (row["bbox"][3] - row["bbox"][1]), 0.0)))
    for row in ordered:
        best_idx = None
        best_score = -1.0
        for idx, existing in enumerate(merged):
            overlap_ab = overlap_fraction(list(row["bbox"]), list(existing["bbox"]))
            overlap_ba = overlap_fraction(list(existing["bbox"]), list(row["bbox"]))
            iou = _bbox_iou(list(row["bbox"]), list(existing["bbox"]))
            score = max(min(overlap_ab, overlap_ba), iou)
            if score >= merge_overlap and score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            row_copy = dict(row)
            row_copy["detector_sources"] = list(dict.fromkeys(row.get("detector_sources") or [str(row.get("detector_source") or "unknown")]))
            merged.append(row_copy)
            continue
        existing = dict(merged[best_idx])
        union_box = [
            round(min(float(existing["bbox"][0]), float(row["bbox"][0])), 2),
            round(min(float(existing["bbox"][1]), float(row["bbox"][1])), 2),
            round(max(float(existing["bbox"][2]), float(row["bbox"][2])), 2),
            round(max(float(existing["bbox"][3]), float(row["bbox"][3])), 2),
        ]
        existing["bbox"] = union_box
        existing["polygon"] = bbox_to_polygon(union_box)
        existing["detection_confidence"] = max(float(existing.get("detection_confidence") or 0.0), float(row.get("detection_confidence") or 0.0))
        existing["detector_sources"] = list(
            dict.fromkeys([*(existing.get("detector_sources") or [str(existing.get("detector_source") or "unknown")]), *(row.get("detector_sources") or [str(row.get("detector_source") or "unknown")])])
        )
        merged[best_idx] = existing
    return merged[:max_detections]


def _degenerate_consensus_text(text: str) -> bool:
    compact = "".join(ch for ch in normalize_answer(str(text)) if ch.isalnum())
    if len(compact) < 4:
        return False
    if len(set(compact)) == 1 and compact[0] in {"1", "l", "i", "|"}:
        return True
    return False


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


def _cluster_relabel_groups(tuple_rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    tuning = load_semantic_dev40_tuning()
    groups: list[list[dict[str, Any]]] = []
    ordered = sorted(
        tuple_rows,
        key=lambda row: (
            str(row["image_id"]),
            str(row["relation"]),
            -float(row.get("anchor_score") or 0.0),
        ),
    )
    for row in ordered:
        box = [float(value) for value in row["anchor_box"]]
        placed = False
        for group in groups:
            exemplar = group[0]
            if str(exemplar["image_id"]) != str(row["image_id"]) or str(exemplar["relation"]) != str(row["relation"]):
                continue
            exemplar_box = [float(value) for value in exemplar["anchor_box"]]
            overlap_ab = overlap_fraction(box, exemplar_box)
            overlap_ba = overlap_fraction(exemplar_box, box)
            iou = _bbox_iou(box, exemplar_box)
            if min(overlap_ab, overlap_ba) >= tuning.anchor_conflict_overlap or iou >= max(0.55, tuning.anchor_conflict_overlap - 0.08):
                group.append(row)
                placed = True
                break
        if not placed:
            groups.append([row])
    return groups


def _box_width(box: list[float]) -> float:
    return max(float(box[2]) - float(box[0]), 1.0)


def _box_height(box: list[float]) -> float:
    return max(float(box[3]) - float(box[1]), 1.0)


def _box_center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _overlap_1d(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _node_text_kind(text: str) -> str:
    compact = "".join(ch for ch in normalize_answer(str(text)) if ch.isalnum())
    if compact.isdigit():
        return "numeric"
    if compact.isalpha():
        return "alpha"
    if compact.isalnum():
        return "alnum"
    return "mixed"


def _node_text_case(text: str) -> str:
    stripped = "".join(ch for ch in str(text) if ch.isalnum())
    if not stripped:
        return "empty"
    if stripped.isupper():
        return "upper"
    if stripped.islower():
        return "lower"
    if stripped[:1].isupper() and stripped[1:].islower():
        return "title"
    return "mixed"


def _styles_compatible(a: dict[str, Any], b: dict[str, Any], *, height_ratio_max: float) -> bool:
    box_a = list(a["bbox"])
    box_b = list(b["bbox"])
    height_ratio = max(_box_height(box_a), _box_height(box_b)) / max(min(_box_height(box_a), _box_height(box_b)), 1.0)
    if height_ratio > height_ratio_max:
        return False
    kind_a = _node_text_kind(str(a.get("text") or ""))
    kind_b = _node_text_kind(str(b.get("text") or ""))
    if "numeric" in {kind_a, kind_b} and kind_a != kind_b:
        return False
    case_a = _node_text_case(str(a.get("text") or ""))
    case_b = _node_text_case(str(b.get("text") or ""))
    if kind_a == kind_b == "alpha" and case_a != case_b and "mixed" not in {case_a, case_b}:
        return False
    return True


def _can_merge_inline(left: dict[str, Any], right: dict[str, Any], tuning: Any) -> bool:
    left_box = list(left["bbox"])
    right_box = list(right["bbox"])
    y_overlap = _overlap_1d(left_box[1], left_box[3], right_box[1], right_box[3]) / max(min(_box_height(left_box), _box_height(right_box)), 1.0)
    gap = float(right_box[0]) - float(left_box[2])
    max_height = max(_box_height(left_box), _box_height(right_box))
    if y_overlap < tuning.text_merge_y_overlap_min:
        return False
    if gap < -0.20 * max_height or gap > tuning.text_merge_gap_ratio_max * max_height:
        return False
    return _styles_compatible(left, right, height_ratio_max=tuning.text_merge_height_ratio_max)


def _can_merge_stacked(top: dict[str, Any], bottom: dict[str, Any], tuning: Any) -> bool:
    top_box = list(top["bbox"])
    bottom_box = list(bottom["bbox"])
    x_overlap = _overlap_1d(top_box[0], top_box[2], bottom_box[0], bottom_box[2]) / max(min(_box_width(top_box), _box_width(bottom_box)), 1.0)
    gap = float(bottom_box[1]) - float(top_box[3])
    max_height = max(_box_height(top_box), _box_height(bottom_box))
    if x_overlap < tuning.text_merge_x_overlap_min:
        return False
    if gap < -0.20 * max_height or gap > tuning.text_merge_gap_ratio_max * max_height:
        return False
    return _styles_compatible(top, bottom, height_ratio_max=tuning.text_merge_height_ratio_max)


def _dedupe_merge_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    ordered = sorted(
        nodes,
        key=lambda row: (
            -float(row.get("confidence") or 0.0),
            -len(str(row.get("text") or "")),
            -((_box_width(list(row["bbox"])) * _box_height(list(row["bbox"])))),
        ),
    )
    for row in ordered:
        row_box = list(row["bbox"])
        row_text = normalize_answer(str(row.get("text") or ""))
        duplicate = False
        for existing in kept:
            existing_box = list(existing["bbox"])
            overlap_ab = overlap_fraction(row_box, existing_box)
            overlap_ba = overlap_fraction(existing_box, row_box)
            iou = _bbox_iou(row_box, existing_box)
            existing_text = normalize_answer(str(existing.get("text") or ""))
            text_match = bool(
                row_text
                and existing_text
                and (row_text == existing_text or row_text in existing_text or existing_text in row_text)
            )
            if (min(overlap_ab, overlap_ba) >= 0.74 or iou >= 0.60) and text_match:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return kept


def _anchor_cluster_rows(
    image_nodes: list[dict[str, Any]],
    best_anchor_by_node: dict[tuple[str, str], dict[str, Any]],
    *,
    image_id: str,
) -> list[dict[str, Any]]:
    tuning = load_semantic_dev40_tuning()
    seed_rows: list[dict[str, Any]] = []
    for node in image_nodes:
        anchor = best_anchor_by_node.get((image_id, str(node["node_id"])))
        if not anchor or str(anchor.get("relation") or "") != "on":
            continue
        seed_rows.append({"node": node, "anchor": anchor})

    clusters: list[list[dict[str, Any]]] = []
    for seed in sorted(seed_rows, key=lambda row: (-float(row["anchor"].get("score") or 0.0), str(row["anchor"].get("label") or ""))):
        placed = False
        seed_box = list(seed["anchor"]["box"])
        for cluster in clusters:
            exemplar = cluster[0]
            exemplar_box = list(exemplar["anchor"]["box"])
            overlap_ab = overlap_fraction(seed_box, exemplar_box)
            overlap_ba = overlap_fraction(exemplar_box, seed_box)
            iou = _bbox_iou(seed_box, exemplar_box)
            if max(min(overlap_ab, overlap_ba), iou) >= tuning.text_merge_anchor_overlap_min:
                cluster.append(seed)
                placed = True
                break
        if not placed:
            clusters.append([seed])

    out: list[dict[str, Any]] = []
    for idx, cluster in enumerate(clusters, start=1):
        anchors = [row["anchor"] for row in cluster]
        cluster_box = union_bbox([list(anchor["box"]) for anchor in anchors])
        best_anchor = max(
            anchors,
            key=lambda anchor: (
                float(anchor.get("selection_score") or anchor.get("relevance") or 0.0),
                float(anchor.get("score") or 0.0),
            ),
        )
        out.append(
            {
                "cluster_id": f"{image_id}::anchor_cluster::{idx}",
                "anchor_box": cluster_box,
                "anchor": best_anchor,
                "member_node_ids": [str(row["node"]["node_id"]) for row in cluster],
                "labels": list(dict.fromkeys(str(anchor.get("label") or "") for anchor in anchors if str(anchor.get("label") or "").strip())),
            }
        )
    return out


def _node_belongs_to_anchor_cluster(node: dict[str, Any], cluster_box: list[float], tuning: Any) -> bool:
    node_box = list(node["bbox"])
    center_x, center_y = _box_center(node_box)
    if float(cluster_box[0]) <= center_x <= float(cluster_box[2]) and float(cluster_box[1]) <= center_y <= float(cluster_box[3]):
        return True
    overlap = overlap_fraction(node_box, cluster_box)
    reverse_overlap = overlap_fraction(cluster_box, node_box)
    return max(overlap, reverse_overlap, _bbox_iou(node_box, cluster_box)) >= tuning.text_merge_anchor_overlap_min


def _component_orientation(nodes: list[dict[str, Any]]) -> str:
    centers = [_box_center(list(node["bbox"])) for node in nodes]
    if len(centers) <= 1:
        return "singleton"
    x_values = [point[0] for point in centers]
    y_values = [point[1] for point in centers]
    x_range = max(x_values) - min(x_values)
    y_range = max(y_values) - min(y_values)
    if y_range > x_range * 1.35:
        return "vertical"
    if x_range > y_range * 1.35:
        return "horizontal"
    return "mixed"


def _is_valid_merged_answer(answer: str, *, node_count: int) -> bool:
    if is_valid_text(answer):
        return True
    compact = "".join(ch for ch in str(answer) if ch.isalnum())
    words = [word for word in str(answer).split() if word.strip()]
    if node_count >= 2 and 4 <= len(str(answer).strip()) <= 64 and 2 <= len(words) <= 8 and len(compact) >= 4 and len(set(compact.lower())) > 1:
        return True
    return False


def _order_component_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order component nodes in natural reading order (left-to-right, top-to-bottom).

    Groups nodes into horizontal rows by y-centroid proximity (tolerance = 0.6 × median
    node height), then sorts left-to-right within each row. This correctly handles:
    - Inline words on a single line (same row → x-sort)
    - Stacked words on a sign (each word in its own row → y-sort)
    - Mixed layouts (multi-line text blocks)

    Replaces the previous orientation-class approach, which misclassified some vertical
    stacks as 'mixed' and applied an x-first sort that scrambled reading order.
    """
    if len(nodes) <= 1:
        return list(nodes)

    heights = sorted(_box_height(list(n["bbox"])) for n in nodes)
    median_h = heights[len(heights) // 2]
    row_tolerance = max(median_h * 0.6, 2.0)

    by_y = sorted(nodes, key=lambda n: (_box_center(list(n["bbox"]))[1], _box_center(list(n["bbox"]))[0]))

    rows: list[list[dict[str, Any]]] = []
    for node in by_y:
        cy = _box_center(list(node["bbox"]))[1]
        placed = False
        for row in rows:
            row_mean_y = sum(_box_center(list(n["bbox"]))[1] for n in row) / len(row)
            if abs(cy - row_mean_y) <= row_tolerance:
                row.append(node)
                placed = True
                break
        if not placed:
            rows.append([node])

    result: list[dict[str, Any]] = []
    for row in rows:
        result.extend(sorted(row, key=lambda n: _box_center(list(n["bbox"]))[0]))
    return result


def build_merged_sign_tuples_for_image(
    spec: dict[str, str],
    image_nodes: list[dict[str, Any]],
    all_image_nodes: list[dict[str, Any]],
    best_anchor_by_node: dict[tuple[str, str], dict[str, Any]],
    image_anchors: list[dict[str, Any]],
    image_source_map: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tuning = load_semantic_dev40_tuning()
    if not tuning.text_merge_enabled or not image_nodes:
        return [], {"merge_groups": 0, "merge_subsumed_candidates": 0}

    image_width = int(image_nodes[0]["image_width"])
    image_height = int(image_nodes[0]["image_height"])
    cluster_rows = _anchor_cluster_rows(image_nodes, best_anchor_by_node, image_id=str(spec["image_id"]))
    merged_rows: list[dict[str, Any]] = []
    seen_node_sets: set[tuple[str, ...]] = set()
    group_count = 0

    for cluster in cluster_rows:
        candidate_nodes = [
            node
            for node in all_image_nodes
            if _node_belongs_to_anchor_cluster(node, list(cluster["anchor_box"]), tuning)
            and is_valid_text(str(node.get("text") or ""))
        ]
        candidate_nodes = _dedupe_merge_nodes(candidate_nodes)
        if len(candidate_nodes) < 2:
            continue

        adjacency: dict[str, set[str]] = {str(node["node_id"]): set() for node in candidate_nodes}
        by_id = {str(node["node_id"]): node for node in candidate_nodes}
        ordered_x = sorted(candidate_nodes, key=lambda row: (_box_center(list(row["bbox"]))[0], _box_center(list(row["bbox"]))[1]))
        ordered_y = sorted(candidate_nodes, key=lambda row: (_box_center(list(row["bbox"]))[1], _box_center(list(row["bbox"]))[0]))
        for left_idx, left in enumerate(ordered_x):
            for right in ordered_x[left_idx + 1 :]:
                if _can_merge_inline(left, right, tuning):
                    adjacency[str(left["node_id"])].add(str(right["node_id"]))
                    adjacency[str(right["node_id"])].add(str(left["node_id"]))
        for top_idx, top in enumerate(ordered_y):
            for bottom in ordered_y[top_idx + 1 :]:
                if _can_merge_stacked(top, bottom, tuning):
                    adjacency[str(top["node_id"])].add(str(bottom["node_id"]))
                    adjacency[str(bottom["node_id"])].add(str(top["node_id"]))

        visited: set[str] = set()
        for node in candidate_nodes:
            node_id = str(node["node_id"])
            if node_id in visited:
                continue
            stack = [node_id]
            component_ids: list[str] = []
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                component_ids.append(current)
                stack.extend(sorted(adjacency[current] - visited))
            if len(component_ids) < 2:
                continue
            component_nodes = [by_id[current] for current in component_ids]
            ordered_component = _order_component_nodes(component_nodes)
            node_ids = tuple(str(item["node_id"]) for item in ordered_component)
            if node_ids in seen_node_sets:
                continue
            seen_node_sets.add(node_ids)
            answer = " ".join(str(item["text"]).strip() for item in ordered_component if str(item["text"]).strip())
            if not answer or not _is_valid_merged_answer(answer, node_count=len(ordered_component)):
                continue
            union_xyxy = union_bbox([list(item["bbox"]) for item in ordered_component])
            polygon = bbox_to_polygon(union_xyxy)
            resolvability = compute_resolvability(polygon, image_width, image_height)
            resolvable_member_count = sum(1 for item in ordered_component if bool(item.get("resolvable")))
            if resolvable_member_count < 1 or not resolvability["passes"]:
                continue
            bbox_xywh = bbox_xyxy_to_xywh(union_xyxy)
            coarse_phrase = REGION_PHRASES.get(region_key_for_bbox(bbox_xywh, image_width, image_height), "area of the image")
            location_meta = local_text_location_metadata(
                primary_box=union_xyxy,
                primary_node_ids=list(node_ids),
                all_text_nodes=all_image_nodes,
                image_size=(image_width, image_height),
                coarse_phrase=coarse_phrase,
            )
            anchor = dict(cluster["anchor"])
            anchor_local = anchor_local_location_metadata(union_xyxy, list(anchor["box"]), "on", str(anchor["label"]))
            kd_metadata = collect_semantic_kd_metadata(
                primary_box=union_xyxy,
                primary_polygon=polygon,
                primary_node_ids=list(node_ids),
                all_text_nodes=all_image_nodes,
                all_anchors=image_anchors,
                image_size=(image_width, image_height),
            )
            unresolved_support = sum(1 for item in ordered_component if not bool(item.get("resolvable")))
            kd_metadata["anchor_conflict_count"] = int(anchor.get("label_conflict_count") or 0)
            kd_metadata["anchor_cluster_size"] = int(anchor.get("cluster_size") or 1)
            kd_metadata["anchor_alternate_labels"] = list(anchor.get("alternate_labels") or [])
            kd_metadata["merge_unresolved_support_count"] = unresolved_support
            kd_metadata["merge_resolvable_support_count"] = resolvable_member_count
            group_count += 1
            merged_rows.append(
                {
                    "tuple_id": f"{spec['image_id']}::merge::{group_count}::{'_'.join(node_ids)}",
                    "image_id": spec["image_id"],
                    "image_path": spec["image_path"],
                    "image_width": image_width,
                    "image_height": image_height,
                    "ann_id": f"merge_{'_'.join(node_ids)}",
                    "answer": answer,
                    "answer_normalized": normalize_answer(answer),
                    "text_polygon": polygon,
                    "text_bbox": [round(v, 2) for v in bbox_xywh],
                    "text_node_ids": list(node_ids),
                    "child_words": [str(item["text"]) for item in ordered_component],
                    "anchor_label": str(anchor["label"]),
                    "anchor_synonyms": list(dict.fromkeys([str(anchor["label"]), *(str(label) for label in anchor.get("alternate_labels") or [])])),
                    "location_phrase": coarse_phrase,
                    "location_synonyms": list(
                        REGION_SYNONYMS.get(
                            region_key_for_bbox(bbox_xywh, image_width, image_height),
                            (REGION_PHRASES.get(region_key_for_bbox(bbox_xywh, image_width, image_height), "area of the image"),),
                        )
                    ),
                    "specific_location_phrase": str(location_meta["specific_location_phrase"]),
                    "specific_location_synonyms": list(location_meta["specific_location_synonyms"]),
                    "anchor_local_phrase": str(anchor_local["phrase"]),
                    "anchor_local_synonyms": list(anchor_local["synonyms"]),
                    "anchor_local_clean": bool(anchor_local["clean"]),
                    "anchor_local_mode": str(anchor_local["mode"]),
                    "anchor_box": [round(float(v), 2) for v in anchor["box"]],
                    "anchor_score": float(anchor["score"]),
                    "anchor_color": str(extract_anchor_color(str(anchor["label"])) or ""),
                    "anchor_category": categorize_anchor(str(anchor["label"])),
                    "anchor_source": str(anchor.get("source") or "grounding_dino"),
                    "relation": "on",
                    "ref_label": None,
                    "ref_box": None,
                    "unique": True,
                    "answer_level": "sign",
                    "ocr_confidence": min(float(item["confidence"]) for item in ordered_component),
                    "consensus_tier": "semantic_merge",
                    "dataset_source": image_source_map.get(spec["image_id"], "textocr_val"),
                    "region_key": region_key_for_bbox(bbox_xywh, image_width, image_height),
                    "resolvable": True,
                    "resolvability": resolvability,
                    "kd_metadata": kd_metadata,
                    "valid_text_count": len(all_image_nodes),
                    "area_fraction": float(((union_xyxy[2] - union_xyxy[0]) * (union_xyxy[3] - union_xyxy[1])) / max(image_width * image_height, 1)),
                    "text_length": len(answer),
                    "density_bucket": density_bucket(len(all_image_nodes)),
                    "area_bucket": area_bucket(float(((union_xyxy[2] - union_xyxy[0]) * (union_xyxy[3] - union_xyxy[1])) / max(image_width * image_height, 1))),
                    "group_kind": "semantic_merge_group",
                    "semantic_debug": {
                        "caption": anchor.get("caption"),
                        "discovered_tags": list(anchor.get("discovered_tags") or []),
                        "final_prompt_tags": list(anchor.get("final_prompt_tags") or anchor.get("discovered_tags") or []),
                        "raw_anchor_label": anchor.get("raw_label"),
                        "best_anchor_relevance": float(anchor["relevance"]),
                        "best_anchor_selection_score": float(anchor.get("selection_score") or anchor["relevance"]),
                        "anchor_support_count": int(anchor.get("support_count") or 1),
                        "anchor_source_support_count": int(anchor.get("source_support_count") or 1),
                        "anchor_support_labels": list(anchor.get("support_labels") or []),
                        "anchor_alternate_labels": list(anchor.get("alternate_labels") or []),
                        "anchor_conflict_count": int(anchor.get("label_conflict_count") or 0),
                        "anchor_cluster_size": int(anchor.get("cluster_size") or 1),
                        "top_candidates": [anchor],
                        "location_detail": str(location_meta["layout_detail"]),
                        "merge_debug": {
                            "cluster_id": cluster["cluster_id"],
                            "cluster_labels": list(cluster["labels"]),
                            "merge_orientation": _component_orientation(ordered_component),
                            "merged_node_count": len(ordered_component),
                            "merged_resolvable_count": resolvable_member_count,
                            "merged_unresolvable_count": unresolved_support,
                        },
                    },
                    "anchor_label_source": "grounding",
                }
            )
    return merged_rows, {"merge_groups": len(merged_rows), "merge_subsumed_candidates": 0}


def _filter_subsumed_rows(tuple_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    tuning = load_semantic_dev40_tuning()
    strong_groups = [
        row
        for row in tuple_rows
        if str(row.get("group_kind") or "") == "semantic_merge_group"
        and len(row.get("text_node_ids") or []) >= tuning.text_merge_subsume_min_nodes
    ]
    if not strong_groups:
        return tuple_rows, 0
    strong_ids = {str(row["tuple_id"]) for row in strong_groups}
    kept: list[dict[str, Any]] = []
    removed = 0
    for row in tuple_rows:
        if str(row["tuple_id"]) in strong_ids:
            kept.append(row)
            continue
        row_ids = set(str(node_id) for node_id in row.get("text_node_ids") or [])
        row_anchor_box = [float(value) for value in row.get("anchor_box") or []]
        subsumed = False
        for group in strong_groups:
            if str(group["image_id"]) != str(row["image_id"]) or str(group.get("relation") or "") != str(row.get("relation") or ""):
                continue
            group_ids = set(str(node_id) for node_id in group.get("text_node_ids") or [])
            if not row_ids or row_ids == group_ids or not row_ids.issubset(group_ids):
                continue
            group_anchor_box = [float(value) for value in group.get("anchor_box") or []]
            if _bbox_iou(row_anchor_box, group_anchor_box) < 0.35 and overlap_fraction(row_anchor_box, group_anchor_box) < 0.20:
                continue
            subsumed = True
            break
        if subsumed:
            removed += 1
            continue
        kept.append(row)
    return kept, removed


def build_dev40_semantic_dataset(
    *,
    source_experiment_dir: Path,
    out_dir: Path,
    intermediate_dir: Path,
    cache_intermediate_dir: Path | None = None,
    model: str = "gemini-2.5-flash",
    device: str = "auto",
    workers: int = 4,
    max_side: int = 768,
    target_per_image: int = 4,
    max_detections: int = 72,
    grounding_threshold: float = 0.36,
    max_tags_per_image: int = 8,
    cache_level: str = "verified",
) -> dict[str, Any]:
    pipeline_start = time.time()
    tuning = load_semantic_dev40_tuning()
    runtime_models = dict(RUNTIME_MODELS)
    runtime_models["ocr_frontend"] = tuning.ocr_frontend
    runtime_models["anchor_tag_discovery_backend"] = tuning.anchor_tag_discovery_backend
    runtime_models["anchor_candidate_backend"] = tuning.anchor_candidate_backend
    runtime_models["qwen_anchor_inventory_mode"] = tuning.qwen_anchor_inventory_mode
    runtime_models["qwen_anchor_inventory_pass_count"] = tuning.qwen_anchor_inventory_pass_count
    runtime_models["qwen_anchor_inventory_temperature"] = tuning.qwen_anchor_inventory_temperature
    runtime_models["qwen_anchor_inventory_consensus_iou"] = tuning.qwen_anchor_inventory_consensus_iou
    runtime_models["qwen_anchor_inventory_min_support"] = tuning.qwen_anchor_inventory_min_support
    if tuning.anchor_candidate_backend == "qwen3_vl_vllm" or tuning.anchor_tag_discovery_backend == "qwen3_vl_vllm":
        runtime_models["grounder"] = tuning.qwen_anchor_model
    elif tuning.anchor_candidate_backend == "gemma4_ollama":
        runtime_models["grounder"] = f"gemma4_ollama:{tuning.gemma_ollama_model}"
    if tuning.sam3_refine_mode != "none":
        runtime_models["sam3_refiner"] = "facebook/sam3"
    image_specs, image_source_map = load_image_specs(source_experiment_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = resolve_device(device)
    cached_verified_tuples: list[dict[str, Any]] | None = None
    cached_anchor_tag_rows: list[dict[str, Any]] | None = None
    cached_grounded_anchor_rows: list[dict[str, Any]] | None = None
    cached_tuple_debug: dict[str, Any] | None = None

    cache_compatible = False
    if cache_intermediate_dir and (cache_intermediate_dir / "runtime_models.json").exists():
        try:
            cached_runtime_models = json.loads((cache_intermediate_dir / "runtime_models.json").read_text(encoding="utf-8"))
            if cache_level == "ocr":
                cache_compatible = _ocr_runtime_signature(cached_runtime_models) == _ocr_runtime_signature(runtime_models)
            elif cache_level == "verified":
                cache_compatible = (
                    _ocr_runtime_signature(cached_runtime_models) == _ocr_runtime_signature(runtime_models)
                    and _semantic_runtime_signature(cached_runtime_models) == _semantic_runtime_signature(runtime_models)
                )
            else:
                cache_compatible = cached_runtime_models == runtime_models
        except Exception:
            cache_compatible = False

    if cache_level not in {"none", "ocr", "verified"}:
        raise ValueError(f"Unsupported cache_level: {cache_level}")

    if cache_compatible and cache_intermediate_dir and cache_level != "none" and (cache_intermediate_dir / "text_nodes.jsonl").exists():
        print(
            f"[stage:pipeline] cache_reuse cache_level={cache_level} cache_dir={cache_intermediate_dir}",
            flush=True,
        )
        text_nodes = load_jsonl(cache_intermediate_dir / "text_nodes.jsonl")
        consensus_stats = json.loads((cache_intermediate_dir / "consensus_stats.json").read_text(encoding="utf-8"))
        detection_rows = load_jsonl(cache_intermediate_dir / "text_detections.jsonl") if (cache_intermediate_dir / "text_detections.jsonl").exists() else []
        detection_summary = {
            "images": len(image_specs),
            "detected_boxes": len(detection_rows),
            "mean_boxes_per_image": statistics.mean(
                [sum(1 for row in detection_rows if str(row["image_id"]) == spec["image_id"]) for spec in image_specs]
            )
            if image_specs
            else 0.0,
            "median_boxes_per_image": statistics.median(
                [sum(1 for row in detection_rows if str(row["image_id"]) == spec["image_id"]) for spec in image_specs]
            )
            if image_specs
            else 0.0,
            "cache_reused": True,
        }
        for filename in (
            "text_detections.jsonl",
            "parseq_readings.jsonl",
            "ppocrv5_server_readings.jsonl",
            "trocr_large_readings.jsonl",
            "consensus_stats.json",
            "text_nodes.jsonl",
            "text_nodes_resolvable.jsonl",
            "resolvability_stats.json",
            "runtime_models.json",
        ):
            src = cache_intermediate_dir / filename
            if src.exists():
                shutil.copy2(src, intermediate_dir / filename)
        if cache_level == "verified":
            for filename in ("anchor_tags.jsonl", "grounded_anchors.jsonl", "verified_tuples.jsonl"):
                src = cache_intermediate_dir / filename
                if src.exists():
                    shutil.copy2(src, intermediate_dir / filename)
            if (cache_intermediate_dir / "verified_tuples.jsonl").exists():
                cached_verified_tuples = load_jsonl(cache_intermediate_dir / "verified_tuples.jsonl")
            if (cache_intermediate_dir / "anchor_tags.jsonl").exists():
                cached_anchor_tag_rows = load_jsonl(cache_intermediate_dir / "anchor_tags.jsonl")
            if (cache_intermediate_dir / "grounded_anchors.jsonl").exists():
                cached_grounded_anchor_rows = load_jsonl(cache_intermediate_dir / "grounded_anchors.jsonl")
            if cached_verified_tuples is not None:
                cached_tuple_debug = {
                    "word_tuples": len([row for row in cached_verified_tuples if row.get("answer_level") == "word"]),
                    "sign_tuples": len([row for row in cached_verified_tuples if row.get("answer_level") == "sign"]),
                    "dropped_no_anchor": None,
                    "cache_reused": True,
                }
    else:
        if tuning.ocr_frontend == "nemotron_v2":
            print(f"[stage:pipeline] start nemotron_ocr images={len(image_specs)}", flush=True)
            detections_by_image, detection_rows, detection_summary, text_nodes, consensus_stats = run_nemotron_ocr_stage(
                image_specs=image_specs,
                image_source_map=image_source_map,
                max_detections=max_detections,
            )
            write_jsonl(intermediate_dir / "text_detections.jsonl", detection_rows)
            write_jsonl(intermediate_dir / "text_nodes.jsonl", text_nodes)
            write_json(intermediate_dir / "consensus_stats.json", consensus_stats)
        else:
            print(f"[stage:pipeline] start classic_ocr images={len(image_specs)}", flush=True)
            detections_by_image, detection_rows, detection_summary = run_detection_stage(
                image_specs=image_specs,
                device=runtime_device,
                max_detections=max_detections,
            )
            write_jsonl(intermediate_dir / "text_detections.jsonl", detection_rows)

            parseq_rows = run_recognition_stage(
                image_specs=image_specs,
                detections_by_image=detections_by_image,
                recognizer=PARSeqRecognizer(device=runtime_device),
                model_key="parseq",
            )
            write_jsonl(intermediate_dir / "parseq_readings.jsonl", parseq_rows)

            ppocr_server_rows = run_recognition_stage(
                image_specs=image_specs,
                detections_by_image=detections_by_image,
                recognizer=PaddleOCRRecognizer("PP-OCRv5_server_rec", device=runtime_device),
                model_key="ppocrv5_server",
            )
            write_jsonl(intermediate_dir / "ppocrv5_server_readings.jsonl", ppocr_server_rows)

            trocr_large_rows = run_recognition_stage(
                image_specs=image_specs,
                detections_by_image=detections_by_image,
                recognizer=TrOCRRecognizer("microsoft/trocr-large-printed", device=runtime_device),
                model_key="trocr_large",
            )
            write_jsonl(intermediate_dir / "trocr_large_readings.jsonl", trocr_large_rows)
            if runtime_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            text_nodes, consensus_stats = build_consensus_nodes(
                image_specs=image_specs,
                image_source_map=image_source_map,
                detections_by_image=detections_by_image,
                reading_rows=parseq_rows + ppocr_server_rows + trocr_large_rows,
            )
            write_jsonl(intermediate_dir / "text_nodes.jsonl", text_nodes)
            write_json(intermediate_dir / "consensus_stats.json", consensus_stats)

    write_json(intermediate_dir / "runtime_models.json", runtime_models)

    if tuning.ocr_frontend != "nemotron_v2":
        text_nodes = recompute_text_node_resolvability(text_nodes)
    write_jsonl(intermediate_dir / "text_nodes.jsonl", text_nodes)

    resolvable_nodes = [node for node in text_nodes if node["resolvable"]]
    print(
        f"[stage:pipeline] resolvability text_nodes={len(text_nodes)} resolvable_nodes={len(resolvable_nodes)} elapsed_s={time.time() - pipeline_start:.1f}",
        flush=True,
    )
    resolvability_stats = build_resolvability_stats(text_nodes)
    write_jsonl(intermediate_dir / "text_nodes_resolvable.jsonl", resolvable_nodes)
    write_json(intermediate_dir / "resolvability_stats.json", resolvability_stats)

    if cached_verified_tuples is not None and cached_anchor_tag_rows is not None and cached_grounded_anchor_rows is not None:
        anchor_tag_rows = cached_anchor_tag_rows
        grounded_anchor_rows = cached_grounded_anchor_rows
        verified_tuples = cached_verified_tuples
        tuple_debug = cached_tuple_debug or {}
        best_anchor_by_node = {
            (str(row["image_id"]), str(row["node_id"])): dict(row["top_candidates"][0])
            for row in grounded_anchor_rows
            if row.get("top_candidates")
        }
    else:
        print(
            f"[stage:pipeline] start anchor_stage resolvable_nodes={len(resolvable_nodes)}",
            flush=True,
        )
        anchor_tag_rows, grounded_anchor_rows, best_anchor_by_node = run_anchor_stage(
            image_specs=image_specs,
            resolvable_nodes=resolvable_nodes,
            device=runtime_device,
            max_tags_per_image=max_tags_per_image,
            grounding_threshold=grounding_threshold,
        )
        write_jsonl(intermediate_dir / "anchor_tags.jsonl", anchor_tag_rows)
        write_jsonl(intermediate_dir / "grounded_anchors.jsonl", grounded_anchor_rows)
        if runtime_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[stage:pipeline] start verified_tuple_build grounded_anchor_rows={len(grounded_anchor_rows)}",
            flush=True,
        )
        verified_tuples, tuple_debug = build_verified_tuples(
            image_specs=image_specs,
            all_text_nodes=text_nodes,
            resolvable_nodes=resolvable_nodes,
            best_anchor_by_node=best_anchor_by_node,
            grounded_anchor_rows=grounded_anchor_rows,
            image_source_map=image_source_map,
        )
    relabel_model = _anchor_relabel_model_name(tuning.anchor_relabel_mode)
    if relabel_model:
        print(
            f"[stage:pipeline] start anchor_relabel verified_tuples={len(verified_tuples)} model={relabel_model}",
            flush=True,
        )
        verified_tuples = refine_anchor_labels(verified_tuples, model_name=relabel_model)
    write_jsonl(intermediate_dir / "verified_tuples.jsonl", verified_tuples)

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    image_batches: list[dict[str, Any]] = []
    question_type_counts = Counter()
    selected_question_type_counts = Counter()
    per_image_tuples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in verified_tuples:
        per_image_tuples[str(row["image_id"])].append(row)
    for spec in image_specs:
        image_tuples = per_image_tuples.get(spec["image_id"], [])
        image_candidates: list[dict[str, Any]] = []
        for tuple_row in image_tuples:
            tuple_candidates = build_question_candidates(tuple_row, image_tuples)
            image_candidates.extend(tuple_candidates)
            question_type_counts.update(candidate["question_type"] for candidate in tuple_candidates)
        candidate_rows.extend(image_candidates)
        selected = select_candidates(image_candidates, target_count=target_per_image)
        selected = enforce_type_constraints(selected, image_candidates, target=target_per_image)
        selected_rows.extend(selected)
        selected_question_type_counts.update(candidate["question_type"] for candidate in selected)
        image_batches.append(
            {
                "image_id": spec["image_id"],
                "image_path": spec["image_path"],
                "selected_candidates": [{**candidate, "candidate_index": idx} for idx, candidate in enumerate(selected, start=1)],
            }
        )
    print(
        f"[stage:pipeline] candidate_selection verified_tuples={len(verified_tuples)} candidate_tuples={len(candidate_rows)} selected_tuples={len(selected_rows)} elapsed_s={time.time() - pipeline_start:.1f}",
        flush=True,
    )

    write_jsonl(intermediate_dir / "candidate_tuples.jsonl", candidate_rows)
    write_jsonl(intermediate_dir / "selected_tuples.jsonl", selected_rows)

    print(
        f"[stage:pipeline] start teacher_generation image_batches={len(image_batches)} selected_tuples={len(selected_rows)}",
        flush=True,
    )
    teacher_results = run_teacher_batches(
        image_batches=image_batches,
        model=model,
        max_side=max_side,
        workers=workers,
    )
    write_jsonl(out_dir / "raw_qa.jsonl", teacher_results["batch_rows"])

    raw_results = []
    final_rows = []
    failure_counts = Counter()
    for row in teacher_results["sample_rows"]:
        raw_results.append(row)
        if row["ok"] and row["summary"]["accepted_count"] == 1:
            final_rows.append(row_to_final_sample(row, model=model, prompt_variant=SEMANTIC_PROMPT_VARIANT))
        else:
            failure_counts[row.get("failure_reason") or "validation_failed"] += 1

    inline_frontier_summary = annotate_inline_frontier(
        final_rows,
        model=str(load_semantic_dev40_tuning().inline_frontier_model),
        max_side=max_side,
        workers=workers,
    )

    # --- Inline frontier gate ---
    # Reject rows where the frontier model (shown the image) got the answer wrong.
    # This is a verification gate using already-computed data — no extra API cost.
    # "Tiny natural error": some genuinely hard questions may fail; accepted here as inherent noise.
    #
    # Error-passthrough: if the frontier eval API call failed (e.g. rate-limited 429), the row is
    # treated as "pass" — only rows with a valid model judgment of "wrong" are rejected.
    # Without this, rate limiting causes 100% rejection (all errors → correct=False → all rejected).
    if tuning.inline_frontier_gate_enabled and final_rows:
        pre_gate = len(final_rows)
        wf1_floor = float(tuning.inline_frontier_gate_word_f1_floor)

        # Determine which question types the gate applies to.
        # "all" (default) applies the gate to every row; a comma-separated list restricts it.
        _gate_types_raw = str(tuning.inline_frontier_gate_question_types or "all").strip()
        if _gate_types_raw.lower() == "all":
            _gate_question_types: frozenset[str] | None = None
        else:
            _gate_question_types = frozenset(t.strip().upper() for t in _gate_types_raw.split(",") if t.strip())

        def _frontier_eval_errored(row: dict) -> bool:
            return bool((row.get("inline_frontier") or {}).get("error"))

        def _gated(row: dict) -> bool:
            """True if this row's question type is subject to the frontier gate."""
            if _gate_question_types is None:
                return True
            return str(row.get("question_type") or "").upper() in _gate_question_types

        if wf1_floor < 0.0:
            # Standard binary gate: keep rows the frontier model answered correctly, or where eval
            # errored, or where the question type is excluded from the gate.
            final_rows = [
                row for row in final_rows
                if not _gated(row) or row.get("inline_frontier_correct") is True or _frontier_eval_errored(row)
            ]
        else:
            # Lenient gate: also accept rows where word-F1 meets the floor, or where eval errored.
            # word_f1 is non-zero only for REVERSE_GROUND; all other types use soft_correct.
            def _passes_lenient_gate(row: dict) -> bool:
                if not _gated(row):
                    return True
                if row.get("inline_frontier_correct") is True or _frontier_eval_errored(row):
                    return True
                word_f1 = float(
                    (row.get("inline_frontier") or {}).get("score", {}).get("word_f1") or 0.0
                )
                return word_f1 >= wf1_floor

            final_rows = [row for row in final_rows if _passes_lenient_gate(row)]

        frontier_gate_errored = sum(1 for row in final_rows if _gated(row) and _frontier_eval_errored(row))
        frontier_gate_rejected = pre_gate - len(final_rows)
        frontier_gate_type_skipped = sum(1 for row in final_rows if not _gated(row))
        failure_counts["inline_frontier_gate_rejected"] = frontier_gate_rejected
        print(
            f"[inline_frontier_gate] pre={pre_gate} accepted={len(final_rows)} rejected={frontier_gate_rejected}"
            f" errored_passthrough={frontier_gate_errored} type_skipped={frontier_gate_type_skipped}"
            f" wf1_floor={wf1_floor:.2f}",
            flush=True,
        )

    # --- Vision dependence gate ---
    # Run text-only eval on each row; reject rows where the answer is derivable without the image.
    # One extra Gemini Flash call per row — the empirical verification that vision is actually required.
    if tuning.vision_dependence_gate_enabled and final_rows:
        from .dev200_eval import apply_vision_dependence_gate
        pre_gate = len(final_rows)
        final_rows, vdep_stats = apply_vision_dependence_gate(
            final_rows,
            model=str(tuning.inline_frontier_model),
            workers=workers,
        )
        failure_counts["vision_dependence_gate_rejected"] = vdep_stats["rejected"]
        print(f"[vision_dependence_gate] {vdep_stats}", flush=True)

    # --- RG vision-dependence check ---
    # Single cross-model text-only call (OpenAI) per REVERSE_GROUND row.
    # Non-RG rows pass through unconditionally. Separate model family from the Gemini teacher
    # avoids self-selection bias. Error → conservative keep.
    # When rg_leakage_correction_enabled: rejected rows where the only leakage is a color/shape
    # token in the question are corrected (token stripped) and re-checked before final discard.
    if tuning.rg_vdep_check_enabled and final_rows:
        from .dev200_eval import apply_rg_vdep_check
        final_rows, rg_vdep_stats = apply_rg_vdep_check(
            final_rows,
            model=str(tuning.rg_vdep_model),
            workers=workers,
            correction_enabled=tuning.rg_leakage_correction_enabled,
        )
        failure_counts["rg_vdep_rejected"] = rg_vdep_stats["rejected_rg"]
        print(f"[rg_vdep_check] {rg_vdep_stats}", flush=True)

    # --- RG leaky-label hard reject ---
    # Structurally reject REVERSE_GROUND rows where the anchor_label contains a color
    # or shape token. These rows expose the visual element's identity in the question
    # text, making them answerable without the image. This is a zero-cost structural
    # filter — no API calls — that directly targets the "leaky-label candidates" flagged
    # in ita08 diagnostics (11–14 per variant, all non-corrected by the broken vdep check).
    if tuning.rg_leaky_label_hard_reject_enabled and final_rows:
        _rg_color_tokens: frozenset[str] = frozenset({
            "red", "blue", "green", "brown", "white", "black", "gray", "grey",
            "yellow", "orange", "purple", "pink", "silver", "gold",
        })
        _rg_shape_tokens: frozenset[str] = frozenset({
            "rectangular", "circular", "square", "oval", "round",
            "triangular", "hexagonal", "cylindrical", "spherical",
            "wedge", "segment", "emblem", "badge", "bar",
        })
        pre_rg_reject = len(final_rows)

        def _has_leaky_label(row: dict) -> bool:
            if str(row.get("question_type") or "") != "REVERSE_GROUND":
                return False
            label_words = str(row.get("anchor_label") or "").lower().split()
            return any(w in _rg_color_tokens or w in _rg_shape_tokens for w in label_words)

        final_rows = [row for row in final_rows if not _has_leaky_label(row)]
        rg_leaky_rejected = pre_rg_reject - len(final_rows)
        failure_counts["rg_leaky_label_rejected"] = rg_leaky_rejected
        print(
            f"[rg_leaky_label_hard_reject] pre={pre_rg_reject} rejected={rg_leaky_rejected}"
            f" remaining={len(final_rows)}",
            flush=True,
        )

    print(
        f"[stage:pipeline] finalize raw_results={len(raw_results)} final_rows={len(final_rows)} elapsed_s={time.time() - pipeline_start:.1f}",
        flush=True,
    )

    write_jsonl(out_dir / "raw_results.jsonl", raw_results)
    write_jsonl(out_dir / "ocr_qa_dataset.jsonl", final_rows)
    write_jsonl(out_dir / "accepted_dataset.jsonl", final_rows)

    stage_counts = {
        "images": len(image_specs),
        "detected_boxes": len(detection_rows),
        "text_nodes": len(text_nodes),
        "resolvable_nodes": len(resolvable_nodes),
        "anchor_tag_rows": len(anchor_tag_rows),
        "grounded_nodes": len(best_anchor_by_node),
        "grounded_anchors": len(grounded_anchor_rows),
        "verified_tuples": len(verified_tuples),
        "candidate_tuples": len(candidate_rows),
        "selected_tuples": len(selected_rows),
        "raw_qa_rows": len(raw_results),
        "final_qa_rows": len(final_rows),
    }
    summary = {
        "experiment": {
            "name": out_dir.name,
            "provider": "gemini",
            "model": model,
            "prompt_variant": SEMANTIC_PROMPT_VARIANT,
            "source_experiment": source_experiment_dir.name,
            "variant": SEMANTIC_PROMPT_VARIANT,
            "workers": workers,
            "max_side": max_side,
            "target_per_image": target_per_image,
            "max_detections": max_detections,
            "grounding_threshold": grounding_threshold,
            "max_tags_per_image": max_tags_per_image,
            "device": runtime_device,
            "runtime_models": runtime_models,
            "tuning": tuning.to_metadata(),
            "cache_level": cache_level,
        },
        "same_image_universe_count": len(image_specs),
        "input_tuple_count": len(verified_tuples),
        "resolvable_tuple_count": len(verified_tuples),
        "dropped_tuple_count": 0,
        "generated_qas": len(raw_results),
        "accepted_qas": len(final_rows),
        "qa_accept_rate": (len(final_rows) / len(raw_results)) if raw_results else 0.0,
        "stage_counts": stage_counts,
        "question_type_counts": dict(question_type_counts),
        "selected_question_type_counts": dict(selected_question_type_counts),
        "disabled_question_types": [],
        "failure_counts": dict(failure_counts),
        "consensus_stats": consensus_stats,
        "resolvability_stats": resolvability_stats,
        "tuple_debug": tuple_debug,
        "detection_summary": detection_summary,
        "mean_question_words": statistics.mean(len(str(row["question"]).split()) for row in final_rows) if final_rows else 0.0,
        "images_with_final_rows": len({row["image_id"] for row in final_rows}),
        "inline_frontier": inline_frontier_summary,
    }
    quality_metrics = compute_run_quality(summary, final_rows)
    summary.update(
        {
            "accepted_rows": int(quality_metrics["accepted_qas"]),
            "inline_frontier_mean": float(quality_metrics["inline_frontier_mean"]),
            "inline_frontier_scored": int(quality_metrics["inline_frontier_scored"]),
            "precision_first_score": float(quality_metrics["precision_first_score"]),
            "sweep_score": float(quality_metrics["sweep_score"]),
            "q3": float(quality_metrics["q3_score"]),
            "quality_score": float(quality_metrics["quality_score"]),
        }
    )
    write_json(out_dir / "summary.json", summary)
    return summary


def resolve_device(device_arg: str) -> str:
    choice = str(device_arg or "auto").strip().lower()
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


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
    sam3_refiner = Sam3Refiner(device=device, confidence_threshold=tuning.sam3_confidence_threshold) if _sam3_prompt_limit() > 0 else None
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

            # --- Structural fallback for degenerate inventory results ---
            if tuning.qwen_structural_fallback_enabled:
                from .qwen_anchor_vllm import STRUCTURAL_FALLBACK_QWEN_PROMPT, STRUCTURAL_FALLBACK_QWEN_PROMPT_ANTI_OCR, STRUCTURAL_FALLBACK_QWEN_PROMPT_NO_TEXT_REF, STRUCTURAL_FALLBACK_QWEN_PROMPT_ITA15, is_degenerate_inventory
                _structural_fallback_prompt = (
                    GEMMA_STRUCTURAL_FALLBACK_PROMPT_ITA15
                    if use_gemma_anchor_backend and tuning.qwen_ita15_prompt_enabled
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
                # Also flag images entirely absent from results (Qwen returned nothing)
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

            # --- Second-pass for low-yield images ---
            if tuning.qwen_min_anchor_detections_per_image > 0:
                low_yield_ids = {
                    image_id
                    for image_id, rows in image_anchors_by_image.items()
                    if len(rows) < tuning.qwen_min_anchor_detections_per_image
                }
                # Also flag images with no results at all
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

    # --- Degenerate anchor label filter ---
    # Removes anchors whose label is too generic (e.g. 'object part', 'surface', 'area')
    # or looks like OCR-copied text content (e.g. 'DAN BROWN BREWING CO') before they can
    # become QA candidates. Prevents these anchors from ever reaching Gemini generation.
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

    # --- Text-reference anchor label filter ---
    # Removes anchors whose labels reference visible text content rather than describing
    # a visual object. Catches Gemma4 compound labels like "x-axis region date labels"
    # (trailing 'labels') or "text block black text" (text-content prefix).
    # These labels leak answer content through the anchor name, harming image-dependence.
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

    # --- Anchor label groundback check ---
    # Re-runs Qwen with only the anchor label text to verify the label uniquely identifies
    # the element in the image. Anchors where Qwen cannot relocate the element (low IoU
    # between the returned bbox and the original) receive a scoring penalty, making them
    # less likely to reach candidate selection. No hard filter — penalty-only approach.
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


def build_verified_tuples(
    *,
    image_specs: list[dict[str, str]],
    all_text_nodes: list[dict[str, Any]],
    resolvable_nodes: list[dict[str, Any]],
    best_anchor_by_node: dict[tuple[str, str], dict[str, Any]],
    grounded_anchor_rows: list[dict[str, Any]],
    image_source_map: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    nodes_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_nodes_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    node_groundings: dict[tuple[str, str], dict[str, Any]] = {}
    for node in resolvable_nodes:
        nodes_by_image[str(node["image_id"])].append(node)
    for node in all_text_nodes:
        all_nodes_by_image[str(node["image_id"])].append(node)
    for row in grounded_anchor_rows:
        node_groundings[(str(row["image_id"]), str(row["node_id"]))] = row

    tuples: list[dict[str, Any]] = []
    dropped_no_anchor = 0
    word_tuple_count = 0
    merged_sign_tuple_count = 0
    subsumed_tuple_count = 0
    verify_progress = _make_stage_progress_logger("verified_tuple_build", len(image_specs))

    for spec_index, spec in enumerate(image_specs, start=1):
        image_nodes = nodes_by_image.get(spec["image_id"], [])
        all_image_nodes = all_nodes_by_image.get(spec["image_id"], [])
        if not image_nodes:
            verify_progress.tick(spec_index, extra=f"image_id={spec['image_id']} skipped=no_nodes")
            continue
        image_width = int(image_nodes[0]["image_width"])
        image_height = int(image_nodes[0]["image_height"])
        image_anchors = dedupe_anchor_rows([candidate for row in grounded_anchor_rows if row["image_id"] == spec["image_id"] for candidate in row["top_candidates"]])
        image_tuples: list[dict[str, Any]] = []
        for node in image_nodes:
            best_anchor = best_anchor_by_node.get((spec["image_id"], str(node["node_id"])))
            if best_anchor is None:
                dropped_no_anchor += 1
                continue
            bbox_xywh = bbox_xyxy_to_xywh(list(node["bbox"]))
            area_fraction = ((node["bbox"][2] - node["bbox"][0]) * (node["bbox"][3] - node["bbox"][1])) / max(image_width * image_height, 1)
            coarse_phrase = REGION_PHRASES.get(str(node["region_key"]), "area of the image")
            location_meta = local_text_location_metadata(
                primary_box=list(node["bbox"]),
                primary_node_ids=[str(node["node_id"])],
                all_text_nodes=all_image_nodes,
                image_size=(image_width, image_height),
                coarse_phrase=coarse_phrase,
            )
            anchor_local = anchor_local_location_metadata(list(node["bbox"]), list(best_anchor["box"]), str(best_anchor["relation"]), str(best_anchor["label"]))
            kd_metadata = collect_semantic_kd_metadata(
                primary_box=list(node["bbox"]),
                primary_polygon=list(node["polygon"]),
                primary_node_ids=[str(node["node_id"])],
                all_text_nodes=all_image_nodes,
                all_anchors=image_anchors,
                image_size=(image_width, image_height),
            )
            kd_metadata["anchor_conflict_count"] = int(best_anchor.get("label_conflict_count") or 0)
            kd_metadata["anchor_cluster_size"] = int(best_anchor.get("cluster_size") or 1)
            kd_metadata["anchor_alternate_labels"] = list(best_anchor.get("alternate_labels") or [])
            semantic_debug = {
                "caption": best_anchor.get("caption"),
                "discovered_tags": list(best_anchor.get("discovered_tags") or []),
                "final_prompt_tags": list(best_anchor.get("final_prompt_tags") or best_anchor.get("discovered_tags") or []),
                "raw_anchor_label": best_anchor.get("raw_label"),
                "top_candidates": list(node_groundings[(spec["image_id"], str(node["node_id"]))]["top_candidates"]),
                "best_anchor_relevance": float(best_anchor["relevance"]),
                "best_anchor_selection_score": float(best_anchor.get("selection_score") or best_anchor["relevance"]),
                "anchor_support_count": int(best_anchor.get("support_count") or 1),
                "anchor_source_support_count": int(best_anchor.get("source_support_count") or 1),
                "anchor_support_labels": list(best_anchor.get("support_labels") or []),
                "anchor_alternate_labels": list(best_anchor.get("alternate_labels") or []),
                "anchor_conflict_count": int(best_anchor.get("label_conflict_count") or 0),
                "anchor_cluster_size": int(best_anchor.get("cluster_size") or 1),
                "location_detail": str(location_meta["layout_detail"]),
            }
            image_tuples.append(
                {
                    "tuple_id": f"{spec['image_id']}::word::{node['node_id']}",
                    "image_id": spec["image_id"],
                    "image_path": spec["image_path"],
                    "image_width": image_width,
                    "image_height": image_height,
                    "ann_id": str(node["node_id"]),
                    "answer": str(node["text"]),
                    "answer_normalized": normalize_answer(str(node["text"])),
                    "text_polygon": list(node["polygon"]),
                    "text_bbox": [round(v, 2) for v in bbox_xywh],
                    "text_node_ids": [str(node["node_id"])],
                    "child_words": [str(node["text"])],
                    "anchor_label": str(best_anchor["label"]),
                    "anchor_synonyms": list(dict.fromkeys([str(best_anchor["label"]), *(str(label) for label in best_anchor.get("alternate_labels") or [])])),
                    "location_phrase": coarse_phrase,
                    "location_synonyms": list(REGION_SYNONYMS.get(str(node["region_key"]), (REGION_PHRASES.get(str(node["region_key"]), "area of the image"),))),
                    "specific_location_phrase": str(location_meta["specific_location_phrase"]),
                    "specific_location_synonyms": list(location_meta["specific_location_synonyms"]),
                    "anchor_local_phrase": str(anchor_local["phrase"]),
                    "anchor_local_synonyms": list(anchor_local["synonyms"]),
                    "anchor_local_clean": bool(anchor_local["clean"]),
                    "anchor_local_mode": str(anchor_local["mode"]),
                    "anchor_box": [round(float(v), 2) for v in best_anchor["box"]],
                    "anchor_score": float(best_anchor["score"]),
                    "anchor_color": str(extract_anchor_color(str(best_anchor["label"])) or ""),
                    "anchor_category": categorize_anchor(str(best_anchor["label"])),
                    "anchor_source": str(best_anchor.get("source") or "grounding_dino"),
                    "anchor_label_source": "grounding",
                    "relation": str(best_anchor["relation"]),
                    "ref_label": None,
                    "ref_box": None,
                    "unique": True,
                    "answer_level": "word",
                    "ocr_confidence": float(node["confidence"]),
                    "consensus_tier": str(node["consensus_tier"]),
                    "dataset_source": image_source_map.get(spec["image_id"], "textocr_val"),
                    "region_key": str(node["region_key"]),
                    "resolvable": True,
                    "resolvability": dict(node["resolvability"]),
                    "kd_metadata": kd_metadata,
                    "valid_text_count": len(all_image_nodes),
                    "area_fraction": float(area_fraction),
                    "text_length": len(str(node["text"])),
                    "density_bucket": density_bucket(len(all_image_nodes)),
                    "area_bucket": area_bucket(float(area_fraction)),
                    "group_kind": "word",
                    "semantic_debug": semantic_debug,
                }
            )
        word_tuple_count += len(image_tuples)
        image_tuples.extend(build_sign_tuples_for_image(spec, image_nodes, all_image_nodes, image_tuples, best_anchor_by_node, image_anchors, image_source_map))
        merged_rows, merge_debug = build_merged_sign_tuples_for_image(
            spec,
            image_nodes,
            all_image_nodes,
            best_anchor_by_node,
            image_anchors,
            image_source_map,
        )
        image_tuples.extend(merged_rows)
        merged_sign_tuple_count += int(merge_debug.get("merge_groups") or 0)
        image_tuples, removed = _filter_subsumed_rows(image_tuples)
        subsumed_tuple_count += removed
        apply_uniqueness(image_tuples)
        fill_competing_tuple_counts(image_tuples)
        tuples.extend(image_tuples)
        verify_progress.tick(
            spec_index,
            extra=(
                f"image_id={spec['image_id']} image_tuples={len(image_tuples)} "
                f"cum_tuples={len(tuples)}"
            ),
        )

    debug = {
        "word_tuples": word_tuple_count,
        "sign_tuples": len([row for row in tuples if row["answer_level"] == "sign"]),
        "merged_sign_tuples": merged_sign_tuple_count,
        "subsumed_tuples": subsumed_tuple_count,
        "dropped_no_anchor": dropped_no_anchor,
    }
    verify_progress.finish(extra=f"tuples={len(tuples)} dropped_no_anchor={dropped_no_anchor}")
    return tuples, debug


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


def build_sign_tuples_for_image(
    spec: dict[str, str],
    image_nodes: list[dict[str, Any]],
    all_image_nodes: list[dict[str, Any]],
    word_tuples: list[dict[str, Any]],
    best_anchor_by_node: dict[tuple[str, str], dict[str, Any]],
    image_anchors: list[dict[str, Any]],
    image_source_map: dict[str, str],
) -> list[dict[str, Any]]:
    image_width = int(image_nodes[0]["image_width"])
    image_height = int(image_nodes[0]["image_height"])
    groups: dict[tuple[str, tuple[float, ...]], list[dict[str, Any]]] = defaultdict(list)
    anchor_meta: dict[tuple[str, tuple[float, ...]], dict[str, Any]] = {}
    for node in image_nodes:
        best_anchor = best_anchor_by_node.get((spec["image_id"], str(node["node_id"])))
        if not best_anchor or str(best_anchor["relation"]) != "on":
            continue
        key = (str(best_anchor["label"]), tuple(round(float(value), 1) for value in best_anchor["box"]))
        groups[key].append(node)
        anchor_meta[key] = best_anchor

    out: list[dict[str, Any]] = []
    seen_children: set[tuple[str, ...]] = set()
    for key, nodes in groups.items():
        if len(nodes) < 2:
            continue
        ordered = sorted(nodes, key=lambda node: (statistics.mean(node["polygon"][1::2]), statistics.mean(node["polygon"][0::2])))
        node_ids = tuple(str(node["node_id"]) for node in ordered)
        if node_ids in seen_children:
            continue
        seen_children.add(node_ids)
        answer = " ".join(str(node["text"]).strip() for node in ordered if str(node["text"]).strip())
        if not answer or not is_valid_text(answer):
            continue
        union_xyxy = union_bbox([list(node["bbox"]) for node in ordered])
        polygon = bbox_to_polygon(union_xyxy)
        resolvability = compute_resolvability(polygon, image_width, image_height)
        if not resolvability["passes"]:
            continue
        anchor = anchor_meta[key]
        area_fraction = ((union_xyxy[2] - union_xyxy[0]) * (union_xyxy[3] - union_xyxy[1])) / max(image_width * image_height, 1)
        bbox_xywh = bbox_xyxy_to_xywh(union_xyxy)
        coarse_phrase = REGION_PHRASES.get(region_key_for_bbox(bbox_xywh, image_width, image_height), "area of the image")
        pseudo_node_ids = list(node_ids)
        location_meta = local_text_location_metadata(
            primary_box=union_xyxy,
            primary_node_ids=pseudo_node_ids,
            all_text_nodes=all_image_nodes,
            image_size=(image_width, image_height),
            coarse_phrase=coarse_phrase,
        )
        anchor_local = anchor_local_location_metadata(union_xyxy, list(anchor["box"]), "on", str(anchor["label"]))
        kd_metadata = collect_semantic_kd_metadata(
            primary_box=union_xyxy,
            primary_polygon=polygon,
            primary_node_ids=pseudo_node_ids,
            all_text_nodes=all_image_nodes,
            all_anchors=image_anchors,
            image_size=(image_width, image_height),
        )
        kd_metadata["anchor_conflict_count"] = int(anchor.get("label_conflict_count") or 0)
        kd_metadata["anchor_cluster_size"] = int(anchor.get("cluster_size") or 1)
        kd_metadata["anchor_alternate_labels"] = list(anchor.get("alternate_labels") or [])
        out.append(
            {
                "tuple_id": f"{spec['image_id']}::sign::{'_'.join(node_ids)}",
                "image_id": spec["image_id"],
                "image_path": spec["image_path"],
                "image_width": image_width,
                "image_height": image_height,
                "ann_id": f"group_{'_'.join(node_ids)}",
                "answer": answer,
                "answer_normalized": normalize_answer(answer),
                "text_polygon": polygon,
                "text_bbox": [round(v, 2) for v in bbox_xywh],
                "text_node_ids": pseudo_node_ids,
                "child_words": [str(node["text"]) for node in ordered],
                "anchor_label": str(anchor["label"]),
                "anchor_synonyms": list(dict.fromkeys([str(anchor["label"]), *(str(label) for label in anchor.get("alternate_labels") or [])])),
                "location_phrase": coarse_phrase,
                "location_synonyms": list(
                    REGION_SYNONYMS.get(
                        region_key_for_bbox(bbox_xywh, image_width, image_height),
                        (REGION_PHRASES.get(region_key_for_bbox(bbox_xywh, image_width, image_height), "area of the image"),),
                    )
                ),
                "specific_location_phrase": str(location_meta["specific_location_phrase"]),
                "specific_location_synonyms": list(location_meta["specific_location_synonyms"]),
                "anchor_local_phrase": str(anchor_local["phrase"]),
                "anchor_local_synonyms": list(anchor_local["synonyms"]),
                "anchor_local_clean": bool(anchor_local["clean"]),
                "anchor_local_mode": str(anchor_local["mode"]),
                "anchor_box": [round(float(v), 2) for v in anchor["box"]],
                "anchor_score": float(anchor["score"]),
                "anchor_color": str(extract_anchor_color(str(anchor["label"])) or ""),
                "anchor_category": categorize_anchor(str(anchor["label"])),
                "anchor_source": str(anchor.get("source") or "grounding_dino"),
                "relation": "on",
                "ref_label": None,
                "ref_box": None,
                "unique": True,
                "answer_level": "sign",
                "ocr_confidence": min(float(node["confidence"]) for node in ordered),
                "consensus_tier": "semantic_group",
                "dataset_source": image_source_map.get(spec["image_id"], "textocr_val"),
                "region_key": region_key_for_bbox(bbox_xywh, image_width, image_height),
                "resolvable": True,
                "resolvability": resolvability,
                "kd_metadata": kd_metadata,
                "valid_text_count": len(all_image_nodes),
                "area_fraction": float(area_fraction),
                "text_length": len(answer),
                "density_bucket": density_bucket(len(all_image_nodes)),
                "area_bucket": area_bucket(float(area_fraction)),
                "group_kind": "semantic_anchor_group",
                "semantic_debug": {
                    "caption": anchor.get("caption"),
                    "discovered_tags": list(anchor.get("discovered_tags") or []),
                    "final_prompt_tags": list(anchor.get("final_prompt_tags") or anchor.get("discovered_tags") or []),
                    "raw_anchor_label": anchor.get("raw_label"),
                    "best_anchor_relevance": float(anchor["relevance"]),
                    "best_anchor_selection_score": float(anchor.get("selection_score") or anchor["relevance"]),
                    "anchor_support_count": int(anchor.get("support_count") or 1),
                    "anchor_source_support_count": int(anchor.get("source_support_count") or 1),
                    "anchor_support_labels": list(anchor.get("support_labels") or []),
                    "anchor_alternate_labels": list(anchor.get("alternate_labels") or []),
                    "anchor_conflict_count": int(anchor.get("label_conflict_count") or 0),
                    "anchor_cluster_size": int(anchor.get("cluster_size") or 1),
                    "top_candidates": [anchor],
                    "location_detail": str(location_meta["layout_detail"]),
                },
                "anchor_label_source": "grounding",
            }
        )
    return out


def fill_competing_tuple_counts(tuple_rows: list[dict[str, Any]]) -> None:
    for row in tuple_rows:
        row_kd = row["kd_metadata"]
        row_anchor_box = [float(value) for value in row.get("anchor_box") or []]
        row_relation = str(row.get("relation") or "")
        row_anchor_label = normalize_answer(str(row.get("anchor_label") or ""))
        row_answer = normalize_answer(str(row.get("answer_normalized") or row.get("answer") or ""))
        row_coarse = normalize_answer(str(row.get("location_phrase") or ""))
        row_specific = normalize_answer(str(row.get("specific_location_phrase") or ""))
        row_bucket = str(row_kd.get("local_text_bucket_key") or "")
        anchor_label_competitors = 0
        anchor_overlap_competitors = 0
        coarse_region_competitors = 0
        bucket_competitors = 0
        competing = 0
        same_anchor_same_answer_count = 1 if row_answer else 0
        same_anchor_same_answer_nonoverlap_instances = 1 if row_answer else 0
        same_anchor_distinct_answers: set[str] = {row_answer} if row_answer else set()
        for other in tuple_rows:
            if other["tuple_id"] == row["tuple_id"]:
                continue
            other_kd = other["kd_metadata"]
            same_relation = str(other.get("relation") or "") == row_relation
            same_anchor_label = normalize_answer(str(other.get("anchor_label") or "")) == row_anchor_label
            same_coarse = bool(row_coarse and normalize_answer(str(other.get("location_phrase") or "")) == row_coarse)
            other_specific = normalize_answer(str(other.get("specific_location_phrase") or ""))
            same_specific = bool(row_specific and other_specific and row_specific == other_specific)
            same_bucket = bool(row_bucket and row_bucket == str(other_kd.get("local_text_bucket_key") or ""))
            other_answer = normalize_answer(str(other.get("answer_normalized") or other.get("answer") or ""))
            other_anchor_box = [float(value) for value in other.get("anchor_box") or []]
            overlap = _bbox_iou(row_anchor_box, other_anchor_box) if len(row_anchor_box) >= 4 and len(other_anchor_box) >= 4 else 0.0
            if same_relation and same_anchor_label:
                anchor_label_competitors += 1
                if other_answer:
                    same_anchor_distinct_answers.add(other_answer)
                if row_answer and other_answer == row_answer:
                    same_anchor_same_answer_count += 1
                    if overlap < 0.35:
                        same_anchor_same_answer_nonoverlap_instances += 1
            if same_relation and overlap >= 0.45:
                anchor_overlap_competitors += 1
            if same_relation and same_coarse:
                coarse_region_competitors += 1
            if same_relation and same_bucket:
                bucket_competitors += 1
            confusable = same_relation and (
                (same_anchor_label and (same_coarse or same_specific or same_bucket))
                or overlap >= 0.60
                or (same_coarse and same_bucket)
            )
            if confusable:
                competing += 1
        row_kd["anchor_label_competitors"] = anchor_label_competitors
        row_kd["anchor_overlap_competitors"] = anchor_overlap_competitors
        row_kd["coarse_region_competitors"] = coarse_region_competitors
        row_kd["bucket_competitors"] = bucket_competitors
        row_kd["competing_tuples"] = competing
        row_kd["same_anchor_text_count"] = anchor_overlap_competitors + 1
        row_kd["same_anchor_same_answer_count"] = same_anchor_same_answer_count
        row_kd["same_anchor_same_answer_nonoverlap_instances"] = same_anchor_same_answer_nonoverlap_instances
        row_kd["same_anchor_distinct_answer_count"] = len(same_anchor_distinct_answers)
        row_kd["diffuse_same_anchor_scene"] = bool(anchor_label_competitors >= 2 and anchor_overlap_competitors <= 1)


def apply_uniqueness(tuple_rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row["answer_normalized"]) for row in tuple_rows)
    for row in tuple_rows:
        row["unique"] = counts[str(row["answer_normalized"])] == 1


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        lowered = str(item).strip().lower()
        if not lowered or lowered in seen:
            continue
        seen.add(lowered)
        out.append(lowered)
    return out


def dedupe_anchor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (-float(item.get("score") or 0.0), item["label"])):
        duplicate = False
        for existing in kept:
            if row["label"] != existing["label"]:
                continue
            if overlap_fraction(list(row["box"]), list(existing["box"])) >= 0.85 and overlap_fraction(list(existing["box"]), list(row["box"])) >= 0.85:
                duplicate = True
                break
        if not duplicate:
            kept.append(row)
    return kept


def _length_bucket(text: str) -> str:
    length = len(str(text).strip())
    if length <= 1:
        return "single_char"
    if length <= 4:
        return "short_2_4"
    if length <= 10:
        return "medium_5_10"
    return "long_11_plus"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def recompute_text_node_resolvability(text_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for node in text_nodes:
        refreshed = compute_resolvability(list(node["polygon"]), int(node["image_width"]), int(node["image_height"]))
        copied = dict(node)
        copied["resolvability"] = refreshed
        copied["resolvable"] = bool(refreshed["passes"])
        out.append(copied)
    return out
