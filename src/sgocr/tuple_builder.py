from __future__ import annotations

import os
import statistics
from collections import Counter, defaultdict
from typing import Any

from .bootstrap import REGION_PHRASES, REGION_SYNONYMS, area_bucket, density_bucket, is_valid_text, normalize_answer, region_key_for_bbox
from .bootstrap_kd import bbox_xywh_to_xyxy, bbox_xyxy_to_xywh, compute_resolvability, overlap_fraction
from .dev40_complete import union_bbox
from .ocr_runtime import bbox_to_polygon
from .semantic_dev40_tuning import load_semantic_dev40_tuning
from .semantic_grounding import (
    GENERIC_TEXT_ANCHORS,
    anchor_local_location_metadata,
    anchor_relevance,
    categorize_anchor,
    collect_semantic_kd_metadata,
    extract_anchor_color,
    local_text_location_metadata,
)


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


def _anchor_area_fraction(box: list[float], image_size: tuple[int, int]) -> float:
    area = max((float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1])), 1.0)
    return area / max(float(image_size[0]) * float(image_size[1]), 1.0)


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


def build_verified_tuples(
    *,
    image_specs: list[dict[str, str]],
    all_text_nodes: list[dict[str, Any]],
    resolvable_nodes: list[dict[str, Any]],
    best_anchor_by_node: dict[tuple[str, str], dict[str, Any]],
    grounded_anchor_rows: list[dict[str, Any]],
    image_source_map: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .pipeline_stages import _make_stage_progress_logger

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


def recompute_text_node_resolvability(text_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for node in text_nodes:
        refreshed = compute_resolvability(list(node["polygon"]), int(node["image_width"]), int(node["image_height"]))
        copied = dict(node)
        copied["resolvability"] = refreshed
        copied["resolvable"] = bool(refreshed["passes"])
        out.append(copied)
    return out
