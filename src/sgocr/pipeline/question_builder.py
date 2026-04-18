from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ..bootstrap import (
    REGION_PHRASES,
    REGION_SYNONYMS,
    area_bucket,
    density_bucket,
    normalize_answer,
    region_key_for_bbox,
    write_json,
    write_jsonl,
)
from ..bootstrap_kd import (
    bbox_xywh_to_xyxy,
    bbox_xyxy_to_xywh,
    build_resolvability_stats,
    center_from_box,
    centroid_from_polygon,
    collect_kd_metadata,
    compute_resolvability,
    load_bootstrap_text_nodes,
    overlap_fraction,
)
from ..semantic_grounding import extract_anchor_color
from ..semantic_dev40_tuning import load_semantic_dev40_tuning

from .dataset_assembly import (
    _anchor_for_region,
    _load_jsonl,
    _maybe_read_json,
    anchor_local_phrase_for_tuple,
    anchor_local_synonyms_for_tuple,
    bbox_to_polygon,
    box_height,
    group_node_id,
    is_blocked_answer,
    location_phrase_for_tuple,
    location_synonyms_for_tuple,
    overlap_1d,
    row_to_final_sample,
    sample_id_for_candidate,
    sanitize_id,
    specific_location_phrase_for_tuple,
    specific_location_synonyms_for_tuple,
    union_bbox,
)

QUESTION_TYPES = ("DIRECT_READ", "YES_NO", "REVERSE_GROUND", "TEXT_PROPERTY", "ANCHOR_PROPERTY")
DISABLED_QUESTION_TYPES: tuple[str, ...] = ()
URL_LIKE_RE = re.compile(r"(https?://|www\.|\.com\b|\.net\b|\.org\b|@)", re.IGNORECASE)
TEXT_APPEARANCE_TERMS = ("text", "word", "words", "letters", "writing", "label")
TEXT_PROPERTY_VISUAL_TYPES = ("text_color", "text_curvature")
TEXT_PROPERTY_TYPE_TERMS = {
    "word_count": ("how many words", "word count", "number of words"),
    "first_word": ("first word", "starts with"),
    "last_word": ("last word", "ends with"),
    "text_color": ("color", "colour", "letters", "text", "writing"),
    "text_orientation": ("orientation", "angle", "horizontal", "vertical", "slanted", "diagonal", "upright"),
    "text_curvature": ("curved", "curve", "arched", "arc", "straight", "bend", "shape"),
}
ANCHOR_PROPERTY_TYPE_TERMS = {
    "anchor_color": ("color", "colour"),
    "anchor_material": ("material", "made of", "metal", "wood", "plastic", "glass", "paper", "cardboard"),
    "anchor_shape": ("shape", "round", "square", "rectangular", "circular", "oval"),
}


def build_dev40_complete_dataset(
    *,
    source_experiment_dir: Path,
    raw_json_path: Path,
    out_dir: Path,
    intermediate_dir: Path,
    model: str = "gemini-2.5-flash",
    max_side: int = 768,
    workers: int = 4,
    target_per_image: int = 4,
) -> dict[str, Any]:
    source_rows = _load_jsonl(source_experiment_dir / "raw_results.jsonl")
    image_ids = sorted({str((row.get("tuple") or {}).get("image_id") or row.get("image_id")) for row in source_rows})
    image_paths = {
        str((row.get("tuple") or {}).get("image_id") or row.get("image_id")): str((row.get("tuple") or {}).get("image_path") or row.get("image_path"))
        for row in source_rows
        if str((row.get("tuple") or {}).get("image_id") or row.get("image_id"))
    }
    source_summary = _maybe_read_json(source_experiment_dir / "summary.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    image_index = load_bootstrap_text_nodes(raw_json_path, image_ids=set(image_ids))
    all_text_nodes = [node for payload in image_index.values() for node in payload["nodes"]]
    resolvability_stats = build_resolvability_stats(image_index)
    write_jsonl(intermediate_dir / "text_nodes.jsonl", all_text_nodes)
    write_jsonl(intermediate_dir / "text_nodes_resolvable.jsonl", [node for node in all_text_nodes if node["resolvable"]])
    write_json(intermediate_dir / "resolvability_stats.json", resolvability_stats)

    verified_tuples: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    image_batches: list[dict[str, Any]] = []
    question_type_counts = Counter()
    selected_question_type_counts = Counter()
    word_tuple_count = 0
    sign_tuple_count = 0

    for image_id in image_ids:
        payload = image_index.get(image_id)
        if not payload:
            continue

        image_width = int(payload["image_width"])
        image_height = int(payload["image_height"])
        image_nodes = list(payload["nodes"])
        resolvable_nodes = [node for node in image_nodes if node["resolvable"] and not is_blocked_answer(str(node["text"]))]
        if not resolvable_nodes:
            image_batches.append(
                {
                    "image_id": image_id,
                    "image_path": image_paths.get(image_id),
                    "candidate_count": 0,
                    "selected_count": 0,
                    "selected_candidates": [],
                    "kept_resolvable_nodes": 0,
                    "all_valid_nodes": len(image_nodes),
                }
            )
            continue

        word_tuples = [
            build_word_tuple(
                image_id=image_id,
                image_path=image_paths[image_id],
                image_width=image_width,
                image_height=image_height,
                node=node,
                image_nodes=image_nodes,
                image_anchors=payload["anchors"],
            )
            for node in resolvable_nodes
        ]
        sign_tuples = build_sign_tuples(
            image_id=image_id,
            image_path=image_paths[image_id],
            image_width=image_width,
            image_height=image_height,
            image_nodes=image_nodes,
            resolvable_nodes=resolvable_nodes,
            image_anchors=payload["anchors"],
        )
        image_tuples = word_tuples + sign_tuples
        apply_uniqueness(image_tuples)

        verified_tuples.extend(image_tuples)
        word_tuple_count += len(word_tuples)
        sign_tuple_count += len(sign_tuples)

        image_candidates = []
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
                "image_id": image_id,
                "image_path": image_paths.get(image_id),
                "candidate_count": len(image_candidates),
                "selected_count": len(selected),
                "selected_candidates": selected,
                "kept_resolvable_nodes": len(resolvable_nodes),
                "all_valid_nodes": len(image_nodes),
            }
        )

    write_jsonl(intermediate_dir / "verified_tuples.jsonl", verified_tuples)
    write_jsonl(intermediate_dir / "candidate_tuples.jsonl", candidate_rows)
    write_jsonl(intermediate_dir / "selected_tuples.jsonl", selected_rows)

    from .teacher_runtime import annotate_inline_frontier, run_teacher_batches

    run_results = run_teacher_batches(
        image_batches=image_batches,
        model=model,
        max_side=max_side,
        workers=workers,
    )
    write_jsonl(out_dir / "raw_qa.jsonl", run_results["batch_rows"])

    raw_results = []
    final_rows = []
    failure_counts = Counter()
    for row in run_results["sample_rows"]:
        raw_results.append(row)
        if row["ok"] and row["summary"]["accepted_count"] == 1:
            final_rows.append(row_to_final_sample(row, model=model, prompt_variant="complete_pipeline_dev40_v1"))
        else:
            failure_counts[row.get("failure_reason") or "validation_failed"] += 1

    inline_frontier_summary = annotate_inline_frontier(
        final_rows,
        model=str(load_semantic_dev40_tuning().inline_frontier_model),
        max_side=max_side,
        workers=workers,
    )

    write_jsonl(out_dir / "raw_results.jsonl", raw_results)
    write_jsonl(out_dir / "ocr_qa_dataset.jsonl", final_rows)
    write_jsonl(out_dir / "accepted_dataset.jsonl", final_rows)

    summary = {
        "experiment": {
            "name": out_dir.name,
            "provider": "gemini",
            "model": model,
            "prompt_variant": "complete_pipeline_dev40_v1",
            "source_experiment": source_experiment_dir.name,
            "variant": "bootstrap_complete_pipeline_dev40_v1",
            "workers": workers,
            "max_side": max_side,
            "target_per_image": target_per_image,
        },
        "same_image_universe_count": len(image_ids),
        "input_tuple_count": len(verified_tuples),
        "resolvable_tuple_count": len(verified_tuples),
        "dropped_tuple_count": 0,
        "generated_qas": len(raw_results),
        "accepted_qas": len(final_rows),
        "qa_accept_rate": (len(final_rows) / len(raw_results)) if raw_results else 0.0,
        "stage_counts": {
            "images": len(image_ids),
            "text_nodes": len(all_text_nodes),
            "resolvable_nodes": sum(1 for node in all_text_nodes if node["resolvable"]),
            "word_tuples": word_tuple_count,
            "sign_tuples": sign_tuple_count,
            "verified_tuples": len(verified_tuples),
            "candidate_tuples": len(candidate_rows),
            "selected_tuples": len(selected_rows),
            "raw_qa_rows": len(raw_results),
            "final_qa_rows": len(final_rows),
        },
        "question_type_counts": dict(question_type_counts),
        "selected_question_type_counts": dict(selected_question_type_counts),
        "disabled_question_types": list(DISABLED_QUESTION_TYPES),
        "failure_counts": dict(failure_counts),
        "resolvability_stats": resolvability_stats,
        "source_summary": source_summary,
        "mean_question_words": statistics.mean(len(str(row["question"]).split()) for row in final_rows) if final_rows else 0.0,
        "inline_frontier": inline_frontier_summary,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def build_word_tuple(
    *,
    image_id: str,
    image_path: str,
    image_width: int,
    image_height: int,
    node: dict[str, Any],
    image_nodes: list[dict[str, Any]],
    image_anchors: list[dict[str, Any]],
) -> dict[str, Any]:
    bbox_xyxy = list(node["bbox"])
    bbox_xywh = bbox_xyxy_to_xywh(bbox_xyxy)
    area_fraction = ((bbox_xyxy[2] - bbox_xyxy[0]) * (bbox_xyxy[3] - bbox_xyxy[1])) / max(image_width * image_height, 1)
    region_key = str(node["region_key"])
    anchor = _anchor_for_region(region_key, image_anchors)
    return {
        "tuple_id": f"{image_id}::word::{node['node_id']}",
        "image_id": image_id,
        "image_path": image_path,
        "image_width": image_width,
        "image_height": image_height,
        "ann_id": str(node["node_id"]),
        "answer": str(node["text"]),
        "answer_normalized": str(node["text_normalized"]),
        "text_polygon": list(node["polygon"]),
        "text_bbox": [round(v, 2) for v in bbox_xywh],
        "text_node_ids": [str(node["node_id"])],
        "child_words": [str(node["text"])],
        "anchor_label": anchor["label"],
        "anchor_synonyms": list(REGION_SYNONYMS.get(region_key, (anchor["label"],))),
        "location_phrase": REGION_PHRASES.get(region_key, "area of the image"),
        "location_synonyms": list(REGION_SYNONYMS.get(region_key, (REGION_PHRASES.get(region_key, "area of the image"),))),
        "anchor_box": [round(v, 2) for v in anchor["box"]],
        "anchor_score": float(anchor.get("score") or 1.0),
        "anchor_category": "scene_region",
        "relation": "in",
        "ref_label": None,
        "ref_box": None,
        "unique": True,
        "answer_level": "word",
        "ocr_confidence": float(node.get("confidence") or 1.0),
        "consensus_tier": str(node.get("consensus_tier") or "bootstrap_gt"),
        "dataset_source": str(node.get("source_dataset") or "textocr_bootstrap"),
        "region_key": region_key,
        "resolvable": True,
        "resolvability": {
            "text_px_w": round(float(node["resolvability"]["text_px_w"]), 4),
            "text_px_h": round(float(node["resolvability"]["text_px_h"]), 4),
            "min_width_px": round(float(node["resolvability"]["min_width_px"]), 4),
            "min_height_px": round(float(node["resolvability"]["min_height_px"]), 4),
            "passes": True,
        },
        "kd_metadata": collect_kd_metadata(node, image_nodes, image_anchors, (image_width, image_height)),
        "valid_text_count": len(image_nodes),
        "area_fraction": float(area_fraction),
        "text_length": len(str(node["text"])),
        "density_bucket": density_bucket(len(image_nodes)),
        "area_bucket": area_bucket(float(area_fraction)),
        "group_kind": "word",
    }


def build_sign_tuples(
    *,
    image_id: str,
    image_path: str,
    image_width: int,
    image_height: int,
    image_nodes: list[dict[str, Any]],
    resolvable_nodes: list[dict[str, Any]],
    image_anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = build_line_groups(resolvable_nodes)
    tuples: list[dict[str, Any]] = []
    for group in groups:
        answer = " ".join(str(node["text"]).strip() for node in group if str(node["text"]).strip())
        if not answer or is_blocked_answer(answer):
            continue
        union_xyxy = union_bbox([list(node["bbox"]) for node in group])
        polygon = bbox_to_polygon(union_xyxy)
        resolvability = compute_resolvability(polygon, image_width, image_height)
        if not resolvability["passes"]:
            continue
        bbox_xywh = bbox_xyxy_to_xywh(union_xyxy)
        region_key = region_key_for_bbox(bbox_xywh, image_width, image_height)
        anchor = _anchor_for_region(region_key, image_anchors)
        pseudo_node = {
            "node_id": group_node_id(group),
            "text": answer,
            "polygon": polygon,
            "bbox": union_xyxy,
            "region_key": region_key,
            "resolvable": True,
            "confidence": min(float(node.get("confidence") or 1.0) for node in group),
            "consensus_tier": "bootstrap_group",
        }
        area_fraction = ((union_xyxy[2] - union_xyxy[0]) * (union_xyxy[3] - union_xyxy[1])) / max(image_width * image_height, 1)
        tuples.append(
            {
                "tuple_id": f"{image_id}::sign::{group_node_id(group)}",
                "image_id": image_id,
                "image_path": image_path,
                "image_width": image_width,
                "image_height": image_height,
                "ann_id": group_node_id(group),
                "answer": answer,
                "answer_normalized": normalize_answer(answer),
                "text_polygon": polygon,
                "text_bbox": [round(v, 2) for v in bbox_xywh],
                "text_node_ids": [str(node["node_id"]) for node in group],
                "child_words": [str(node["text"]) for node in group],
                "anchor_label": anchor["label"],
                "anchor_synonyms": list(REGION_SYNONYMS.get(region_key, (anchor["label"],))),
                "location_phrase": REGION_PHRASES.get(region_key, "area of the image"),
                "location_synonyms": list(REGION_SYNONYMS.get(region_key, (REGION_PHRASES.get(region_key, "area of the image"),))),
                "anchor_box": [round(v, 2) for v in anchor["box"]],
                "anchor_score": float(anchor.get("score") or 1.0),
                "anchor_category": "scene_region",
                "relation": "in",
                "ref_label": None,
                "ref_box": None,
                "unique": True,
                "answer_level": "sign",
                "ocr_confidence": min(float(node.get("confidence") or 1.0) for node in group),
                "consensus_tier": "bootstrap_group",
                "dataset_source": str(group[0].get("source_dataset") or "textocr_bootstrap"),
                "region_key": region_key,
                "resolvable": True,
                "resolvability": {
                    "text_px_w": round(float(resolvability["text_px_w"]), 4),
                    "text_px_h": round(float(resolvability["text_px_h"]), 4),
                    "min_width_px": round(float(resolvability["min_width_px"]), 4),
                    "min_height_px": round(float(resolvability["min_height_px"]), 4),
                    "passes": True,
                },
                "kd_metadata": collect_kd_metadata(pseudo_node, image_nodes, image_anchors, (image_width, image_height)),
                "valid_text_count": len(image_nodes),
                "area_fraction": float(area_fraction),
                "text_length": len(answer),
                "density_bucket": density_bucket(len(image_nodes)),
                "area_bucket": area_bucket(float(area_fraction)),
                "group_kind": "line_group",
            }
        )
    return tuples


def build_line_groups(nodes: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if len(nodes) < 2:
        return []

    line_clusters: list[list[dict[str, Any]]] = []
    for node in sorted(nodes, key=lambda item: (centroid_from_polygon(item["polygon"])[1], centroid_from_polygon(item["polygon"])[0])):
        node_cy = centroid_from_polygon(node["polygon"])[1]
        node_h = box_height(list(node["bbox"]))
        placed = False
        for cluster in line_clusters:
            cluster_cys = [centroid_from_polygon(item["polygon"])[1] for item in cluster]
            cluster_hs = [box_height(list(item["bbox"])) for item in cluster]
            if abs(node_cy - statistics.mean(cluster_cys)) <= max(node_h, statistics.mean(cluster_hs)) * 0.65:
                cluster.append(node)
                placed = True
                break
        if not placed:
            line_clusters.append([node])

    seen: set[tuple[str, ...]] = set()
    groups: list[list[dict[str, Any]]] = []
    for cluster in line_clusters:
        if len(cluster) < 2:
            continue
        ordered = sorted(cluster, key=lambda item: centroid_from_polygon(item["polygon"])[0])
        segment = [ordered[0]]
        for node in ordered[1:]:
            if can_join_inline(segment[-1], node):
                segment.append(node)
            else:
                groups.extend(groups_from_segment(segment, seen))
                segment = [node]
        groups.extend(groups_from_segment(segment, seen))
    return groups


def can_join_inline(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_box = list(left["bbox"])
    right_box = list(right["bbox"])
    y_overlap = overlap_1d(left_box[1], left_box[3], right_box[1], right_box[3]) / max(min(box_height(left_box), box_height(right_box)), 1e-6)
    gap = right_box[0] - left_box[2]
    max_height = max(box_height(left_box), box_height(right_box))
    return bool(y_overlap >= 0.35 and gap >= -0.1 * max_height and gap <= 1.6 * max_height)


def groups_from_segment(segment: list[dict[str, Any]], seen: set[tuple[str, ...]]) -> list[list[dict[str, Any]]]:
    if len(segment) < 2:
        return []
    out: list[list[dict[str, Any]]] = []
    max_window = min(4, len(segment))
    for window_size in range(2, max_window + 1):
        for start in range(0, len(segment) - window_size + 1):
            window = segment[start : start + window_size]
            key = tuple(str(node["node_id"]) for node in window)
            if key in seen:
                continue
            text = " ".join(str(node["text"]).strip() for node in window if str(node["text"]).strip())
            if len(text) < 4 or len(text) > 48:
                continue
            if any(is_blocked_answer(str(node["text"])) for node in window):
                continue
            seen.add(key)
            out.append(window)
    return out


def apply_uniqueness(tuple_rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row["answer_normalized"]) for row in tuple_rows)
    for row in tuple_rows:
        row["unique"] = counts[str(row["answer_normalized"])] == 1


def grounding_context_from_tuple(tuple_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_anchor_label": str(tuple_row["anchor_label"]),
        "query_anchor_synonyms": list(tuple_row.get("anchor_synonyms") or [tuple_row["anchor_label"]]),
        "query_anchor_color": str(tuple_row.get("anchor_color") or extract_anchor_color(str(tuple_row.get("anchor_label") or "")) or ""),
        "query_anchor_box": list(tuple_row["anchor_box"]),
        "query_anchor_local_phrase": anchor_local_phrase_for_tuple(tuple_row),
        "query_anchor_local_synonyms": anchor_local_synonyms_for_tuple(tuple_row),
        "query_location_phrase": location_phrase_for_tuple(tuple_row),
        "query_location_synonyms": location_synonyms_for_tuple(tuple_row),
        "query_specific_location_phrase": specific_location_phrase_for_tuple(tuple_row),
        "query_specific_location_synonyms": specific_location_synonyms_for_tuple(tuple_row),
        "query_relation": str(tuple_row.get("relation") or ""),
        "query_anchor_disambiguation_required": bool(int((tuple_row.get("kd_metadata") or {}).get("anchor_label_competitors") or 0) >= 1),
    }


def candidate_anchor_label(candidate: dict[str, Any]) -> str:
    label = str(candidate.get("query_anchor_label") or candidate["tuple"]["anchor_label"])
    tuning = load_semantic_dev40_tuning()
    if (
        str(candidate.get("question_type") or "").upper() == "REVERSE_GROUND"
        and tuning.rg_scrub_color_anchor_phrases_enabled
    ):
        return _strip_simple_color_tokens_from_phrase(label)
    return label


def candidate_anchor_phrases(candidate: dict[str, Any]) -> list[str]:
    phrases = (
        list(candidate["query_anchor_synonyms"])
        if candidate.get("query_anchor_synonyms")
        else list(candidate["tuple"].get("anchor_synonyms") or [candidate["tuple"]["anchor_label"]])
    )
    tuning = load_semantic_dev40_tuning()
    if (
        str(candidate.get("question_type") or "").upper() == "REVERSE_GROUND"
        and tuning.rg_scrub_color_anchor_phrases_enabled
    ):
        cleaned: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            text = _strip_simple_color_tokens_from_phrase(str(phrase))
            norm = normalize_answer(text)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            cleaned.append(text)
        if cleaned:
            return cleaned
    return phrases


def candidate_anchor_box(candidate: dict[str, Any]) -> list[float] | None:
    box = candidate.get("query_anchor_box")
    if isinstance(box, list) and len(box) >= 4:
        return [float(value) for value in box[:4]]
    tuple_box = candidate["tuple"].get("anchor_box")
    if isinstance(tuple_box, list) and len(tuple_box) >= 4:
        return [float(value) for value in tuple_box[:4]]
    return None


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def candidate_anchor_region_key(candidate: dict[str, Any]) -> str:
    box = candidate_anchor_box(candidate)
    if not box:
        return ""
    tuple_row = candidate["tuple"]
    bbox_xywh = _xyxy_to_xywh(box)
    return region_key_for_bbox(bbox_xywh, int(tuple_row["image_width"]), int(tuple_row["image_height"]))


def candidate_anchor_region_phrase(candidate: dict[str, Any]) -> str:
    key = candidate_anchor_region_key(candidate)
    if not key:
        return ""
    return REGION_PHRASES.get(key, "area of the image")


def candidate_anchor_region_synonyms(candidate: dict[str, Any]) -> list[str]:
    key = candidate_anchor_region_key(candidate)
    if not key:
        return []
    if key in REGION_SYNONYMS:
        return list(REGION_SYNONYMS[key])
    phrase = REGION_PHRASES.get(key, "area of the image")
    return [phrase]


def candidate_location_phrase(candidate: dict[str, Any]) -> str:
    if candidate.get("query_location_phrase"):
        return str(candidate["query_location_phrase"])
    return location_phrase_for_tuple(candidate["tuple"])


def candidate_location_synonyms(candidate: dict[str, Any]) -> list[str]:
    if candidate.get("query_location_synonyms"):
        return list(candidate["query_location_synonyms"])
    return location_synonyms_for_tuple(candidate["tuple"])


def candidate_anchor_local_phrase(candidate: dict[str, Any]) -> str:
    if not _anchor_local_phrase_allowed(candidate):
        return ""
    if candidate.get("query_anchor_local_phrase"):
        return str(candidate["query_anchor_local_phrase"])
    return anchor_local_phrase_for_tuple(candidate["tuple"])


def candidate_anchor_local_synonyms(candidate: dict[str, Any]) -> list[str]:
    if not _anchor_local_phrase_allowed(candidate):
        return []
    if candidate.get("query_anchor_local_synonyms"):
        return list(candidate["query_anchor_local_synonyms"])
    return anchor_local_synonyms_for_tuple(candidate["tuple"])


def candidate_specific_location_phrase(candidate: dict[str, Any]) -> str:
    if candidate.get("query_specific_location_phrase"):
        return str(candidate["query_specific_location_phrase"])
    return specific_location_phrase_for_tuple(candidate["tuple"])


def candidate_specific_location_synonyms(candidate: dict[str, Any]) -> list[str]:
    if candidate.get("query_specific_location_synonyms"):
        return list(candidate["query_specific_location_synonyms"])
    return specific_location_synonyms_for_tuple(candidate["tuple"])


def candidate_anchor_color(candidate: dict[str, Any]) -> str:
    tuning = load_semantic_dev40_tuning()
    if not tuning.anchor_reference_color_enabled:
        return ""
    if (
        str(candidate.get("question_type") or "").upper() == "REVERSE_GROUND"
        and tuning.rg_scrub_color_anchor_phrases_enabled
    ):
        return ""
    explicit = str(candidate.get("query_anchor_color") or candidate["tuple"].get("anchor_color") or "").strip()
    if explicit:
        return explicit
    return str(extract_anchor_color(candidate_anchor_label(candidate)) or "")


def candidate_anchor_label_competitors(candidate: dict[str, Any]) -> int:
    return int((candidate["tuple"].get("kd_metadata") or {}).get("anchor_label_competitors") or 0)


_IRREGULAR_ANCHOR_PLURALS = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "foot": "feet",
    "tooth": "teeth",
    "mouse": "mice",
}


def _pluralize_word(word: str) -> str:
    raw = str(word or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered in _IRREGULAR_ANCHOR_PLURALS:
        plural = _IRREGULAR_ANCHOR_PLURALS[lowered]
        return plural if raw.islower() else plural.title() if raw.istitle() else plural.upper() if raw.isupper() else plural
    if lowered.endswith("y") and len(lowered) >= 2 and lowered[-2] not in "aeiou":
        return raw[:-1] + "ies"
    if lowered.endswith(("s", "x", "z", "ch", "sh")):
        return raw + "es"
    return raw + "s"


def pluralize_anchor_phrase(text: str) -> str:
    parts = [part for part in str(text or "").strip().split() if part]
    if not parts:
        return ""
    parts[-1] = _pluralize_word(parts[-1])
    return " ".join(parts)


def repeated_anchor_group_context(tuple_row: dict[str, Any]) -> dict[str, Any] | None:
    tuning = load_semantic_dev40_tuning()
    if not tuning.repeated_anchor_grouping_enabled:
        return None
    kd = tuple_row.get("kd_metadata") or {}
    group_instances = int(kd.get("same_anchor_same_answer_nonoverlap_instances") or 0)
    if group_instances < int(tuning.repeated_anchor_group_min_instances):
        return None
    anchor_label = str(tuple_row.get("anchor_label") or "").strip()
    if not anchor_label:
        return None
    base_synonyms = list(tuple_row.get("anchor_synonyms") or [anchor_label])
    plural_synonyms: list[str] = []
    for phrase in base_synonyms:
        plural = pluralize_anchor_phrase(phrase)
        if plural:
            plural_synonyms.append(plural)
            plural_synonyms.append(f"the {plural}")
            plural_synonyms.append(f"all the {plural}")
    plural_synonyms = list(dict.fromkeys(item.strip() for item in plural_synonyms if item.strip()))
    if not plural_synonyms:
        return None
    return {
        "query_group_mode": "scene_repeat_same_text",
        "query_group_count": group_instances,
        "query_anchor_label": plural_synonyms[0],
        "query_anchor_synonyms": plural_synonyms,
        "query_anchor_local_phrase": "",
        "query_anchor_local_synonyms": [],
        "query_location_phrase": "",
        "query_location_synonyms": [],
        "query_specific_location_phrase": "",
        "query_specific_location_synonyms": [],
        "query_anchor_disambiguation_required": False,
        "query_location_required": False,
    }


def candidate_scene_repeat_group_mode(candidate: dict[str, Any]) -> str:
    return str(candidate.get("query_group_mode") or "")


def _anchor_local_phrase_allowed(candidate: dict[str, Any]) -> bool:
    phrase = str(candidate.get("query_anchor_local_phrase") or candidate["tuple"].get("anchor_local_phrase") or "").strip()
    if not phrase:
        return False
    if candidate_scene_repeat_group_mode(candidate):
        return False
    tuning = load_semantic_dev40_tuning()
    if not tuning.suppress_anchor_local_without_competing_text:
        return True
    kd = candidate["tuple"].get("kd_metadata") or {}
    same_anchor_text_count = int(kd.get("same_anchor_text_count") or (int(kd.get("anchor_overlap_competitors") or 0) + 1))
    return same_anchor_text_count >= 2


def candidate_requires_anchor_disambiguation(candidate: dict[str, Any]) -> bool:
    tuning = load_semantic_dev40_tuning()
    if not tuning.sibling_disambiguation_enabled:
        return False
    if candidate_scene_repeat_group_mode(candidate):
        return False
    if candidate.get("query_anchor_disambiguation_required") is not None:
        return bool(candidate.get("query_anchor_disambiguation_required"))
    kd = candidate["tuple"].get("kd_metadata") or {}
    return bool(
        int(kd.get("anchor_label_competitors") or 0) >= 1
        or int(kd.get("anchor_cluster_size") or 1) >= 2
    )


def _minimal_location_phrase(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    replacements = {
        "upper-left area of the image": "upper left of the image",
        "upper-right area of the image": "upper right of the image",
        "lower-left area of the image": "lower left of the image",
        "lower-right area of the image": "lower right of the image",
        "top-center area of the image": "top center of the image",
        "bottom-center area of the image": "bottom center of the image",
        "center-left area of the image": "left side of the image",
        "center-right area of the image": "right side of the image",
        "center area of the image": "center of the image",
    }
    normalized = raw.lower()
    if normalized in replacements:
        return replacements[normalized]
    compact = raw.replace(" area of the image", " of the image")
    compact = compact.replace(" part of the image", " of the image")
    compact = re.sub(r"\bupper-left\b", "upper left", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\bupper-right\b", "upper right", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\blower-left\b", "lower left", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\blower-right\b", "lower right", compact, flags=re.IGNORECASE)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact


def candidate_minimal_location_synonyms(candidate: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in candidate_location_synonyms(candidate):
        text = _minimal_location_phrase(item)
        norm = normalize_answer(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def candidate_minimal_specific_location_synonyms(candidate: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in candidate_specific_location_synonyms(candidate):
        text = _minimal_location_phrase(item)
        norm = normalize_answer(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def candidate_disambiguation_cues(candidate: dict[str, Any]) -> list[str]:
    tuning = load_semantic_dev40_tuning()
    cues: list[str] = []
    color = candidate_anchor_color(candidate)
    label = candidate_anchor_label(candidate)
    if tuning.anchor_reference_color_enabled and color and label and normalize_answer(color) not in normalize_answer(label):
        cues.append(f"{color} {label}")
    elif tuning.anchor_reference_color_enabled and color:
        cues.append(color)
    for item in candidate_anchor_local_synonyms(candidate):
        if item:
            cues.append(item)
    for item in candidate_minimal_specific_location_synonyms(candidate):
        if item:
            cues.append(item)
    for item in candidate_anchor_region_synonyms(candidate):
        if item:
            cues.append(item)
    for item in candidate_minimal_location_synonyms(candidate):
        if item:
            cues.append(item)
    seen: set[str] = set()
    out: list[str] = []
    for item in cues:
        norm = normalize_answer(item)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(item)
    return out


def candidate_cheap_ambiguity_proxy_score(candidate: dict[str, Any]) -> int:
    tuple_row = candidate["tuple"]
    kd = tuple_row.get("kd_metadata") or {}
    same_label_competitors = int(kd.get("anchor_label_competitors") or 0)
    same_anchor_text_count = int(kd.get("same_anchor_text_count") or (int(kd.get("anchor_overlap_competitors") or 0) + 1))
    same_answer_instances = int(kd.get("same_anchor_same_answer_nonoverlap_instances") or 1)
    distinct_answers = int(kd.get("same_anchor_distinct_answer_count") or 1)
    coarse_competitors = int(kd.get("coarse_region_competitors") or 0)
    bucket_competitors = int(kd.get("bucket_competitors") or 0)
    overlap_competitors = int(kd.get("anchor_overlap_competitors") or 0)
    competing = int(kd.get("competing_tuples") or 0)
    score = 0
    if same_label_competitors >= 6 and overlap_competitors <= 1:
        score += 3
    elif same_label_competitors >= 3 and overlap_competitors <= 1:
        score += 2
    elif same_label_competitors >= 1 and overlap_competitors <= 1:
        score += 1
    if same_anchor_text_count >= 3:
        score += 2
    elif same_anchor_text_count >= 2:
        score += 1
    if same_answer_instances >= 4:
        score += 2
    elif same_answer_instances >= 3:
        score += 1
    if distinct_answers >= 5:
        score += 2
    elif distinct_answers >= 3:
        score += 1
    if coarse_competitors >= 1:
        score += 1
    if bucket_competitors >= 1:
        score += 1
    if competing >= 2:
        score += 1
    if candidate_scene_repeat_group_mode(candidate):
        score = max(score - 2, 0)
    return score


_SIMPLE_COLORS = frozenset({
    "red", "blue", "green", "brown", "white", "black", "gray", "grey",
    "yellow", "orange", "purple", "pink", "silver", "gold",
})

_GENERIC_DIRECT_READ_ANCHOR_TOKENS: frozenset[str] = frozenset({
    "area", "axis", "badge", "bar", "board", "button", "cell", "chart",
    "display", "label", "legend", "panel", "plot", "region", "screen",
    "sign", "surface", "table", "wall",
})

_GENERIC_TEXT_PROPERTY_ANCHOR_TOKENS: frozenset[str] = frozenset({
    "area", "axis", "background", "button", "cell", "chart", "label",
    "legend", "panel", "plot", "region", "surface", "table",
})

_HIGH_PRIOR_TEXT_PROPERTY_ANSWERS: dict[str, frozenset[str]] = {
    "text_color": frozenset({"white", "black", "gray", "grey", "blue"}),
    "text_curvature": frozenset({"straight"}),
}


def _strip_simple_color_tokens_from_phrase(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    kept = [part for part in raw.split() if normalize_answer(part) not in _SIMPLE_COLORS]
    cleaned = " ".join(kept).strip()
    cleaned = re.sub(r"\bwith\s+(?=$)", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,")
    return cleaned or raw


def _anchor_label_has_generic_tokens(anchor_label: str, tokens: frozenset[str]) -> bool:
    label_tokens = set(normalize_answer(anchor_label).split())
    return bool(label_tokens & tokens)


def _is_high_prior_text_property_answer(answer: str, property_type: str) -> bool:
    return normalize_answer(answer) in _HIGH_PRIOR_TEXT_PROPERTY_ANSWERS.get(str(property_type or ""), frozenset())

_ANCHOR_SHAPE_TOKENS: frozenset[str] = frozenset({
    "rectangular", "circular", "square", "oval", "round",
    "triangular", "hexagonal", "cylindrical", "spherical",
})

_STRUCTURAL_SHAPE_TOKENS: frozenset[str] = frozenset({
    "rectangular", "circular", "square", "oval", "wedge",
    "segment", "panel", "emblem", "badge", "bar",
})


def _anchor_label_has_circular_shape(anchor_label: str) -> bool:
    """Return True if the anchor label contains a shape token that would make
    anchor_shape ANCHOR_PROPERTY circular (answer is already in the label)."""
    tokens = set(normalize_answer(anchor_label).split())
    return bool(tokens & _ANCHOR_SHAPE_TOKENS)


def _anchor_label_is_structural_fallback(anchor_label: str) -> bool:
    """Return True if the anchor label looks like a structural-fallback generated description.

    Structural fallback labels combine a color with a structural shape type
    (e.g. 'blue rectangular bar', 'orange wedge segment'). These make REVERSE_GROUND
    questions text-leaky because the color+shape description encodes visual context
    that text-only models can exploit via chart-knowledge priors.
    """
    tokens = set(normalize_answer(anchor_label).split())
    return bool(tokens & _SIMPLE_COLORS) and bool(tokens & _STRUCTURAL_SHAPE_TOKENS)


def candidate_has_answer_leakage(candidate: dict[str, Any], question: str) -> bool:
    """Return True when the expected answer is trivially derivable from the question text alone.

    Two cases:
    - TEXT_PROPERTY word_count: the full text blob is quoted in the question, so the model can
      count the words without looking at the image.
    - ANCHOR_PROPERTY anchor_color: the expected color word appears verbatim in the anchor label
      embedded in the question (e.g. "What color is the *brown* bottle?" → "brown" is leaked).
    """
    qtype = candidate.get("question_type", "")
    norm_q = normalize_answer(question)
    if qtype == "TEXT_PROPERTY" and str(candidate.get("text_property_type") or "") == "word_count":
        tuple_row = candidate["tuple"]
        text_blob = normalize_answer(str(tuple_row.get("answer") or ""))
        if text_blob and text_blob in norm_q:
            expected = str(candidate.get("expected_answer") or "").strip()
            derived = str(len(str(tuple_row.get("answer") or "").split()))
            if expected == derived:
                return True
    if qtype == "ANCHOR_PROPERTY" and str(candidate.get("anchor_property_type") or "anchor_color") == "anchor_color":
        expected_tokens = set(normalize_answer(str(candidate.get("expected_answer") or "")).split())
        anchor_tokens = set(normalize_answer(candidate_anchor_label(candidate)).split())
        color_hits = expected_tokens & _SIMPLE_COLORS & anchor_tokens
        if color_hits:
            return True
    return False


def stable_choice(options: list[str], seed_text: str) -> str:
    if not options:
        return ""
    seed = sum(ord(ch) for ch in seed_text)
    return options[seed % len(options)]


def preferred_location_phrase_for_candidate(candidate: dict[str, Any]) -> str:
    tuning = load_semantic_dev40_tuning()
    if candidate.get("query_location_required") and candidate_specific_location_synonyms(candidate):
        options = (
            candidate_minimal_specific_location_synonyms(candidate)
            if tuning.location_wording_mode == "finalv0"
            else candidate_specific_location_synonyms(candidate)
        )
    else:
        options = (
            candidate_minimal_location_synonyms(candidate)
            if tuning.location_wording_mode == "finalv0"
            else candidate_location_synonyms(candidate)
        )
    return stable_choice(options, str(candidate["candidate_id"]))


def ambiguity_requires_explicit_specific(tuple_row: dict[str, Any]) -> bool:
    kd = tuple_row.get("kd_metadata", {}) or {}
    return bool(
        int(kd.get("competing_tuples") or 0) >= 1
        or int(kd.get("bucket_competitors") or 0) >= 1
        or int(kd.get("anchor_overlap_competitors") or 0) >= 1
        or int(kd.get("coarse_region_competitors") or 0) >= 1
        or int(kd.get("local_text_bucket_occupancy") or 1) >= 2
        or str(kd.get("local_text_cluster_shape") or "") == "grid"
    )


def property_reference_text(tuple_row: dict[str, Any]) -> str:
    answer = str(tuple_row.get("answer") or "").strip()
    child_words = [str(word).strip() for word in tuple_row.get("child_words") or [] if str(word).strip()]
    if tuple_row.get("answer_level") == "word" or len(answer) <= 16:
        return answer
    long_words = [word for word in child_words if len(word) >= 4]
    if long_words:
        return max(long_words, key=len)
    if child_words:
        return child_words[0]
    return answer


def ordered_text_visual_property_types(tuple_row: dict[str, Any]) -> list[str]:
    base = list(TEXT_PROPERTY_VISUAL_TYPES)
    if not base:
        return []
    offset = sum(ord(ch) for ch in str(tuple_row["tuple_id"])) % len(base)
    return base[offset:] + base[:offset]


def ordered_anchor_property_types(tuple_row: dict[str, Any]) -> list[str]:
    base = ["anchor_color", "anchor_material", "anchor_shape"]
    if not base:
        return []
    offset = sum(ord(ch) for ch in str(tuple_row["tuple_id"])) % len(base)
    return base[offset:] + base[:offset]


def location_ambiguity_score(tuple_row: dict[str, Any]) -> int:
    kd = tuple_row.get("kd_metadata", {}) or {}
    cluster_size = int(kd.get("local_text_cluster_size") or 1)
    unresolved = int(kd.get("local_text_cluster_unresolvable") or kd.get("local_text_cluster_unresolved") or 0)
    nearby_anchor_count = len(kd.get("nearby_anchors") or [])
    competing = int(kd.get("competing_tuples") or 0)
    coarse_competitors = int(kd.get("coarse_region_competitors") or 0)
    bucket_competitors = int(kd.get("bucket_competitors") or 0)
    overlap_competitors = int(kd.get("anchor_overlap_competitors") or 0)
    anchor_conflicts = int(kd.get("anchor_conflict_count") or 0)
    bucket_occupancy = int(kd.get("local_text_bucket_occupancy") or 1)
    cluster_shape = str(kd.get("local_text_cluster_shape") or "")
    score = 0
    if competing >= 1:
        score += 3
    if coarse_competitors >= 1:
        score += 2
    if bucket_competitors >= 1:
        score += 2
    if overlap_competitors >= 1:
        score += 2
    if anchor_conflicts >= 1:
        score += 1
    if cluster_size >= 4:
        score += 2
    elif cluster_size >= 3:
        score += 1
    if cluster_shape == "grid":
        score += 2
    if bucket_occupancy >= 2:
        score += 2
    if unresolved >= 2:
        score += 1
    if nearby_anchor_count >= 4:
        score += 1
    return score


def ambiguity_level_from_score(score: int) -> str:
    if score <= 1:
        return "low"
    if score <= 4:
        return "medium"
    return "high"


def _anchor_centroid_offset(tuple_row: dict[str, Any]) -> float:
    """Fractional displacement of the anchor centroid from image center (0=center, 0.5=corner)."""
    box = tuple_row.get("anchor_box")
    if not isinstance(box, list) or len(box) < 4:
        return 0.5
    w = float(tuple_row.get("image_width") or 1)
    h = float(tuple_row.get("image_height") or 1)
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    return max(abs(cx / w - 0.5), abs(cy / h - 0.5))


def requires_specific_location(tuple_row: dict[str, Any], question_type: str, *, yesno_polarity: str | None = None) -> bool:
    tuning = load_semantic_dev40_tuning()
    if not specific_location_synonyms_for_tuple(tuple_row):
        return False
    if tuning.spatial_min_centroid_offset > 0.0:
        if _anchor_centroid_offset(tuple_row) < tuning.spatial_min_centroid_offset:
            return False
    score = location_ambiguity_score(tuple_row)
    competing = int((tuple_row.get("kd_metadata") or {}).get("competing_tuples") or 0)
    if question_type == "YES_NO" and yesno_polarity == "negative":
        return score >= tuning.yesno_negative_specific_threshold
    if question_type == "YES_NO":
        return score >= tuning.yesno_positive_specific_threshold
    if question_type == "DIRECT_READ":
        return score >= tuning.direct_read_specific_threshold
    if question_type == "TEXT_PROPERTY":
        return score >= tuning.property_specific_threshold and competing >= 1
    if question_type == "ANCHOR_PROPERTY":
        return score >= tuning.anchor_property_specific_threshold and competing >= 1
    return False


def reverse_ground_scope_preference(tuple_row: dict[str, Any]) -> str:
    tuning = load_semantic_dev40_tuning()
    local_synonyms = anchor_local_synonyms_for_tuple(tuple_row)
    if not local_synonyms:
        return "global"
    seed = sum(ord(ch) for ch in str(tuple_row["tuple_id"]))
    ambiguity = location_ambiguity_score(tuple_row)
    relation = str(tuple_row.get("relation") or "")
    clean_local = bool(str(tuple_row.get("anchor_local_clean") or "").strip()) or len(local_synonyms) >= 2
    seed_float = (seed % 1000) / 1000.0
    if relation in ("above", "below", "left_of", "right_of"):
        if clean_local and ambiguity >= 2:
            if seed_float < tuning.reverse_ground_directional_mixed_bias:
                return "mixed"
            if seed_float < tuning.reverse_ground_directional_mixed_bias + tuning.reverse_ground_directional_local_bias:
                return "local"
        return "global"
    if relation == "on":
        if clean_local and ambiguity >= 2:
            if seed_float < tuning.reverse_ground_on_mixed_bias:
                return "mixed"
            if seed_float < tuning.reverse_ground_on_mixed_bias + tuning.reverse_ground_on_local_bias:
                return "local"
        return "global"
    return "global"


def _normalize_yesno_query_text(text: str) -> str:
    """Strip trailing OCR punctuation artifacts before embedding in YES/NO questions.

    Prevents questions like 'Does the text say "REFRIGERATORS]"?' which frontier models
    reject because the literal bracket doesn't appear in the image.
    """
    return str(text).rstrip("][(.,;:!?)'\"").strip()


def build_question_candidates(tuple_row: dict[str, Any], answer_units: list[dict[str, Any]], tuning=None) -> list[dict[str, Any]]:
    if tuning is None:
        tuning = load_semantic_dev40_tuning()
    candidates: list[dict[str, Any]] = []
    base_quality = candidate_quality(tuple_row)
    actual_grounding = grounding_context_from_tuple(tuple_row)
    repeated_group_context = repeated_anchor_group_context(tuple_row)
    direct_read_grounding = dict(repeated_group_context or actual_grounding)
    yesno_positive_grounding = dict(repeated_group_context or actual_grounding)
    grouped_scene_read = bool(repeated_group_context)
    reference_text = property_reference_text(tuple_row)
    needs_specific_read_location = False if grouped_scene_read else requires_specific_location(tuple_row, "DIRECT_READ")
    needs_specific_property_location = requires_specific_location(tuple_row, "TEXT_PROPERTY")
    needs_specific_anchor_property_location = requires_specific_location(tuple_row, "ANCHOR_PROPERTY")

    candidates.append(
        {
            "candidate_id": f"{tuple_row['tuple_id']}::DIRECT_READ",
            "tuple": tuple_row,
            "question_type": "DIRECT_READ",
            "quality": base_quality + (0.40 if grouped_scene_read else 0.35),
            "answer_source": "mechanical",
            "answer_type": "text_string",
            "expected_answer": tuple_row["answer"],
            "queried_text": None,
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            **direct_read_grounding,
            "query_location_required": needs_specific_read_location,
        }
    )

    candidates.append(
        {
            "candidate_id": f"{tuple_row['tuple_id']}::YES_NO::POS",
            "tuple": tuple_row,
            "question_type": "YES_NO",
            "quality": base_quality + 0.15,
            "answer_source": "mechanical",
            "answer_type": "yes",
            "expected_answer": "Yes",
            "queried_text": _normalize_yesno_query_text(tuple_row["answer"]),
            "yesno_polarity": "positive",
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            **yesno_positive_grounding,
            "query_location_required": False if grouped_scene_read else requires_specific_location(tuple_row, "YES_NO", yesno_polarity="positive"),
        }
    )

    exclusion = None if grouped_scene_read else choose_grounded_exclusion(tuple_row, answer_units)
    if exclusion is not None:
        candidates.append(
            {
                "candidate_id": f"{tuple_row['tuple_id']}::YES_NO::NEG",
                "tuple": tuple_row,
                "question_type": "YES_NO",
                "quality": base_quality + 0.22 + float(exclusion["strength"]) * 0.04,
                "answer_source": "mechanical",
                "answer_type": "no",
                "expected_answer": "No",
                "queried_text": _normalize_yesno_query_text(tuple_row["answer"]),
                "yesno_polarity": "negative",
                "yesno_distractor_source": "grounded_exclusion",
                "text_property_type": None,
                "anchor_property_type": None,
                "query_anchor_label": exclusion["anchor_label"],
                "query_anchor_synonyms": list(exclusion["anchor_synonyms"]),
                "query_anchor_color": str(exclusion.get("anchor_color") or extract_anchor_color(str(exclusion["anchor_label"])) or ""),
                "query_anchor_box": list(exclusion["anchor_box"]),
                "query_anchor_local_phrase": str(exclusion.get("anchor_local_phrase") or ""),
                "query_anchor_local_synonyms": list(exclusion.get("anchor_local_synonyms") or []),
                "query_location_phrase": str(exclusion["location_phrase"]),
                "query_location_synonyms": list(exclusion["location_synonyms"]),
                "query_specific_location_phrase": str(exclusion.get("specific_location_phrase") or exclusion["location_phrase"]),
                "query_specific_location_synonyms": list(exclusion.get("specific_location_synonyms") or exclusion["location_synonyms"]),
                "query_relation": str(exclusion["relation"]),
                "query_anchor_disambiguation_required": bool(int((exclusion.get("kd_metadata") or {}).get("anchor_label_competitors") or 0) >= 1),
                "query_location_required": requires_specific_location(exclusion, "YES_NO", yesno_polarity="negative"),
                "grounded_exclusion_score": round(float(exclusion["strength"]), 4),
                "grounded_exclusion_source_tuple_id": str(exclusion["tuple_id"]),
            }
        )

    _rg_structural_blocked = (
        tuning.rg_structural_anchor_filter_enabled
        and _anchor_label_is_structural_fallback(str(tuple_row.get("anchor_label") or ""))
    )
    if tuple_row["unique"] and not grouped_scene_read and not _rg_structural_blocked:
        candidates.append(
            {
                "candidate_id": f"{tuple_row['tuple_id']}::REVERSE_GROUND",
                "tuple": tuple_row,
                "question_type": "REVERSE_GROUND",
                "quality": base_quality + (0.22 if tuple_row["answer_level"] == "sign" else 0.08),
                "answer_source": "teacher_spatial",
                "answer_type": "spatial_phrase",
                "expected_answer": None,
                "queried_text": tuple_row["answer"],
                "yesno_polarity": None,
                "yesno_distractor_source": None,
                "text_property_type": None,
                "anchor_property_type": None,
                **actual_grounding,
                "query_location_required": False,
                "reverse_ground_scope_preference": reverse_ground_scope_preference(tuple_row),
            }
        )

    if tuple_row["answer_level"] == "sign" and len(tuple_row["text_node_ids"]) >= 2:
        words = list(tuple_row["child_words"])
        candidates.append(
            {
                "candidate_id": f"{tuple_row['tuple_id']}::TEXT_PROPERTY::WORD_COUNT",
                "tuple": tuple_row,
                "question_type": "TEXT_PROPERTY",
                "quality": base_quality + 0.2,
                "answer_source": "mechanical",
                "answer_type": "number",
                "expected_answer": str(len(words)),
                "queried_text": None,
                "yesno_polarity": None,
                "yesno_distractor_source": None,
                "text_property_type": "word_count",
                "anchor_property_type": None,
                **actual_grounding,
                "query_location_required": needs_specific_property_location,
            }
        )
        candidates.append(
            {
                "candidate_id": f"{tuple_row['tuple_id']}::TEXT_PROPERTY::FIRST_WORD",
                "tuple": tuple_row,
                "question_type": "TEXT_PROPERTY",
                "quality": base_quality + 0.18,
                "answer_source": "mechanical",
                "answer_type": "text_string",
                "expected_answer": words[0],
                "queried_text": None,
                "yesno_polarity": None,
                "yesno_distractor_source": None,
                "text_property_type": "first_word",
                "anchor_property_type": None,
                **actual_grounding,
                "query_location_required": needs_specific_property_location,
            }
        )
        if len(words) >= 3:
            candidates.append(
                {
                    "candidate_id": f"{tuple_row['tuple_id']}::TEXT_PROPERTY::LAST_WORD",
                    "tuple": tuple_row,
                    "question_type": "TEXT_PROPERTY",
                    "quality": base_quality + 0.16,
                    "answer_source": "mechanical",
                    "answer_type": "text_string",
                    "expected_answer": words[-1],
                    "queried_text": None,
                    "yesno_polarity": None,
                    "yesno_distractor_source": None,
                    "text_property_type": "last_word",
                    "anchor_property_type": None,
                    **actual_grounding,
                    "query_location_required": needs_specific_property_location,
                }
            )
    for rank, property_type in enumerate(ordered_text_visual_property_types(tuple_row)[:2]):
        candidates.append(
            {
                "candidate_id": f"{tuple_row['tuple_id']}::TEXT_PROPERTY::{property_type.upper()}",
                "tuple": tuple_row,
                "question_type": "TEXT_PROPERTY",
                "quality": base_quality + 0.07 - 0.02 * rank,
                "answer_source": "teacher_visual",
                "answer_type": "attribute",
                "expected_answer": None,
                "queried_text": reference_text,
                "query_text_reference": reference_text,
                "yesno_polarity": None,
                "yesno_distractor_source": None,
                "text_property_type": property_type,
                "anchor_property_type": None,
                **actual_grounding,
                "query_location_required": needs_specific_property_location,
            }
        )
    if tuple_row["relation"] == "on" and float(tuple_row.get("anchor_score") or 0.0) >= 0.60:
        object_bonus = 0.08 if str(tuple_row.get("anchor_category") or "") in {"container", "device", "clothing", "vehicle"} else 0.02
        _anchor_color_mech = None
        if tuning.anchor_color_mechanical_answer_enabled:
            _raw_color = extract_anchor_color(str(tuple_row.get("anchor_label") or ""))
            if _raw_color and normalize_answer(_raw_color) in _SIMPLE_COLORS:
                _anchor_color_mech = _raw_color
        _label_has_circular_shape = _anchor_label_has_circular_shape(
            str(tuple_row.get("anchor_label") or "")
        )
        for rank, property_type in enumerate(ordered_anchor_property_types(tuple_row)):
            if property_type == "anchor_shape" and _label_has_circular_shape:
                continue
            quality_bonus = object_bonus + (0.09, 0.07, 0.045)[rank]
            _use_mechanical_color = property_type == "anchor_color" and _anchor_color_mech is not None
            candidates.append(
                {
                    "candidate_id": f"{tuple_row['tuple_id']}::ANCHOR_PROPERTY::{property_type.upper()}",
                    "tuple": tuple_row,
                    "question_type": "ANCHOR_PROPERTY",
                    "quality": base_quality + quality_bonus,
                    "answer_source": "mechanical_color" if _use_mechanical_color else "teacher_visual",
                    "answer_type": "attribute",
                    "expected_answer": _anchor_color_mech if _use_mechanical_color else None,
                    "queried_text": reference_text,
                    "query_text_reference": reference_text,
                    "yesno_polarity": None,
                    "yesno_distractor_source": None,
                    "text_property_type": None,
                    "anchor_property_type": property_type,
                    **actual_grounding,
                    "query_location_required": needs_specific_anchor_property_location,
                }
            )
    return candidates


def choose_grounded_exclusion(tuple_row: dict[str, Any], answer_units: list[dict[str, Any]]) -> dict[str, Any] | None:
    tuning = load_semantic_dev40_tuning()
    answer_norm = normalize_answer(str(tuple_row["answer"]))
    actual_anchor = normalize_answer(str(tuple_row["anchor_label"]))
    actual_location = normalize_answer(location_phrase_for_tuple(tuple_row))
    actual_specific = normalize_answer(specific_location_phrase_for_tuple(tuple_row))
    actual_relation = str(tuple_row.get("relation") or "")
    actual_bucket = str((tuple_row.get("kd_metadata") or {}).get("local_text_bucket_key") or "")
    actual_box = [float(value) for value in tuple_row.get("anchor_box") or []]
    actual_center = center_from_box(actual_box) if len(actual_box) >= 4 else [0.0, 0.0]
    image_scale = max(float(tuple_row.get("image_width") or 1), float(tuple_row.get("image_height") or 1), 1.0)
    candidates = []
    for other in answer_units:
        if other["tuple_id"] == tuple_row["tuple_id"]:
            continue
        other_anchor = normalize_answer(str(other["anchor_label"]))
        other_location = normalize_answer(location_phrase_for_tuple(other))
        other_specific = normalize_answer(specific_location_phrase_for_tuple(other))
        other_relation = str(other.get("relation") or "")
        other_bucket = str((other.get("kd_metadata") or {}).get("local_text_bucket_key") or "")
        if other_anchor == actual_anchor and other_location == actual_location and other_relation == actual_relation:
            continue
        other_box = [float(value) for value in other.get("anchor_box") or []]
        if len(actual_box) >= 4 and len(other_box) >= 4:
            overlap = max(overlap_fraction(actual_box, other_box), overlap_fraction(other_box, actual_box))
            other_center = center_from_box(other_box)
            center_gap = (((actual_center[0] - other_center[0]) ** 2 + (actual_center[1] - other_center[1]) ** 2) ** 0.5) / image_scale
        else:
            overlap = 0.0
            center_gap = 0.0
        if overlap > 0.12:
            continue
        if actual_specific and other_specific and actual_specific == other_specific:
            continue
        if actual_bucket and other_bucket and actual_bucket == other_bucket and actual_location == other_location:
            continue
        if other_anchor == actual_anchor and center_gap < 0.20:
            continue
        conflict = False
        for existing in answer_units:
            if normalize_answer(str(existing["answer"])) != answer_norm:
                continue
            if (
                normalize_answer(str(existing["anchor_label"])) == other_anchor
                and normalize_answer(location_phrase_for_tuple(existing)) == other_location
                and str(existing.get("relation") or "") == other_relation
            ):
                conflict = True
                break
        if conflict:
            continue
        strength = 0.0
        strength += 1.5 if other_anchor != actual_anchor else 0.0
        strength += 1.2 if other_location != actual_location else 0.0
        strength += 0.35 if other_relation != actual_relation else 0.0
        strength += 0.55 if overlap < 0.04 else 0.0
        strength += 0.75 if center_gap >= 0.30 else 0.35 if center_gap >= 0.18 else 0.0
        strength += 0.45 if actual_specific and other_specific and actual_specific != other_specific else 0.0
        if other_anchor == actual_anchor and other_location == actual_location:
            strength -= 1.0
        if other_anchor in {"sign", "poster", "label", "screen", "board"} and other_location == actual_location:
            strength -= 0.55
        strength += min(float(other.get("anchor_score") or 0.0), 1.0) * 0.08
        if strength < tuning.grounded_exclusion_min_strength:
            continue
        candidates.append((-(strength), -float(other.get("anchor_score") or 0.0), str(other["tuple_id"]), other, strength))
    if not candidates:
        return None
    candidates.sort()
    chosen = dict(candidates[0][3])
    chosen["strength"] = float(candidates[0][4])
    return chosen


def candidate_quality(tuple_row: dict[str, Any]) -> float:
    size_score = min(float(tuple_row["resolvability"]["text_px_w"]) / 96.0, 1.0)
    unique_bonus = 1.0 if tuple_row["unique"] else 0.45
    level_bonus = 0.18 if tuple_row["answer_level"] == "sign" else 0.0
    density_penalty = 0.12 if tuple_row["kd_metadata"]["text_density"] >= 12 else 0.0
    ambiguity_penalty = 0.05 * float(location_ambiguity_score(tuple_row))
    return float(
        tuple_row["ocr_confidence"] * 0.38
        + tuple_row["anchor_score"] * 0.12
        + size_score * 0.23
        + unique_bonus * 0.17
        + level_bonus
        - density_penalty
        - ambiguity_penalty
    )


def select_candidates(candidates: list[dict[str, Any]], *, target_count: int) -> list[dict[str, Any]]:
    tuning = load_semantic_dev40_tuning()
    if tuning.upstream_centroid_filter_enabled and tuning.spatial_min_centroid_offset > 0.0:
        candidates = [
            c for c in candidates
            if _anchor_centroid_offset(c["tuple"]) >= tuning.spatial_min_centroid_offset
        ]
    if tuning.rg_per_image_hard_cap == 0:
        candidates = [c for c in candidates if c["question_type"] != "REVERSE_GROUND"]
    if tuning.tp_per_image_hard_cap == 0:
        candidates = [c for c in candidates if c["question_type"] != "TEXT_PROPERTY"]
    elif tuning.tp_visual_only_enabled:
        candidates = [
            c for c in candidates
            if c["question_type"] != "TEXT_PROPERTY"
            or str(c.get("text_property_type") or "") in TEXT_PROPERTY_VISUAL_TYPES
        ]
    if len(candidates) <= target_count:
        return sorted(candidates, key=lambda item: (-float(item["quality"]), item["candidate_id"]))

    selected: list[dict[str, Any]] = []
    used_text_nodes: set[tuple[str, ...]] = set()
    used_qtypes: Counter[str] = Counter()
    used_anchor_labels: Counter[str] = Counter()
    remaining = sorted(candidates, key=lambda row: (-float(row["quality"]), row["candidate_id"]))
    while remaining and len(selected) < target_count:
        best = None
        best_score = -1e9
        for candidate in remaining:
            text_key = tuple(candidate["tuple"]["text_node_ids"])
            score = float(candidate["quality"])
            anchor_label_norm = normalize_answer(str(candidate["tuple"].get("anchor_label") or ""))
            score += 1.7 if text_key not in used_text_nodes else 0.0
            score += 1.1 if used_qtypes[candidate["question_type"]] == 0 else 0.0
            if candidate["question_type"] == "DIRECT_READ":
                score += float(tuning.direct_read_selection_bonus)
                if tuning.dr_generic_anchor_penalty > 0.0 and _anchor_label_has_generic_tokens(
                    anchor_label_norm,
                    _GENERIC_DIRECT_READ_ANCHOR_TOKENS,
                ):
                    score -= float(tuning.dr_generic_anchor_penalty)
                if tuning.dr_same_anchor_repeat_penalty > 0.0 and anchor_label_norm:
                    score -= float(tuning.dr_same_anchor_repeat_penalty) * float(used_anchor_labels[anchor_label_norm])
            if tuning.per_image_anchor_diversity_bonus > 0.0 and anchor_label_norm and used_anchor_labels[anchor_label_norm] == 0:
                score += tuning.per_image_anchor_diversity_bonus
            if candidate["question_type"] == "YES_NO" and candidate.get("yesno_polarity") == "negative":
                score += 0.08 + float(candidate.get("grounded_exclusion_score") or 0.0) * 0.025
            if tuning.rg_candidate_oversample_boost > 0.0 and candidate["question_type"] == "REVERSE_GROUND":
                score += float(tuning.rg_candidate_oversample_boost)
            if tuning.property_candidate_selection_bonus > 0.0 and candidate["question_type"] in {"TEXT_PROPERTY", "ANCHOR_PROPERTY"}:
                score += float(tuning.property_candidate_selection_bonus)
            if (
                tuning.anchor_type_soft_cap_count > 0
                and anchor_label_norm
                and _anchor_label_subject_to_soft_cap(anchor_label_norm)
                and used_anchor_labels[anchor_label_norm] >= tuning.anchor_type_soft_cap_count
            ):
                score -= float(tuning.anchor_type_soft_cap_penalty)
            if score > best_score:
                best_score = score
                best = candidate
        assert best is not None
        selected.append(best)
        remaining.remove(best)
        used_text_nodes.add(tuple(best["tuple"]["text_node_ids"]))
        used_qtypes[best["question_type"]] += 1
        used_anchor_labels[normalize_answer(str(best["tuple"].get("anchor_label") or ""))] += 1
    return selected


def _anchor_label_subject_to_soft_cap(label_norm: str) -> bool:
    tokens = set(str(label_norm or "").split())
    return bool(tokens & {"sign", "wall", "board"})


def _unique_anchor_can_skip_global_location(candidate: dict[str, Any]) -> bool:
    tuple_row = candidate["tuple"]
    return (
        bool(tuple_row.get("unique"))
        and candidate["question_type"] in {"DIRECT_READ", "YES_NO"}
        and not ambiguity_requires_explicit_specific(tuple_row)
    )


def enforce_type_constraints(selected: list[dict[str, Any]], candidates: list[dict[str, Any]], *, target: int) -> list[dict[str, Any]]:
    tuning = load_semantic_dev40_tuning()
    selected = list(selected)
    distinct_nodes = {tuple(candidate["tuple"]["text_node_ids"]) for candidate in selected}
    min_direct = 2 if len(distinct_nodes) >= 2 else 1

    while sum(1 for item in selected if item["question_type"] == "DIRECT_READ") < min_direct:
        direct_candidates = [candidate for candidate in candidates if candidate not in selected and candidate["question_type"] == "DIRECT_READ"]
        if not direct_candidates:
            break
        pick = max(direct_candidates, key=lambda candidate: float(candidate["quality"]))
        if len(selected) >= target:
            removable = [candidate for candidate in selected if candidate["question_type"] != "DIRECT_READ"]
            if removable:
                selected.remove(min(removable, key=lambda candidate: float(candidate["quality"])))
        selected.append(pick)

    rg_cap = max(0, int(tuning.rg_per_image_hard_cap))
    tp_cap = max(0, int(tuning.tp_per_image_hard_cap))
    for limited_type in ("REVERSE_GROUND", "TEXT_PROPERTY", "ANCHOR_PROPERTY"):
        cap = rg_cap if limited_type == "REVERSE_GROUND" else (tp_cap if limited_type == "TEXT_PROPERTY" else 1)
        while sum(1 for item in selected if item["question_type"] == limited_type) > cap:
            items = [item for item in selected if item["question_type"] == limited_type]
            selected.remove(min(items, key=lambda item: float(item["quality"])))

    while sum(1 for item in selected if item["question_type"] == "YES_NO" and item.get("yesno_polarity") == "negative") > tuning.max_negative_yesno_per_image:
        items = [item for item in selected if item["question_type"] == "YES_NO" and item.get("yesno_polarity") == "negative"]
        selected.remove(min(items, key=lambda item: float(item["quality"])))

    if tuning.max_yesno_per_image >= 0:
        while sum(1 for item in selected if item["question_type"] == "YES_NO") > tuning.max_yesno_per_image:
            items = [item for item in selected if item["question_type"] == "YES_NO"]
            selected.remove(min(items, key=lambda item: float(item["quality"])))

    while len(selected) > target:
        selected.remove(min(selected, key=lambda item: float(item["quality"])))

    selected.sort(key=lambda item: (-float(item["quality"]), item["candidate_id"]))
    return selected
