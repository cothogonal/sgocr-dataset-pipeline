from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from .bootstrap import (
    REGION_PHRASES,
    REGION_SYNONYMS,
    area_bucket,
    density_bucket,
    normalize_answer,
    region_key_for_bbox,
    write_json,
    write_jsonl,
)
from .bootstrap_kd import (
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
from .semantic_dev40_tuning import load_semantic_dev40_tuning
from .secrets import GEMINI, get_secret
from .teacher.http_clients import encode_image

QUESTION_TYPES = ("DIRECT_READ", "YES_NO", "REVERSE_GROUND", "TEXT_PROPERTY", "ANCHOR_PROPERTY")
DISABLED_QUESTION_TYPES: tuple[str, ...] = ()
URL_LIKE_RE = re.compile(r"(https?://|www\.|\.com\b|\.net\b|\.org\b|@)", re.IGNORECASE)
TEXT_APPEARANCE_TERMS = ("text", "word", "words", "letters", "writing", "label")
TEXT_PROPERTY_VISUAL_TYPES = ("text_color", "text_orientation", "text_curvature")
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
        "query_anchor_box": list(tuple_row["anchor_box"]),
        "query_anchor_local_phrase": anchor_local_phrase_for_tuple(tuple_row),
        "query_anchor_local_synonyms": anchor_local_synonyms_for_tuple(tuple_row),
        "query_location_phrase": location_phrase_for_tuple(tuple_row),
        "query_location_synonyms": location_synonyms_for_tuple(tuple_row),
        "query_specific_location_phrase": specific_location_phrase_for_tuple(tuple_row),
        "query_specific_location_synonyms": specific_location_synonyms_for_tuple(tuple_row),
        "query_relation": str(tuple_row.get("relation") or ""),
    }


def candidate_anchor_label(candidate: dict[str, Any]) -> str:
    return str(candidate.get("query_anchor_label") or candidate["tuple"]["anchor_label"])


def candidate_anchor_phrases(candidate: dict[str, Any]) -> list[str]:
    if candidate.get("query_anchor_synonyms"):
        return list(candidate["query_anchor_synonyms"])
    return list(candidate["tuple"].get("anchor_synonyms") or [candidate["tuple"]["anchor_label"]])


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
    if candidate.get("query_anchor_local_phrase"):
        return str(candidate["query_anchor_local_phrase"])
    return anchor_local_phrase_for_tuple(candidate["tuple"])


def candidate_anchor_local_synonyms(candidate: dict[str, Any]) -> list[str]:
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


def stable_choice(options: list[str], seed_text: str) -> str:
    if not options:
        return ""
    seed = sum(ord(ch) for ch in seed_text)
    return options[seed % len(options)]


def preferred_location_phrase_for_candidate(candidate: dict[str, Any]) -> str:
    if candidate.get("query_location_required") and candidate_specific_location_synonyms(candidate):
        options = candidate_specific_location_synonyms(candidate)
    else:
        options = candidate_location_synonyms(candidate)
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


def requires_specific_location(tuple_row: dict[str, Any], question_type: str, *, yesno_polarity: str | None = None) -> bool:
    tuning = load_semantic_dev40_tuning()
    if not specific_location_synonyms_for_tuple(tuple_row):
        return False
    score = location_ambiguity_score(tuple_row)
    competing = int((tuple_row.get("kd_metadata") or {}).get("competing_tuples") or 0)
    if question_type == "YES_NO" and yesno_polarity == "negative":
        return score >= tuning.yesno_negative_specific_threshold
    if question_type == "YES_NO":
        return score >= tuning.yesno_positive_specific_threshold
    if question_type == "DIRECT_READ":
        return score >= tuning.direct_read_specific_threshold
    if question_type in ("TEXT_PROPERTY", "ANCHOR_PROPERTY"):
        return score >= tuning.property_specific_threshold and competing >= 1
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


def build_question_candidates(tuple_row: dict[str, Any], answer_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    base_quality = candidate_quality(tuple_row)
    actual_grounding = grounding_context_from_tuple(tuple_row)
    reference_text = property_reference_text(tuple_row)
    needs_specific_read_location = requires_specific_location(tuple_row, "DIRECT_READ")
    needs_specific_property_location = requires_specific_location(tuple_row, "TEXT_PROPERTY")
    needs_specific_anchor_property_location = requires_specific_location(tuple_row, "ANCHOR_PROPERTY")

    candidates.append(
        {
            "candidate_id": f"{tuple_row['tuple_id']}::DIRECT_READ",
            "tuple": tuple_row,
            "question_type": "DIRECT_READ",
            "quality": base_quality + 0.35,
            "answer_source": "mechanical",
            "answer_type": "text_string",
            "expected_answer": tuple_row["answer"],
            "queried_text": None,
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            **actual_grounding,
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
            "queried_text": tuple_row["answer"],
            "yesno_polarity": "positive",
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            **actual_grounding,
            "query_location_required": requires_specific_location(tuple_row, "YES_NO", yesno_polarity="positive"),
        }
    )

    exclusion = choose_grounded_exclusion(tuple_row, answer_units)
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
                "queried_text": tuple_row["answer"],
                "yesno_polarity": "negative",
                "yesno_distractor_source": "grounded_exclusion",
                "text_property_type": None,
                "anchor_property_type": None,
                "query_anchor_label": exclusion["anchor_label"],
                "query_anchor_synonyms": list(exclusion["anchor_synonyms"]),
                "query_anchor_box": list(exclusion["anchor_box"]),
                "query_anchor_local_phrase": str(exclusion.get("anchor_local_phrase") or ""),
                "query_anchor_local_synonyms": list(exclusion.get("anchor_local_synonyms") or []),
                "query_location_phrase": str(exclusion["location_phrase"]),
                "query_location_synonyms": list(exclusion["location_synonyms"]),
                "query_specific_location_phrase": str(exclusion.get("specific_location_phrase") or exclusion["location_phrase"]),
                "query_specific_location_synonyms": list(exclusion.get("specific_location_synonyms") or exclusion["location_synonyms"]),
                "query_relation": str(exclusion["relation"]),
                "query_location_required": requires_specific_location(exclusion, "YES_NO", yesno_polarity="negative"),
                "grounded_exclusion_score": round(float(exclusion["strength"]), 4),
                "grounded_exclusion_source_tuple_id": str(exclusion["tuple_id"]),
            }
        )

    if tuple_row["unique"]:
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
        for rank, property_type in enumerate(ordered_anchor_property_types(tuple_row)):
            quality_bonus = object_bonus + (0.09, 0.07, 0.045)[rank]
            candidates.append(
                {
                    "candidate_id": f"{tuple_row['tuple_id']}::ANCHOR_PROPERTY::{property_type.upper()}",
                    "tuple": tuple_row,
                    "question_type": "ANCHOR_PROPERTY",
                    "quality": base_quality + quality_bonus,
                    "answer_source": "teacher_visual",
                    "answer_type": "attribute",
                    "expected_answer": None,
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
    if len(candidates) <= target_count:
        return sorted(candidates, key=lambda item: (-float(item["quality"]), item["candidate_id"]))

    selected: list[dict[str, Any]] = []
    used_text_nodes: set[tuple[str, ...]] = set()
    used_qtypes: Counter[str] = Counter()
    remaining = sorted(candidates, key=lambda row: (-float(row["quality"]), row["candidate_id"]))
    while remaining and len(selected) < target_count:
        best = None
        best_score = -1e9
        for candidate in remaining:
            text_key = tuple(candidate["tuple"]["text_node_ids"])
            score = float(candidate["quality"])
            score += 1.7 if text_key not in used_text_nodes else 0.0
            score += 1.1 if used_qtypes[candidate["question_type"]] == 0 else 0.0
            if candidate["question_type"] == "DIRECT_READ":
                score += 0.25
            if candidate["question_type"] == "YES_NO" and candidate.get("yesno_polarity") == "negative":
                score += 0.08 + float(candidate.get("grounded_exclusion_score") or 0.0) * 0.025
            if score > best_score:
                best_score = score
                best = candidate
        assert best is not None
        selected.append(best)
        remaining.remove(best)
        used_text_nodes.add(tuple(best["tuple"]["text_node_ids"]))
        used_qtypes[best["question_type"]] += 1
    return selected


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

    for limited_type in ("REVERSE_GROUND", "TEXT_PROPERTY", "ANCHOR_PROPERTY"):
        while sum(1 for item in selected if item["question_type"] == limited_type) > 1:
            items = [item for item in selected if item["question_type"] == limited_type]
            selected.remove(min(items, key=lambda item: float(item["quality"])))

    while sum(1 for item in selected if item["question_type"] == "YES_NO" and item.get("yesno_polarity") == "negative") > tuning.max_negative_yesno_per_image:
        items = [item for item in selected if item["question_type"] == "YES_NO" and item.get("yesno_polarity") == "negative"]
        selected.remove(min(items, key=lambda item: float(item["quality"])))

    while len(selected) > target:
        selected.remove(min(selected, key=lambda item: float(item["quality"])))

    selected.sort(key=lambda item: (-float(item["quality"]), item["candidate_id"]))
    return selected


def run_teacher_batches(
    *,
    image_batches: list[dict[str, Any]],
    model: str,
    max_side: int,
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    batch_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    def run_one(batch: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        image_path = batch["image_path"]
        selected_candidates = list(batch["selected_candidates"])
        indexed_candidates = [{**candidate, "candidate_index": index} for index, candidate in enumerate(selected_candidates, start=1)]
        if not image_path or not indexed_candidates:
            batch_row = {
                "image_id": batch["image_id"],
                "image_path": image_path,
                "selected_count": len(indexed_candidates),
                "ok": True,
                "items": [],
            }
            return batch_row, []

        try:
            payload = encode_image(Path(image_path), max_side=max_side)
            result = call_gemini_batched(model=model, image_payload=payload, selected_candidates=indexed_candidates)
            items = list(result.get("parsed", {}).get("items") or [])
            by_index = {int(entry.get("candidate_index")): entry for entry in items if entry.get("candidate_index") is not None}
            normalized_rows = []
            for position, candidate in enumerate(indexed_candidates, start=1):
                item = by_index.get(position)
                if item is None and position - 1 < len(items):
                    item = items[position - 1]
                normalized_rows.append(normalize_candidate_result(candidate, item, result))
            batch_row = {
                "image_id": batch["image_id"],
                "image_path": image_path,
                "selected_count": len(indexed_candidates),
                "ok": True,
                "usage": result.get("usage"),
                "raw_text": result.get("raw_text"),
                "items": items,
            }
            return batch_row, normalized_rows
        except Exception as exc:  # pragma: no cover - exercised only on remote API failure
            result = {"provider": "gemini", "model": model, "raw_text": "", "parsed": {"items": []}, "usage": {}}
            rows = [normalize_candidate_result(candidate, None, result, batch_error=str(exc)) for candidate in indexed_candidates]
            batch_row = {
                "image_id": batch["image_id"],
                "image_path": image_path,
                "selected_count": len(indexed_candidates),
                "ok": False,
                "error": str(exc),
                "items": [],
            }
            return batch_row, rows

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, batch) for batch in image_batches]
            for future in as_completed(futures):
                batch_row, rows = future.result()
                batch_rows.append(batch_row)
                sample_rows.extend(rows)
    else:
        for batch in image_batches:
            batch_row, rows = run_one(batch)
            batch_rows.append(batch_row)
            sample_rows.extend(rows)

    batch_rows.sort(key=lambda row: row["image_id"])
    sample_rows.sort(key=lambda row: row["sample_id"])
    return {"batch_rows": batch_rows, "sample_rows": sample_rows}


def call_gemini_batched(*, model: str, image_payload: Any, selected_candidates: list[dict[str, Any]], timeout_s: int = 120) -> dict[str, Any]:
    prompt = build_batched_prompt(selected_candidates)
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
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseJsonSchema": batched_response_schema(len(selected_candidates)),
        },
    }
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "x-goog-api-key": get_secret(GEMINI),
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    raw_text = extract_gemini_text(payload)
    return {
        "provider": "gemini",
        "model": model,
        "raw_text": raw_text,
        "parsed": json.loads(raw_text),
        "usage": payload.get("usageMetadata", {}),
    }


def build_batched_prompt(selected_candidates: list[dict[str, Any]]) -> str:
    tuning = load_semantic_dev40_tuning()
    blocks = []
    for candidate in selected_candidates:
        tuple_row = candidate["tuple"]
        qtype = candidate["question_type"]
        location_phrase = candidate_location_phrase(candidate)
        preferred_location_phrase = preferred_location_phrase_for_candidate(candidate)
        specific_location_phrase = candidate_specific_location_phrase(candidate)
        anchor_region_phrase = candidate_anchor_region_phrase(candidate)
        lines = [
            f"--- Candidate {candidate['candidate_index']} ---",
            f"candidate_index: {candidate['candidate_index']}",
            f"question_type: {qtype}",
            f'text: "{tuple_row["answer"]}"',
            f'anchor_label: "{candidate_anchor_label(candidate)}"',
            f"allowed_anchor_phrases: {json.dumps(candidate_anchor_phrases(candidate), ensure_ascii=False)}",
            f'anchor_local_location: "{candidate_anchor_local_phrase(candidate)}"',
            f"allowed_anchor_local_phrases: {json.dumps(candidate_anchor_local_synonyms(candidate), ensure_ascii=False)}",
            f'anchor_region: "{anchor_region_phrase}"',
            f"allowed_anchor_region_phrases: {json.dumps(candidate_anchor_region_synonyms(candidate), ensure_ascii=False)}",
            f'coarse_location: "{location_phrase}"',
            f'preferred_location_phrase: "{preferred_location_phrase}"',
            f"allowed_location_phrases: {json.dumps(candidate_location_synonyms(candidate), ensure_ascii=False)}",
            f'specific_location: "{specific_location_phrase}"',
            f"allowed_specific_location_phrases: {json.dumps(candidate_specific_location_synonyms(candidate), ensure_ascii=False)}",
            f"answer_level: {tuple_row['answer_level']}",
            f"unique: {'yes' if tuple_row['unique'] else 'no'}",
            f'relation: "{candidate.get("query_relation") or tuple_row["relation"]}"',
            f"competing_tuples: {int(tuple_row.get('kd_metadata', {}).get('competing_tuples') or 0)}",
            f'local_layout_detail: "{tuple_row.get("semantic_debug", {}).get("location_detail") or tuple_row.get("kd_metadata", {}).get("layout_detail") or ""}"',
        ]
        if qtype == "DIRECT_READ":
            lines.append(f'Write 1 natural question asking what the text says at this location. The answer must be exactly "{tuple_row["answer"]}".')
            if tuning.location_wording_mode == "lite":
                lines.append("Use the preferred location phrase directly. Keep the wording low-entropy, literal, and brief.")
            elif tuning.location_wording_mode == "varied":
                lines.append("Use a varied spatial phrase instead of repeating the same canned wording when possible, but keep the question literal and brief.")
            else:
                lines.append("Use a small amount of spatial wording variation when possible, but keep the question literal and brief.")
            if candidate.get("query_location_required"):
                lines.append(f'Because nearby text boxes share this region, make the question more specific by mentioning both an allowed anchor phrase and the preferred specific location phrase "{preferred_location_phrase}".')
                lines.append("Do not stack multiple near-synonymous global phrases together; use one clean specific phrase instead of layered wording.")
        elif qtype == "YES_NO":
            lines.append(f'Write 1 yes/no question asking whether the text at this location says "{candidate["queried_text"]}".')
            lines.append(f'The answer must be exactly "{candidate["expected_answer"]}".')
            if candidate.get("yesno_polarity") == "negative":
                lines.append("This is a grounded exclusion check: the queried location is intentionally false for this image. Ask whether the text appears at this provided location, not whether it appears anywhere else in the image.")
                lines.append(f'Mention the preferred location phrase "{preferred_location_phrase}" so the question is clearly about the false grounded location.')
            elif candidate.get("query_location_required"):
                lines.append(f'Because the anchor has multiple text regions, mention the preferred location phrase "{preferred_location_phrase}" to disambiguate the target text.')
        elif qtype == "REVERSE_GROUND":
            scope_preference = str(candidate.get("reverse_ground_scope_preference") or "global")
            lines.append(f'Write 1 question asking where the text "{tuple_row["answer"]}" appears.')
            lines.append("The answer must be a short visible location phrase of 2 to 10 words and should mention an allowed anchor phrase, the specific location, or both.")
            if candidate_anchor_local_synonyms(candidate):
                global_examples = [
                    f"on the {candidate_anchor_label(candidate)} in the {anchor_region_phrase}" if anchor_region_phrase else f"{candidate_anchor_label(candidate)} at the {preferred_location_phrase}",
                    f"the {anchor_region_phrase} {candidate_anchor_label(candidate)}" if anchor_region_phrase else f"the {preferred_location_phrase} on the {candidate_anchor_label(candidate)}",
                ]
                mixed_examples = [
                    f"{candidate_anchor_local_phrase(candidate)} in the {anchor_region_phrase}" if anchor_region_phrase else f"{candidate_anchor_local_phrase(candidate)} near the {preferred_location_phrase}",
                    f"{candidate_anchor_local_phrase(candidate)} on the {candidate_anchor_label(candidate)} in the {anchor_region_phrase}" if anchor_region_phrase else f"the {preferred_location_phrase}, {candidate_anchor_local_phrase(candidate)}",
                ]
                lines.append(f'reverse_ground_scope_preference: "{scope_preference}"')
                lines.append("If the local anchor phrase is clean, you may answer with an anchor-scoped phrase instead of a global image phrase.")
                lines.append(f'Local examples: {json.dumps(candidate_anchor_local_synonyms(candidate), ensure_ascii=False)}')
                lines.append(f'Global examples: {json.dumps(global_examples, ensure_ascii=False)}')
                lines.append(f'Mixed examples: {json.dumps(mixed_examples, ensure_ascii=False)}')
                lines.append("Follow the provided scope preference: `local` prefers anchor-local phrasing, `global` prefers anchor+image phrasing, and `mixed` may combine both when concise.")
                if scope_preference == "mixed":
                    lines.append("When `mixed`, prefer a concise answer that combines one clean anchor-local phrase with one coarse image-location phrase, for example `on the bottom left of the car in the center of the image`.")
            else:
                if anchor_region_phrase:
                    lines.append(f'Use only the provided allowed anchor/location phrases plus simple connectors, for example "on the {candidate_anchor_label(candidate)} in the {anchor_region_phrase}" or "the {anchor_region_phrase} {candidate_anchor_label(candidate)}".')
                else:
                    lines.append(f'Use only the provided allowed anchor/location phrases plus simple connectors, for example "{candidate_anchor_label(candidate)} at the {preferred_location_phrase}" or "the {preferred_location_phrase} on the {candidate_anchor_label(candidate)}".')
        elif qtype == "TEXT_PROPERTY":
            prop = candidate["text_property_type"]
            if prop in {"word_count", "first_word", "last_word"}:
                lines.append(f"Write 1 question about the text property `{prop}` for this text at this location.")
                lines.append(f'The answer must be exactly "{candidate["expected_answer"]}".')
            elif prop == "text_color":
                lines.append(f'Write 1 question about the visible color of the text that says "{candidate.get("query_text_reference") or tuple_row["answer"]}" at this location.')
                lines.append('The answer must be a short color phrase of 1 to 3 words, such as "white" or "bright red".')
            elif prop == "text_orientation":
                lines.append(f'Write 1 question about the visible orientation of the text that says "{candidate.get("query_text_reference") or tuple_row["answer"]}" at this location.')
                lines.append("The answer must be a short orientation phrase of 1 to 3 words, such as horizontal, vertical, or diagonal.")
            elif prop == "text_curvature":
                lines.append(f'Write 1 question about whether the text that says "{candidate.get("query_text_reference") or tuple_row["answer"]}" looks straight, curved, or arched.')
                lines.append("The answer must be a short shape phrase of 1 to 3 words.")
            if candidate.get("query_location_required"):
                lines.append(f'Because nearby text boxes share this region, mention the preferred specific location phrase "{preferred_location_phrase}" so the property question targets the correct text.')
        elif qtype == "ANCHOR_PROPERTY":
            property_type = str(candidate.get("anchor_property_type") or "anchor_color")
            reference_text = str(candidate.get("query_text_reference") or tuple_row["answer"])
            if property_type == "anchor_color":
                lines.append(f'Write 1 question about the visible color of the {candidate_anchor_label(candidate)} that has "{reference_text}" on it.')
                lines.append('The answer must be a short color phrase of 1 to 3 words, such as "green" or "dark blue".')
            elif property_type == "anchor_material":
                lines.append(f'Write 1 question about the visible material or surface type of the {candidate_anchor_label(candidate)} that has "{reference_text}" on it.')
                lines.append('The answer must be a short material phrase of 1 to 4 words, such as "metal" or "painted wood".')
            elif property_type == "anchor_shape":
                lines.append(f'Write 1 question about the visible shape of the {candidate_anchor_label(candidate)} that has "{reference_text}" on it.')
                lines.append('The answer must be a short shape phrase of 1 to 3 words, such as "round" or "rectangular".')
            if candidate.get("query_location_required"):
                lines.append(f'Mention the preferred specific location phrase "{preferred_location_phrase}" so the question clearly targets the correct text region.')
            lines.append("Keep it OCR-adjacent: use the text as the reference for which object to describe, but ask about the object or surface itself.")
        blocks.append("\n".join(lines))

    location_instruction = (
        "Prefer stable, low-entropy location wording. Reuse the preferred location phrase rather than inventing extra variants.\n"
        if tuning.location_wording_mode == "lite"
        else "Vary the location wording across candidates when possible instead of repeating the same phrase every time.\n"
        if tuning.location_wording_mode == "varied"
        else "Use some location wording variation across candidates, but keep the phrasing stable and literal.\n"
    )
    strictness_instruction = "Prefer short literal questions and short literal answers over expressive phrasing.\n" if tuning.teacher_strictness == "very_strict" else ""
    return (
        "You are generating OCR spatial QA training data for a single image.\n\n"
        "Use only the verified facts below.\n"
        "Every question must make the location identifiable using one of the allowed anchor phrases.\n"
        + location_instruction
        + "Do not invent new text, objects, materials, activities, scene context, or semantic interpretations.\n"
        + "Do not paraphrase the anchor into a richer description than the allowed anchor phrases.\n"
        + "If the allowed anchor phrase is generic like sign, poster, label, board, screen, bottle, or box, use that exact noun instead of inventing a more specific object description.\n"
        + "Do not use the image to guess subject matter such as medical notes, menus, therapy, music, sports, or brands unless that wording already appears in the verified text itself.\n"
        + strictness_instruction
        + "Do not ask questions that require reading two separate text regions, comparing multiple text regions, or using world knowledge.\n"
        + "Do not ask about URLs, email addresses, or phone numbers.\n"
        + "Keep every question between 5 and 35 words.\n"
        + "If location disambiguation is needed, prefer one clean specific phrase or one clean anchor-local plus anchor-region phrase; avoid redundant stacking like `upper-left part of the upper-left section`.\n"
        + "For DIRECT_READ, YES_NO, and mechanical TEXT_PROPERTY questions, the answer must exactly match the required value.\n"
        + "For REVERSE_GROUND, the answer must be a short visible location phrase that uses only the provided anchor/location wording.\n"
        + "When an anchor-local phrase is provided, you may use it, but keep the answer short and literal.\n"
        + "For visual TEXT_PROPERTY questions, the answer must be a short visible attribute phrase of 1 to 4 words and the question must stay about the text itself.\n"
        + "For ANCHOR_PROPERTY, the answer must be a short visible attribute phrase of 1 to 4 words and the question must use the text as the reference for which object or surface to describe.\n"
        + "Return one JSON item per candidate in the same order.\n\n"
        + "\n\n".join(blocks)
    )


def batched_response_schema(candidate_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {"type": "integer"},
                        "question_type": {"type": "string"},
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "required": ["candidate_index", "question_type", "question", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def extract_gemini_text(payload: dict[str, Any]) -> str:
    for candidate in payload.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                return str(text)
    raise RuntimeError("Gemini response did not contain text")


def normalize_candidate_result(
    candidate: dict[str, Any],
    item: dict[str, Any] | None,
    result: dict[str, Any],
    *,
    batch_error: str | None = None,
) -> dict[str, Any]:
    tuple_row = candidate["tuple"]
    validations, summary, failure_reason = validate_candidate_output(candidate, item)
    if batch_error and failure_reason is None:
        failure_reason = "teacher_batch_failed"
    question = str((item or {}).get("question") or "").strip()
    answer = str((item or {}).get("answer") or "").strip()
    return {
        "sample_id": sample_id_for_candidate(candidate),
        "ok": item is not None and not batch_error,
        "image_id": tuple_row["image_id"],
        "ann_id": tuple_row["ann_id"],
        "tuple": {
            **tuple_row,
            "question_type": candidate["question_type"],
            "answer_source": candidate["answer_source"],
            "answer_type": candidate["answer_type"],
            "yesno_polarity": candidate.get("yesno_polarity"),
            "yesno_distractor_source": candidate.get("yesno_distractor_source"),
            "text_property_type": candidate.get("text_property_type"),
            "anchor_property_type": candidate.get("anchor_property_type"),
            "expected_answer": candidate.get("expected_answer"),
            "queried_text": candidate.get("queried_text"),
            "query_text_reference": candidate.get("query_text_reference"),
            "query_anchor_label": candidate.get("query_anchor_label"),
            "query_anchor_synonyms": list(candidate.get("query_anchor_synonyms") or []),
            "query_anchor_box": candidate.get("query_anchor_box"),
            "query_anchor_local_phrase": candidate.get("query_anchor_local_phrase"),
            "query_anchor_local_synonyms": list(candidate.get("query_anchor_local_synonyms") or []),
            "query_location_phrase": candidate.get("query_location_phrase"),
            "query_location_synonyms": list(candidate.get("query_location_synonyms") or []),
            "query_specific_location_phrase": candidate.get("query_specific_location_phrase"),
            "query_specific_location_synonyms": list(candidate.get("query_specific_location_synonyms") or []),
            "query_relation": candidate.get("query_relation"),
            "query_location_required": bool(candidate.get("query_location_required")),
            "reverse_ground_scope_preference": candidate.get("reverse_ground_scope_preference"),
            "grounded_exclusion_score": candidate.get("grounded_exclusion_score"),
            "grounded_exclusion_source_tuple_id": candidate.get("grounded_exclusion_source_tuple_id"),
            "candidate_id": candidate["candidate_id"],
            "candidate_index": candidate["candidate_index"],
        },
        "result": result,
        "items": [{"question": question, "answer": answer}] if item is not None else [],
        "validations": [validations],
        "summary": summary,
        "failure_reason": failure_reason,
        "filter_stage": {"reason": failure_reason} if failure_reason else None,
    }


def validate_candidate_output(candidate: dict[str, Any], item: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    tuple_row = candidate["tuple"]
    tuning = load_semantic_dev40_tuning()
    if item is None:
        validation = {
            "answer_ok": False,
            "anchor_ok": False,
            "length_ok": False,
            "duplicate_ok": True,
            "accepted": False,
            "question_type_ok": False,
            "mechanical_ok": False,
        }
        return validation, {"generated_count": 0, "accepted_count": 0}, "missing_candidate"

    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    normalized_question = normalize_answer(question)
    normalized_answer = normalize_answer(answer)
    anchor_ok = any(normalize_answer(anchor) in normalized_question or normalize_answer(anchor) in normalized_answer for anchor in candidate_anchor_phrases(candidate))
    location_ok = any(normalize_answer(loc) in normalized_question or normalize_answer(loc) in normalized_answer for loc in candidate_location_synonyms(candidate))
    specific_location_ok = any(normalize_answer(loc) in normalized_question or normalize_answer(loc) in normalized_answer for loc in candidate_specific_location_synonyms(candidate))
    anchor_local_ok = any(normalize_answer(loc) in normalized_question or normalize_answer(loc) in normalized_answer for loc in candidate_anchor_local_synonyms(candidate))
    length_ok = 5 <= len(question.split()) <= 35
    question_type_ok = str(item.get("question_type") or "").strip().upper() == candidate["question_type"]
    answer_ok = False
    mechanical_ok = False
    failure_reason = None
    ambiguity_score = location_ambiguity_score(tuple_row)
    ambiguity_hard = ambiguity_score >= tuning.ambiguity_hard_reject_score
    ambiguity_reverse = ambiguity_score >= tuning.ambiguity_reverse_reject_score
    disambiguated_ok = bool(specific_location_ok or anchor_local_ok)
    explicit_specific_required = bool(ambiguity_hard and ambiguity_requires_explicit_specific(tuple_row))
    reverse_disambiguated_ok = bool(specific_location_ok or (anchor_local_ok and location_ok))

    if candidate["question_type"] == "DIRECT_READ":
        answer_ok = normalized_answer == normalize_answer(str(candidate["expected_answer"]))
        mechanical_ok = answer_ok
        if not answer_ok:
            failure_reason = "direct_read_answer_mismatch"
        elif candidate.get("query_location_required") and candidate_specific_location_synonyms(candidate) and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "direct_read_specific_location_missing"
        elif candidate.get("query_location_required") and not location_ok:
            mechanical_ok = False
            failure_reason = "direct_read_location_missing"
        elif explicit_specific_required and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "direct_read_specific_location_missing"
        elif ambiguity_hard and not disambiguated_ok:
            mechanical_ok = False
            failure_reason = "ambiguous_grounding"
    elif candidate["question_type"] == "YES_NO":
        answer_ok = normalized_answer == normalize_answer(str(candidate["expected_answer"]))
        queried_ok = normalize_answer(str(candidate["queried_text"])) in normalized_question
        location_required = bool(candidate.get("query_location_required"))
        location_match_ok = specific_location_ok if candidate_specific_location_synonyms(candidate) else (location_ok or anchor_local_ok)
        mechanical_ok = bool(answer_ok and queried_ok and (not location_required or location_match_ok))
        if not queried_ok:
            failure_reason = "yesno_query_missing"
        elif not answer_ok:
            failure_reason = "yesno_answer_mismatch"
        elif location_required and candidate_specific_location_synonyms(candidate) and not specific_location_ok:
            failure_reason = "yesno_specific_location_missing"
        elif location_required and not location_match_ok:
            failure_reason = "yesno_location_missing"
        elif explicit_specific_required and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "yesno_specific_location_missing"
        elif ambiguity_hard and not disambiguated_ok:
            mechanical_ok = False
            failure_reason = "ambiguous_grounding"
    elif candidate["question_type"] == "TEXT_PROPERTY":
        property_type = str(candidate["text_property_type"] or "")
        property_terms = TEXT_PROPERTY_TYPE_TERMS.get(property_type, ())
        property_ok = any(normalize_answer(term) in normalized_question for term in property_terms)
        location_required = bool(candidate.get("query_location_required"))
        location_match_ok = specific_location_ok if candidate_specific_location_synonyms(candidate) else (location_ok or anchor_local_ok)
        if property_type in {"word_count", "first_word", "last_word"}:
            answer_ok = normalized_answer == normalize_answer(str(candidate["expected_answer"]))
            mechanical_ok = bool(answer_ok and property_ok and (not location_required or location_match_ok))
        else:
            answer_ok = 1 <= len(answer.split()) <= 4 and len(answer.strip()) >= 2
            reference_text = normalize_answer(str(candidate.get("query_text_reference") or tuple_row["answer"]))
            text_focus_ok = reference_text in normalized_question or any(term in normalized_question.split() for term in TEXT_APPEARANCE_TERMS)
            mechanical_ok = bool(answer_ok and property_ok and text_focus_ok and (not location_required or location_match_ok))
        if not property_ok:
            failure_reason = "text_property_query_missing"
        elif not answer_ok:
            failure_reason = "text_property_answer_mismatch"
        elif property_type not in {"word_count", "first_word", "last_word"} and not (reference_text in normalized_question or any(term in normalized_question.split() for term in TEXT_APPEARANCE_TERMS)):
            failure_reason = "text_property_text_reference_missing"
        elif location_required and candidate_specific_location_synonyms(candidate) and not specific_location_ok:
            failure_reason = "text_property_specific_location_missing"
        elif location_required and not location_match_ok:
            failure_reason = "text_property_location_missing"
        elif explicit_specific_required and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "text_property_specific_location_missing"
        elif ambiguity_hard and not disambiguated_ok:
            mechanical_ok = False
            failure_reason = "ambiguous_grounding"
    elif candidate["question_type"] == "REVERSE_GROUND":
        query_ok = normalize_answer(str(tuple_row["answer"])) in normalized_question
        answer_ok = 2 <= len(answer.split()) <= 10 and (anchor_ok or location_ok or specific_location_ok or anchor_local_ok)
        mechanical_ok = bool(query_ok and answer_ok)
        if not query_ok:
            failure_reason = "reverse_ground_query_missing"
        elif not answer_ok:
            failure_reason = "reverse_ground_answer_invalid"
        elif ambiguity_reverse and ambiguity_requires_explicit_specific(tuple_row) and not reverse_disambiguated_ok:
            mechanical_ok = False
            failure_reason = "reverse_ground_ambiguous"
        elif ambiguity_reverse and not (specific_location_ok or anchor_local_ok):
            mechanical_ok = False
            failure_reason = "reverse_ground_ambiguous"
    elif candidate["question_type"] == "ANCHOR_PROPERTY":
        answer_ok = 1 <= len(answer.split()) <= 4 and len(answer.strip()) >= 2
        property_terms = ANCHOR_PROPERTY_TYPE_TERMS.get(str(candidate.get("anchor_property_type") or "anchor_color"), ())
        property_ok = any(normalize_answer(term) in normalized_question for term in property_terms)
        reference_text = normalize_answer(str(candidate.get("query_text_reference") or tuple_row["answer"]))
        reference_text_ok = reference_text in normalized_question or any(normalize_answer(word) in normalized_question for word in tuple_row.get("child_words") or [])
        location_required = bool(candidate.get("query_location_required"))
        location_match_ok = specific_location_ok if candidate_specific_location_synonyms(candidate) else (location_ok or anchor_local_ok)
        mechanical_ok = bool(answer_ok and anchor_ok and reference_text_ok and property_ok and (not location_required or location_match_ok))
        if not answer_ok:
            failure_reason = "anchor_property_answer_invalid"
        elif not reference_text_ok:
            failure_reason = "anchor_property_text_reference_missing"
        elif not property_ok:
            failure_reason = "anchor_property_signal_missing"
        elif location_required and candidate_specific_location_synonyms(candidate) and not specific_location_ok:
            failure_reason = "anchor_property_specific_location_missing"
        elif location_required and not location_match_ok:
            failure_reason = "anchor_property_location_missing"
        elif explicit_specific_required and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "anchor_property_specific_location_missing"
        elif ambiguity_hard and not disambiguated_ok:
            mechanical_ok = False
            failure_reason = "ambiguous_grounding"

    grounding_ok = bool(anchor_ok or anchor_local_ok or (candidate["question_type"] == "REVERSE_GROUND" and (location_ok or specific_location_ok)))
    accepted = bool(grounding_ok and length_ok and question_type_ok and mechanical_ok)
    validation = {
        "answer_ok": answer_ok,
        "anchor_ok": anchor_ok,
        "length_ok": length_ok,
        "duplicate_ok": True,
        "accepted": accepted,
        "question_type_ok": question_type_ok,
        "mechanical_ok": mechanical_ok,
    }
    if accepted:
        failure_reason = None
    elif failure_reason is None:
        if not grounding_ok:
            failure_reason = "anchor_missing"
        elif not question_type_ok:
            failure_reason = "question_type_mismatch"
        else:
            failure_reason = "validation_failed"
    return validation, {"generated_count": 1, "accepted_count": int(accepted)}, failure_reason


def row_to_final_sample(row: dict[str, Any], *, model: str, prompt_variant: str) -> dict[str, Any]:
    tuple_row = row["tuple"]
    item = row["items"][0]
    tags = build_tags(tuple_row)
    grounding = {
        "text_polygon": tuple_row["text_polygon"],
        "text_bbox": tuple_row["text_bbox"],
        "text_node_ids": tuple_row["text_node_ids"],
        "anchor_box": tuple_row["anchor_box"],
        "anchor_score": tuple_row["anchor_score"],
        "anchor_label_source": tuple_row.get("anchor_label_source"),
        "anchor_synonyms": tuple_row["anchor_synonyms"],
        "anchor_local_phrase": tuple_row.get("anchor_local_phrase"),
        "anchor_local_synonyms": tuple_row.get("anchor_local_synonyms"),
        "anchor_region_phrase": candidate_anchor_region_phrase({"tuple": tuple_row}),
        "anchor_region_synonyms": candidate_anchor_region_synonyms({"tuple": tuple_row}),
        "relation": tuple_row["relation"],
        "ref_label": tuple_row["ref_label"],
        "ref_box": tuple_row["ref_box"],
        "ocr_confidence": tuple_row["ocr_confidence"],
        "region_key": tuple_row["region_key"],
        "location_phrase": tuple_row.get("location_phrase"),
        "specific_location_phrase": tuple_row.get("specific_location_phrase"),
        "specific_location_synonyms": tuple_row.get("specific_location_synonyms"),
        "query_anchor_label": tuple_row.get("query_anchor_label"),
        "query_anchor_synonyms": tuple_row.get("query_anchor_synonyms"),
        "query_anchor_box": tuple_row.get("query_anchor_box"),
        "query_anchor_local_phrase": tuple_row.get("query_anchor_local_phrase"),
        "query_anchor_local_synonyms": tuple_row.get("query_anchor_local_synonyms"),
        "query_anchor_region_phrase": candidate_anchor_region_phrase({"tuple": tuple_row, "query_anchor_box": tuple_row.get("query_anchor_box")}),
        "query_anchor_region_synonyms": candidate_anchor_region_synonyms({"tuple": tuple_row, "query_anchor_box": tuple_row.get("query_anchor_box")}),
        "query_location_phrase": tuple_row.get("query_location_phrase"),
        "query_location_synonyms": tuple_row.get("query_location_synonyms"),
        "query_specific_location_phrase": tuple_row.get("query_specific_location_phrase"),
        "query_specific_location_synonyms": tuple_row.get("query_specific_location_synonyms"),
        "query_relation": tuple_row.get("query_relation"),
        "query_text_reference": tuple_row.get("query_text_reference"),
        "reverse_ground_scope_preference": tuple_row.get("reverse_ground_scope_preference"),
        "query_grounding_matches_text": bool(
            normalize_answer(str(tuple_row.get("query_anchor_label") or tuple_row["anchor_label"]))
            == normalize_answer(str(tuple_row["anchor_label"]))
            and normalize_answer(str(tuple_row.get("query_location_phrase") or location_phrase_for_tuple(tuple_row)))
            == normalize_answer(location_phrase_for_tuple(tuple_row))
        ),
    }
    if tuple_row.get("anchor_source") is not None:
        grounding["anchor_source"] = tuple_row["anchor_source"]
    if tuple_row.get("semantic_debug") is not None:
        grounding["semantic_debug"] = tuple_row["semantic_debug"]
    kd_metadata = dict(tuple_row["kd_metadata"])
    kd_metadata["teacher_answer_logprobs"] = None
    return {
        "sample_id": row["sample_id"],
        "image_id": tuple_row["image_id"],
        "image_path": tuple_row["image_path"],
        "image_width": tuple_row["image_width"],
        "image_height": tuple_row["image_height"],
        "ann_id": tuple_row["ann_id"],
        "question": item["question"],
        "answer": item["answer"],
        "answer_level": tuple_row["answer_level"],
        "anchor_label": tuple_row["anchor_label"],
        "anchor_box": tuple_row["anchor_box"],
        "relation": tuple_row["relation"],
        "ref_label": tuple_row["ref_label"],
        "ref_box": tuple_row["ref_box"],
        "text_polygon": tuple_row["text_polygon"],
        "text_bbox": tuple_row["text_bbox"],
        "ocr_confidence": tuple_row["ocr_confidence"],
        "consensus_tier": tuple_row["consensus_tier"],
        "unique": tuple_row["unique"],
        "dataset_source": tuple_row["dataset_source"],
        "resolvable": tuple_row["resolvable"],
        "resolvability": tuple_row["resolvability"],
        "question_type": tuple_row["question_type"],
        "tags": tags,
        "grounding": grounding,
        "kd_metadata": kd_metadata,
        "teacher_provider": "gemini",
        "teacher_model": model,
        "prompt_variant": prompt_variant,
    }


def build_tags(tuple_row: dict[str, Any]) -> dict[str, Any]:
    question_type = str(tuple_row["question_type"])
    answer_source = str(tuple_row["answer_source"])
    area_frac = float(tuple_row["area_fraction"])
    ambiguity_score = location_ambiguity_score(tuple_row)
    tags = {
        "question_type": question_type,
        "answer_type": answer_type_for(tuple_row),
        "answer_source": answer_source,
        "consensus_tier": str(tuple_row["consensus_tier"]),
        "ocr_confidence_bin": confidence_bin(float(tuple_row["ocr_confidence"])),
        "text_length_bin": text_length_bin(str(tuple_row["answer"])),
        "text_case": text_case(str(tuple_row["answer"])),
        "answer_level": str(tuple_row["answer_level"]),
        "relation": str(tuple_row["relation"]),
        "anchor_category": str(tuple_row["anchor_category"]),
        "anchor_label": str(tuple_row["anchor_label"]),
        "has_reference_object": bool(tuple_row["ref_label"]),
        "unique": bool(tuple_row["unique"]),
        "text_density": text_density_bin(int(tuple_row["kd_metadata"]["text_density"])),
        "text_size_bin": text_size_bin(area_frac),
        "image_source": str(tuple_row["dataset_source"]),
        "anchor_score_bin": anchor_score_bin(float(tuple_row["anchor_score"])),
        "answer_verified_by": "mechanical" if answer_source == "mechanical" else "teacher_only",
        "yesno_polarity": tuple_row.get("yesno_polarity"),
        "yesno_distractor_source": tuple_row.get("yesno_distractor_source"),
        "text_property_type": tuple_row.get("text_property_type"),
        "anchor_property_type": tuple_row.get("anchor_property_type"),
        "ambiguity_level": ambiguity_level_from_score(ambiguity_score),
    }
    tags["difficulty"] = estimate_difficulty(tags)
    return tags


def answer_type_for(tuple_row: dict[str, Any]) -> str:
    if tuple_row["question_type"] == "YES_NO":
        return "yes" if normalize_answer(str(tuple_row["expected_answer"])) == "yes" else "no"
    if tuple_row["question_type"] == "REVERSE_GROUND":
        return "spatial_phrase"
    if tuple_row["question_type"] == "ANCHOR_PROPERTY":
        return "attribute"
    if tuple_row["question_type"] == "TEXT_PROPERTY" and tuple_row.get("text_property_type") in TEXT_PROPERTY_VISUAL_TYPES:
        return "attribute"
    if tuple_row.get("text_property_type") == "word_count":
        return "number"
    return "text_string"


def confidence_bin(value: float) -> str:
    if value > 0.95:
        return "high_>0.95"
    if value >= 0.85:
        return "medium_0.85-0.95"
    return "low_0.80-0.85"


def text_length_bin(text: str) -> str:
    length = len(text.strip())
    if length <= 1:
        return "single_char"
    if length <= 4:
        return "short_2-4"
    if length <= 10:
        return "medium_5-10"
    return "long_11+"


def text_case(text: str) -> str:
    stripped = "".join(ch for ch in text if not ch.isspace())
    if stripped.isdigit():
        return "numeric"
    if stripped.isalnum() and not stripped.isalpha():
        return "alphanumeric"
    if stripped.isupper():
        return "upper"
    if stripped.islower():
        return "lower"
    return "mixed"


def text_density_bin(count: int) -> str:
    if count <= 2:
        return "sparse_1-2"
    if count <= 5:
        return "moderate_3-5"
    return "dense_6+"


def text_size_bin(area_fraction: float) -> str:
    if area_fraction > 0.10:
        return "large_>10pct"
    if area_fraction >= 0.03:
        return "medium_3-10pct"
    if area_fraction >= 0.01:
        return "small_1-3pct"
    return "tiny_<1pct"


def anchor_score_bin(score: float) -> str:
    if score > 0.75:
        return "high_>0.75"
    if score >= 0.50:
        return "medium_0.50-0.75"
    return "low_<0.50"


def estimate_difficulty(tags: dict[str, Any]) -> str:
    score = 0
    if tags["text_size_bin"] == "small_1-3pct":
        score += 2
    elif tags["text_size_bin"] == "medium_3-10pct":
        score += 1
    elif tags["text_size_bin"] == "tiny_<1pct":
        score += 3
    if tags["text_density"] == "dense_6+":
        score += 2
    elif tags["text_density"] == "moderate_3-5":
        score += 1
    if not tags["unique"]:
        score += 1
    if tags["text_length_bin"] == "long_11+":
        score += 1
    if tags["ocr_confidence_bin"] == "low_0.80-0.85":
        score += 1
    if tags["question_type"] in ("REVERSE_GROUND", "TEXT_PROPERTY", "ANCHOR_PROPERTY"):
        score += 1
    if tags.get("ambiguity_level") == "high":
        score += 2
    elif tags.get("ambiguity_level") == "medium":
        score += 1
    if score <= 1:
        return "easy"
    if score <= 3:
        return "medium"
    return "hard"


def sample_id_for_candidate(candidate: dict[str, Any]) -> str:
    return sanitize_id(str(candidate["candidate_id"]))


def location_phrase_for_tuple(tuple_row: dict[str, Any]) -> str:
    if tuple_row.get("location_phrase"):
        return str(tuple_row["location_phrase"])
    return REGION_PHRASES.get(str(tuple_row.get("region_key") or ""), "area of the image")


def location_synonyms_for_tuple(tuple_row: dict[str, Any]) -> list[str]:
    if tuple_row.get("location_synonyms"):
        return list(tuple_row["location_synonyms"])
    key = str(tuple_row.get("region_key") or "")
    if key in REGION_SYNONYMS:
        return list(REGION_SYNONYMS[key])
    return [location_phrase_for_tuple(tuple_row)]


def anchor_local_phrase_for_tuple(tuple_row: dict[str, Any]) -> str:
    phrase = str(tuple_row.get("anchor_local_phrase") or "").strip()
    return phrase


def anchor_local_synonyms_for_tuple(tuple_row: dict[str, Any]) -> list[str]:
    raw = tuple_row.get("anchor_local_synonyms") or []
    if not isinstance(raw, list):
        raw = [raw]
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if not text:
            continue
        norm = normalize_answer(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def specific_location_phrase_for_tuple(tuple_row: dict[str, Any]) -> str:
    phrase = str(tuple_row.get("specific_location_phrase") or "").strip()
    if not phrase:
        return ""
    if normalize_answer(phrase) == normalize_answer(location_phrase_for_tuple(tuple_row)):
        return ""
    return phrase


def specific_location_synonyms_for_tuple(tuple_row: dict[str, Any]) -> list[str]:
    raw = tuple_row.get("specific_location_synonyms") or []
    if not isinstance(raw, list):
        raw = [raw]
    coarse = normalize_answer(location_phrase_for_tuple(tuple_row))
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if not text:
            continue
        norm = normalize_answer(text)
        if not norm or norm == coarse or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def sanitize_id(text: str) -> str:
    return text.replace("::", "__")


def group_node_id(group: list[dict[str, Any]]) -> str:
    return "group_" + "_".join(str(node["node_id"]) for node in group)


def union_bbox(boxes: list[list[float]]) -> list[float]:
    xs1 = [box[0] for box in boxes]
    ys1 = [box[1] for box in boxes]
    xs2 = [box[2] for box in boxes]
    ys2 = [box[3] for box in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def bbox_to_polygon(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def overlap_1d(left_a: float, right_a: float, left_b: float, right_b: float) -> float:
    return max(0.0, min(right_a, right_b) - max(left_a, left_b))


def box_height(box: list[float]) -> float:
    return max(float(box[3]) - float(box[1]), 1e-6)


def _anchor_for_region(region_key: str, image_anchors: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        (anchor for anchor in image_anchors if str(anchor.get("region_key")) == region_key),
        {
            "label": REGION_PHRASES.get(region_key, "region of the image"),
            "box": image_anchors[0]["box"] if image_anchors else [0.0, 0.0, 1.0, 1.0],
            "score": 1.0,
        },
    )


def is_blocked_answer(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return True
    if URL_LIKE_RE.search(stripped):
        return True
    digits = sum(ch.isdigit() for ch in stripped)
    if digits >= 7 and any(ch in stripped for ch in "-()"):
        return True
    return False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _maybe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
