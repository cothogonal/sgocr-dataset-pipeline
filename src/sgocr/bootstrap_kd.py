from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .bootstrap import REGION_PHRASES, REGION_SYNONYMS, is_valid_text, normalize_answer, region_key_for_bbox, write_json, write_jsonl
from .semantic_dev40_tuning import load_semantic_dev40_tuning

MODEL_RESOLUTION = 224
PATCH_SIZE = 16
MIN_WIDTH_PATCHES = 1.75
MIN_HEIGHT_PATCHES = 0.7


def resolvability_thresholds() -> tuple[float, float, float, float]:
    tuning = load_semantic_dev40_tuning()
    min_width_patches = float(tuning.resolvability_min_width_patches)
    min_height_patches = float(tuning.resolvability_min_height_patches)
    return (
        min_width_patches,
        min_height_patches,
        min_width_patches * PATCH_SIZE,
        min_height_patches * PATCH_SIZE,
    )


def polygon_to_bbox_xyxy(points: list[float]) -> list[float]:
    xs = [float(points[idx]) for idx in range(0, len(points), 2)]
    ys = [float(points[idx]) for idx in range(1, len(points), 2)]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = [float(value) for value in bbox]
    return [x, y, x + w, y + h]


def bbox_xyxy_to_xywh(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return [x1, y1, x2 - x1, y2 - y1]


def centroid_from_polygon(points: list[float]) -> list[float]:
    xs = [float(points[idx]) for idx in range(0, len(points), 2)]
    ys = [float(points[idx]) for idx in range(1, len(points), 2)]
    return [sum(xs) / max(len(xs), 1), sum(ys) / max(len(ys), 1)]


def center_from_box(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


def overlap_fraction(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    return float(inter / area_a)


def distance_point_to_box(point: list[float], box: list[float]) -> float:
    x, y = point
    x1, y1, x2, y2 = box
    dx = 0.0 if x1 <= x <= x2 else min(abs(x - x1), abs(x - x2))
    dy = 0.0 if y1 <= y <= y2 else min(abs(y - y1), abs(y - y2))
    return math.sqrt(dx * dx + dy * dy)


def determine_relation(primary_box: list[float], other_box: list[float], image_size: tuple[int, int]) -> str:
    if overlap_fraction(primary_box, other_box) > 0.05 or overlap_fraction(other_box, primary_box) > 0.05:
        return "overlapping"
    px, py = center_from_box(primary_box)
    ox, oy = center_from_box(other_box)
    width, height = image_size
    dx = ox - px
    dy = oy - py
    if abs(dx) > abs(dy):
        return "right_of" if ox > px else "left_of"
    if abs(dy) > abs(dx):
        return "below" if oy > py else "above"
    if abs(dx) >= 0.05 * width:
        return "right_of" if dx > 0 else "left_of"
    if abs(dy) >= 0.05 * height:
        return "below" if dy > 0 else "above"
    return "overlapping"


def compute_resolvability(text_polygon: list[float], image_w: int, image_h: int) -> dict[str, Any]:
    min_width_patches, min_height_patches, min_width_px, min_height_px = resolvability_thresholds()
    bbox = polygon_to_bbox_xyxy(text_polygon)
    bbox_w_frac = max((bbox[2] - bbox[0]) / max(image_w, 1), 0.0)
    bbox_h_frac = max((bbox[3] - bbox[1]) / max(image_h, 1), 0.0)
    text_px_w = bbox_w_frac * MODEL_RESOLUTION
    text_px_h = bbox_h_frac * MODEL_RESOLUTION
    return {
        "bbox_xyxy": bbox,
        "bbox_xywh": bbox_xyxy_to_xywh(bbox),
        "text_px_w": float(text_px_w),
        "text_px_h": float(text_px_h),
        "min_width_px": float(min_width_px),
        "min_height_px": float(min_height_px),
        "min_width_patches": float(min_width_patches),
        "min_height_patches": float(min_height_patches),
        "passes": bool(text_px_w >= min_width_px and text_px_h >= min_height_px),
    }


def bootstrap_anchor_boxes(image_w: int, image_h: int) -> list[dict[str, Any]]:
    anchors: list[dict[str, Any]] = []
    cell_w = float(image_w) / 3.0
    cell_h = float(image_h) / 3.0
    for region_key, label in REGION_PHRASES.items():
        col = {"l": 0, "c": 1, "r": 2}[region_key[1]]
        row = {"u": 0, "c": 1, "l": 2}[region_key[0]]
        box = [col * cell_w, row * cell_h, (col + 1) * cell_w, (row + 1) * cell_h]
        anchors.append(
            {
                "label": label,
                "box": box,
                "score": 1.0,
                "region_key": region_key,
                "source": "bootstrap_region",
            }
        )
    return anchors


def load_bootstrap_text_nodes(raw_json_path: Path, *, image_ids: set[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
    imgs = payload.get("imgs", {})
    anns = payload.get("anns", {})
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in anns.values():
        image_id = str(ann.get("image_id") or "")
        if image_id in image_ids:
            by_image[image_id].append(ann)

    out: dict[str, dict[str, Any]] = {}
    for image_id in sorted(image_ids):
        meta = imgs.get(image_id) or {}
        image_w = int(meta.get("width") or 0)
        image_h = int(meta.get("height") or 0)
        nodes: list[dict[str, Any]] = []
        for ann in by_image.get(image_id, []):
            text = str(ann.get("utf8_string") or "").strip()
            if not is_valid_text(text):
                continue
            bbox = ann.get("bbox") or []
            points = ann.get("points") or []
            if len(bbox) != 4 or len(points) < 8:
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 1.0 or h <= 1.0:
                continue
            polygon = [float(v) for v in points[:8]]
            resolvability = compute_resolvability(polygon, image_w, image_h)
            bbox_xyxy = resolvability["bbox_xyxy"]
            nodes.append(
                {
                    "image_id": image_id,
                    "node_id": str(ann.get("id") or ""),
                    "text": text,
                    "text_normalized": normalize_answer(text),
                    "polygon": polygon,
                    "bbox": bbox_xyxy,
                    "bbox_xywh": [x, y, w, h],
                    "confidence": 1.0,
                    "consensus_tier": "bootstrap_gt",
                    "region_key": region_key_for_bbox([x, y, w, h], image_w, image_h),
                    "resolvable": bool(resolvability["passes"]),
                    "resolvability": resolvability,
                    "source_dataset": "textocr_bootstrap",
                }
            )
        out[image_id] = {
            "image_id": image_id,
            "image_width": image_w,
            "image_height": image_h,
            "nodes": sorted(nodes, key=lambda node: (node["node_id"], node["text"])),
            "anchors": bootstrap_anchor_boxes(image_w, image_h),
        }
    return out


def collect_kd_metadata(primary_node: dict[str, Any], all_text_nodes: list[dict[str, Any]], all_anchors: list[dict[str, Any]], image_size: tuple[int, int]) -> dict[str, Any]:
    text_bbox = list(primary_node["bbox"])
    text_center = centroid_from_polygon(primary_node["polygon"])
    neighbors = []
    for other in all_text_nodes:
        if other["node_id"] == primary_node["node_id"]:
            continue
        other_center = centroid_from_polygon(other["polygon"])
        dist = euclidean(text_center, other_center)
        rel = determine_relation(text_bbox, list(other["bbox"]), image_size)
        neighbors.append(
            {
                "node_id": other["node_id"],
                "text": other["text"],
                "confidence": other.get("confidence"),
                "consensus_tier": other.get("consensus_tier"),
                "distance_px": round(dist, 1),
                "relation_to_primary": rel,
                "bbox": [round(v, 2) for v in other["bbox"]],
                "resolvable": bool(other.get("resolvable", True)),
            }
        )
    neighbors.sort(key=lambda row: (row["distance_px"], row["node_id"]))

    nearby_anchors = []
    max_image_dim = float(max(image_size))
    primary_region = str(primary_node["region_key"])
    for anchor in all_anchors:
        overlap = overlap_fraction(text_bbox, list(anchor["box"]))
        proximity = distance_point_to_box(text_center, list(anchor["box"]))
        if overlap > 0.05 or proximity < 0.3 * max_image_dim or str(anchor.get("region_key")) == primary_region:
            score = 1.0 if str(anchor.get("region_key")) == primary_region else max(0.0, 1.0 - (proximity / max(max_image_dim, 1.0)))
            nearby_anchors.append(
                {
                    "label": anchor["label"],
                    "box": [round(v, 2) for v in anchor["box"]],
                    "score": round(float(score), 4),
                    "overlap_with_text": round(float(overlap), 4),
                    "source": anchor.get("source"),
                    "region_key": anchor.get("region_key"),
                }
            )
    nearby_anchors.sort(key=lambda row: (-float(row["score"]), row["label"]))

    competing_tuples = sum(
        1
        for other in all_text_nodes
        if other["node_id"] != primary_node["node_id"] and other.get("resolvable") and str(other.get("region_key")) == primary_region
    )

    return {
        "neighboring_text": neighbors[:10],
        "nearby_anchors": nearby_anchors,
        "text_density": len(all_text_nodes),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "competing_tuples": competing_tuples,
        "teacher_answer_logprobs": None,
    }


def build_resolvability_stats(image_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    min_width_patches, min_height_patches, min_width_px, min_height_px = resolvability_thresholds()
    nodes = [node for payload in image_index.values() for node in payload["nodes"]]
    accepted = [node for node in nodes if node["resolvable"]]
    dropped = [node for node in nodes if not node["resolvable"]]
    by_source = defaultdict(lambda: {"total": 0, "accepted": 0, "dropped": 0})
    for node in nodes:
        bucket = by_source[str(node.get("source_dataset") or "unknown")]
        bucket["total"] += 1
        if node["resolvable"]:
            bucket["accepted"] += 1
        else:
            bucket["dropped"] += 1

    return {
        "model_resolution": MODEL_RESOLUTION,
        "patch_size": PATCH_SIZE,
        "min_width_patches": min_width_patches,
        "min_height_patches": min_height_patches,
        "min_width_px": min_width_px,
        "min_height_px": min_height_px,
        "total_text_nodes": len(nodes),
        "passing_text_nodes": len(accepted),
        "dropped_text_nodes": len(dropped),
        "drop_rate": (len(dropped) / len(nodes)) if nodes else 0.0,
        "drop_rate_by_image_source": {
            key: {
                **value,
                "drop_rate": (value["dropped"] / value["total"]) if value["total"] else 0.0,
            }
            for key, value in sorted(by_source.items())
        },
        "dropped_text_px_w_hist": _histogram([float(node["resolvability"]["text_px_w"]) for node in dropped]),
        "accepted_text_px_w_hist": _histogram([float(node["resolvability"]["text_px_w"]) for node in accepted]),
        "dropped_text_px_w_summary": _summary([float(node["resolvability"]["text_px_w"]) for node in dropped]),
        "accepted_text_px_w_summary": _summary([float(node["resolvability"]["text_px_w"]) for node in accepted]),
    }


def materialize_bootstrap_kd_dataset(
    *,
    source_experiment_dir: Path,
    raw_json_path: Path,
    out_dir: Path,
    intermediate_dir: Path,
) -> dict[str, Any]:
    source_rows = _load_jsonl(source_experiment_dir / "raw_results.jsonl")
    source_summary = json.loads((source_experiment_dir / "summary.json").read_text(encoding="utf-8"))
    image_ids = {str(row["tuple"]["image_id"]) for row in source_rows}
    image_index = load_bootstrap_text_nodes(raw_json_path, image_ids=image_ids)

    all_nodes = [node for payload in image_index.values() for node in payload["nodes"]]
    resolvable_nodes = [node for node in all_nodes if node["resolvable"]]
    write_jsonl(intermediate_dir / "text_nodes.jsonl", all_nodes)
    write_jsonl(intermediate_dir / "text_nodes_resolvable.jsonl", resolvable_nodes)
    resolvability_stats = build_resolvability_stats(image_index)
    write_json(intermediate_dir / "resolvability_stats.json", resolvability_stats)

    enriched_rows = []
    accepted_dataset_rows = []
    verified_tuples = []
    dropped_rows = []
    for row in source_rows:
        base_tuple = dict(row["tuple"])
        image_id = str(base_tuple["image_id"])
        node_id = str(base_tuple["ann_id"])
        image_payload = image_index.get(image_id) or {}
        nodes = list(image_payload.get("nodes") or [])
        anchors = list(image_payload.get("anchors") or [])
        primary = next((node for node in nodes if str(node["node_id"]) == node_id), None)
        if primary is None:
            row = dict(row)
            row["ok"] = False
            row["filter_stage"] = {"resolvable_pass": False, "reason": "missing_primary_node"}
            dropped_rows.append(row)
            enriched_rows.append(row)
            continue

        kd_metadata = collect_kd_metadata(primary, nodes, anchors, (int(image_payload["image_width"]), int(image_payload["image_height"])))
        primary_anchor = next((anchor for anchor in anchors if str(anchor.get("region_key")) == str(primary["region_key"])), None)
        enriched_tuple = {
            **base_tuple,
            "anchor_box": [round(v, 2) for v in (primary_anchor["box"] if primary_anchor else [])],
            "ocr_confidence": primary.get("confidence"),
            "consensus_tier": primary.get("consensus_tier"),
            "resolvable": bool(primary["resolvable"]),
            "resolvability": {
                "text_px_w": round(float(primary["resolvability"]["text_px_w"]), 4),
                "text_px_h": round(float(primary["resolvability"]["text_px_h"]), 4),
                "min_width_px": round(float(primary["resolvability"]["min_width_px"]), 4),
                "min_height_px": round(float(primary["resolvability"]["min_height_px"]), 4),
                "passes": bool(primary["resolvability"]["passes"]),
            },
            "kd_metadata": kd_metadata,
        }
        enriched_row = {
            **row,
            "tuple": enriched_tuple,
            "filter_stage": {
                "resolvable_pass": bool(primary["resolvable"]),
                "reason": None if primary["resolvable"] else "fails_224_patch_filter",
            },
        }
        enriched_rows.append(enriched_row)
        if not primary["resolvable"]:
            dropped_rows.append(enriched_row)
            continue

        verified_tuples.append(enriched_tuple)
        for item, validation in zip(row.get("items") or [], row.get("validations") or []):
            if not validation.get("accepted"):
                continue
            accepted_dataset_rows.append(
                {
                    "image_id": enriched_tuple["image_id"],
                    "image_path": enriched_tuple["image_path"],
                    "ann_id": enriched_tuple["ann_id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "answer_level": enriched_tuple["answer_level"],
                    "anchor_label": enriched_tuple["anchor_label"],
                    "anchor_box": enriched_tuple["anchor_box"],
                    "relation": enriched_tuple["relation"],
                    "ref_label": None,
                    "ref_box": None,
                    "text_polygon": enriched_tuple["text_polygon"],
                    "text_bbox": enriched_tuple["text_bbox"],
                    "ocr_confidence": enriched_tuple["ocr_confidence"],
                    "consensus_tier": enriched_tuple["consensus_tier"],
                    "unique": enriched_tuple["unique"],
                    "dataset_source": enriched_tuple["source_dataset"],
                    "resolvable": enriched_tuple["resolvable"],
                    "resolvability": enriched_tuple["resolvability"],
                    "kd_metadata": {
                        **kd_metadata,
                        "teacher_answer_logprobs": None,
                    },
                    "teacher_provider": row["result"].get("provider") or source_summary.get("experiment", {}).get("provider"),
                    "teacher_model": row["result"].get("model") or source_summary.get("experiment", {}).get("model"),
                    "prompt_variant": source_summary.get("experiment", {}).get("prompt_variant"),
                    "bootstrap_region_key": enriched_tuple["region_key"],
                }
            )

    write_jsonl(out_dir / "raw_results.jsonl", enriched_rows)
    write_jsonl(out_dir / "accepted_dataset.jsonl", accepted_dataset_rows)
    write_jsonl(out_dir / "ocr_qa_dataset.jsonl", accepted_dataset_rows)
    write_jsonl(out_dir / "dropped_tuples.jsonl", [row["tuple"] for row in dropped_rows if "tuple" in row])
    write_jsonl(intermediate_dir / "verified_tuples.jsonl", verified_tuples)

    success_rows = [row for row in enriched_rows if row.get("ok")]
    kept_rows = [row for row in success_rows if row.get("filter_stage", {}).get("resolvable_pass")]
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    question_lengths = []
    accepted_qas = 0
    generated_qas = 0
    for row in kept_rows:
        usage = (row.get("result") or {}).get("usage") or {}
        prompt_tokens.append(int(usage.get("prompt_tokens") or usage.get("promptTokenCount") or 0))
        completion_tokens.append(int(usage.get("completion_tokens") or usage.get("candidatesTokenCount") or 0))
        total_tokens.append(int(usage.get("total_tokens") or usage.get("totalTokenCount") or 0))
        generated_qas += len(row.get("items") or [])
        accepted_qas += sum(1 for item in row.get("validations") or [] if item.get("accepted"))
        for item in row.get("items") or []:
            question_lengths.append(len(str(item.get("question") or "").split()))

    summary = {
        "experiment": {
            **(source_summary.get("experiment") or {}),
            "name": out_dir.name,
            "source_experiment": source_experiment_dir.name,
            "variant": "resolvable_kd_bootstrap_v1",
        },
        "input_tuple_count": len(source_rows),
        "same_image_universe_count": len(image_ids),
        "resolvable_tuple_count": len(kept_rows),
        "dropped_tuple_count": len(dropped_rows),
        "generated_qas": generated_qas,
        "accepted_qas": accepted_qas,
        "qa_accept_rate": (accepted_qas / generated_qas) if generated_qas else 0.0,
        "tuple_full_accept_rate": (
            sum(1 for row in kept_rows if row.get("summary", {}).get("accepted_count") == len(row.get("items") or [])) / len(kept_rows)
            if kept_rows
            else 0.0
        ),
        "mean_prompt_tokens": statistics.mean(prompt_tokens) if prompt_tokens else 0.0,
        "mean_completion_tokens": statistics.mean(completion_tokens) if completion_tokens else 0.0,
        "mean_total_tokens": statistics.mean(total_tokens) if total_tokens else 0.0,
        "mean_question_words": statistics.mean(question_lengths) if question_lengths else 0.0,
        "resolvability_stats": resolvability_stats,
        "intermediate_dir": str(intermediate_dir),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def _histogram(values: list[float]) -> dict[str, int]:
    bins = [
        ("<16", lambda x: x < 16.0),
        ("16-24", lambda x: 16.0 <= x < 24.0),
        ("24-32", lambda x: 24.0 <= x < 32.0),
        ("32-48", lambda x: 32.0 <= x < 48.0),
        ("48-64", lambda x: 48.0 <= x < 64.0),
        ("64+", lambda x: x >= 64.0),
    ]
    counts = Counter()
    for value in values:
        for name, predicate in bins:
            if predicate(value):
                counts[name] += 1
                break
    return {name: counts.get(name, 0) for name, _ in bins}


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(float(ordered[0]), 4),
        "p25": round(float(_percentile(ordered, 0.25)), 4),
        "median": round(float(_percentile(ordered, 0.5)), 4),
        "p75": round(float(_percentile(ordered, 0.75)), 4),
        "max": round(float(ordered[-1]), 4),
        "mean": round(float(statistics.mean(ordered)), 4),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    idx = (len(values) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return values[lo]
    frac = idx - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
