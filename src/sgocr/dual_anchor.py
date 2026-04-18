from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lib.infra.io import load_json, load_jsonl, write_json, write_jsonl


_GENERIC_RESCUE_ANCHORS = frozenset({
    "sign",
    "label",
    "poster",
    "panel",
    "board",
    "screen",
    "display",
    "wall",
    "surface",
    "object",
    "area",
    "region",
})


@dataclass(frozen=True)
class RescueSelectionConfig:
    max_images: int
    min_types_per_image: int = 4
    missing_type_weight: float = 2.0
    missing_anchor_property_weight: float = 2.5
    missing_reverse_ground_weight: float = 0.7
    missing_yesno_weight: float = 0.5
    missing_text_property_weight: float = 0.7
    target_answer_entropy: float = 1.25
    low_answer_entropy_weight: float = 0.6
    target_anchor_diversity: float = 2.0
    low_anchor_diversity_weight: float = 0.4
    generic_anchor_ratio_weight: float = 0.6
    empty_image_weight: float = 4.0


@dataclass(frozen=True)
class RescueImageScore:
    image_id: str
    score: float
    final_rows: int
    final_type_count: int
    final_type_counts: dict[str, int]
    answer_entropy: float
    unique_anchor_labels: int
    generic_anchor_ratio: float
    missing_types: list[str]


def wait_for_gpu_free_memory(
    *,
    min_free_mib: int = 10_000,
    poll_interval_s: float = 1.0,
    timeout_s: float = 30.0,
) -> None:
    deadline = time.time() + max(timeout_s, 0.0)
    while True:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            first = proc.stdout.strip().splitlines()[0].strip()
            free_mib = int(first)
        except Exception:
            return
        if free_mib >= min_free_mib:
            return
        if time.time() >= deadline:
            return
        time.sleep(max(poll_interval_s, 0.1))


def drain_ollama_model(
    model_name: str,
    *,
    min_free_mib: int = 10_000,
    poll_interval_s: float = 1.0,
    timeout_s: float = 30.0,
) -> None:
    subprocess.run(
        ["ollama", "stop", model_name],
        check=False,
        capture_output=True,
        text=True,
    )
    deadline = time.time() + max(timeout_s, 0.0)
    while True:
        try:
            proc = subprocess.run(
                ["ollama", "ps"],
                check=False,
                capture_output=True,
                text=True,
            )
            rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
            active = [line for line in rows[1:] if line and not line.startswith("NAME")]
        except Exception:
            active = []
        if not active:
            break
        if time.time() >= deadline:
            break
        time.sleep(max(poll_interval_s, 0.1))
    wait_for_gpu_free_memory(
        min_free_mib=min_free_mib,
        poll_interval_s=poll_interval_s,
        timeout_s=timeout_s,
    )


def write_source_subset(
    *,
    source_dir: Path,
    out_dir: Path,
    image_ids: set[str],
    role: str,
    note: str,
) -> dict[str, Any]:
    source_manifest = load_json(source_dir / "manifest.json") or {}
    raw_rows = load_jsonl(source_dir / "raw_results.jsonl")
    filtered_rows = [
        row for row in raw_rows
        if str((row.get("tuple") or {}).get("image_id") or row.get("image_id") or "") in image_ids
    ]
    filtered_items = [
        item for item in list(source_manifest.get("items") or [])
        if str(item.get("image_id") or "") in image_ids
    ]
    manifest = dict(source_manifest)
    manifest["name"] = out_dir.name
    manifest["image_count"] = len(filtered_items)
    manifest["selection_policy"] = note
    manifest["parent_source_name"] = str(source_manifest.get("name") or source_dir.name)
    manifest["items"] = filtered_items
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "manifest.json", manifest)
    write_jsonl(out_dir / "raw_results.jsonl", filtered_rows)
    write_jsonl(out_dir / "bootstrap_tuples.jsonl", [])
    notes = "\n".join([
        f"# {out_dir.name}",
        "",
        f"- role: `{role}`",
        f"- parent source: `{source_dir.name}`",
        f"- images: `{len(filtered_items)}`",
        f"- selection: `{note}`",
    ]) + "\n"
    (out_dir / "notes.md").write_text(notes, encoding="utf-8")
    return manifest


def build_subset_ocr_cache(
    *,
    source_intermediate_dir: Path,
    out_dir: Path,
    image_ids: set[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_models = load_json(source_intermediate_dir / "runtime_models.json") or {}
    text_nodes = [
        row for row in load_jsonl(source_intermediate_dir / "text_nodes.jsonl")
        if str(row.get("image_id") or "") in image_ids
    ]
    detection_rows = [
        row for row in load_jsonl(source_intermediate_dir / "text_detections.jsonl")
        if str(row.get("image_id") or "") in image_ids
    ]
    resolvable_nodes = [row for row in text_nodes if bool(row.get("resolvable"))]
    for filename in (
        "parseq_readings.jsonl",
        "ppocrv5_server_readings.jsonl",
        "trocr_large_readings.jsonl",
    ):
        src = source_intermediate_dir / filename
        if src.exists():
            rows = [row for row in load_jsonl(src) if str(row.get("image_id") or "") in image_ids]
            write_jsonl(out_dir / filename, rows)
    write_jsonl(out_dir / "text_nodes.jsonl", text_nodes)
    write_jsonl(out_dir / "text_nodes_resolvable.jsonl", resolvable_nodes)
    write_jsonl(out_dir / "text_detections.jsonl", detection_rows)
    write_json(out_dir / "runtime_models.json", runtime_models)
    source_counts = Counter(str(row.get("dataset_source") or "unknown") for row in text_nodes)
    consensus_stats = {
        "frontend": str((load_json(source_intermediate_dir / "consensus_stats.json") or {}).get("frontend") or "subset_cache"),
        "accepted_nodes": len(text_nodes),
        "total_candidates": len(text_nodes),
        "dropped_nodes": 0,
        "drop_reasons": {},
        "by_source": dict(source_counts),
        "cache_reused": True,
    }
    write_json(out_dir / "consensus_stats.json", consensus_stats)
    resolvability_stats = {
        "total_text_nodes": len(text_nodes),
        "passing_text_nodes": len(resolvable_nodes),
        "dropped_text_nodes": len(text_nodes) - len(resolvable_nodes),
        "by_source": {
            key: {
                "total": count,
                "passed": sum(
                    1 for row in resolvable_nodes if str(row.get("dataset_source") or "unknown") == key
                ),
                "dropped": count - sum(
                    1 for row in resolvable_nodes if str(row.get("dataset_source") or "unknown") == key
                ),
            }
            for key, count in source_counts.items()
        },
    }
    write_json(out_dir / "resolvability_stats.json", resolvability_stats)


def _group_rows_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("image_id") or "")].append(row)
    return grouped


def merge_intermediate_by_image(
    *,
    primary_intermediate_dir: Path,
    rescue_intermediate_dir: Path,
    out_dir: Path,
    rescue_image_ids: set[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "runtime_models.json",
        "text_detections.jsonl",
        "parseq_readings.jsonl",
        "ppocrv5_server_readings.jsonl",
        "trocr_large_readings.jsonl",
        "consensus_stats.json",
        "text_nodes.jsonl",
        "text_nodes_resolvable.jsonl",
        "resolvability_stats.json",
    ):
        src = primary_intermediate_dir / filename
        if src.exists():
            shutil.copy2(src, out_dir / filename)

    for filename in ("anchor_tags.jsonl", "grounded_anchors.jsonl", "verified_tuples.jsonl"):
        primary_rows = load_jsonl(primary_intermediate_dir / filename)
        rescue_rows = load_jsonl(rescue_intermediate_dir / filename)
        rescue_by_image = _group_rows_by_image(rescue_rows)
        merged_rows = [
            row for row in primary_rows
            if str(row.get("image_id") or "") not in rescue_image_ids
        ]
        for image_id in sorted(rescue_image_ids):
            merged_rows.extend(rescue_by_image.get(image_id, []))
        write_jsonl(out_dir / filename, merged_rows)


def score_rescue_images(
    *,
    final_rows: list[dict[str, Any]],
    verified_tuples: list[dict[str, Any]],
    config: RescueSelectionConfig,
) -> list[RescueImageScore]:
    rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tuples_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        rows_by_image[str(row.get("image_id") or "")].append(row)
    for row in verified_tuples:
        tuples_by_image[str(row.get("image_id") or "")].append(row)

    all_image_ids = sorted(set(rows_by_image) | set(tuples_by_image))
    scored: list[RescueImageScore] = []
    for image_id in all_image_ids:
        image_rows = rows_by_image.get(image_id, [])
        image_tuples = tuples_by_image.get(image_id, [])
        qtype_counts = Counter(
            str((row.get("tags") or {}).get("question_type") or row.get("question_type") or "")
            for row in image_rows
        )
        qtype_counts.pop("", None)
        answer_counts = Counter(str(row.get("answer") or "").strip().lower() for row in image_rows if row.get("answer"))
        entropy = 0.0
        total_answers = sum(answer_counts.values())
        if total_answers > 0:
            for count in answer_counts.values():
                p = count / total_answers
                entropy -= p * math.log(p)
        anchor_labels = [
            str(row.get("anchor_label") or "").strip().lower()
            for row in image_tuples
            if str(row.get("anchor_label") or "").strip()
        ]
        unique_anchor_labels = len(set(anchor_labels))
        generic_anchor_ratio = (
            sum(1 for label in anchor_labels if label in _GENERIC_RESCUE_ANCHORS) / len(anchor_labels)
            if anchor_labels else 1.0
        )
        missing_types = [
            qtype for qtype in ("ANCHOR_PROPERTY", "REVERSE_GROUND", "YES_NO", "TEXT_PROPERTY")
            if qtype_counts.get(qtype, 0) <= 0
        ]
        missing_count = max(0, config.min_types_per_image - len(qtype_counts))
        score = 0.0
        score += float(config.empty_image_weight) if not image_rows else 0.0
        score += float(config.missing_type_weight) * missing_count
        if "ANCHOR_PROPERTY" in missing_types:
            score += float(config.missing_anchor_property_weight)
        if "REVERSE_GROUND" in missing_types:
            score += float(config.missing_reverse_ground_weight)
        if "YES_NO" in missing_types:
            score += float(config.missing_yesno_weight)
        if "TEXT_PROPERTY" in missing_types:
            score += float(config.missing_text_property_weight)
        score += max(0.0, float(config.target_answer_entropy) - float(entropy)) * float(config.low_answer_entropy_weight)
        score += max(0.0, float(config.target_anchor_diversity) - float(unique_anchor_labels)) * float(config.low_anchor_diversity_weight)
        score += float(generic_anchor_ratio) * float(config.generic_anchor_ratio_weight)
        scored.append(
            RescueImageScore(
                image_id=image_id,
                score=round(score, 6),
                final_rows=len(image_rows),
                final_type_count=len(qtype_counts),
                final_type_counts=dict(qtype_counts),
                answer_entropy=round(entropy, 6),
                unique_anchor_labels=unique_anchor_labels,
                generic_anchor_ratio=round(generic_anchor_ratio, 6),
                missing_types=missing_types,
            )
        )
    scored.sort(
        key=lambda row: (
            -row.score,
            row.final_type_count,
            row.final_rows,
            row.answer_entropy,
            row.image_id,
        )
    )
    return scored


def select_rescue_image_ids(
    *,
    final_rows: list[dict[str, Any]],
    verified_tuples: list[dict[str, Any]],
    config: RescueSelectionConfig,
) -> tuple[list[str], list[dict[str, Any]]]:
    scored = score_rescue_images(
        final_rows=final_rows,
        verified_tuples=verified_tuples,
        config=config,
    )
    chosen = [row.image_id for row in scored if row.score > 0.0][: max(0, int(config.max_images))]
    metadata = [
        {
            "image_id": row.image_id,
            "score": row.score,
            "final_rows": row.final_rows,
            "final_type_count": row.final_type_count,
            "final_type_counts": row.final_type_counts,
            "answer_entropy": row.answer_entropy,
            "unique_anchor_labels": row.unique_anchor_labels,
            "generic_anchor_ratio": row.generic_anchor_ratio,
            "missing_types": row.missing_types,
            "selected_for_rescue": row.image_id in set(chosen),
        }
        for row in scored
    ]
    return chosen, metadata


def summarize_rescue_selection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if bool(row.get("selected_for_rescue"))]
    missing_type_counts = Counter(
        missing_type
        for row in selected
        for missing_type in list(row.get("missing_types") or [])
    )
    return {
        "selected_images": len(selected),
        "mean_score": round(statistics.mean([float(row.get("score") or 0.0) for row in selected]), 4) if selected else 0.0,
        "mean_final_type_count": round(
            statistics.mean([int(row.get("final_type_count") or 0) for row in selected]),
            4,
        ) if selected else 0.0,
        "missing_type_counts": dict(missing_type_counts),
    }
