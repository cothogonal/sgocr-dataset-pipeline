from __future__ import annotations

import json
from pathlib import Path

from sgocr.bootstrap import write_json, write_jsonl
from sgocr.dual_anchor import (
    RescueSelectionConfig,
    build_subset_ocr_cache,
    merge_intermediate_by_image,
    select_rescue_image_ids,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_select_rescue_image_ids_prioritizes_missing_ap_and_low_diversity() -> None:
    final_rows = [
        {
            "image_id": "img_low",
            "answer": "yes",
            "tags": {"question_type": "YES_NO"},
        },
        {
            "image_id": "img_mid",
            "answer": "red",
            "tags": {"question_type": "DIRECT_READ"},
        },
        {
            "image_id": "img_mid",
            "answer": "stop",
            "tags": {"question_type": "REVERSE_GROUND"},
        },
        {
            "image_id": "img_good",
            "answer": "red",
            "tags": {"question_type": "DIRECT_READ"},
        },
        {
            "image_id": "img_good",
            "answer": "no",
            "tags": {"question_type": "YES_NO"},
        },
        {
            "image_id": "img_good",
            "answer": "metal",
            "tags": {"question_type": "ANCHOR_PROPERTY"},
        },
        {
            "image_id": "img_good",
            "answer": "straight",
            "tags": {"question_type": "TEXT_PROPERTY"},
        },
    ]
    verified_tuples = [
        {"image_id": "img_low", "anchor_label": "sign"},
        {"image_id": "img_low", "anchor_label": "panel"},
        {"image_id": "img_mid", "anchor_label": "display"},
        {"image_id": "img_mid", "anchor_label": "blue bottle"},
        {"image_id": "img_good", "anchor_label": "metal sign with red border"},
        {"image_id": "img_good", "anchor_label": "plastic label with white fill"},
    ]

    chosen, metadata = select_rescue_image_ids(
        final_rows=final_rows,
        verified_tuples=verified_tuples,
        config=RescueSelectionConfig(max_images=2),
    )

    assert chosen == ["img_low", "img_mid"]
    assert metadata[0]["selected_for_rescue"] is True
    assert "ANCHOR_PROPERTY" in metadata[0]["missing_types"]


def test_build_subset_ocr_cache_filters_to_requested_images(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    write_json(source_dir / "runtime_models.json", {"ocr_frontend": "nemotron_v2"})
    write_json(
        source_dir / "consensus_stats.json",
        {"frontend": "nemotron_v2", "accepted_nodes": 3, "total_candidates": 3},
    )
    write_jsonl(
        source_dir / "text_nodes.jsonl",
        [
            {"image_id": "keep", "dataset_source": "chartqa_train", "resolvable": True},
            {"image_id": "drop", "dataset_source": "textocr_train", "resolvable": False},
        ],
    )
    write_jsonl(
        source_dir / "text_detections.jsonl",
        [
            {"image_id": "keep", "bbox": [0, 0, 1, 1]},
            {"image_id": "drop", "bbox": [0, 0, 2, 2]},
        ],
    )
    write_jsonl(
        source_dir / "parseq_readings.jsonl",
        [
            {"image_id": "keep", "text": "A"},
            {"image_id": "drop", "text": "B"},
        ],
    )

    out_dir = tmp_path / "subset"
    build_subset_ocr_cache(
        source_intermediate_dir=source_dir,
        out_dir=out_dir,
        image_ids={"keep"},
    )

    assert _read_jsonl(out_dir / "text_nodes.jsonl") == [
        {"image_id": "keep", "dataset_source": "chartqa_train", "resolvable": True}
    ]
    assert _read_jsonl(out_dir / "text_detections.jsonl") == [
        {"image_id": "keep", "bbox": [0, 0, 1, 1]}
    ]
    assert _read_jsonl(out_dir / "parseq_readings.jsonl") == [
        {"image_id": "keep", "text": "A"}
    ]
    assert json.loads((out_dir / "runtime_models.json").read_text(encoding="utf-8"))["ocr_frontend"] == "nemotron_v2"


def test_merge_intermediate_by_image_replaces_rescue_rows(tmp_path: Path) -> None:
    primary_dir = tmp_path / "primary"
    rescue_dir = tmp_path / "rescue"
    write_json(primary_dir / "runtime_models.json", {"anchor_candidate_backend": "gemma4_ollama"})
    write_jsonl(primary_dir / "text_nodes.jsonl", [{"image_id": "a"}, {"image_id": "b"}])
    write_jsonl(primary_dir / "text_detections.jsonl", [])
    write_json(primary_dir / "consensus_stats.json", {"frontend": "nemotron_v2"})
    write_jsonl(primary_dir / "text_nodes_resolvable.jsonl", [])
    write_json(primary_dir / "resolvability_stats.json", {"total_text_nodes": 2})
    write_jsonl(primary_dir / "anchor_tags.jsonl", [{"image_id": "a", "tag": "gemma-a"}, {"image_id": "b", "tag": "gemma-b"}])
    write_jsonl(primary_dir / "grounded_anchors.jsonl", [{"image_id": "a", "anchor": "ga"}, {"image_id": "b", "anchor": "gb"}])
    write_jsonl(primary_dir / "verified_tuples.jsonl", [{"image_id": "a", "tuple_id": "a1"}, {"image_id": "b", "tuple_id": "b1"}])

    write_jsonl(rescue_dir / "anchor_tags.jsonl", [{"image_id": "b", "tag": "qwen-b"}])
    write_jsonl(rescue_dir / "grounded_anchors.jsonl", [{"image_id": "b", "anchor": "qb"}])
    write_jsonl(rescue_dir / "verified_tuples.jsonl", [{"image_id": "b", "tuple_id": "bq"}])

    out_dir = tmp_path / "merged"
    merge_intermediate_by_image(
        primary_intermediate_dir=primary_dir,
        rescue_intermediate_dir=rescue_dir,
        out_dir=out_dir,
        rescue_image_ids={"b"},
    )

    assert _read_jsonl(out_dir / "anchor_tags.jsonl") == [
        {"image_id": "a", "tag": "gemma-a"},
        {"image_id": "b", "tag": "qwen-b"},
    ]
    assert _read_jsonl(out_dir / "verified_tuples.jsonl") == [
        {"image_id": "a", "tuple_id": "a1"},
        {"image_id": "b", "tuple_id": "bq"},
    ]
