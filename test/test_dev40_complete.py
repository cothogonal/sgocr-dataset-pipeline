from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from sgocr.bootstrap_kd import bootstrap_anchor_boxes, compute_resolvability
from sgocr.dev40_complete import (
    _anchor_centroid_offset,
    apply_answer_probe_policy,
    build_batched_prompt,
    _unique_anchor_can_skip_global_location,
    candidate_anchor_label,
    candidate_anchor_phrases,
    candidate_anchor_local_phrase,
    candidate_anchor_local_synonyms,
    candidate_has_answer_leakage,
    build_question_candidates,
    build_inline_frontier_prompt,
    inline_frontier_response_schema,
    build_sign_tuples,
    build_tags,
    location_ambiguity_score,
    normalize_candidate_result,
    requires_specific_location,
    sample_id_for_candidate,
    select_candidates,
    validate_candidate_output,
)


def make_node(
    node_id: str,
    text: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    image_w: int = 224,
    image_h: int = 224,
    region_key: str = "uc",
) -> dict:
    polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
    resolvability = compute_resolvability(polygon, image_w, image_h)
    return {
        "image_id": "img-1",
        "node_id": node_id,
        "text": text,
        "text_normalized": text.lower(),
        "polygon": polygon,
        "bbox": [x1, y1, x2, y2],
        "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
        "confidence": 1.0,
        "consensus_tier": "bootstrap_gt",
        "region_key": region_key,
        "resolvable": True,
        "resolvability": resolvability,
        "source_dataset": "textocr_bootstrap",
    }


class TestDev40Complete(unittest.TestCase):
    def _make_rescued_specific_location_candidate(self, *, question_type: str, answer: str = "HELLO") -> dict:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": answer,
            "answer_normalized": answer.lower(),
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": [answer],
            "anchor_label": "box",
            "anchor_synonyms": ["box"],
            "location_phrase": "center of the image",
            "location_synonyms": ["center of the image"],
            "specific_location_phrase": "lower text in the center of the image",
            "specific_location_synonyms": [
                "lower text in the center of the image",
                "lower text near the center of the image",
            ],
            "anchor_local_phrase": "",
            "anchor_local_synonyms": [],
            "anchor_box": [0, 0, 120, 80],
            "anchor_score": 0.8,
            "anchor_category": "container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": False,
            "answer_level": "word",
            "ocr_confidence": 1.0,
            "consensus_tier": "bootstrap_gt",
            "dataset_source": "textocr_bootstrap",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {
                "text_density": 6,
                "competing_tuples": 1,
                "coarse_region_competitors": 1,
                "bucket_competitors": 1,
                "anchor_overlap_competitors": 1,
                "anchor_conflict_count": 1,
                "local_text_bucket_occupancy": 2,
                "local_text_cluster_shape": "grid",
                "neighboring_text": [],
                "nearby_anchors": [],
            },
            "valid_text_count": 6,
            "area_fraction": 0.03,
            "text_length": len(answer),
            "density_bucket": "medium",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        return {
            "candidate_id": f"img-1::word::a::{question_type}",
            "candidate_index": 1,
            "tuple": tuple_row,
            "question_type": question_type,
            "quality": 1.0,
            "answer_source": "mechanical" if question_type in {"DIRECT_READ", "YES_NO"} else "teacher_visual",
            "answer_type": "text_string" if question_type != "TEXT_PROPERTY" else "attribute",
            "expected_answer": answer if question_type == "DIRECT_READ" else ("Yes" if question_type == "YES_NO" else "horizontal"),
            "queried_text": answer if question_type == "YES_NO" else None,
            "yesno_polarity": "positive" if question_type == "YES_NO" else None,
            "yesno_distractor_source": None,
            "text_property_type": "text_orientation" if question_type == "TEXT_PROPERTY" else None,
            "anchor_property_type": None,
            "query_text_reference": answer if question_type == "TEXT_PROPERTY" else None,
            "query_anchor_label": "box",
            "query_anchor_synonyms": ["box"],
            "query_anchor_box": [0, 0, 120, 80],
            "query_anchor_local_phrase": "",
            "query_anchor_local_synonyms": [],
            "query_location_phrase": "center of the image",
            "query_location_synonyms": ["center of the image"],
            "query_specific_location_phrase": "lower text in the center of the image",
            "query_specific_location_synonyms": [
                "lower text in the center of the image",
                "lower text near the center of the image",
            ],
            "query_relation": "on",
            "query_location_required": True,
        }

    def _make_repeated_scene_tuple_row(self) -> dict:
        return {
            "tuple_id": "img-1::word::repeat",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "repeat",
            "answer": "BUD LIGHT",
            "answer_normalized": "bud light",
            "text_polygon": [20, 20, 90, 20, 90, 42, 20, 42],
            "text_bbox": [20, 20, 70, 22],
            "text_node_ids": ["repeat"],
            "child_words": ["BUD", "LIGHT"],
            "anchor_label": "silver can",
            "anchor_synonyms": ["silver can"],
            "location_phrase": "center of the image",
            "location_synonyms": ["center of the image"],
            "specific_location_phrase": "",
            "specific_location_synonyms": [],
            "anchor_local_phrase": "toward the left of the silver can",
            "anchor_local_synonyms": [
                "toward the left of the silver can",
                "on the left side of the silver can",
            ],
            "anchor_box": [10, 0, 110, 200],
            "anchor_score": 0.88,
            "anchor_category": "container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": False,
            "answer_level": "word",
            "ocr_confidence": 1.0,
            "consensus_tier": "bootstrap_gt",
            "dataset_source": "coco_text_train",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 22.0},
            "kd_metadata": {
                "text_density": 8,
                "competing_tuples": 2,
                "coarse_region_competitors": 1,
                "bucket_competitors": 1,
                "anchor_overlap_competitors": 0,
                "anchor_conflict_count": 0,
                "local_text_bucket_occupancy": 1,
                "local_text_cluster_shape": "line",
                "neighboring_text": [],
                "nearby_anchors": ["a", "b", "c"],
                "anchor_label_competitors": 9,
                "same_anchor_text_count": 1,
                "same_anchor_same_answer_count": 5,
                "same_anchor_same_answer_nonoverlap_instances": 5,
                "same_anchor_distinct_answer_count": 4,
                "diffuse_same_anchor_scene": True,
            },
            "valid_text_count": 8,
            "area_fraction": 0.02,
            "text_length": 9,
            "density_bucket": "medium",
            "area_bucket": "medium",
            "group_kind": "word",
        }

    def test_build_sign_tuples_creates_multiword_group(self) -> None:
        nodes = [
            make_node("a", "World-Class", 10, 10, 90, 30, region_key="ul"),
            make_node("b", "Recordings", 95, 11, 175, 31, region_key="uc"),
            make_node("c", "Poster", 20, 100, 60, 120, region_key="cl"),
        ]
        tuples = build_sign_tuples(
            image_id="img-1",
            image_path="data/fake.jpg",
            image_width=224,
            image_height=224,
            image_nodes=nodes,
            resolvable_nodes=nodes,
            image_anchors=bootstrap_anchor_boxes(224, 224),
        )
        answers = {row["answer"] for row in tuples}
        self.assertIn("World-Class Recordings", answers)
        sign = next(row for row in tuples if row["answer"] == "World-Class Recordings")
        self.assertEqual(sign["answer_level"], "sign")
        self.assertEqual(sign["text_node_ids"], ["a", "b"])

    def test_question_candidates_include_text_property_for_sign(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::sign::a_b",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "group_a_b",
            "answer": "World Class",
            "answer_normalized": "world class",
            "text_polygon": [10, 10, 150, 10, 150, 30, 10, 30],
            "text_bbox": [10, 10, 140, 20],
            "text_node_ids": ["a", "b"],
            "child_words": ["World", "Class"],
            "anchor_label": "top-center area of the image",
            "anchor_synonyms": ["top-center area of the image"],
            "anchor_box": [0, 0, 224, 74.6],
            "anchor_score": 1.0,
            "anchor_category": "scene_region",
            "relation": "in",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "sign",
            "ocr_confidence": 1.0,
            "consensus_tier": "bootstrap_group",
            "dataset_source": "textocr_bootstrap",
            "region_key": "uc",
            "resolvable": True,
            "resolvability": {"text_px_w": 64.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 3, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 3,
            "area_fraction": 0.04,
            "text_length": 11,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "line_group",
        }
        candidates = build_question_candidates(tuple_row, [tuple_row])
        qtypes = {(row["question_type"], row.get("text_property_type")) for row in candidates}
        self.assertIn(("DIRECT_READ", None), qtypes)
        self.assertIn(("TEXT_PROPERTY", "word_count"), qtypes)
        self.assertIn(("TEXT_PROPERTY", "first_word"), qtypes)

    def test_repeated_anchor_grouping_produces_plural_direct_read(self) -> None:
        tuple_row = self._make_repeated_scene_tuple_row()
        with patch.dict(
            os.environ,
            {
                "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
                "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": "3",
            },
            clear=True,
        ):
            candidates = build_question_candidates(tuple_row, [tuple_row])
        direct = next(candidate for candidate in candidates if candidate["question_type"] == "DIRECT_READ")
        self.assertEqual(direct.get("query_group_mode"), "scene_repeat_same_text")
        self.assertIn("silver cans", direct.get("query_anchor_label") or "")
        self.assertIn("all the silver cans", direct.get("query_anchor_synonyms") or [])
        self.assertFalse(direct.get("query_location_required"))

    def test_anchor_local_phrase_can_be_suppressed_without_same_anchor_text(self) -> None:
        tuple_row = self._make_repeated_scene_tuple_row()
        with patch.dict(
            os.environ,
            {
                "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
            },
            clear=True,
        ):
            candidates = build_question_candidates(tuple_row, [tuple_row])
            direct = next(candidate for candidate in candidates if candidate["question_type"] == "DIRECT_READ")
            self.assertEqual(candidate_anchor_local_phrase(direct), "")
            self.assertEqual(candidate_anchor_local_synonyms(direct), [])

    def test_validate_candidate_output_can_use_cheap_ambiguity_proxy(self) -> None:
        tuple_row = self._make_repeated_scene_tuple_row()
        candidate = {
            "candidate_id": "img-1::word::repeat::DIRECT_READ",
            "candidate_index": 1,
            "tuple": tuple_row,
            "question_type": "DIRECT_READ",
            "quality": 1.0,
            "answer_source": "mechanical",
            "answer_type": "text_string",
            "expected_answer": "BUD LIGHT",
            "queried_text": None,
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            "query_anchor_label": "silver can",
            "query_anchor_synonyms": ["silver can"],
            "query_anchor_box": [10, 0, 110, 200],
            "query_anchor_local_phrase": "",
            "query_anchor_local_synonyms": [],
            "query_location_phrase": "center of the image",
            "query_location_synonyms": ["center of the image"],
            "query_specific_location_phrase": "",
            "query_specific_location_synonyms": [],
            "query_relation": "on",
            "query_location_required": False,
        }
        with patch.dict(
            os.environ,
            {
                "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "1",
                "SGOCR_CHEAP_AMBIGUITY_PROXY_REJECT_SCORE": "5",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "99",
            },
            clear=True,
        ):
            validation, summary, failure_reason = validate_candidate_output(
                candidate,
                {"question": "What text is on the silver can?", "answer": "BUD LIGHT", "question_type": "DIRECT_READ"},
            )
        self.assertFalse(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 0)
        self.assertEqual(failure_reason, "cheap_proxy_ambiguous")

    def test_sample_ids_are_unique_for_yesno_polarity(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "HELLO",
            "answer_normalized": "hello",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["HELLO"],
            "anchor_label": "sign",
            "anchor_synonyms": ["sign"],
            "location_phrase": "upper-left area of the image",
            "location_synonyms": ["upper-left area of the image", "near the top-left of the image"],
            "anchor_box": [0, 0, 90, 70],
            "anchor_score": 1.0,
            "anchor_category": "text_container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 1.0,
            "consensus_tier": "bootstrap_gt",
            "dataset_source": "textocr_bootstrap",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 2, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        other = dict(tuple_row)
        other["tuple_id"] = "img-1::word::b"
        other["ann_id"] = "b"
        other["answer"] = "WORLD"
        other["answer_normalized"] = "world"
        other["anchor_label"] = "poster"
        other["anchor_synonyms"] = ["poster"]
        other["location_phrase"] = "bottom-center area of the image"
        other["location_synonyms"] = ["bottom-center area of the image", "near the bottom of the image"]
        other["anchor_box"] = [40, 120, 180, 210]
        other["region_key"] = "lc"
        candidates = build_question_candidates(tuple_row, [tuple_row, other])
        yesno = [row for row in candidates if row["question_type"] == "YES_NO"]
        sample_ids = {sample_id_for_candidate({**row, "candidate_index": idx + 1}) for idx, row in enumerate(yesno)}
        self.assertEqual(len(sample_ids), len(yesno))
        self.assertTrue(any(row.get("yesno_polarity") == "negative" for row in yesno))

    def test_reverse_ground_validation_accepts_anchor_in_answer(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "HELLO",
            "answer_normalized": "hello",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["HELLO"],
            "anchor_label": "center of the image",
            "anchor_synonyms": ["center of the image", "middle of the image"],
            "anchor_box": [0, 0, 224, 224],
            "anchor_score": 1.0,
            "anchor_category": "scene_region",
            "relation": "in",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 1.0,
            "consensus_tier": "bootstrap_gt",
            "dataset_source": "textocr_bootstrap",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 2, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
            "question_type": "REVERSE_GROUND",
            "answer_source": "teacher_spatial",
            "answer_type": "spatial_phrase",
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "expected_answer": None,
            "queried_text": "HELLO",
        }
        candidate = {
            "candidate_id": "img-1::word::a::REVERSE_GROUND",
            "candidate_index": 1,
            "tuple": tuple_row,
            "question_type": "REVERSE_GROUND",
            "quality": 1.0,
            "answer_source": "teacher_spatial",
            "answer_type": "spatial_phrase",
            "expected_answer": None,
            "queried_text": "HELLO",
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
        }
        item = {
            "candidate_index": 1,
            "question_type": "REVERSE_GROUND",
            "question": "Where does the text HELLO appear in the image?",
            "answer": "in the center of the image",
        }
        validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

        row = normalize_candidate_result(candidate, item, {"provider": "gemini", "model": "gemini-2.5-flash", "parsed": {"items": [item]}, "usage": {}})
        self.assertEqual(row["sample_id"], "img-1__word__a__REVERSE_GROUND")
        tags = build_tags(row["tuple"])
        self.assertEqual(tags["question_type"], "REVERSE_GROUND")
        self.assertEqual(tags["answer_verified_by"], "teacher_only")

    def test_anchor_property_candidate_and_validation(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "PIZZA",
            "answer_normalized": "pizza",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["PIZZA"],
            "anchor_label": "box",
            "anchor_synonyms": ["box"],
            "anchor_box": [0, 0, 100, 60],
            "anchor_score": 0.82,
            "anchor_category": "container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 0.95,
            "consensus_tier": "strong_consensus",
            "dataset_source": "textocr_val",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 2, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        candidates = build_question_candidates(tuple_row, [tuple_row])
        anchor_property = next(
            row for row in candidates if row["question_type"] == "ANCHOR_PROPERTY" and row.get("anchor_property_type") == "anchor_color"
        )
        self.assertEqual(anchor_property["answer_source"], "teacher_visual")
        property_types = {row.get("anchor_property_type") for row in candidates if row["question_type"] == "ANCHOR_PROPERTY"}
        self.assertEqual(property_types, {"anchor_color", "anchor_material", "anchor_shape"})

        item = {
            "candidate_index": 1,
            "question_type": "ANCHOR_PROPERTY",
            "question": "What color is the box that has PIZZA on it?",
            "answer": "red",
        }
        candidate = {**anchor_property, "candidate_index": 1}
        validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_anchor_shape_circular_filter_suppresses_anchor_shape_candidate(self) -> None:
        """ANCHOR_PROPERTY anchor_shape must be dropped when the anchor_label already contains
        a shape token — e.g. 'blue rectangular bar' → asking 'what shape?' is circular."""
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "70",
            "answer_normalized": "70",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["70"],
            "anchor_label": "blue rectangular bar",
            "anchor_synonyms": ["blue rectangular bar"],
            "anchor_box": [0, 0, 180, 30],
            "anchor_score": 0.80,
            "anchor_category": "container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 0.92,
            "consensus_tier": "strong_consensus",
            "dataset_source": "chartqa",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 2, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 2,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        candidates = build_question_candidates(tuple_row, [tuple_row])
        ap_types = {c.get("anchor_property_type") for c in candidates if c["question_type"] == "ANCHOR_PROPERTY"}
        # anchor_shape must be absent — it would be circular given "rectangular" is in the label
        self.assertNotIn("anchor_shape", ap_types)
        # anchor_color and anchor_material should still be present
        self.assertIn("anchor_color", ap_types)
        self.assertIn("anchor_material", ap_types)

    def test_anchor_shape_not_filtered_for_non_shape_label(self) -> None:
        """ANCHOR_PROPERTY anchor_shape must remain when the anchor_label has no shape token."""
        tuple_row = {
            "tuple_id": "img-1::word::b",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "b",
            "answer": "PIZZA",
            "answer_normalized": "pizza",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["b"],
            "child_words": ["PIZZA"],
            "anchor_label": "box",
            "anchor_synonyms": ["box"],
            "anchor_box": [0, 0, 100, 60],
            "anchor_score": 0.82,
            "anchor_category": "container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 0.95,
            "consensus_tier": "strong_consensus",
            "dataset_source": "textocr_val",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 2, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        candidates = build_question_candidates(tuple_row, [tuple_row])
        ap_types = {c.get("anchor_property_type") for c in candidates if c["question_type"] == "ANCHOR_PROPERTY"}
        self.assertEqual(ap_types, {"anchor_color", "anchor_material", "anchor_shape"})

    def test_rg_structural_anchor_filter_suppresses_rg_for_color_shape_label(self) -> None:
        """REVERSE_GROUND must be suppressed when rg_structural_anchor_filter_enabled=1 and
        the anchor_label contains both a color and a structural shape token."""
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "99",
            "answer_normalized": "99",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["99"],
            "anchor_label": "blue rectangular bar",
            "anchor_synonyms": ["blue rectangular bar"],
            "anchor_box": [0, 0, 180, 30],
            "anchor_score": 0.80,
            "anchor_category": "container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 0.92,
            "consensus_tier": "strong_consensus",
            "dataset_source": "chartqa",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 2, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 2,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        with patch.dict(os.environ, {"SGOCR_RG_STRUCTURAL_ANCHOR_FILTER_ENABLED": "1"}, clear=True):
            candidates = build_question_candidates(tuple_row, [tuple_row])
        rg = [c for c in candidates if c["question_type"] == "REVERSE_GROUND"]
        self.assertEqual(rg, [], "REVERSE_GROUND should be suppressed for structural fallback anchor labels")

    def test_rg_structural_anchor_filter_off_by_default(self) -> None:
        """REVERSE_GROUND must not be suppressed when rg_structural_anchor_filter_enabled is off."""
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "99",
            "answer_normalized": "99",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["99"],
            "anchor_label": "blue rectangular bar",
            "anchor_synonyms": ["blue rectangular bar"],
            "anchor_box": [0, 0, 180, 30],
            "anchor_score": 0.80,
            "anchor_category": "container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 0.92,
            "consensus_tier": "strong_consensus",
            "dataset_source": "chartqa",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 2, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 2,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        with patch.dict(os.environ, {}, clear=True):
            candidates = build_question_candidates(tuple_row, [tuple_row])
        rg = [c for c in candidates if c["question_type"] == "REVERSE_GROUND"]
        self.assertEqual(len(rg), 1, "REVERSE_GROUND should be present when filter is off")

    def test_direct_read_high_ambiguity_requires_specific_location_not_only_anchor_local(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "HELLO",
            "answer_normalized": "hello",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["HELLO"],
            "anchor_label": "sign",
            "anchor_synonyms": ["sign"],
            "location_phrase": "center-left area of the image",
            "location_synonyms": ["center-left area of the image"],
            "specific_location_phrase": "upper text in the center-left area of the image",
            "specific_location_synonyms": ["upper text in the center-left area of the image"],
            "anchor_local_phrase": "on the left of the sign",
            "anchor_local_synonyms": ["on the left of the sign"],
            "anchor_box": [0, 0, 160, 120],
            "anchor_score": 1.0,
            "anchor_category": "text_container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": False,
            "answer_level": "word",
            "ocr_confidence": 1.0,
            "consensus_tier": "bootstrap_gt",
            "dataset_source": "textocr_bootstrap",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {
                "text_density": 6,
                "competing_tuples": 1,
                "coarse_region_competitors": 1,
                "bucket_competitors": 1,
                "anchor_overlap_competitors": 1,
                "anchor_conflict_count": 1,
                "local_text_cluster_shape": "grid",
                "local_text_bucket_occupancy": 2,
                "neighboring_text": [],
                "nearby_anchors": [],
            },
            "valid_text_count": 6,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "medium",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        candidate = {
            "candidate_id": "img-1::word::a::DIRECT_READ",
            "candidate_index": 1,
            "tuple": tuple_row,
            "question_type": "DIRECT_READ",
            "quality": 1.0,
            "answer_source": "mechanical",
            "answer_type": "text_string",
            "expected_answer": "HELLO",
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            "query_location_required": False,
        }
        item = {
            "candidate_index": 1,
            "question_type": "DIRECT_READ",
            "question": "What does the text on the left of the sign say?",
            "answer": "HELLO",
        }
        validation, _summary, failure = validate_candidate_output(candidate, item)
        self.assertFalse(validation["accepted"])
        self.assertEqual(failure, "direct_read_specific_location_missing")

    def test_unique_direct_read_skips_global_location_requirement(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "HELLO",
            "answer_normalized": "hello",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["HELLO"],
            "anchor_label": "red can",
            "anchor_synonyms": ["red can", "can"],
            "location_phrase": "center-left area of the image",
            "location_synonyms": ["center-left area of the image"],
            "specific_location_phrase": "upper text in the center-left area of the image",
            "specific_location_synonyms": ["upper text in the center-left area of the image"],
            "anchor_local_phrase": "on the red can",
            "anchor_local_synonyms": ["on the red can"],
            "anchor_box": [0, 0, 160, 120],
            "anchor_score": 1.0,
            "anchor_category": "container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 1.0,
            "consensus_tier": "bootstrap_gt",
            "dataset_source": "textocr_bootstrap",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {
                "text_density": 2,
                "competing_tuples": 0,
                "coarse_region_competitors": 0,
                "bucket_competitors": 0,
                "anchor_overlap_competitors": 0,
                "anchor_conflict_count": 0,
                "local_text_cluster_shape": "line",
                "local_text_bucket_occupancy": 1,
                "neighboring_text": [],
                "nearby_anchors": [],
            },
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        candidate = {
            "candidate_id": "img-1::word::a::DIRECT_READ",
            "candidate_index": 1,
            "tuple": tuple_row,
            "question_type": "DIRECT_READ",
            "quality": 1.0,
            "answer_source": "mechanical",
            "answer_type": "text_string",
            "expected_answer": "HELLO",
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            "query_location_required": True,
        }
        self.assertTrue(_unique_anchor_can_skip_global_location(candidate))
        item = {
            "candidate_index": 1,
            "question_type": "DIRECT_READ",
            "question": "What does the text on the red can say?",
            "answer": "HELLO",
        }
        validation, _summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertIsNone(failure)

    def test_text_property_visual_candidate_validation(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "PIZZA",
            "answer_normalized": "pizza",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["PIZZA"],
            "anchor_label": "box",
            "anchor_synonyms": ["box"],
            "anchor_box": [0, 0, 100, 60],
            "anchor_score": 0.82,
            "anchor_category": "container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 0.95,
            "consensus_tier": "strong_consensus",
            "dataset_source": "textocr_val",
            "region_key": "cc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 2, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        candidates = build_question_candidates(tuple_row, [tuple_row])
        visual = next(
            row for row in candidates if row["question_type"] == "TEXT_PROPERTY" and row.get("text_property_type") == "text_color"
        )
        item = {
            "candidate_index": 1,
            "question_type": "TEXT_PROPERTY",
            "question": "What color is the text that says PIZZA on the box?",
            "answer": "white",
        }
        candidate = {**visual, "candidate_index": 1}
        validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_rg_anchor_phrase_scrubs_color_when_enabled(self) -> None:
        candidate = {
            "question_type": "REVERSE_GROUND",
            "query_anchor_label": "blue sign",
            "query_anchor_synonyms": ["blue sign", "the blue sign"],
            "tuple": {
                "anchor_label": "blue sign",
                "anchor_synonyms": ["blue sign"],
            },
        }
        with patch.dict(os.environ, {"SGOCR_RG_SCRUB_COLOR_ANCHOR_PHRASES_ENABLED": "1"}, clear=True):
            self.assertEqual(candidate_anchor_label(candidate), "sign")
            self.assertEqual(candidate_anchor_phrases(candidate), ["sign", "the sign"])

    def test_text_property_prompt_uses_location_first_wording_when_enabled(self) -> None:
        candidate = self._make_rescued_specific_location_candidate(question_type="TEXT_PROPERTY", answer="PIZZA")
        candidate["text_property_type"] = "text_color"
        candidate["query_text_reference"] = "PIZZA"
        with patch.dict(
            os.environ,
            {"SGOCR_TP_VISUAL_AVOID_TEXT_REFERENCE_WITH_SPECIFIC_LOCATION_ENABLED": "1"},
            clear=True,
        ):
            prompt = build_batched_prompt([candidate])
        self.assertIn("visible color of the target text at this location", prompt)
        self.assertNotIn('that says "PIZZA"', prompt)

    def test_text_property_high_prior_answer_filter_rejects_generic_anchor(self) -> None:
        candidate = self._make_rescued_specific_location_candidate(question_type="TEXT_PROPERTY", answer="PIZZA")
        candidate["text_property_type"] = "text_color"
        candidate["query_anchor_label"] = "panel"
        candidate["query_anchor_synonyms"] = ["panel"]
        candidate["tuple"]["anchor_label"] = "panel"
        candidate["tuple"]["anchor_synonyms"] = ["panel"]
        candidate["query_location_required"] = False
        item = {
            "candidate_index": 1,
            "question_type": "TEXT_PROPERTY",
            "question": "What color is the text on the panel?",
            "answer": "white",
        }
        with patch.dict(os.environ, {"SGOCR_TP_VISUAL_HIGH_PRIOR_ANSWER_FILTER_ENABLED": "1"}, clear=True):
            validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertFalse(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 0)
        self.assertEqual(failure, "text_property_prior_leaky")

    def test_yesno_negative_uses_grounded_exclusion_location(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "MENU",
            "answer_normalized": "menu",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["MENU"],
            "anchor_label": "sign",
            "anchor_synonyms": ["sign"],
            "location_phrase": "upper-right area of the image",
            "location_synonyms": ["upper-right area of the image", "near the top-right of the image"],
            "anchor_box": [100, 0, 200, 80],
            "anchor_score": 0.82,
            "anchor_category": "text_container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 0.95,
            "consensus_tier": "strong_consensus",
            "dataset_source": "textocr_val",
            "region_key": "ur",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 3, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 3,
            "area_fraction": 0.03,
            "text_length": 4,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        other = dict(tuple_row)
        other["tuple_id"] = "img-1::word::b"
        other["ann_id"] = "b"
        other["answer"] = "EXIT"
        other["answer_normalized"] = "exit"
        other["anchor_label"] = "poster"
        other["anchor_synonyms"] = ["poster"]
        other["location_phrase"] = "bottom-center area of the image"
        other["location_synonyms"] = ["bottom-center area of the image", "near the bottom of the image"]
        other["anchor_box"] = [50, 140, 170, 220]
        candidates = build_question_candidates(tuple_row, [tuple_row, other])
        negative = next(row for row in candidates if row["question_type"] == "YES_NO" and row.get("yesno_polarity") == "negative")
        self.assertEqual(negative["queried_text"], "MENU")
        self.assertEqual(negative["yesno_distractor_source"], "grounded_exclusion")
        self.assertEqual(negative["query_anchor_label"], "poster")
        self.assertEqual(negative["query_location_phrase"], "bottom-center area of the image")

        item = {
            "candidate_index": 1,
            "question_type": "YES_NO",
            "question": "Is the text MENU on the poster near the bottom of the image?",
            "answer": "No",
        }
        validation, summary, failure = validate_candidate_output({**negative, "candidate_index": 1}, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_reverse_ground_accepts_location_only_answer(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "HELLO",
            "answer_normalized": "hello",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["HELLO"],
            "anchor_label": "sign",
            "anchor_synonyms": ["sign"],
            "location_phrase": "upper-right area of the image",
            "location_synonyms": ["upper-right area of the image", "near the top-right of the image"],
            "anchor_box": [100, 0, 200, 80],
            "anchor_score": 0.9,
            "anchor_category": "text_container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 1.0,
            "consensus_tier": "strong_consensus",
            "dataset_source": "textocr_val",
            "region_key": "ur",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 2, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 2,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        candidate = {
            "candidate_id": "img-1::word::a::REVERSE_GROUND",
            "candidate_index": 1,
            "tuple": tuple_row,
            "question_type": "REVERSE_GROUND",
            "quality": 1.0,
            "answer_source": "teacher_spatial",
            "answer_type": "spatial_phrase",
            "expected_answer": None,
            "queried_text": "HELLO",
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            "query_anchor_label": "sign",
            "query_anchor_synonyms": ["sign"],
            "query_anchor_box": [100, 0, 200, 80],
            "query_location_phrase": "upper-right area of the image",
            "query_location_synonyms": ["upper-right area of the image", "near the top-right of the image"],
            "query_relation": "on",
            "query_location_required": False,
        }
        item = {
            "candidate_index": 1,
            "question_type": "REVERSE_GROUND",
            "question": "Where does the text HELLO appear?",
            "answer": "near the top-right of the image",
        }
        validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_inline_frontier_prompt_and_schema(self) -> None:
        row = {
            "sample_id": "sample-1",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "question": "What does the text on the red can say?",
            "answer": "HELLO",
            "tags": {"question_type": "DIRECT_READ", "answer_type": "text_string"},
        }
        prompt = build_inline_frontier_prompt([row])
        self.assertIn("Candidate 1", prompt)
        self.assertIn("What does the text on the red can say?", prompt)
        schema = inline_frontier_response_schema(2)
        self.assertEqual(schema["properties"]["items"]["minItems"], 2)

    def test_direct_read_requires_coarse_location_when_competing(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "HELLO",
            "answer_normalized": "hello",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["HELLO"],
            "anchor_label": "sign",
            "anchor_synonyms": ["sign"],
            "location_phrase": "upper-right area of the image",
            "location_synonyms": ["upper-right area of the image", "upper right area of the image"],
            "anchor_box": [0, 0, 100, 60],
            "anchor_score": 0.82,
            "anchor_category": "text_container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": False,
            "answer_level": "word",
            "ocr_confidence": 0.95,
            "consensus_tier": "strong_consensus",
            "dataset_source": "textocr_val",
            "region_key": "ur",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {"text_density": 4, "competing_tuples": 2, "neighboring_text": [], "nearby_anchors": []},
            "valid_text_count": 4,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "low",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        candidate = {
            "candidate_id": "img-1::word::a::DIRECT_READ",
            "candidate_index": 1,
            "tuple": tuple_row,
            "question_type": "DIRECT_READ",
            "quality": 1.0,
            "answer_source": "mechanical",
            "answer_type": "text_string",
            "expected_answer": "HELLO",
            "queried_text": None,
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            "query_anchor_label": "sign",
            "query_anchor_synonyms": ["sign"],
            "query_anchor_box": [0, 0, 100, 60],
            "query_location_phrase": "upper-right area of the image",
            "query_location_synonyms": ["upper-right area of the image", "upper right area of the image"],
            "query_relation": "on",
            "query_location_required": True,
        }
        bad_item = {
            "candidate_index": 1,
            "question_type": "DIRECT_READ",
            "question": "What text is on the sign?",
            "answer": "HELLO",
        }
        validation, _, failure = validate_candidate_output(candidate, bad_item)
        self.assertFalse(validation["accepted"])
        self.assertEqual(failure, "direct_read_location_missing")

        good_item = {
            "candidate_index": 1,
            "question_type": "DIRECT_READ",
            "question": "What text is on the sign in the upper-right area of the image?",
            "answer": "HELLO",
        }
        validation, summary, failure = validate_candidate_output(candidate, good_item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_high_ambiguity_requires_stronger_disambiguation(self) -> None:
        tuple_row = {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "MENU",
            "answer_normalized": "menu",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["MENU"],
            "anchor_label": "document",
            "anchor_synonyms": ["document"],
            "location_phrase": "center-left area of the image",
            "location_synonyms": ["center-left area of the image"],
            "specific_location_phrase": "upper-left quadrant text",
            "specific_location_synonyms": ["upper-left quadrant text"],
            "anchor_local_phrase": "",
            "anchor_local_synonyms": [],
            "anchor_box": [0, 0, 160, 200],
            "anchor_score": 0.82,
            "anchor_category": "text_container",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 0.96,
            "consensus_tier": "strong_consensus",
            "dataset_source": "textocr_val",
            "region_key": "cl",
            "resolvable": True,
            "resolvability": {"text_px_w": 48.0, "text_px_h": 18.0},
            "kd_metadata": {
                "text_density": 6,
                "competing_tuples": 2,
                "coarse_region_competitors": 2,
                "bucket_competitors": 2,
                "anchor_overlap_competitors": 1,
                "anchor_conflict_count": 1,
                "local_text_bucket_key": "upper_left",
                "local_text_bucket_occupancy": 2,
                "local_text_cluster_size": 4,
                "local_text_cluster_shape": "grid",
                "local_text_cluster_unresolvable": 2,
                "nearby_anchors": [{"label": "document"}],
            },
            "valid_text_count": 6,
            "area_fraction": 0.03,
            "text_length": 4,
            "density_bucket": "medium",
            "area_bucket": "medium",
            "group_kind": "word",
        }
        self.assertGreaterEqual(location_ambiguity_score(tuple_row), 5)
        candidate = {
            "candidate_id": "img-1::word::a::DIRECT_READ",
            "candidate_index": 1,
            "tuple": tuple_row,
            "question_type": "DIRECT_READ",
            "quality": 1.0,
            "answer_source": "mechanical",
            "answer_type": "text_string",
            "expected_answer": "MENU",
            "queried_text": None,
            "yesno_polarity": None,
            "yesno_distractor_source": None,
            "text_property_type": None,
            "anchor_property_type": None,
            "query_anchor_label": "document",
            "query_anchor_synonyms": ["document"],
            "query_anchor_box": [0, 0, 160, 200],
            "query_anchor_local_phrase": "",
            "query_anchor_local_synonyms": [],
            "query_location_phrase": "center-left area of the image",
            "query_location_synonyms": ["center-left area of the image"],
            "query_specific_location_phrase": "upper-left quadrant text",
            "query_specific_location_synonyms": ["upper-left quadrant text"],
            "query_relation": "on",
            "query_location_required": False,
        }
        item = {
            "candidate_index": 1,
            "question_type": "DIRECT_READ",
            "question": "What does the document in the center-left area of the image say?",
            "answer": "MENU",
        }
        validation, _, failure = validate_candidate_output(candidate, item)
        self.assertFalse(validation["accepted"])
        self.assertEqual(failure, "direct_read_specific_location_missing")

    def test_direct_read_specific_location_rescues_high_ambiguity(self) -> None:
        candidate = self._make_rescued_specific_location_candidate(question_type="DIRECT_READ", answer="HELLO")
        item = {
            "candidate_index": 1,
            "question_type": "DIRECT_READ",
            "question": "What does the lower text in the center of the image on the box say?",
            "answer": "HELLO",
        }
        validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_yesno_specific_location_can_pass_without_anchor_phrase(self) -> None:
        candidate = self._make_rescued_specific_location_candidate(question_type="YES_NO", answer="HELLO")
        item = {
            "candidate_index": 1,
            "question_type": "YES_NO",
            "question": 'Does the lower text in the center of the image say "HELLO"?',
            "answer": "Yes",
        }
        validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_text_property_specific_location_can_pass_without_anchor_phrase(self) -> None:
        candidate = self._make_rescued_specific_location_candidate(question_type="TEXT_PROPERTY", answer="HELLO")
        item = {
            "candidate_index": 1,
            "question_type": "TEXT_PROPERTY",
            "question": 'What is the orientation of the lower text in the center of the image that says "HELLO"?',
            "answer": "horizontal",
        }
        validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_anchor_instance_disambiguation_uses_color_when_enabled(self) -> None:
        candidate = self._make_rescued_specific_location_candidate(question_type="DIRECT_READ", answer="HELLO")
        candidate["tuple"]["anchor_label"] = "jersey"
        candidate["tuple"]["anchor_color"] = "blue"
        candidate["tuple"]["kd_metadata"]["anchor_label_competitors"] = 2
        candidate["tuple"]["kd_metadata"]["competing_tuples"] = 0
        candidate["tuple"]["kd_metadata"]["coarse_region_competitors"] = 0
        candidate["tuple"]["kd_metadata"]["bucket_competitors"] = 0
        candidate["tuple"]["kd_metadata"]["anchor_overlap_competitors"] = 0
        candidate["tuple"]["kd_metadata"]["local_text_bucket_occupancy"] = 1
        candidate["tuple"]["kd_metadata"]["local_text_cluster_shape"] = "line"
        candidate["query_specific_location_phrase"] = ""
        candidate["query_specific_location_synonyms"] = []
        candidate["query_anchor_label"] = "jersey"
        candidate["query_anchor_synonyms"] = ["jersey"]
        candidate["query_anchor_color"] = "blue"
        candidate["query_location_required"] = False
        item = {
            "candidate_index": 1,
            "question_type": "DIRECT_READ",
            "question": "What does the text on the blue jersey say?",
            "answer": "HELLO",
        }
        with patch.dict(
            os.environ,
            {
                "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
                "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
            },
            clear=True,
        ):
            validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_anchor_instance_disambiguation_is_disabled_by_default(self) -> None:
        candidate = self._make_rescued_specific_location_candidate(question_type="DIRECT_READ", answer="HELLO")
        candidate["tuple"]["anchor_label"] = "jersey"
        candidate["tuple"]["anchor_color"] = "blue"
        candidate["tuple"]["kd_metadata"]["anchor_label_competitors"] = 2
        candidate["tuple"]["kd_metadata"]["competing_tuples"] = 0
        candidate["tuple"]["kd_metadata"]["coarse_region_competitors"] = 0
        candidate["tuple"]["kd_metadata"]["bucket_competitors"] = 0
        candidate["tuple"]["kd_metadata"]["anchor_overlap_competitors"] = 0
        candidate["tuple"]["kd_metadata"]["local_text_bucket_occupancy"] = 1
        candidate["tuple"]["kd_metadata"]["local_text_cluster_shape"] = "line"
        candidate["query_specific_location_phrase"] = ""
        candidate["query_specific_location_synonyms"] = []
        candidate["query_anchor_label"] = "jersey"
        candidate["query_anchor_synonyms"] = ["jersey"]
        candidate["query_anchor_color"] = "blue"
        candidate["query_location_required"] = False
        item = {
            "candidate_index": 1,
            "question_type": "DIRECT_READ",
            "question": "What does the text on the jersey say?",
            "answer": "HELLO",
        }
        with patch.dict(os.environ, {}, clear=True):
            validation, summary, failure = validate_candidate_output(candidate, item)
        self.assertTrue(validation["accepted"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertIsNone(failure)

    def test_answer_probe_policy_rejects_strong_direct_read_ambiguity(self) -> None:
        row = {
            "tuple": {"question_type": "DIRECT_READ"},
            "validations": [{"accepted": True, "mechanical_ok": True}],
            "summary": {"generated_count": 1, "accepted_count": 1},
            "answer_probe": {
                "enabled": True,
                "ambiguous": True,
                "distinct_count": 3,
                "plurality_count": 1,
            },
        }
        with patch.dict(os.environ, {"SGOCR_TEACHER_ANSWER_PROBE_COUNT": "3"}, clear=True):
            from sgocr.semantic_dev40_tuning import load_semantic_dev40_tuning

            apply_answer_probe_policy([row], tuning=load_semantic_dev40_tuning())
        self.assertEqual(row["failure_reason"], "answer_probe_ambiguous")
        self.assertEqual(row["summary"]["accepted_count"], 0)
        self.assertFalse(row["validations"][0]["accepted"])


class TestAnswerLeakageFilter(unittest.TestCase):
    def _make_word_count_candidate(self, answer: str, question: str) -> dict:
        return {
            "question_type": "TEXT_PROPERTY",
            "text_property_type": "word_count",
            "anchor_property_type": None,
            "expected_answer": str(len(answer.split())),
            "queried_text": None,
            "query_anchor_label": "sign",
            "query_anchor_synonyms": ["sign"],
            "tuple": {"answer": answer},
        }

    def _make_anchor_color_candidate(self, anchor_label: str, expected_color: str) -> dict:
        return {
            "question_type": "ANCHOR_PROPERTY",
            "anchor_property_type": "anchor_color",
            "text_property_type": None,
            "expected_answer": expected_color,
            "queried_text": None,
            "query_anchor_label": anchor_label,
            "query_anchor_synonyms": [anchor_label],
            "tuple": {"answer": "HELLO", "anchor_label": anchor_label},
        }

    def test_word_count_leaky_when_full_text_quoted(self) -> None:
        c = self._make_word_count_candidate("the car is here", "How many words are in the text 'the car is here' on the sign?")
        self.assertTrue(candidate_has_answer_leakage(c, "how many words are in the text the car is here on the sign"))

    def test_word_count_not_leaky_when_text_not_quoted(self) -> None:
        c = self._make_word_count_candidate("the car is here", "How many words are in the text on the sign?")
        self.assertFalse(candidate_has_answer_leakage(c, "how many words are in the text on the sign"))

    def test_anchor_color_leaky_when_color_in_label(self) -> None:
        c = self._make_anchor_color_candidate("brown bottle", "brown")
        self.assertTrue(candidate_has_answer_leakage(c, "what color is the brown bottle that has hello on it"))

    def test_anchor_color_not_leaky_generic_label(self) -> None:
        c = self._make_anchor_color_candidate("bottle", "brown")
        self.assertFalse(candidate_has_answer_leakage(c, "what color is the bottle that has hello on it"))

    def test_anchor_color_not_leaky_different_color(self) -> None:
        # anchor_label has "blue" but expected answer is "red" — not leaky
        c = self._make_anchor_color_candidate("blue sign", "red")
        self.assertFalse(candidate_has_answer_leakage(c, "what color is the blue sign that has hello on it"))

    def test_direct_read_never_leaky(self) -> None:
        c = {
            "question_type": "DIRECT_READ",
            "anchor_property_type": None,
            "text_property_type": None,
            "expected_answer": "OPEN",
            "query_anchor_label": "sign",
            "query_anchor_synonyms": ["sign"],
            "tuple": {"answer": "OPEN", "anchor_label": "sign"},
        }
        self.assertFalse(candidate_has_answer_leakage(c, "what text is on the sign near the top of the image"))


class TestAnchorCentroidOffset(unittest.TestCase):
    def _make_tuple_row(self, anchor_box: list, image_w: int = 200, image_h: int = 200) -> dict:
        return {
            "anchor_box": anchor_box,
            "image_width": image_w,
            "image_height": image_h,
        }

    def test_centered_anchor_has_small_offset(self) -> None:
        row = self._make_tuple_row([80, 80, 120, 120])  # centered in 200x200
        self.assertAlmostEqual(_anchor_centroid_offset(row), 0.0, places=5)

    def test_corner_anchor_has_large_offset(self) -> None:
        row = self._make_tuple_row([0, 0, 20, 20])  # top-left corner
        offset = _anchor_centroid_offset(row)
        self.assertGreater(offset, 0.4)

    def test_side_anchor_has_medium_offset(self) -> None:
        row = self._make_tuple_row([0, 80, 20, 120])  # left edge, centered vertically
        offset = _anchor_centroid_offset(row)
        self.assertAlmostEqual(offset, 0.45, places=2)

    def test_requires_specific_location_suppressed_when_centered(self) -> None:
        tuple_row = {
            "image_width": 200,
            "image_height": 200,
            "anchor_box": [80, 80, 120, 120],  # dead center
            "specific_location_phrase": "lower text in the center of the image",
            "specific_location_synonyms": ["lower text in the center of the image"],
            "kd_metadata": {
                "competing_tuples": 1,
                "coarse_region_competitors": 1,
                "bucket_competitors": 1,
                "anchor_overlap_competitors": 1,
                "anchor_conflict_count": 0,
                "local_text_cluster_size": 1,
                "local_text_cluster_shape": "",
                "local_text_bucket_occupancy": 1,
                "local_text_cluster_unresolvable": 0,
                "nearby_anchors": [],
            },
        }
        with patch.dict(os.environ, {"SGOCR_SPATIAL_MIN_CENTROID_OFFSET": "0.10"}, clear=True):
            result = requires_specific_location(tuple_row, "DIRECT_READ")
        self.assertFalse(result)

    def test_requires_specific_location_not_suppressed_when_off_center(self) -> None:
        tuple_row = {
            "image_width": 200,
            "image_height": 200,
            "anchor_box": [0, 0, 20, 20],  # top-left corner, clearly off center
            "specific_location_phrase": "upper left of the image",
            "specific_location_synonyms": ["upper left of the image"],
            "kd_metadata": {
                "competing_tuples": 1,
                "coarse_region_competitors": 1,
                "bucket_competitors": 1,
                "anchor_overlap_competitors": 1,
                "anchor_conflict_count": 0,
                "local_text_cluster_size": 1,
                "local_text_cluster_shape": "",
                "local_text_bucket_occupancy": 1,
                "local_text_cluster_unresolvable": 0,
                "nearby_anchors": [],
            },
        }
        with patch.dict(os.environ, {"SGOCR_SPATIAL_MIN_CENTROID_OFFSET": "0.10"}, clear=True):
            result = requires_specific_location(tuple_row, "DIRECT_READ")
        self.assertTrue(result)


class TestAnchorColorMechanical(unittest.TestCase):
    def _base_tuple_row(self, anchor_label: str) -> dict:
        return {
            "tuple_id": "img-1::word::a",
            "image_id": "img-1",
            "image_path": "data/fake.jpg",
            "image_width": 224,
            "image_height": 224,
            "ann_id": "a",
            "answer": "HELLO",
            "answer_normalized": "hello",
            "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
            "text_bbox": [10, 10, 70, 20],
            "text_node_ids": ["a"],
            "child_words": ["HELLO"],
            "anchor_label": anchor_label,
            "anchor_synonyms": [anchor_label],
            "anchor_color": None,
            "anchor_material": None,
            "anchor_shape": None,
            "location_phrase": "left of the image",
            "location_synonyms": ["left of the image"],
            "specific_location_phrase": "",
            "specific_location_synonyms": [],
            "anchor_local_phrase": "",
            "anchor_local_synonyms": [],
            "anchor_box": [0, 50, 40, 100],
            "anchor_score": 0.75,
            "anchor_category": "clothing",
            "relation": "on",
            "ref_label": None,
            "ref_box": None,
            "unique": True,
            "answer_level": "word",
            "ocr_confidence": 0.92,
            "consensus_tier": "nemotron_v2",
            "dataset_source": "coco_text_train",
            "region_key": "lc",
            "resolvable": True,
            "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
            "kd_metadata": {
                "text_density": 3,
                "competing_tuples": 0,
                "coarse_region_competitors": 0,
                "bucket_competitors": 0,
                "anchor_overlap_competitors": 0,
                "anchor_conflict_count": 0,
                "local_text_bucket_occupancy": 1,
                "local_text_cluster_shape": "",
                "neighboring_text": [],
                "nearby_anchors": [],
                "anchor_label_competitors": 0,
                "same_anchor_text_count": 1,
                "same_anchor_same_answer_nonoverlap_instances": 1,
                "same_anchor_distinct_answer_count": 1,
            },
            "valid_text_count": 3,
            "area_fraction": 0.03,
            "text_length": 5,
            "density_bucket": "low",
            "area_bucket": "small",
            "group_kind": "word",
            "question_type": "ANCHOR_PROPERTY",
            "answer_source": "teacher_visual",
            "answer_level": "word",
        }

    def test_mechanical_color_set_when_color_in_label(self) -> None:
        tuple_row = self._base_tuple_row("brown bottle")
        answer_units: list = []
        with patch.dict(os.environ, {"SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1"}, clear=True):
            candidates = build_question_candidates(tuple_row, answer_units)
        ap_candidates = [c for c in candidates if c["question_type"] == "ANCHOR_PROPERTY" and c.get("anchor_property_type") == "anchor_color"]
        self.assertTrue(len(ap_candidates) >= 1)
        c = ap_candidates[0]
        self.assertEqual(c["answer_source"], "mechanical_color")
        self.assertEqual(c["expected_answer"], "brown")

    def test_no_mechanical_color_when_label_has_no_color(self) -> None:
        tuple_row = self._base_tuple_row("bottle")
        answer_units: list = []
        with patch.dict(os.environ, {"SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1"}, clear=True):
            candidates = build_question_candidates(tuple_row, answer_units)
        ap_candidates = [c for c in candidates if c["question_type"] == "ANCHOR_PROPERTY" and c.get("anchor_property_type") == "anchor_color"]
        if ap_candidates:
            self.assertNotEqual(ap_candidates[0]["answer_source"], "mechanical_color")

    def test_no_mechanical_color_when_flag_off(self) -> None:
        tuple_row = self._base_tuple_row("brown bottle")
        answer_units: list = []
        with patch.dict(os.environ, {"SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "0"}, clear=True):
            candidates = build_question_candidates(tuple_row, answer_units)
        ap_candidates = [c for c in candidates if c["question_type"] == "ANCHOR_PROPERTY" and c.get("anchor_property_type") == "anchor_color"]
        if ap_candidates:
            self.assertNotEqual(ap_candidates[0]["answer_source"], "mechanical_color")


def _make_candidate_with_centroid(anchor_box: list, image_w: int = 224, image_h: int = 224, *, qtype: str = "DIRECT_READ") -> dict:
    """Build a minimal candidate dict for centroid filter tests."""
    tuple_row = {
        "tuple_id": f"img-1::word::n_{anchor_box[0]}",
        "image_id": "img-1",
        "image_path": "data/fake.jpg",
        "image_width": image_w,
        "image_height": image_h,
        "ann_id": "n",
        "answer": "HELLO",
        "answer_normalized": "hello",
        "text_polygon": [10, 10, 80, 10, 80, 30, 10, 30],
        "text_bbox": [10, 10, 70, 20],
        "text_node_ids": [f"n_{anchor_box[0]}"],
        "child_words": ["HELLO"],
        "anchor_label": "sign",
        "anchor_synonyms": ["sign"],
        "anchor_box": anchor_box,
        "anchor_score": 1.0,
        "anchor_category": "sign",
        "relation": "on",
        "ref_label": None,
        "ref_box": None,
        "unique": True,
        "answer_level": "word",
        "ocr_confidence": 1.0,
        "consensus_tier": "bootstrap_gt",
        "dataset_source": "textocr_bootstrap",
        "region_key": "cc",
        "resolvable": True,
        "resolvability": {"text_px_w": 70.0, "text_px_h": 20.0},
        "kd_metadata": {"text_density": 1, "competing_tuples": 0, "neighboring_text": [], "nearby_anchors": []},
        "valid_text_count": 2,
        "area_fraction": 0.03,
        "text_length": 5,
        "density_bucket": "low",
        "area_bucket": "medium",
        "group_kind": "word",
        "anchor_local_clean": None,
        "anchor_local_synonyms": [],
        "location_synonyms": ["in the image"],
        "specific_location_synonyms": [],
        "location_phrase": "in the image",
    }
    return {
        "candidate_id": f"img-1::word::n_{anchor_box[0]}::{qtype}",
        "tuple": tuple_row,
        "question_type": qtype,
        "quality": 1.0,
        "answer_source": "mechanical",
        "answer_type": "text_string",
        "expected_answer": "HELLO",
        "queried_text": None,
        "yesno_polarity": None,
        "yesno_distractor_source": None,
        "text_property_type": None,
        "anchor_property_type": None,
        "query_location_required": False,
    }


class TestUpstreamCentroidFilter(unittest.TestCase):
    """Test that select_candidates filters by centroid when upstream_centroid_filter_enabled=1."""

    def _make_candidates(self) -> list:
        # Centered anchor: cx=112, cy=112 in 224x224 → offset=0.0
        center = _make_candidate_with_centroid([72, 72, 152, 152])  # centered, offset ~0.0
        # Off-center anchor: cx=20, cy=20 in 224x224 → offset ≈ 0.41
        corner = _make_candidate_with_centroid([0, 0, 40, 40])
        return [center, corner]

    def test_no_filter_when_disabled(self) -> None:
        candidates = self._make_candidates()
        env = {"SGOCR_UPSTREAM_CENTROID_FILTER_ENABLED": "0", "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": "0.15"}
        with patch.dict(os.environ, env, clear=True):
            result = select_candidates(candidates, target_count=10)
        self.assertEqual(len(result), 2)

    def test_filter_removes_centered_anchor(self) -> None:
        candidates = self._make_candidates()
        env = {"SGOCR_UPSTREAM_CENTROID_FILTER_ENABLED": "1", "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": "0.15"}
        with patch.dict(os.environ, env, clear=True):
            result = select_candidates(candidates, target_count=10)
        # Centered anchor should be removed; only off-center remains
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(_anchor_centroid_offset(result[0]["tuple"]), 0.41, delta=0.02)

    def test_filter_keeps_all_when_threshold_zero(self) -> None:
        candidates = self._make_candidates()
        env = {"SGOCR_UPSTREAM_CENTROID_FILTER_ENABLED": "1", "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": "0.0"}
        with patch.dict(os.environ, env, clear=True):
            result = select_candidates(candidates, target_count=10)
        self.assertEqual(len(result), 2)

    def test_filter_returns_empty_if_all_central(self) -> None:
        candidates = [_make_candidate_with_centroid([72, 72, 152, 152])]
        env = {"SGOCR_UPSTREAM_CENTROID_FILTER_ENABLED": "1", "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": "0.30"}
        with patch.dict(os.environ, env, clear=True):
            result = select_candidates(candidates, target_count=10)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
