from __future__ import annotations

import unittest

from sgocr.bootstrap_kd import bootstrap_anchor_boxes, compute_resolvability
from sgocr.dev40_complete import (
    _unique_anchor_can_skip_global_location,
    build_question_candidates,
    build_inline_frontier_prompt,
    inline_frontier_response_schema,
    build_sign_tuples,
    build_tags,
    location_ambiguity_score,
    normalize_candidate_result,
    sample_id_for_candidate,
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


if __name__ == "__main__":
    unittest.main()
