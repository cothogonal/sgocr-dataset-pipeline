from __future__ import annotations

import unittest

from sgocr.semantic_grounding import (
    annotate_anchor_candidate_support,
    anchor_local_location_metadata,
    anchor_relevance,
    categorize_anchor,
    consolidate_anchor_candidates,
    expand_grounding_tags_for_node,
    extract_caption_tags,
    is_generic_anchor_label,
    relation_between_text_and_anchor,
    sanitize_anchor_label,
    sanitize_anchor_tags,
)


class TestSemanticGrounding(unittest.TestCase):
    def test_extract_caption_tags_prefers_semantic_terms(self) -> None:
        tags = extract_caption_tags("a woman holding up a box of cell phones")
        self.assertIn("box", tags)
        self.assertIn("phone", tags)

    def test_categorize_anchor_maps_known_labels(self) -> None:
        self.assertEqual(categorize_anchor("awning"), "building")
        self.assertEqual(categorize_anchor("box"), "container")
        self.assertEqual(categorize_anchor("poster"), "text_container")

    def test_sanitize_anchor_label_strips_editorial_caption(self) -> None:
        self.assertEqual(
            sanitize_anchor_label("whiteboard with handwritten notes about different types of physical therapy"),
            "whiteboard",
        )
        self.assertEqual(sanitize_anchor_label("sign-covered wall behind the player"), "sign wall")
        self.assertIsNone(sanitize_anchor_label("No object detected"))

    def test_sanitize_anchor_label_preserves_visible_color(self) -> None:
        self.assertEqual(sanitize_anchor_label("bright red can on a shelf"), "bright red can")
        self.assertEqual(sanitize_anchor_label("blue jersey with numbers"), "blue jersey")

    def test_is_generic_anchor_label_flags_text_container_fallbacks(self) -> None:
        self.assertTrue(is_generic_anchor_label("sign wall"))
        self.assertTrue(is_generic_anchor_label("display panel"))
        self.assertFalse(is_generic_anchor_label("red airplane tail"))
        self.assertFalse(is_generic_anchor_label("silver car door"))

    def test_sanitize_anchor_tags_prefers_safe_nouns(self) -> None:
        tags = sanitize_anchor_tags(
            "pellegrino sparkling mineral water bottle",
            "guinness beer bottle",
            "label",
        )
        self.assertIn("bottle", tags)
        self.assertNotIn("pellegrino sparkling mineral water bottle", tags)

    def test_relation_between_text_and_anchor(self) -> None:
        image_size = (640, 480)
        self.assertEqual(relation_between_text_and_anchor([10, 10, 40, 30], [0, 0, 80, 60], image_size), "on")
        self.assertEqual(relation_between_text_and_anchor([10, 10, 40, 30], [10, 80, 50, 140], image_size), "above")
        self.assertEqual(relation_between_text_and_anchor([120, 10, 160, 40], [20, 10, 80, 40], image_size), "right_of")

    def test_anchor_relevance_penalizes_full_image_boxes(self) -> None:
        image_size = (640, 480)
        text_box = [120, 120, 220, 170]
        text_polygon = [120, 120, 220, 120, 220, 170, 120, 170]
        local_anchor = [100, 100, 260, 210]
        full_anchor = [0, 0, 640, 480]
        self.assertGreater(
            anchor_relevance(text_box, text_polygon, local_anchor, image_size),
            anchor_relevance(text_box, text_polygon, full_anchor, image_size),
        )

    def test_anchor_local_location_metadata_prefers_clean_anchor_scopes(self) -> None:
        meta = anchor_local_location_metadata([300, 120, 360, 170], [100, 80, 400, 240], "on", "car")
        self.assertTrue(meta["clean"])
        self.assertIn("car", meta["phrase"])
        self.assertTrue(any("right" in item for item in meta["synonyms"]))

    def test_anchor_local_location_metadata_supports_corner_scopes(self) -> None:
        meta = anchor_local_location_metadata([105, 190, 150, 230], [100, 80, 400, 240], "on", "car")
        self.assertTrue(meta["clean"])
        self.assertEqual(meta["mode"], "corner")
        self.assertIn("bottom left", meta["phrase"])

    def test_expand_grounding_tags_for_dense_wall_text(self) -> None:
        node = {"node_id": "a", "bbox": [10, 10, 50, 30]}
        image_nodes = [
            {"node_id": "a", "bbox": [10, 10, 50, 30]},
            {"node_id": "b", "bbox": [55, 12, 95, 32]},
            {"node_id": "c", "bbox": [100, 14, 145, 34]},
        ]
        expanded = expand_grounding_tags_for_node(
            base_tags=["wall", "poster"],
            node=node,
            image_nodes=image_nodes,
            image_size=(300, 200),
            mode="supportive",
        )
        self.assertIn("sign wall", expanded)

    def test_annotate_anchor_candidate_support_counts_overlap(self) -> None:
        rows = annotate_anchor_candidate_support(
            [
                {"label": "wall", "box": [10, 10, 100, 100], "source": "grounding_dino", "relation": "on"},
                {"label": "sign wall", "box": [12, 12, 102, 102], "source": "florence_region", "relation": "on"},
                {"label": "bottle", "box": [140, 10, 180, 90], "source": "grounding_dino", "relation": "on"},
            ]
        )
        wall = rows[0]
        self.assertGreaterEqual(wall["support_count"], 2)
        self.assertGreaterEqual(wall["source_support_count"], 2)

    def test_consolidate_anchor_candidates_merges_conflicting_labels(self) -> None:
        rows = consolidate_anchor_candidates(
            [
                {"label": "door", "box": [40, 40, 140, 220], "source": "grounding_dino", "relation": "on", "relevance": 0.61, "score": 0.66, "support_count": 1},
                {"label": "can", "box": [42, 42, 142, 222], "source": "grounding_dino", "relation": "on", "relevance": 0.64, "score": 0.68, "support_count": 2},
                {"label": "can", "box": [43, 43, 141, 221], "source": "florence_region", "relation": "on", "relevance": 0.62, "score": 0.60, "support_count": 2},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], "can")
        self.assertGreaterEqual(rows[0]["label_conflict_count"], 1)
        self.assertIn("door", rows[0]["alternate_labels"])


if __name__ == "__main__":
    unittest.main()
