from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from sgocr.bootstrap_kd import compute_resolvability
from sgocr.full_pipeline_dev40 import _filter_subsumed_rows, build_merged_sign_tuples_for_image


def make_text_node(
    node_id: str,
    text: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    resolvable: bool = True,
    image_id: str = "img-merge",
    image_w: int = 256,
    image_h: int = 256,
) -> dict:
    polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
    resolvability = compute_resolvability(polygon, image_w, image_h)
    resolvability["passes"] = bool(resolvable)
    return {
        "image_id": image_id,
        "node_id": node_id,
        "text": text,
        "polygon": polygon,
        "bbox": [x1, y1, x2, y2],
        "image_width": image_w,
        "image_height": image_h,
        "confidence": 0.98,
        "consensus_tier": "strong_consensus",
        "region_key": "cc",
        "resolvable": resolvable,
        "resolvability": resolvability,
    }


def make_anchor(label: str = "album cover") -> dict:
    return {
        "label": label,
        "box": [24.0, 52.0, 192.0, 212.0],
        "score": 0.92,
        "source": "sam3",
        "relevance": 0.88,
        "selection_score": 0.94,
        "caption": "album cover with stacked names",
        "support_count": 2,
        "source_support_count": 2,
        "support_labels": [label, "cover"],
        "alternate_labels": [label, "cover"],
        "label_conflict_count": 0,
        "cluster_size": 1,
        "relation": "on",
    }


class TestFullPipelineDev40(unittest.TestCase):
    def test_merge_stage_groups_stacked_names_and_ignores_remote_noise(self) -> None:
        nodes = [
            make_text_node("a", "Lazar Berman", 42, 62, 168, 86),
            make_text_node("b", "Chopin", 48, 95, 154, 118),
            make_text_node("c", "Liszt", 56, 128, 146, 151),
            make_text_node("d", "Kazhlaev", 44, 161, 162, 185),
        ]
        all_nodes = [
            make_text_node("top", "EMI", 36, 18, 80, 34, resolvable=False),
            *nodes,
            make_text_node("bottom", "STEREO", 68, 226, 150, 242, resolvable=False),
        ]
        best_anchor_by_node = {(str(node["image_id"]), str(node["node_id"])): make_anchor() for node in nodes}
        with patch.dict(os.environ, {"SGOCR_TEXT_MERGE_ENABLED": "1"}, clear=True):
            merged, debug = build_merged_sign_tuples_for_image(
                {"image_id": "img-merge", "image_path": "data/fake.jpg"},
                nodes,
                all_nodes,
                best_anchor_by_node,
                [make_anchor()],
                {"img-merge": "textocr_val"},
            )
        self.assertEqual(debug["merge_groups"], 1)
        self.assertEqual(len(merged), 1)
        row = merged[0]
        self.assertEqual(row["child_words"], ["Lazar Berman", "Chopin", "Liszt", "Kazhlaev"])
        self.assertEqual(row["group_kind"], "semantic_merge_group")
        self.assertEqual(row["semantic_debug"]["merge_debug"]["merged_node_count"], 4)
        self.assertEqual(row["semantic_debug"]["merge_debug"]["merged_unresolvable_count"], 0)

    def test_merge_groups_subsume_smaller_child_rows(self) -> None:
        merge_row = {
            "tuple_id": "img-merge::merge::1",
            "image_id": "img-merge",
            "relation": "on",
            "group_kind": "semantic_merge_group",
            "text_node_ids": ["a", "b", "c"],
            "anchor_box": [24.0, 52.0, 192.0, 212.0],
        }
        child_row = {
            "tuple_id": "img-merge::word::a",
            "image_id": "img-merge",
            "relation": "on",
            "group_kind": "word",
            "text_node_ids": ["a"],
            "anchor_box": [30.0, 60.0, 188.0, 206.0],
        }
        pair_row = {
            "tuple_id": "img-merge::sign::a_b",
            "image_id": "img-merge",
            "relation": "on",
            "group_kind": "semantic_anchor_group",
            "text_node_ids": ["a", "b"],
            "anchor_box": [24.0, 52.0, 192.0, 212.0],
        }
        unrelated_row = {
            "tuple_id": "img-merge::word::z",
            "image_id": "img-merge",
            "relation": "on",
            "group_kind": "word",
            "text_node_ids": ["z"],
            "anchor_box": [200.0, 10.0, 230.0, 30.0],
        }
        with patch.dict(os.environ, {"SGOCR_TEXT_MERGE_ENABLED": "1", "SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES": "3"}, clear=True):
            kept, removed = _filter_subsumed_rows([child_row, pair_row, merge_row, unrelated_row])
        self.assertEqual(removed, 2)
        kept_ids = {row["tuple_id"] for row in kept}
        self.assertEqual(kept_ids, {"img-merge::merge::1", "img-merge::word::z"})


if __name__ == "__main__":
    unittest.main()
