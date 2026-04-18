from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from sgocr.bootstrap_kd import compute_resolvability
from sgocr.full_pipeline_dev40 import (
    _filter_subsumed_rows,
    _order_component_nodes,
    build_merged_sign_tuples_for_image,
    fill_competing_tuple_counts,
    run_anchor_stage,
)
from sgocr.semantic_grounding import QWEN_GLOBAL_INVENTORY_CATEGORIES


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
    def test_fill_competing_tuple_counts_tracks_repeated_anchor_scene_stats(self) -> None:
        rows = [
            {
                "tuple_id": "t1",
                "relation": "on",
                "anchor_label": "silver can",
                "anchor_box": [0.0, 0.0, 20.0, 40.0],
                "location_phrase": "left side of the image",
                "specific_location_phrase": "",
                "answer": "BUD LIGHT",
                "answer_normalized": "bud light",
                "kd_metadata": {"local_text_bucket_key": "a"},
            },
            {
                "tuple_id": "t2",
                "relation": "on",
                "anchor_label": "silver can",
                "anchor_box": [40.0, 0.0, 60.0, 40.0],
                "location_phrase": "center of the image",
                "specific_location_phrase": "",
                "answer": "BUD LIGHT",
                "answer_normalized": "bud light",
                "kd_metadata": {"local_text_bucket_key": "b"},
            },
            {
                "tuple_id": "t3",
                "relation": "on",
                "anchor_label": "silver can",
                "anchor_box": [80.0, 0.0, 100.0, 40.0],
                "location_phrase": "right side of the image",
                "specific_location_phrase": "",
                "answer": "BUD LIGHT",
                "answer_normalized": "bud light",
                "kd_metadata": {"local_text_bucket_key": "c"},
            },
        ]
        fill_competing_tuple_counts(rows)
        kd = rows[0]["kd_metadata"]
        self.assertEqual(kd["same_anchor_same_answer_count"], 3)
        self.assertEqual(kd["same_anchor_same_answer_nonoverlap_instances"], 3)
        self.assertEqual(kd["same_anchor_distinct_answer_count"], 1)
        self.assertTrue(kd["diffuse_same_anchor_scene"])

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

    def test_order_component_nodes_keeps_left_to_right_priority(self) -> None:
        nodes = [
            make_text_node("a", "LEFT", 20, 40, 60, 60),
            make_text_node("b", "RIGHT", 90, 34, 140, 54),
            make_text_node("c", "FAR", 160, 20, 210, 40),
        ]
        ordered = _order_component_nodes(nodes)
        self.assertEqual([node["node_id"] for node in ordered], ["a", "b", "c"])

    def test_run_anchor_stage_can_use_qwen_for_local_tag_discovery(self) -> None:
        class FakeQwenGrounder:
            def __init__(self, **_: object) -> None:
                pass

            def detect_many(self, requests, **kwargs):
                out = {}
                for req in requests:
                    if req.image_obj is not None:
                        if str(req.image_id).endswith("::n1"):
                            out[req.image_id] = [
                                {
                                    "label": "bottle",
                                    "raw_label": "brown bottle",
                                    "box": [5.0, 5.0, 55.0, 95.0],
                                    "score": 0.62,
                                    "source": "qwen3_vl_vllm",
                                    "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"bottle"}]',
                                }
                            ]
                        else:
                            out[req.image_id] = [
                                {
                                    "label": "screen",
                                    "raw_label": "screen",
                                    "box": [0.0, 0.0, 70.0, 50.0],
                                    "score": 0.62,
                                    "source": "qwen3_vl_vllm",
                                    "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"screen"}]',
                                }
                            ]
                    else:
                        out[req.image_id] = [
                            {
                                "label": "bottle",
                                "raw_label": "bottle",
                                "box": [8.0, 8.0, 84.0, 128.0],
                                "score": 0.62,
                                "source": "qwen3_vl_vllm",
                                "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"bottle"}]',
                            },
                            {
                                "label": "screen",
                                "raw_label": "screen",
                                "box": [96.0, 16.0, 214.0, 92.0],
                                "score": 0.62,
                                "source": "qwen3_vl_vllm",
                                "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"screen"}]',
                            },
                        ]
                return out

            def close(self) -> None:
                return None

        nodes = [
            make_text_node("n1", "BROWN", 20, 30, 50, 52, image_id="img-qwen"),
            make_text_node("n2", "LCD", 130, 32, 168, 54, image_id="img-qwen"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "img.jpg")
            Image.new("RGB", (256, 256), color="white").save(image_path)
            with patch.dict(
                os.environ,
                {
                    "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
                    "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
                    "SGOCR_QWEN_ANCHOR_BATCH_SIZE": "1",
                },
                clear=True,
            ):
                with patch("sgocr.pipeline.stages.QwenAnchorGrounderVLLM", FakeQwenGrounder):
                    anchor_tag_rows, grounded_anchor_rows, best_anchor_by_node = run_anchor_stage(
                        image_specs=[{"image_id": "img-qwen", "image_path": image_path}],
                        resolvable_nodes=nodes,
                        device="cpu",
                        max_tags_per_image=8,
                        grounding_threshold=0.28,
                    )
        self.assertEqual(len(anchor_tag_rows), 2)
        self.assertEqual(anchor_tag_rows[0]["discovered_tags"][0], "brown bottle")
        self.assertEqual(anchor_tag_rows[1]["discovered_tags"][0], "screen")
        self.assertIn("brown bottle", anchor_tag_rows[0]["final_prompt_tags"])
        self.assertIn("screen", anchor_tag_rows[1]["final_prompt_tags"])
        self.assertEqual(best_anchor_by_node[("img-qwen", "n1")]["label"], "brown bottle")
        self.assertEqual(best_anchor_by_node[("img-qwen", "n2")]["label"], "screen")
        nonempty = [row for row in grounded_anchor_rows if row["top_candidates"]]
        self.assertEqual(len(nonempty), 2)

    def test_run_anchor_stage_can_use_qwen_global_inventory_join(self) -> None:
        seen_full_image_categories: list[list[str]] = []

        class FakeQwenGrounder:
            def __init__(self, **_: object) -> None:
                pass

            def detect_many(self, requests, **kwargs):
                out = {}
                for req in requests:
                    if req.image_obj is not None:
                        out[req.image_id] = [
                            {
                                "label": "screen",
                                "raw_label": "screen",
                                "box": [0.0, 0.0, 70.0, 50.0],
                                "score": 0.62,
                                "source": "qwen3_vl_vllm",
                                "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"screen"}]',
                            }
                        ]
                    else:
                        seen_full_image_categories.append(list(req.categories))
                        out[req.image_id] = [
                            {
                                "label": "bottle",
                                "raw_label": "bottle",
                                "box": [12.0, 10.0, 96.0, 136.0],
                                "score": 0.62,
                                "source": "qwen3_vl_vllm",
                                "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"bottle"}]',
                            }
                        ]
                return out

            def close(self) -> None:
                return None

        nodes = [make_text_node("n1", "BROWN", 20, 30, 50, 52, image_id="img-qwen-global")]
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "img.jpg")
            Image.new("RGB", (256, 256), color="white").save(image_path)
            with patch.dict(
                os.environ,
                {
                    "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
                    "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
                    "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "global_inventory",
                    "SGOCR_QWEN_ANCHOR_BATCH_SIZE": "1",
                },
                clear=True,
            ):
                with patch("sgocr.pipeline.stages.QwenAnchorGrounderVLLM", FakeQwenGrounder):
                    _, grounded_anchor_rows, best_anchor_by_node = run_anchor_stage(
                        image_specs=[{"image_id": "img-qwen-global", "image_path": image_path}],
                        resolvable_nodes=nodes,
                        device="cpu",
                        max_tags_per_image=8,
                        grounding_threshold=0.28,
                    )
        self.assertEqual(seen_full_image_categories, [list(QWEN_GLOBAL_INVENTORY_CATEGORIES)])
        self.assertEqual(best_anchor_by_node[("img-qwen-global", "n1")]["label"], "bottle")
        self.assertIn("bottle", [row["label"] for row in grounded_anchor_rows[0]["top_candidates"]])

    def test_run_anchor_stage_uses_color_specific_open_qwen_prompt(self) -> None:
        seen_prompts: list[str] = []

        class FakeQwenGrounder:
            def __init__(self, **_: object) -> None:
                pass

            def detect_many(self, requests, **kwargs):
                out = {}
                for req in requests:
                    seen_prompts.append(str(req.prompt_text or ""))
                    if req.image_obj is not None:
                        out[req.image_id] = [
                            {
                                "label": "blue jersey",
                                "raw_label": "blue jersey",
                                "box": [0.0, 0.0, 64.0, 64.0],
                                "score": 0.75,
                                "source": "qwen3_vl_vllm",
                                "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"blue jersey"}]',
                            }
                        ]
                    else:
                        out[req.image_id] = [
                            {
                                "label": "jersey",
                                "raw_label": "jersey",
                                "box": [8.0, 8.0, 84.0, 128.0],
                                "score": 0.62,
                                "source": "qwen3_vl_vllm",
                                "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"jersey"}]',
                            }
                        ]
                return out

            def close(self) -> None:
                return None

        nodes = [make_text_node("n1", "BROWN", 20, 30, 50, 52, image_id="img-qwen-open")]
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "img.jpg")
            Image.new("RGB", (256, 256), color="white").save(image_path)
            with patch.dict(
                os.environ,
                {
                    "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
                    "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "open",
                    "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
                    "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
                    "SGOCR_QWEN_ANCHOR_BATCH_SIZE": "1",
                },
                clear=True,
            ):
                with patch("sgocr.pipeline.stages.QwenAnchorGrounderVLLM", FakeQwenGrounder):
                    run_anchor_stage(
                        image_specs=[{"image_id": "img-qwen-open", "image_path": image_path}],
                        resolvable_nodes=nodes,
                        device="cpu",
                        max_tags_per_image=8,
                        grounding_threshold=0.28,
                    )
        self.assertTrue(any("Include a visible color adjective" in prompt for prompt in seen_prompts))

    def test_run_anchor_stage_can_use_qwen_independent_raw_inventory(self) -> None:
        seen_requests: list[dict[str, object]] = []

        class FakeQwenGrounder:
            def __init__(self, **_: object) -> None:
                pass

            def detect_many(self, requests, **kwargs):
                seen_requests.extend(
                    {
                        "image_id": req.image_id,
                        "prompt_text": req.prompt_text,
                        "categories": list(req.categories),
                        "image_obj": req.image_obj is not None,
                        "normalize_open_labels": req.normalize_open_labels,
                    }
                    for req in requests
                )
                return {
                    "img-qwen-independent": [
                        {
                            "label": "blue jersey",
                            "raw_label": "blue jersey",
                            "box": [16.0, 8.0, 94.0, 138.0],
                            "score": 0.62,
                            "source": "qwen3_vl_vllm",
                            "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"blue jersey"}]',
                        },
                        {
                            "label": "scoreboard",
                            "raw_label": "scoreboard",
                            "box": [102.0, 12.0, 220.0, 84.0],
                            "score": 0.62,
                            "source": "qwen3_vl_vllm",
                            "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"scoreboard"}]',
                        },
                    ]
                }

            def close(self) -> None:
                return None

        nodes = [
            make_text_node("n1", "BROWN", 20, 30, 50, 52, image_id="img-qwen-independent"),
            make_text_node("n2", "42", 130, 24, 160, 46, image_id="img-qwen-independent"),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "img.jpg")
            Image.new("RGB", (256, 256), color="white").save(image_path)
            with patch.dict(
                os.environ,
                {
                    "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "florence",
                    "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
                    "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "independent_raw",
                    "SGOCR_QWEN_ANCHOR_BATCH_SIZE": "1",
                },
                clear=True,
            ):
                with patch("sgocr.pipeline.stages.QwenAnchorGrounderVLLM", FakeQwenGrounder):
                    anchor_tag_rows, grounded_anchor_rows, best_anchor_by_node = run_anchor_stage(
                        image_specs=[{"image_id": "img-qwen-independent", "image_path": image_path}],
                        resolvable_nodes=nodes,
                        device="cpu",
                        max_tags_per_image=8,
                        grounding_threshold=0.28,
                    )
        self.assertEqual(len(anchor_tag_rows), 2)
        self.assertTrue(all(row["inventory_mode"] == "independent_raw" for row in anchor_tag_rows))
        self.assertTrue(any("useful as anchors for referring to text later" in str(req["prompt_text"]) for req in seen_requests))
        self.assertTrue(all(req["categories"] == [] for req in seen_requests))
        self.assertTrue(all(req["image_obj"] is False for req in seen_requests))
        self.assertTrue(all(req["normalize_open_labels"] is True for req in seen_requests))
        self.assertEqual(best_anchor_by_node[("img-qwen-independent", "n1")]["label"], "blue jersey")
        self.assertTrue(any(row["top_candidates"] for row in grounded_anchor_rows))


if __name__ == "__main__":
    unittest.main()
