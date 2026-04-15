from __future__ import annotations

import unittest

from sgocr.qwen_anchor_vllm import (
    INDEPENDENT_QWEN_INVENTORY_PROMPT,
    INDEPENDENT_QWEN_INVENTORY_PROMPT_ANTI_OCR,
    OFFICIAL_QWEN_BBOX_PROMPT,
    OPEN_QWEN_LOCAL_DISCOVERY_PROMPT,
    OPEN_QWEN_LOCAL_DISCOVERY_PROMPT_COLOR_SPECIFIC,
    STRUCTURAL_FALLBACK_QWEN_PROMPT,
    STRUCTURAL_FALLBACK_QWEN_PROMPT_ANTI_OCR,
    _iter_json_candidates,
    _scale_relative_bbox,
    is_degenerate_anchor_label,
    is_degenerate_inventory,
    is_ocr_text_label,
    merge_qwen_inventory_passes,
    normalize_qwen_description,
)


class TestQwenAnchorVLLMHelpers(unittest.TestCase):
    def test_official_prompt_shape(self) -> None:
        prompt = OFFICIAL_QWEN_BBOX_PROMPT.format(categories="car, bus")
        self.assertIn('Locate every instance that belongs to the following categories: "car, bus".', prompt)
        self.assertIn('"bbox_2d": [x1, y1, x2, y2]', prompt)
        self.assertIn('"label": "category"', prompt)

    def test_open_prompt_shape(self) -> None:
        self.assertIn("Locate the visible objects, surfaces, or object parts", OPEN_QWEN_LOCAL_DISCOVERY_PROMPT)
        self.assertIn('"bbox_2d": [x1, y1, x2, y2]', OPEN_QWEN_LOCAL_DISCOVERY_PROMPT)
        self.assertIn('"label": "object"', OPEN_QWEN_LOCAL_DISCOVERY_PROMPT)

    def test_color_specific_open_prompt_shape(self) -> None:
        self.assertIn("Include a visible color adjective", OPEN_QWEN_LOCAL_DISCOVERY_PROMPT_COLOR_SPECIFIC)
        self.assertIn("Avoid generic labels like", OPEN_QWEN_LOCAL_DISCOVERY_PROMPT_COLOR_SPECIFIC)
        self.assertIn('"bbox_2d": [x1, y1, x2, y2]', OPEN_QWEN_LOCAL_DISCOVERY_PROMPT_COLOR_SPECIFIC)

    def test_independent_inventory_prompt_shape(self) -> None:
        self.assertIn('Locate every instance that belongs to the following categories:', INDEPENDENT_QWEN_INVENTORY_PROMPT)
        self.assertIn("useful as anchors for referring to text later", INDEPENDENT_QWEN_INVENTORY_PROMPT)
        self.assertIn('"bbox_2d": [x1, y1, x2, y2]', INDEPENDENT_QWEN_INVENTORY_PROMPT)

    def test_parses_markdown_json_list(self) -> None:
        parsed = _iter_json_candidates(
            """```json
            [
              {"bbox_2d": [10, 20, 30, 40], "label": "car"},
              {"bbox_2d": [100, 200, 300, 400], "label": "bus"}
            ]
            ```"""
        )
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["label"], "car")
        self.assertEqual(parsed[1]["bbox_2d"], [100, 200, 300, 400])

    def test_scales_relative_bbox(self) -> None:
        box = _scale_relative_bbox([100, 250, 900, 750], width=200, height=100)
        self.assertEqual(box, [20.0, 25.0, 180.0, 75.0])

    def test_normalizes_qwen_description(self) -> None:
        desc = normalize_qwen_description(
            [
                {
                    "label": "bottle",
                    "raw_label": "brown bottle",
                    "box": [10, 20, 40, 80],
                    "raw_text": '[{"bbox_2d":[1,2,3,4],"label":"bottle"}]',
                },
                {
                    "label": "label",
                    "raw_label": "label",
                    "box": [12, 50, 30, 70],
                    "raw_text": "",
                },
            ]
        )
        self.assertEqual(desc["caption"], "brown bottle; label")
        self.assertEqual(desc["labels"], ["brown bottle", "label"])
        self.assertEqual(desc["regions"][0]["label"], "brown bottle")
        self.assertEqual(desc["regions"][1]["box"], [12.0, 50.0, 30.0, 70.0])

    def test_merges_inventory_passes_by_label_and_iou(self) -> None:
        merged = merge_qwen_inventory_passes(
            [
                {
                    "img": [
                        {"label": "red jersey", "raw_label": "red jersey", "box": [10, 10, 80, 100], "score": 0.62},
                        {"label": "scoreboard", "raw_label": "scoreboard", "box": [100, 20, 180, 80], "score": 0.62},
                    ]
                },
                {
                    "img": [
                        {"label": "red jersey", "raw_label": "red jersey", "box": [12, 12, 82, 102], "score": 0.62},
                        {"label": "person", "raw_label": "person", "box": [190, 20, 240, 120], "score": 0.62},
                    ]
                },
            ],
            iou_threshold=0.55,
            min_support=1,
        )
        self.assertIn("img", merged)
        labels = {row["label"] for row in merged["img"]}
        self.assertIn("red jersey", labels)
        self.assertIn("scoreboard", labels)
        self.assertIn("person", labels)
        jersey = next(row for row in merged["img"] if row["label"] == "red jersey")
        self.assertEqual(jersey["pass_support_count"], 2)


class TestStructuralFallback(unittest.TestCase):
    def test_structural_fallback_prompt_shape(self) -> None:
        self.assertIn("bbox_2d", STRUCTURAL_FALLBACK_QWEN_PROMPT)
        # The prompt must discourage generic labels (mentioned as an example to avoid)
        self.assertIn("Avoid generic labels", STRUCTURAL_FALLBACK_QWEN_PROMPT)
        self.assertIn("bars", STRUCTURAL_FALLBACK_QWEN_PROMPT)
        self.assertIn("segments", STRUCTURAL_FALLBACK_QWEN_PROMPT)
        # The prompt must NOT use "blue rectangular bar" as an example — that biases Qwen
        self.assertNotIn("blue rectangular bar", STRUCTURAL_FALLBACK_QWEN_PROMPT)
        # The prompt should use neutral structural examples instead
        self.assertIn("bar chart segment", STRUCTURAL_FALLBACK_QWEN_PROMPT)

    def test_is_degenerate_empty(self) -> None:
        self.assertTrue(is_degenerate_inventory([]))

    def test_is_degenerate_all_same_generic_label(self) -> None:
        rows = [{"label": "all visible objects"} for _ in range(5)]
        self.assertTrue(is_degenerate_inventory(rows))

    def test_is_degenerate_unknown_label(self) -> None:
        rows = [{"label": "unknown"} for _ in range(3)]
        self.assertTrue(is_degenerate_inventory(rows))

    def test_is_degenerate_single_label_above_threshold(self) -> None:
        # 4 out of 5 rows share a non-generic, non-valid-object label → degenerate at threshold=0.75
        rows = [{"label": "blue bar"}] * 4 + [{"label": "red sign"}]
        self.assertTrue(is_degenerate_inventory(rows, threshold=0.75))

    def test_not_degenerate_diverse_labels(self) -> None:
        rows = [
            {"label": "bar chart segment"},
            {"label": "pie chart wedge"},
            {"label": "white legend panel"},
            {"label": "table cell"},
        ]
        self.assertFalse(is_degenerate_inventory(rows))

    def test_not_degenerate_majority_below_threshold(self) -> None:
        rows = [{"label": "blue bar"}] * 3 + [{"label": "red sign"}, {"label": "white label"}]
        self.assertFalse(is_degenerate_inventory(rows, threshold=0.75))

    def test_degenerate_uses_raw_label_fallback(self) -> None:
        rows = [{"raw_label": "all visible objects", "label": ""} for _ in range(3)]
        self.assertTrue(is_degenerate_inventory(rows))

    def test_not_degenerate_concentrated_valid_object(self) -> None:
        # Airport image: all planes — should NOT trigger structural fallback
        rows = [{"label": "plane"} for _ in range(4)]
        self.assertFalse(is_degenerate_inventory(rows))

    def test_not_degenerate_concentrated_person(self) -> None:
        rows = [{"label": "person"} for _ in range(5)]
        self.assertFalse(is_degenerate_inventory(rows))

    def test_degenerate_concentrated_chart_label_above_threshold(self) -> None:
        # High concentration of a non-valid-object label → degenerate (12/13 ≈ 0.923 ≥ 0.92)
        rows = [{"label": "data bar"}] * 12 + [{"label": "axis label"}]
        self.assertTrue(is_degenerate_inventory(rows, threshold=0.92))


class TestAntiOcrPrompts(unittest.TestCase):
    def test_anti_ocr_inventory_prompt_adds_instruction(self) -> None:
        self.assertIn("never copy verbatim text content", INDEPENDENT_QWEN_INVENTORY_PROMPT_ANTI_OCR)
        self.assertIn("visual object or shape type", INDEPENDENT_QWEN_INVENTORY_PROMPT_ANTI_OCR)

    def test_anti_ocr_inventory_prompt_preserves_base_content(self) -> None:
        self.assertIn("Locate every instance", INDEPENDENT_QWEN_INVENTORY_PROMPT_ANTI_OCR)
        self.assertIn("bbox_2d", INDEPENDENT_QWEN_INVENTORY_PROMPT_ANTI_OCR)

    def test_anti_ocr_fallback_prompt_adds_instruction(self) -> None:
        self.assertIn("never copy verbatim text content", STRUCTURAL_FALLBACK_QWEN_PROMPT_ANTI_OCR)

    def test_anti_ocr_fallback_preserves_base_content(self) -> None:
        self.assertIn("bars", STRUCTURAL_FALLBACK_QWEN_PROMPT_ANTI_OCR)
        self.assertIn("bbox_2d", STRUCTURAL_FALLBACK_QWEN_PROMPT_ANTI_OCR)
        self.assertNotIn("blue rectangular bar", STRUCTURAL_FALLBACK_QWEN_PROMPT_ANTI_OCR)


class TestDegenerateAnchorLabel(unittest.TestCase):
    def test_empty_label_is_degenerate(self) -> None:
        self.assertTrue(is_degenerate_anchor_label(""))

    def test_single_char_is_degenerate(self) -> None:
        self.assertTrue(is_degenerate_anchor_label("a"))

    def test_object_part_is_degenerate(self) -> None:
        self.assertTrue(is_degenerate_anchor_label("object part"))

    def test_surface_is_degenerate(self) -> None:
        self.assertTrue(is_degenerate_anchor_label("surface"))

    def test_area_is_degenerate(self) -> None:
        self.assertTrue(is_degenerate_anchor_label("area"))

    def test_unknown_is_degenerate(self) -> None:
        self.assertTrue(is_degenerate_anchor_label("unknown"))

    def test_all_visible_objects_is_degenerate(self) -> None:
        self.assertTrue(is_degenerate_anchor_label("all visible objects"))

    def test_specific_label_is_not_degenerate(self) -> None:
        self.assertFalse(is_degenerate_anchor_label("red jersey"))

    def test_blue_bar_is_not_degenerate(self) -> None:
        self.assertFalse(is_degenerate_anchor_label("blue rectangular bar"))

    def test_brewery_sign_is_not_degenerate(self) -> None:
        self.assertFalse(is_degenerate_anchor_label("brewery sign"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(is_degenerate_anchor_label("Surface"))
        self.assertTrue(is_degenerate_anchor_label("AREA"))


class TestOcrTextLabel(unittest.TestCase):
    def test_all_caps_multi_word_is_ocr_text(self) -> None:
        self.assertTrue(is_ocr_text_label("DAN BROWN BREWING COMPANY"))

    def test_all_caps_short_is_not_ocr_text(self) -> None:
        # Only 1 token → too short to be a copied text string
        self.assertFalse(is_ocr_text_label("STOP"))

    def test_lowercase_specific_label_is_not_ocr_text(self) -> None:
        self.assertFalse(is_ocr_text_label("red jersey"))

    def test_mixed_specific_label_is_not_ocr_text(self) -> None:
        self.assertFalse(is_ocr_text_label("blue rectangular bar"))

    def test_title_case_long_proper_noun_is_ocr_text(self) -> None:
        # ≥3 words, all start uppercase, avg token len ≥ 5
        self.assertTrue(is_ocr_text_label("Springfield Community Center"))

    def test_two_word_title_case_not_flagged(self) -> None:
        # Only 2 words — short enough to be a legitimate label like "Coca Cola"
        self.assertFalse(is_ocr_text_label("Coca Cola"))

    def test_empty_is_not_ocr_text(self) -> None:
        self.assertFalse(is_ocr_text_label(""))


if __name__ == "__main__":
    unittest.main()
