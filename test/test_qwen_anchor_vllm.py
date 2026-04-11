from __future__ import annotations

import unittest

from sgocr.qwen_anchor_vllm import (
    OFFICIAL_QWEN_BBOX_PROMPT,
    OPEN_QWEN_LOCAL_DISCOVERY_PROMPT,
    _iter_json_candidates,
    _scale_relative_bbox,
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


if __name__ == "__main__":
    unittest.main()
