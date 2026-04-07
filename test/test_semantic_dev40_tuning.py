from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from sgocr.semantic_dev40_tuning import load_semantic_dev40_tuning


class TestSemanticDev40Tuning(unittest.TestCase):
    def test_defaults_load_cleanly(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.location_wording_mode, "balanced")
        self.assertEqual(tuning.teacher_strictness, "strict")
        self.assertEqual(tuning.max_negative_yesno_per_image, 1)
        self.assertEqual(tuning.anchor_prompt_expansion_mode, "none")
        self.assertEqual(tuning.anchor_relabel_mode, "none")
        self.assertEqual(tuning.sam3_apply_mode, "all")

    def test_env_overrides_are_applied(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.5",
                "SGOCR_LOCATION_WORDING_MODE": "lite",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.35",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.15",
                "SGOCR_MAX_NEGATIVE_YESNO_PER_IMAGE": "0",
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
                "SGOCR_ANCHOR_RELABEL_MODE": "flash",
                "SGOCR_DETECTOR_MODE": "ppocr_craft_ensemble",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "6",
                "SGOCR_SAM3_REFINE_MODE": "top2",
                "SGOCR_SAM3_TOPK_PROMPTS": "3",
                "SGOCR_SAM3_APPLY_MODE": "targeted",
                "SGOCR_TEXT_MERGE_ENABLED": "1",
                "SGOCR_TEXT_MERGE_GAP_RATIO_MAX": "2.1",
            },
            clear=True,
        ):
            tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.grounded_exclusion_min_strength, 2.5)
        self.assertEqual(tuning.location_wording_mode, "lite")
        self.assertEqual(tuning.reverse_ground_directional_local_bias, 0.35)
        self.assertEqual(tuning.reverse_ground_directional_mixed_bias, 0.15)
        self.assertEqual(tuning.max_negative_yesno_per_image, 0)
        self.assertEqual(tuning.anchor_prompt_expansion_mode, "supportive")
        self.assertEqual(tuning.anchor_relabel_mode, "flash")
        self.assertEqual(tuning.detector_mode, "ppocr_craft_ensemble")
        self.assertEqual(tuning.ambiguity_hard_reject_score, 6)
        self.assertEqual(tuning.sam3_refine_mode, "top2")
        self.assertEqual(tuning.sam3_topk_prompts, 3)
        self.assertEqual(tuning.sam3_apply_mode, "targeted")
        self.assertTrue(tuning.text_merge_enabled)
        self.assertEqual(tuning.text_merge_gap_ratio_max, 2.1)

    def test_invalid_mode_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_LOCATION_WORDING_MODE": "wild"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()

    def test_invalid_text_merge_threshold_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES": "1"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()


if __name__ == "__main__":
    unittest.main()
