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
        self.assertEqual(tuning.ocr_frontend, "classic")
        self.assertEqual(tuning.anchor_tag_discovery_backend, "florence")
        self.assertEqual(tuning.qwen_anchor_tag_discovery_vocab_mode, "constrained")
        self.assertEqual(tuning.anchor_candidate_backend, "florence_dino")
        self.assertEqual(tuning.qwen_anchor_inventory_mode, "selected_tags")
        self.assertEqual(tuning.qwen_anchor_inventory_pass_count, 1)
        self.assertEqual(tuning.qwen_anchor_inventory_temperature, 0.0)
        self.assertEqual(tuning.qwen_anchor_inventory_consensus_iou, 0.55)
        self.assertEqual(tuning.qwen_anchor_inventory_min_support, 1)
        self.assertEqual(tuning.teacher_strictness, "strict")
        self.assertEqual(tuning.max_negative_yesno_per_image, 1)
        self.assertEqual(tuning.anchor_prompt_expansion_mode, "none")
        self.assertEqual(tuning.anchor_relabel_mode, "none")
        self.assertEqual(tuning.anchor_relabel_generic_mode, "standard")
        self.assertEqual(tuning.sam3_apply_mode, "all")
        self.assertEqual(tuning.gemini_api_mode, "sync")
        self.assertEqual(tuning.gemini_batch_chunk_size, 48)
        self.assertFalse(tuning.sibling_disambiguation_enabled)
        self.assertFalse(tuning.anchor_reference_color_enabled)
        self.assertFalse(tuning.suppress_anchor_local_without_competing_text)
        self.assertFalse(tuning.repeated_anchor_grouping_enabled)
        self.assertEqual(tuning.repeated_anchor_group_min_instances, 3)
        self.assertFalse(tuning.cheap_ambiguity_proxy_enabled)
        self.assertEqual(tuning.cheap_ambiguity_proxy_reject_score, 6)
        self.assertEqual(tuning.qwen_open_tag_prompt_mode, "basic")
        self.assertEqual(tuning.teacher_answer_probe_count, 1)
        self.assertEqual(tuning.teacher_answer_probe_temperature, 0.35)

    def test_env_overrides_are_applied(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.5",
                "SGOCR_OCR_FRONTEND": "nemotron_v2",
                "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
                "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "open",
                "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
                "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "independent_raw",
                "SGOCR_QWEN_ANCHOR_INVENTORY_PASS_COUNT": "2",
                "SGOCR_QWEN_ANCHOR_INVENTORY_TEMPERATURE": "0.15",
                "SGOCR_QWEN_ANCHOR_INVENTORY_CONSENSUS_IOU": "0.61",
                "SGOCR_QWEN_ANCHOR_INVENTORY_MIN_SUPPORT": "2",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.35",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.15",
                "SGOCR_MAX_NEGATIVE_YESNO_PER_IMAGE": "0",
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
                "SGOCR_ANCHOR_RELABEL_MODE": "flash",
                "SGOCR_ANCHOR_RELABEL_GENERIC_MODE": "anti_generic",
                "SGOCR_DETECTOR_MODE": "ppocr_craft_ensemble",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "6",
                "SGOCR_SAM3_REFINE_MODE": "top2",
                "SGOCR_SAM3_TOPK_PROMPTS": "3",
                "SGOCR_SAM3_APPLY_MODE": "targeted",
                "SGOCR_TEXT_MERGE_ENABLED": "1",
                "SGOCR_TEXT_MERGE_GAP_RATIO_MAX": "2.1",
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier_rich",
                "SGOCR_LOCATION_WORDING_MODE": "rich_local",
                "SGOCR_QWEN_ANCHOR_MODEL": "Qwen/Qwen3-VL-8B-Instruct-FP8",
                "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": "0.66",
                "SGOCR_QWEN_ANCHOR_BATCH_SIZE": "3",
                "SGOCR_GEMINI_API_MODE": "batch",
                "SGOCR_GEMINI_BATCH_CHUNK_SIZE": "16",
                "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
                "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
                "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
                "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
                "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": "4",
                "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "1",
                "SGOCR_CHEAP_AMBIGUITY_PROXY_REJECT_SCORE": "8",
                "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
                "SGOCR_TEACHER_ANSWER_PROBE_COUNT": "3",
                "SGOCR_TEACHER_ANSWER_PROBE_TEMPERATURE": "0.55",
            },
            clear=True,
        ):
            tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.grounded_exclusion_min_strength, 2.5)
        self.assertEqual(tuning.ocr_frontend, "nemotron_v2")
        self.assertEqual(tuning.anchor_tag_discovery_backend, "qwen3_vl_vllm")
        self.assertEqual(tuning.qwen_anchor_tag_discovery_vocab_mode, "open")
        self.assertEqual(tuning.anchor_candidate_backend, "qwen3_vl_vllm")
        self.assertEqual(tuning.qwen_anchor_inventory_mode, "independent_raw")
        self.assertEqual(tuning.qwen_anchor_inventory_pass_count, 2)
        self.assertEqual(tuning.qwen_anchor_inventory_temperature, 0.15)
        self.assertEqual(tuning.qwen_anchor_inventory_consensus_iou, 0.61)
        self.assertEqual(tuning.qwen_anchor_inventory_min_support, 2)
        self.assertEqual(tuning.location_wording_mode, "rich_local")
        self.assertEqual(tuning.reverse_ground_directional_local_bias, 0.35)
        self.assertEqual(tuning.reverse_ground_directional_mixed_bias, 0.15)
        self.assertEqual(tuning.max_negative_yesno_per_image, 0)
        self.assertEqual(tuning.anchor_prompt_expansion_mode, "supportive")
        self.assertEqual(tuning.anchor_relabel_mode, "flash")
        self.assertEqual(tuning.anchor_relabel_generic_mode, "anti_generic")
        self.assertEqual(tuning.detector_mode, "ppocr_craft_ensemble")
        self.assertEqual(tuning.ambiguity_hard_reject_score, 6)
        self.assertEqual(tuning.sam3_refine_mode, "top2")
        self.assertEqual(tuning.sam3_topk_prompts, 3)
        self.assertEqual(tuning.sam3_apply_mode, "targeted")
        self.assertTrue(tuning.text_merge_enabled)
        self.assertEqual(tuning.text_merge_gap_ratio_max, 2.1)
        self.assertEqual(tuning.reverse_ground_answer_style, "frontier_rich")
        self.assertEqual(tuning.qwen_anchor_model, "Qwen/Qwen3-VL-8B-Instruct-FP8")
        self.assertEqual(tuning.qwen_anchor_gpu_memory_utilization, 0.66)
        self.assertEqual(tuning.qwen_anchor_batch_size, 3)
        self.assertEqual(tuning.gemini_api_mode, "batch")
        self.assertEqual(tuning.gemini_batch_chunk_size, 16)
        self.assertTrue(tuning.sibling_disambiguation_enabled)
        self.assertTrue(tuning.anchor_reference_color_enabled)
        self.assertTrue(tuning.suppress_anchor_local_without_competing_text)
        self.assertTrue(tuning.repeated_anchor_grouping_enabled)
        self.assertEqual(tuning.repeated_anchor_group_min_instances, 4)
        self.assertTrue(tuning.cheap_ambiguity_proxy_enabled)
        self.assertEqual(tuning.cheap_ambiguity_proxy_reject_score, 8)
        self.assertEqual(tuning.qwen_open_tag_prompt_mode, "color_specific")
        self.assertEqual(tuning.teacher_answer_probe_count, 3)
        self.assertEqual(tuning.teacher_answer_probe_temperature, 0.55)

    def test_invalid_mode_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_LOCATION_WORDING_MODE": "wild"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()

    def test_invalid_ocr_frontend_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_OCR_FRONTEND": "mystery"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()

    def test_invalid_anchor_backend_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_ANCHOR_CANDIDATE_BACKEND": "mystery"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()

    def test_invalid_anchor_tag_backend_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "mystery"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()

    def test_invalid_qwen_tag_vocab_mode_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "wild"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()

    def test_invalid_qwen_inventory_mode_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "wild"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()

    def test_invalid_text_merge_threshold_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES": "1"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()

    def test_invalid_gemini_mode_raises(self) -> None:
        with patch.dict(os.environ, {"SGOCR_GEMINI_API_MODE": "weird"}, clear=True):
            with self.assertRaises(ValueError):
                load_semantic_dev40_tuning()


if __name__ == "__main__":
    unittest.main()
