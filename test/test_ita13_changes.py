"""Tests for ITA13 pipeline changes:
  1. rg_per_image_hard_cap — enforce_type_constraints respects configurable RG cap
  2. property_candidate_selection_bonus — TP/AP get selection bonus in select_candidates
  3. Qwen 'with' connector — QWEN_INDEPENDENT_RAW_INVENTORY_CATEGORY_TEXT contains instruction
"""
from __future__ import annotations

import os
import unittest
import unittest.mock

from sgocr.semantic_dev40_tuning import load_semantic_dev40_tuning


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate(question_type: str, quality: float = 1.0, candidate_id: str = "c1") -> dict:
    return {
        "candidate_id": candidate_id,
        "question_type": question_type,
        "quality": quality,
        "tuple": {
            "text_node_ids": [candidate_id],
            "anchor_label": "test_label",
            "image_id": "img-1",
        },
    }


def _run_select(candidates, target=4):
    from sgocr.dev40_complete import select_candidates
    return select_candidates(candidates, target_count=target)


def _run_enforce(selected, candidates, target=4):
    from sgocr.dev40_complete import enforce_type_constraints
    return enforce_type_constraints(selected, candidates, target=target)


# ---------------------------------------------------------------------------
# 1. rg_per_image_hard_cap tuning + enforce_type_constraints
# ---------------------------------------------------------------------------

class TestRgPerImageHardCap(unittest.TestCase):
    def test_default_cap_is_one(self):
        tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.rg_per_image_hard_cap, 1)

    def test_env_var_loads(self):
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_PER_IMAGE_HARD_CAP": "2"}):
            tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.rg_per_image_hard_cap, 2)

    def test_cap1_removes_second_rg(self):
        """Default cap=1: two RG candidates → only 1 survives enforce."""
        candidates = [
            _candidate("REVERSE_GROUND", quality=2.0, candidate_id="rg-1"),
            _candidate("REVERSE_GROUND", quality=1.0, candidate_id="rg-2"),
            _candidate("DIRECT_READ", quality=1.5, candidate_id="dr-1"),
        ]
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_PER_IMAGE_HARD_CAP": "1"}):
            result = _run_enforce(list(candidates), candidates, target=3)
        rg = [c for c in result if c["question_type"] == "REVERSE_GROUND"]
        self.assertEqual(len(rg), 1)
        self.assertEqual(rg[0]["candidate_id"], "rg-1")  # higher quality kept

    def test_cap2_keeps_two_rg(self):
        """Cap=2: two RG candidates both survive enforce."""
        candidates = [
            _candidate("REVERSE_GROUND", quality=2.0, candidate_id="rg-1"),
            _candidate("REVERSE_GROUND", quality=1.0, candidate_id="rg-2"),
            _candidate("DIRECT_READ", quality=1.5, candidate_id="dr-1"),
            _candidate("YES_NO", quality=1.5, candidate_id="yn-1"),
        ]
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_PER_IMAGE_HARD_CAP": "2"}):
            result = _run_enforce(list(candidates), candidates, target=4)
        rg = [c for c in result if c["question_type"] == "REVERSE_GROUND"]
        self.assertEqual(len(rg), 2)

    def test_cap2_does_not_affect_tp_ap_cap(self):
        """RG cap raise does NOT raise TP or AP cap — still capped at 1 each."""
        candidates = [
            _candidate("REVERSE_GROUND", quality=2.0, candidate_id="rg-1"),
            _candidate("TEXT_PROPERTY", quality=2.0, candidate_id="tp-1"),
            _candidate("TEXT_PROPERTY", quality=1.0, candidate_id="tp-2"),
            _candidate("ANCHOR_PROPERTY", quality=2.0, candidate_id="ap-1"),
            _candidate("ANCHOR_PROPERTY", quality=1.0, candidate_id="ap-2"),
        ]
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_PER_IMAGE_HARD_CAP": "2"}):
            result = _run_enforce(list(candidates), candidates, target=5)
        self.assertLessEqual(sum(1 for c in result if c["question_type"] == "TEXT_PROPERTY"), 1)
        self.assertLessEqual(sum(1 for c in result if c["question_type"] == "ANCHOR_PROPERTY"), 1)


# ---------------------------------------------------------------------------
# 2. property_candidate_selection_bonus
# ---------------------------------------------------------------------------

class TestPropertyCandidateSelectionBonus(unittest.TestCase):
    def test_default_is_zero(self):
        tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.property_candidate_selection_bonus, 0.0)

    def test_env_var_loads(self):
        with unittest.mock.patch.dict(os.environ, {"SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": "0.5"}):
            tuning = load_semantic_dev40_tuning()
        self.assertAlmostEqual(tuning.property_candidate_selection_bonus, 0.5)

    def test_no_bonus_dr_beats_tp_same_quality(self):
        """Without bonus, DR gets +0.25 so it beats same-quality TP."""
        dr = _candidate("DIRECT_READ", quality=1.0, candidate_id="dr-1")
        tp = _candidate("TEXT_PROPERTY", quality=1.0, candidate_id="tp-1")
        with unittest.mock.patch.dict(os.environ, {"SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": "0.0"}):
            result = _run_select([dr, tp], target=1)
        self.assertEqual(result[0]["question_type"], "DIRECT_READ")

    def test_large_bonus_tp_beats_dr(self):
        """With bonus=1.0, TP wins over DR (+0.25)."""
        dr = _candidate("DIRECT_READ", quality=1.0, candidate_id="dr-1")
        tp = _candidate("TEXT_PROPERTY", quality=1.0, candidate_id="tp-1")
        with unittest.mock.patch.dict(os.environ, {"SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": "1.0"}):
            result = _run_select([dr, tp], target=1)
        self.assertEqual(result[0]["question_type"], "TEXT_PROPERTY")

    def test_bonus_applies_to_anchor_property(self):
        """Bonus applies to ANCHOR_PROPERTY, not just TEXT_PROPERTY."""
        dr = _candidate("DIRECT_READ", quality=1.0, candidate_id="dr-1")
        ap = _candidate("ANCHOR_PROPERTY", quality=1.0, candidate_id="ap-1")
        with unittest.mock.patch.dict(os.environ, {"SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": "1.0"}):
            result = _run_select([dr, ap], target=1)
        self.assertEqual(result[0]["question_type"], "ANCHOR_PROPERTY")

    def test_bonus_does_not_affect_rg_or_yn(self):
        """Bonus only applies to TP/AP — RG and YN are unaffected."""
        rg = _candidate("REVERSE_GROUND", quality=5.0, candidate_id="rg-1")
        tp = _candidate("TEXT_PROPERTY", quality=1.0, candidate_id="tp-1")
        with unittest.mock.patch.dict(os.environ, {"SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": "2.0"}):
            result = _run_select([rg, tp], target=1)
        # RG quality=5.0 >> TP quality=1.0+2.0=3.0; RG wins
        self.assertEqual(result[0]["question_type"], "REVERSE_GROUND")


# ---------------------------------------------------------------------------
# 3. Qwen 'with' connector in inventory category text
# ---------------------------------------------------------------------------

class TestQwenWithConnector(unittest.TestCase):
    def test_with_instruction_present(self):
        from sgocr.semantic_grounding import QWEN_INDEPENDENT_RAW_INVENTORY_CATEGORY_TEXT
        self.assertIn("with", QWEN_INDEPENDENT_RAW_INVENTORY_CATEGORY_TEXT)
        self.assertIn("man with dark hoodie", QWEN_INDEPENDENT_RAW_INVENTORY_CATEGORY_TEXT)

    def test_anti_example_present(self):
        from sgocr.semantic_grounding import QWEN_INDEPENDENT_RAW_INVENTORY_CATEGORY_TEXT
        self.assertIn("man dark hoodie", QWEN_INDEPENDENT_RAW_INVENTORY_CATEGORY_TEXT)


if __name__ == "__main__":
    unittest.main()
