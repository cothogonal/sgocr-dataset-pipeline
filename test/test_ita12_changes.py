"""Tests for ITA12 pipeline changes:
  1. rg_candidate_oversample_boost tuning parameter + env-var loading
  2. select_candidates() applies boost to REVERSE_GROUND candidates
  3. No regression when boost=0.0 (default)
"""
from __future__ import annotations

import os
import unittest

from sgocr.semantic_dev40_tuning import load_semantic_dev40_tuning


# ---------------------------------------------------------------------------
# Helpers: minimal candidate builders
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


# ---------------------------------------------------------------------------
# 1. Tuning field: rg_candidate_oversample_boost
# ---------------------------------------------------------------------------

class TestRgOversampleBoostTuning(unittest.TestCase):
    def test_default_is_zero(self) -> None:
        tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.rg_candidate_oversample_boost, 0.0)

    def test_env_var_loads_float(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": "1.5"}):
            tuning = load_semantic_dev40_tuning()
        self.assertAlmostEqual(tuning.rg_candidate_oversample_boost, 1.5)

    def test_env_var_zero(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": "0.0"}):
            tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.rg_candidate_oversample_boost, 0.0)


# ---------------------------------------------------------------------------
# 2. select_candidates: RG boost causes RG to win over DR when boosted
# ---------------------------------------------------------------------------

import unittest.mock


class TestSelectCandidatesRgBoost(unittest.TestCase):
    """
    We test select_candidates indirectly by verifying that adding
    SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST causes REVERSE_GROUND to be preferred
    over DIRECT_READ when quality is otherwise equal.
    """

    def _run_select(self, candidates: list[dict], target: int = 1) -> list[dict]:
        from sgocr.dev40_complete import select_candidates
        return select_candidates(candidates, target_count=target)

    def test_no_boost_prefers_direct_read_bonus(self) -> None:
        """Without boost, DIRECT_READ gets +0.25 so it beats same-quality RG."""
        dr = _candidate("DIRECT_READ", quality=1.0, candidate_id="dr-1")
        rg = _candidate("REVERSE_GROUND", quality=1.0, candidate_id="rg-1")
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": "0.0"}):
            selected = self._run_select([dr, rg], target=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["question_type"], "DIRECT_READ")

    def test_large_boost_prefers_rg(self) -> None:
        """With boost=2.0, REVERSE_GROUND wins over DIRECT_READ (+0.25)."""
        dr = _candidate("DIRECT_READ", quality=1.0, candidate_id="dr-1")
        rg = _candidate("REVERSE_GROUND", quality=1.0, candidate_id="rg-1")
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": "2.0"}):
            selected = self._run_select([dr, rg], target=1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["question_type"], "REVERSE_GROUND")

    def test_boost_does_not_affect_non_rg(self) -> None:
        """Boost does not change score of non-RG candidates."""
        dr = _candidate("DIRECT_READ", quality=5.0, candidate_id="dr-1")
        yn = _candidate("YES_NO", quality=1.0, candidate_id="yn-1")
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": "3.0"}):
            selected = self._run_select([dr, yn], target=1)
        # DR quality=5.0 >> YN quality=1.0 even with no RG boost; DR still wins
        self.assertEqual(selected[0]["question_type"], "DIRECT_READ")

    def test_boost_1point5_matches_plan(self) -> None:
        """With boost=1.5 (the ita12 value), RG beats DR when qualities are equal."""
        # novel-text-node bonus +1.7 is equal for both since they're both novel.
        # novel-question-type bonus +1.1 is equal for both (first of each type).
        # DR gets +0.25 extra. RG gets +1.5 boost.
        # Expected: RG wins because 1.5 > 0.25.
        dr = _candidate("DIRECT_READ", quality=1.0, candidate_id="dr-1")
        rg = _candidate("REVERSE_GROUND", quality=1.0, candidate_id="rg-1")
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": "1.5"}):
            selected = self._run_select([dr, rg], target=1)
        self.assertEqual(selected[0]["question_type"], "REVERSE_GROUND")

    def test_no_boost_regression_multiple_types(self) -> None:
        """With boost=0, existing selection logic is unchanged (regression test)."""
        candidates = [
            _candidate("DIRECT_READ", quality=2.0, candidate_id="dr-1"),
            _candidate("YES_NO", quality=1.5, candidate_id="yn-1"),
            _candidate("REVERSE_GROUND", quality=1.0, candidate_id="rg-1"),
        ]
        with unittest.mock.patch.dict(os.environ, {"SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": "0.0"}):
            selected = self._run_select(candidates, target=2)
        # DR (quality=2.0 + 0.25 DR bonus) and YN (1.5 quality) should be in top 2
        types = {c["question_type"] for c in selected}
        self.assertIn("DIRECT_READ", types)


if __name__ == "__main__":
    unittest.main()
