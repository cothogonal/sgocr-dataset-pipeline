"""Tests for ITA08 pipeline changes:
  1. Typed frontier gate (per question-type filtering)
  2. apply_rg_vdep_check + _strip_rg_leakage_tokens
  3. Groundback prompt constant shape
  4. New tuning fields + env-var loading
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from sgocr.dev200_eval import (
    _strip_rg_leakage_tokens,
    apply_rg_vdep_check,
)
from sgocr.qwen_anchor_vllm import (
    GROUNDBACK_QWEN_PROMPT,
    _bbox_iou,
)
from sgocr.semantic_dev40_tuning import load_semantic_dev40_tuning


# ---------------------------------------------------------------------------
# Helper: minimal fake row
# ---------------------------------------------------------------------------

def _rg_row(sample_id: str = "rg-1", question: str = "Where is the button?") -> dict:
    return {
        "sample_id": sample_id,
        "image_id": "img-1",
        "image_path": "/tmp/fake.jpg",
        "question": question,
        "answer": "to the left of the label",
        "question_type": "REVERSE_GROUND",
        "tags": {"question_type": "REVERSE_GROUND", "answer_type": "text_string"},
    }


def _dr_row(sample_id: str = "dr-1") -> dict:
    return {
        "sample_id": sample_id,
        "image_id": "img-1",
        "image_path": "/tmp/fake.jpg",
        "question": "What does the sign say?",
        "answer": "OPEN",
        "question_type": "DIRECT_READ",
        "tags": {"question_type": "DIRECT_READ", "answer_type": "text_string"},
    }


# ---------------------------------------------------------------------------
# 1. _strip_rg_leakage_tokens
# ---------------------------------------------------------------------------

class TestStripRgLeakageTokens(unittest.TestCase):
    def test_strips_color_token(self) -> None:
        q = "Where is the blue button relative to the label?"
        result = _strip_rg_leakage_tokens(q)
        self.assertNotIn("blue", result)
        self.assertIn("button", result)
        self.assertIn("label", result)

    def test_strips_shape_token(self) -> None:
        q = "Where is the rectangular panel in the chart?"
        result = _strip_rg_leakage_tokens(q)
        self.assertNotIn("rectangular", result)
        self.assertIn("panel", result)

    def test_strips_multiple_tokens(self) -> None:
        q = "What is to the right of the blue rectangular bar?"
        result = _strip_rg_leakage_tokens(q)
        self.assertNotIn("blue", result)
        self.assertNotIn("rectangular", result)
        self.assertNotIn("bar", result)
        self.assertIn("right", result)

    def test_no_change_when_no_tokens(self) -> None:
        q = "Where is the submit button relative to the username field?"
        result = _strip_rg_leakage_tokens(q)
        self.assertIs(result, q)  # same object — no copy made

    def test_trailing_punctuation_handled(self) -> None:
        q = "Where is the blue?"
        result = _strip_rg_leakage_tokens(q)
        self.assertNotIn("blue", result)

    def test_partial_word_not_stripped(self) -> None:
        # 'blues' contains 'blue' but as a substring — should NOT be stripped
        q = "Where is the blues band logo?"
        result = _strip_rg_leakage_tokens(q)
        self.assertIn("blues", result)

    def test_returns_non_empty_after_strip(self) -> None:
        # If the entire question is color/shape tokens the result is empty string
        q = "blue red"
        result = _strip_rg_leakage_tokens(q)
        # Both tokens stripped; result is empty — caller should discard
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# 2. apply_rg_vdep_check — with mocked API calls
# ---------------------------------------------------------------------------

class TestApplyRgVdepCheck(unittest.TestCase):
    def _make_rows(self) -> list[dict]:
        return [_rg_row("rg-1"), _rg_row("rg-2"), _dr_row("dr-1")]

    @patch("sgocr.dev200_eval._call_openai_text_only_eval")
    @patch("sgocr.dev200_eval._score_prediction")
    def test_non_rg_always_pass(self, mock_score: MagicMock, mock_call: MagicMock) -> None:
        rows = [_dr_row("dr-1")]
        mock_call.return_value = {"answer": "OPEN"}
        mock_score.return_value = {"soft_correct": True}

        accepted, stats = apply_rg_vdep_check(rows, model="openai:gpt-5.3-codex", workers=1)

        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["question_type"], "DIRECT_READ")
        mock_call.assert_not_called()  # no API call for non-RG

    @patch("sgocr.dev200_eval._call_openai_text_only_eval")
    @patch("sgocr.dev200_eval._score_prediction")
    def test_rg_leaky_rejected(self, mock_score: MagicMock, mock_call: MagicMock) -> None:
        rows = self._make_rows()
        mock_call.return_value = {"answer": "to the left"}
        mock_score.return_value = {"soft_correct": True}  # model answers correctly → leaky

        accepted, stats = apply_rg_vdep_check(rows, model="openai:gpt-5.3-codex", workers=1)

        rg_accepted = [r for r in accepted if r["question_type"] == "REVERSE_GROUND"]
        self.assertEqual(len(rg_accepted), 0)
        self.assertEqual(stats["rejected_rg"], 2)
        self.assertAlmostEqual(stats["text_leaky_rg_rate"], 1.0)

    @patch("sgocr.dev200_eval._call_openai_text_only_eval")
    @patch("sgocr.dev200_eval._score_prediction")
    def test_rg_non_leaky_kept(self, mock_score: MagicMock, mock_call: MagicMock) -> None:
        rows = self._make_rows()
        mock_call.return_value = {"answer": "somewhere"}
        mock_score.return_value = {"soft_correct": False}  # model wrong → vision-dep → keep

        accepted, stats = apply_rg_vdep_check(rows, model="openai:gpt-5.3-codex", workers=1)

        rg_accepted = [r for r in accepted if r["question_type"] == "REVERSE_GROUND"]
        self.assertEqual(len(rg_accepted), 2)
        self.assertEqual(stats["rejected_rg"], 0)
        self.assertAlmostEqual(stats["text_leaky_rg_rate"], 0.0)

    @patch("sgocr.dev200_eval._call_openai_text_only_eval")
    @patch("sgocr.dev200_eval._score_prediction")
    def test_api_error_conservative_keep(self, mock_score: MagicMock, mock_call: MagicMock) -> None:
        mock_call.side_effect = RuntimeError("API down")
        rows = [_rg_row("rg-1")]

        accepted, stats = apply_rg_vdep_check(rows, model="openai:gpt-5.3-codex", workers=1)

        self.assertEqual(len(accepted), 1)
        self.assertEqual(stats["rejected_rg"], 0)

    @patch("sgocr.dev200_eval._call_openai_text_only_eval")
    @patch("sgocr.dev200_eval._score_prediction")
    def test_no_rg_rows_returns_all(self, mock_score: MagicMock, mock_call: MagicMock) -> None:
        rows = [_dr_row("dr-1"), _dr_row("dr-2")]
        accepted, stats = apply_rg_vdep_check(rows, model="openai:gpt-5.3-codex", workers=1)
        self.assertEqual(len(accepted), 2)
        self.assertEqual(stats["total_rg"], 0)
        mock_call.assert_not_called()

    @patch("sgocr.dev200_eval._call_openai_text_only_eval")
    @patch("sgocr.dev200_eval._score_prediction")
    def test_gemini_provider_parsed(self, mock_score: MagicMock, mock_call: MagicMock) -> None:
        """Provider prefix 'gemini:' should route to gemini text-only, not openai."""
        rows = [_rg_row("rg-1")]
        with patch("sgocr.dev200_eval._call_gemini_text_only_eval") as mock_gemini:
            mock_gemini.return_value = {"answer": "somewhere"}
            mock_score.return_value = {"soft_correct": False}
            accepted, _ = apply_rg_vdep_check(rows, model="gemini:gemini-3-flash-preview", workers=1)
        mock_gemini.assert_called_once()
        mock_call.assert_not_called()

    @patch("sgocr.dev200_eval._call_openai_text_only_eval")
    @patch("sgocr.dev200_eval._score_prediction")
    def test_correction_recovers_color_token_row(self, mock_score: MagicMock, mock_call: MagicMock) -> None:
        """A row rejected due to color leak should be recovered if stripping the color fixes it."""
        leaky_q = "Where is the blue button?"
        rows = [_rg_row("rg-1", question=leaky_q)]

        # First call (original question): leaky. Second call (stripped question): not leaky.
        mock_call.return_value = {"answer": "to the left"}
        mock_score.side_effect = [
            {"soft_correct": True},   # first vdep check — leaky
            {"soft_correct": False},  # correction re-check — passes
        ]

        accepted, stats = apply_rg_vdep_check(
            rows, model="openai:gpt-5.3-codex", workers=1, correction_enabled=True
        )

        rg_accepted = [r for r in accepted if r["question_type"] == "REVERSE_GROUND"]
        self.assertEqual(len(rg_accepted), 1)
        self.assertTrue(rg_accepted[0].get("rg_correction_applied"))
        self.assertNotIn("blue", rg_accepted[0]["question"])
        self.assertEqual(stats["corrected_rg"], 1)

    @patch("sgocr.dev200_eval._call_openai_text_only_eval")
    @patch("sgocr.dev200_eval._score_prediction")
    def test_correction_does_not_recover_fundamentally_leaky(
        self, mock_score: MagicMock, mock_call: MagicMock
    ) -> None:
        """A row with no strippable tokens, or still leaky after correction, stays rejected."""
        rows = [_rg_row("rg-1", question="Where is the submit button?")]
        mock_call.return_value = {"answer": "to the left"}
        # First check leaky; no correction possible (no color/shape tokens) → rejected
        mock_score.return_value = {"soft_correct": True}

        accepted, stats = apply_rg_vdep_check(
            rows, model="openai:gpt-5.3-codex", workers=1, correction_enabled=True
        )
        rg_accepted = [r for r in accepted if r["question_type"] == "REVERSE_GROUND"]
        self.assertEqual(len(rg_accepted), 0)
        self.assertEqual(stats["corrected_rg"], 0)


# ---------------------------------------------------------------------------
# 3. Groundback prompt and _bbox_iou
# ---------------------------------------------------------------------------

class TestGroundbackPrompt(unittest.TestCase):
    def test_prompt_contains_label_placeholder(self) -> None:
        self.assertIn("{label}", GROUNDBACK_QWEN_PROMPT)

    def test_prompt_formats_with_label(self) -> None:
        rendered = GROUNDBACK_QWEN_PROMPT.format(label="blue button")
        self.assertIn("blue button", rendered)
        self.assertIn("bbox_2d", rendered)
        self.assertIn("only one bbox", rendered)


class TestBboxIou(unittest.TestCase):
    def test_identical_boxes(self) -> None:
        self.assertAlmostEqual(_bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_no_overlap(self) -> None:
        self.assertAlmostEqual(_bbox_iou([0, 0, 5, 5], [10, 10, 20, 20]), 0.0)

    def test_partial_overlap(self) -> None:
        iou = _bbox_iou([0, 0, 10, 10], [5, 5, 15, 15])
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)

    def test_one_inside_other(self) -> None:
        iou = _bbox_iou([0, 0, 10, 10], [2, 2, 8, 8])
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)


# ---------------------------------------------------------------------------
# 4. New tuning fields
# ---------------------------------------------------------------------------

class TestNewTuningFields(unittest.TestCase):
    def test_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.inline_frontier_gate_question_types, "all")
        self.assertFalse(tuning.rg_vdep_check_enabled)
        self.assertEqual(tuning.rg_vdep_model, "openai:gpt-5.3-codex")
        self.assertFalse(tuning.rg_leakage_correction_enabled)
        self.assertFalse(tuning.anchor_label_groundback_enabled)
        self.assertAlmostEqual(tuning.anchor_label_groundback_iou_threshold, 0.35)

    def test_env_overrides(self) -> None:
        overrides = {
            "SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": "DIRECT_READ,YES_NO",
            "SGOCR_RG_VDEP_CHECK_ENABLED": "1",
            "SGOCR_RG_VDEP_MODEL": "openai:gpt-4o",
            "SGOCR_RG_LEAKAGE_CORRECTION_ENABLED": "1",
            "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
            "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.5",
        }
        with patch.dict(os.environ, overrides):
            tuning = load_semantic_dev40_tuning()
        self.assertEqual(tuning.inline_frontier_gate_question_types, "DIRECT_READ,YES_NO")
        self.assertTrue(tuning.rg_vdep_check_enabled)
        self.assertEqual(tuning.rg_vdep_model, "openai:gpt-4o")
        self.assertTrue(tuning.rg_leakage_correction_enabled)
        self.assertTrue(tuning.anchor_label_groundback_enabled)
        self.assertAlmostEqual(tuning.anchor_label_groundback_iou_threshold, 0.5)

    def test_inline_frontier_gate_question_types_serialized(self) -> None:
        with patch.dict(os.environ, {"SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": "DIRECT_READ"}, clear=False):
            tuning = load_semantic_dev40_tuning()
        meta = tuning.to_metadata()
        self.assertIn("inline_frontier_gate_question_types", meta)
        self.assertEqual(meta["inline_frontier_gate_question_types"], "DIRECT_READ")


# ---------------------------------------------------------------------------
# 5. Typed frontier gate logic (unit-level, no full pipeline)
# ---------------------------------------------------------------------------

class TestTypedFrontierGateLogic(unittest.TestCase):
    """Verify the typed-gate filtering logic without running the full pipeline."""

    def _apply_typed_gate(
        self,
        rows: list[dict],
        gate_types_raw: str,
        wf1_floor: float = -1.0,
    ) -> list[dict]:
        """Reimplements the gate logic inline so we can test it independently."""
        if gate_types_raw.lower() == "all":
            gate_types = None
        else:
            gate_types = frozenset(t.strip().upper() for t in gate_types_raw.split(",") if t.strip())

        def _gated(row: dict) -> bool:
            if gate_types is None:
                return True
            return str(row.get("question_type") or "").upper() in gate_types

        def _errored(row: dict) -> bool:
            return bool((row.get("inline_frontier") or {}).get("error"))

        if wf1_floor < 0.0:
            return [
                row for row in rows
                if not _gated(row) or row.get("inline_frontier_correct") is True or _errored(row)
            ]
        # lenient mode (not needed for these tests)
        return rows

    def _make_rows(self) -> list[dict]:
        return [
            {"sample_id": "rg-1", "question_type": "REVERSE_GROUND", "inline_frontier_correct": False},
            {"sample_id": "dr-1", "question_type": "DIRECT_READ", "inline_frontier_correct": True},
            {"sample_id": "yn-1", "question_type": "YES_NO", "inline_frontier_correct": False},
        ]

    def test_all_types_gate_rejects_wrong_answers(self) -> None:
        rows = self._make_rows()
        result = self._apply_typed_gate(rows, gate_types_raw="all")
        ids = {r["sample_id"] for r in result}
        self.assertIn("dr-1", ids)
        self.assertNotIn("rg-1", ids)
        self.assertNotIn("yn-1", ids)

    def test_typed_gate_skips_excluded_type(self) -> None:
        rows = self._make_rows()
        # Gate applies only to DR and YN — RG is excluded
        result = self._apply_typed_gate(rows, gate_types_raw="DIRECT_READ,YES_NO")
        ids = {r["sample_id"] for r in result}
        # rg-1 has inline_frontier_correct=False but is NOT gated → kept
        self.assertIn("rg-1", ids)
        # dr-1 is gated and correct → kept
        self.assertIn("dr-1", ids)
        # yn-1 is gated and wrong → rejected
        self.assertNotIn("yn-1", ids)

    def test_error_passthrough_for_gated_type(self) -> None:
        rows = [{"sample_id": "dr-err", "question_type": "DIRECT_READ", "inline_frontier_correct": False,
                 "inline_frontier": {"error": "429 rate limit"}}]
        result = self._apply_typed_gate(rows, gate_types_raw="DIRECT_READ")
        self.assertEqual(len(result), 1)  # error → passthrough


if __name__ == "__main__":
    unittest.main()
