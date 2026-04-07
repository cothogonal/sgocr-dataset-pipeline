from __future__ import annotations

import unittest

from sgocr.consensus import OCRVote, choose_consensus


class TestConsensus(unittest.TestCase):
    def test_strong_consensus_accepts_three_way_match(self) -> None:
        decision = choose_consensus(
            [
                OCRVote("parseq", "OPEN", 0.98),
                OCRVote("trocr_small", "OPEN", 0.91),
                OCRVote("trocr_base", "OPEN", 0.95),
            ]
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.consensus_tier, "strong_consensus")
        self.assertEqual(decision.text, "OPEN")

    def test_standard_consensus_accepts_two_way_match(self) -> None:
        decision = choose_consensus(
            [
                OCRVote("parseq", "HELLO", 0.90),
                OCRVote("trocr_small", "HELLO", 0.88),
                OCRVote("trocr_base", "WORLD", 0.92),
            ]
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.consensus_tier, "standard_consensus")
        self.assertEqual(decision.text, "HELLO")

    def test_edit_rescue_accepts_distance_one(self) -> None:
        decision = choose_consensus(
            [
                OCRVote("parseq", "APPRENTICE", 0.83),
                OCRVote("trocr_small", "APPRENTlCE", 0.81),
                OCRVote("trocr_base", "APPFENTICE", 0.72),
            ]
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.consensus_tier, "edit_rescue")

    def test_low_confidence_drops(self) -> None:
        decision = choose_consensus(
            [
                OCRVote("parseq", "NOPE", 0.21),
                OCRVote("trocr_small", "N0PE", 0.14),
                OCRVote("trocr_base", "MAYBE", 0.19),
            ]
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.failure_reason, "low_confidence")

    def test_single_non_empty_vote_does_not_become_strong_consensus(self) -> None:
        decision = choose_consensus(
            [
                OCRVote("parseq", "111111111111", 0.99),
                OCRVote("ppocr", "", 0.91),
                OCRVote("trocr_large", "***", 0.88),
            ]
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.consensus_tier, "dropped")


if __name__ == "__main__":
    unittest.main()
