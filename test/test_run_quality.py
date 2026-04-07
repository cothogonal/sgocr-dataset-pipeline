from __future__ import annotations

import unittest

from sgocr.run_quality import compute_run_quality


class TestRunQuality(unittest.TestCase):
    def test_quality_uses_rates_and_ambiguity_fraction(self) -> None:
        summary = {
            "accepted_qas": 4,
            "generated_qas": 10,
            "qa_accept_rate": 0.4,
            "images_with_final_rows": 3,
            "failure_counts": {
                "anchor_missing": 2,
                "ambiguous_grounding": 1,
                "reverse_ground_ambiguous": 1,
                "reverse_ground_answer_invalid": 1,
            },
        }
        rows = [
            {"tags": {"question_type": "DIRECT_READ", "ambiguity_level": "high"}, "grounding": {}},
            {"tags": {"question_type": "YES_NO", "ambiguity_level": "high", "yesno_polarity": "negative"}, "grounding": {}},
            {"tags": {"question_type": "REVERSE_GROUND", "ambiguity_level": "low"}, "grounding": {"reverse_ground_scope_preference": "local"}},
            {"tags": {"question_type": "REVERSE_GROUND", "ambiguity_level": "medium"}, "grounding": {"reverse_ground_scope_preference": "mixed"}},
        ]
        metrics = compute_run_quality(summary, rows)
        self.assertEqual(metrics["accepted_qas"], 4)
        self.assertEqual(metrics["yesno_negative"], 1)
        self.assertEqual(metrics["reverse_local"], 1)
        self.assertEqual(metrics["reverse_mixed"], 1)
        self.assertAlmostEqual(metrics["high_ambiguity_fraction"], 0.5)
        self.assertAlmostEqual(metrics["anchor_missing_rate"], 0.2)
        self.assertAlmostEqual(metrics["quality_score"], 7.13)


if __name__ == "__main__":
    unittest.main()
