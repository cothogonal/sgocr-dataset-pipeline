from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sgocr.dev200_eval import SweepRun, _build_pointing_row, compute_frontier_agreement, prepare_bridge_ft_eval


def _sample_row() -> dict:
    return {
        "sample_id": "sample-1",
        "image_id": "img-1",
        "image_path": "/tmp/fake.jpg",
        "image_width": 640,
        "image_height": 480,
        "question": "What does the sign say?",
        "answer": "OPEN",
        "answer_level": "word",
        "anchor_label": "sign",
        "relation": "on",
        "text_bbox": [10.0, 20.0, 30.0, 40.0],
        "tags": {
            "question_type": "DIRECT_READ",
            "answer_type": "text_string",
            "difficulty": "easy",
            "ambiguity_level": "low",
            "image_source": "textocr_val",
        },
        "grounding": {
            "text_bbox": [10.0, 20.0, 30.0, 40.0],
        },
    }


class TestDev200Eval(unittest.TestCase):
    def test_build_pointing_row_sets_zero_grounding_and_vqa_target(self) -> None:
        row = _build_pointing_row(_sample_row())
        self.assertTrue(row["has_vqa_target"])
        self.assertFalse(row["has_grounding_target"])
        self.assertEqual(len(row["soft_target"]), 196)
        self.assertTrue(all(v == 0.0 for v in row["soft_target"]))
        self.assertEqual(row["bbox_xyxy"], [10.0, 20.0, 40.0, 60.0])
        self.assertEqual(row["metadata"]["question_type"], "DIRECT_READ")

    def test_prepare_bridge_ft_eval_writes_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            experiment_dir = root / "exp"
            experiment_dir.mkdir(parents=True)
            with (experiment_dir / "ocr_qa_dataset.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(_sample_row()) + "\n")
            run = SweepRun(
                bundle_id="bundle-x",
                name="run-a",
                experiment_dir=experiment_dir,
                intermediate_dir=root / "intermediate",
            )
            out_dir = prepare_bridge_ft_eval(run, max_steps=123, batch_size=8, grad_accum_steps=4)
            self.assertTrue((out_dir / "sgocr_pointing_train_index.jsonl").exists())
            self.assertTrue((out_dir / "manifest.json").exists())
            script = (out_dir / "run_bridge_ft_1k.sh").read_text(encoding="utf-8")
            self.assertIn("--grounding_loss_weight 0.0", script)
            self.assertIn("--pointing_mix_ratio 0.25", script)
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_rows"], 1)
            self.assertEqual(manifest["train_plan"]["max_steps"], 123)

    def test_compute_frontier_agreement_outputs_pairwise_rates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bench = Path(td)
            rows = [
                {
                    "sample_id": "s1",
                    "provider": "openai",
                    "model": "gpt-5.3-codex",
                    "prediction_norm": "open",
                    "question_type": "DIRECT_READ",
                    "difficulty": "easy",
                    "ambiguity_level": "low",
                },
                {
                    "sample_id": "s1",
                    "provider": "gemini",
                    "model": "gemini-3.1-flash",
                    "prediction_norm": "open",
                    "question_type": "DIRECT_READ",
                    "difficulty": "easy",
                    "ambiguity_level": "low",
                },
                {
                    "sample_id": "s2",
                    "provider": "openai",
                    "model": "gpt-5.3-codex",
                    "prediction_norm": "closed",
                    "question_type": "YES_NO",
                    "difficulty": "medium",
                    "ambiguity_level": "medium",
                },
                {
                    "sample_id": "s2",
                    "provider": "gemini",
                    "model": "gemini-3.1-flash",
                    "prediction_norm": "no",
                    "question_type": "YES_NO",
                    "difficulty": "medium",
                    "ambiguity_level": "medium",
                },
            ]
            with (bench / "predictions.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            summary = compute_frontier_agreement(benchmark_dir=bench)
            self.assertEqual(summary["rows"], 2)
            self.assertAlmostEqual(summary["overall_unanimous_rate"], 0.5)
            self.assertEqual(len(summary["pairwise"]), 1)
            self.assertAlmostEqual(summary["pairwise"][0]["agreement_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
