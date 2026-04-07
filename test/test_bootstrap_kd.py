from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sgocr.bootstrap_kd import collect_kd_metadata, compute_resolvability, materialize_bootstrap_kd_dataset


class TestBootstrapKD(unittest.TestCase):
    def test_compute_resolvability_thresholds(self) -> None:
        resolvable = compute_resolvability([0, 0, 120, 0, 120, 60, 0, 60], 224, 224)
        self.assertTrue(resolvable["passes"])
        tiny = compute_resolvability([0, 0, 20, 0, 20, 10, 0, 10], 224, 224)
        self.assertFalse(tiny["passes"])
        self.assertLess(tiny["text_px_w"], resolvable["text_px_w"])

    def test_collect_kd_metadata_tracks_neighbors_and_competing(self) -> None:
        primary = {
            "node_id": "a",
            "text": "HELLO",
            "polygon": [10, 10, 30, 10, 30, 30, 10, 30],
            "bbox": [10, 10, 30, 30],
            "region_key": "cc",
            "resolvable": True,
            "confidence": 1.0,
            "consensus_tier": "bootstrap_gt",
        }
        other_same = {
            "node_id": "b",
            "text": "WORLD",
            "polygon": [40, 10, 60, 10, 60, 30, 40, 30],
            "bbox": [40, 10, 60, 30],
            "region_key": "cc",
            "resolvable": True,
            "confidence": 1.0,
            "consensus_tier": "bootstrap_gt",
        }
        other_far = {
            "node_id": "c",
            "text": "EDGE",
            "polygon": [120, 10, 140, 10, 140, 30, 120, 30],
            "bbox": [120, 10, 140, 30],
            "region_key": "cr",
            "resolvable": False,
            "confidence": 1.0,
            "consensus_tier": "bootstrap_gt",
        }
        anchors = [
            {"label": "center of the image", "box": [0, 0, 100, 100], "region_key": "cc", "source": "bootstrap_region"},
            {"label": "right side of the image", "box": [100, 0, 200, 100], "region_key": "cr", "source": "bootstrap_region"},
        ]
        kd = collect_kd_metadata(primary, [primary, other_same, other_far], anchors, (200, 100))
        self.assertEqual(kd["competing_tuples"], 1)
        self.assertEqual(kd["neighboring_text"][0]["node_id"], "b")
        self.assertEqual(kd["neighboring_text"][0]["relation_to_primary"], "right_of")
        self.assertEqual(kd["nearby_anchors"][0]["region_key"], "cc")

    def test_materialize_bootstrap_kd_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_json = root / "textocr.json"
            source_dir = root / "source_exp"
            out_dir = root / "out_exp"
            intermediate_dir = root / "intermediate"
            source_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "imgs": {
                    "img-1": {"width": 224, "height": 224, "file_name": "img-1.jpg"},
                },
                "anns": {
                    "img-1_0": {
                        "id": "img-1_0",
                        "image_id": "img-1",
                        "utf8_string": "HELLO",
                        "bbox": [10, 10, 60, 30],
                        "points": [10, 10, 70, 10, 70, 40, 10, 40],
                    },
                    "img-1_1": {
                        "id": "img-1_1",
                        "image_id": "img-1",
                        "utf8_string": "WORLD",
                        "bbox": [120, 10, 60, 30],
                        "points": [120, 10, 180, 10, 180, 40, 120, 40],
                    },
                },
            }
            raw_json.write_text(json.dumps(payload), encoding="utf-8")
            (source_dir / "raw_results.jsonl").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "image_id": "img-1",
                        "ann_id": "img-1_0",
                        "tuple": {
                            "image_id": "img-1",
                            "image_path": "data/vm_ssl/raw/textocr_trainval/img-1.jpg",
                            "image_width": 224,
                            "image_height": 224,
                            "source_split": "val",
                            "source_dataset": "textocr_bootstrap",
                            "ann_id": "img-1_0",
                            "answer": "HELLO",
                            "answer_normalized": "hello",
                            "text_polygon": [10, 10, 70, 10, 70, 40, 10, 40],
                            "text_bbox": [10, 10, 60, 30],
                            "region_key": "cc",
                            "anchor_label": "center of the image",
                            "anchor_synonyms": ["center of the image"],
                            "relation": "in",
                            "unique": True,
                            "answer_level": "word",
                            "valid_text_count": 2,
                            "area_fraction": 0.04,
                            "text_length": 5,
                            "score": 4.0,
                            "density_bucket": "low",
                            "area_bucket": "large",
                        },
                        "result": {"provider": "gemini", "model": "gemini-2.5-flash", "usage": {"promptTokenCount": 100, "candidatesTokenCount": 10, "totalTokenCount": 150}},
                        "items": [{"question": "What text is in the center of the image?", "answer": "HELLO"}],
                        "validations": [{"accepted": True}],
                        "summary": {"accepted_count": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (source_dir / "summary.json").write_text(
                json.dumps({"experiment": {"provider": "gemini", "model": "gemini-2.5-flash", "prompt_variant": "natural_2q"}}),
                encoding="utf-8",
            )
            summary = materialize_bootstrap_kd_dataset(
                source_experiment_dir=source_dir,
                raw_json_path=raw_json,
                out_dir=out_dir,
                intermediate_dir=intermediate_dir,
            )
            self.assertEqual(summary["input_tuple_count"], 1)
            self.assertEqual(summary["resolvable_tuple_count"], 1)
            accepted_rows = [json.loads(line) for line in (out_dir / "accepted_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(accepted_rows), 1)
            self.assertIn("kd_metadata", accepted_rows[0])
            self.assertEqual(accepted_rows[0]["kd_metadata"]["competing_tuples"], 0)


if __name__ == "__main__":
    unittest.main()
