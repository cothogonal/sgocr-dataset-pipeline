import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from sgocr.bootstrap import (
    REGION_PHRASES,
    build_and_write_dev_subset,
    is_valid_text,
    normalize_answer,
    region_key_for_bbox,
)


class TestBootstrap(unittest.TestCase):
    def test_normalize_answer(self) -> None:
        self.assertEqual(normalize_answer(" OPEN 24 HOURS! "), "open 24 hours")

    def test_valid_text_filter(self) -> None:
        self.assertTrue(is_valid_text("RICHARD"))
        self.assertFalse(is_valid_text("."))
        self.assertFalse(is_valid_text(" "))

    def test_region_key(self) -> None:
        self.assertEqual(region_key_for_bbox([0, 0, 10, 10], 90, 90), "ul")
        self.assertEqual(region_key_for_bbox([40, 40, 10, 10], 90, 90), "cc")
        self.assertEqual(region_key_for_bbox([80, 80, 5, 5], 90, 90), "lr")
        self.assertEqual(REGION_PHRASES["cc"], "center of the image")

    def test_build_dev_subset_on_tiny_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            images_root = tmp_path / "images"
            images_root.mkdir()
            Image.new("RGB", (100, 100), color=(255, 255, 255)).save(images_root / "img1.jpg")
            Image.new("RGB", (100, 100), color=(255, 255, 255)).save(images_root / "img2.jpg")
            payload = {
                "imgs": {
                    "img1": {"id": "img1", "width": 100, "height": 100, "file_name": "train/img1.jpg"},
                    "img2": {"id": "img2", "width": 100, "height": 100, "file_name": "train/img2.jpg"},
                },
                "anns": {
                    "a1": {"id": "a1", "image_id": "img1", "bbox": [0, 0, 20, 20], "utf8_string": "HELLO", "points": [0, 0, 20, 0, 20, 20, 0, 20]},
                    "a2": {"id": "a2", "image_id": "img2", "bbox": [60, 60, 20, 20], "utf8_string": "WORLD", "points": [60, 60, 80, 60, 80, 80, 60, 80]},
                },
            }
            raw_json = tmp_path / "val.json"
            raw_json.write_text(json.dumps(payload), encoding="utf-8")
            manifest = build_and_write_dev_subset(
                raw_json_path=raw_json,
                images_root=images_root,
                manifest_path=tmp_path / "manifest.json",
                tuples_path=tmp_path / "tuples.jsonl",
                notes_path=tmp_path / "notes.md",
                limit=2,
                seed=42,
            )
            self.assertEqual(manifest["image_count"], 2)
            self.assertTrue((tmp_path / "manifest.json").is_file())
            self.assertTrue((tmp_path / "tuples.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
