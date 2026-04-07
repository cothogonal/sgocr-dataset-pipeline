import unittest

from sgocr.layout import BuildLayout
from sgocr.paths import DATA_ROOT, OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, OCR_SPATIAL_QA_RAW_ROOT


class TestBuildLayout(unittest.TestCase):
    def setUp(self) -> None:
        self.layout = BuildLayout()

    def test_layout_roots_stay_in_shared_data_tree(self) -> None:
        self.assertEqual(self.layout.raw_root, OCR_SPATIAL_QA_RAW_ROOT)
        self.assertEqual(self.layout.intermediate_root, OCR_SPATIAL_QA_INTERMEDIATE_ROOT)
        self.assertEqual(self.layout.final_root, OCR_SPATIAL_QA_FINAL_ROOT)
        self.assertTrue(str(self.layout.raw_root).startswith(str(DATA_ROOT)))
        self.assertTrue(str(self.layout.intermediate_root).startswith(str(DATA_ROOT)))
        self.assertTrue(str(self.layout.final_root).startswith(str(DATA_ROOT)))

    def test_expected_artifact_paths(self) -> None:
        self.assertEqual(self.layout.source_manifest.name, "source_manifest.json")
        self.assertEqual(self.layout.consensus_stats.name, "consensus_stats.json")
        self.assertEqual(self.layout.verified_tuples.name, "verified_tuples.jsonl")
        self.assertEqual(self.layout.final_dataset.name, "ocr_spatial_qa_dataset.jsonl")
        self.assertEqual(self.layout.audit_summary.name, "audit_summary.json")


if __name__ == "__main__":
    unittest.main()
