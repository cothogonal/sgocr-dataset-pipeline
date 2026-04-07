import unittest
from pathlib import Path

from sgocr.paths import DATA_ROOT, LOGS_ROOT, REPO_ROOT, SGOCR_ROOT, SRC_ROOT, TEST_ROOT


class TestPaths(unittest.TestCase):
    def test_repo_root_is_core_checkout(self) -> None:
        self.assertEqual(REPO_ROOT, Path("/home/wdree/percy/cothogonal/core"))

    def test_sgocr_roots_live_under_repo(self) -> None:
        self.assertEqual(SGOCR_ROOT, REPO_ROOT / "sgocr")
        self.assertEqual(SRC_ROOT, SGOCR_ROOT / "src")
        self.assertEqual(TEST_ROOT, SGOCR_ROOT / "test")

    def test_data_and_logs_resolve_to_shared_asset_roots(self) -> None:
        self.assertTrue(DATA_ROOT.is_absolute())
        self.assertTrue(LOGS_ROOT.is_absolute())
        self.assertEqual(DATA_ROOT.name, "data")
        self.assertEqual(LOGS_ROOT.name, "logs")


if __name__ == "__main__":
    unittest.main()
