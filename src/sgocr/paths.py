from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SGOCR_ROOT = REPO_ROOT / "sgocr"
SRC_ROOT = SGOCR_ROOT / "src"
TEST_ROOT = SGOCR_ROOT / "test"

# Shared assets may resolve into the linked old repo; use resolved paths for data and logs.
DATA_ROOT = (REPO_ROOT / "data").resolve()
LOGS_ROOT = (REPO_ROOT / "logs").resolve()

OCR_SPATIAL_QA_ROOT = DATA_ROOT / "ocr_spatial_qa"
OCR_SPATIAL_QA_RAW_ROOT = OCR_SPATIAL_QA_ROOT / "raw"
OCR_SPATIAL_QA_INTERMEDIATE_ROOT = OCR_SPATIAL_QA_ROOT / "intermediate"
OCR_SPATIAL_QA_FINAL_ROOT = OCR_SPATIAL_QA_ROOT / "final"


def repo_relative(path: Path) -> str:
    """Return a repo-relative path string when possible."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
