from __future__ import annotations

import os
from pathlib import Path


SGOCR_ROOT = Path(os.environ.get("SGOCR_ROOT", Path(__file__).resolve().parents[2])).resolve()
_MONOREPO_ROOT = SGOCR_ROOT.parent
REPO_ROOT = Path(
    os.environ.get(
        "SGOCR_REPO_ROOT",
        _MONOREPO_ROOT
        if (_MONOREPO_ROOT / "sgocr").resolve() == SGOCR_ROOT and (_MONOREPO_ROOT / "DATASETS.md").exists()
        else SGOCR_ROOT,
    )
).resolve()
SRC_ROOT = SGOCR_ROOT / "src"
TEST_ROOT = SGOCR_ROOT / "test"

DATA_ROOT = Path(os.environ.get("SGOCR_DATA_ROOT", REPO_ROOT / "data")).resolve()
LOGS_ROOT = Path(os.environ.get("SGOCR_LOGS_ROOT", REPO_ROOT / "logs")).resolve()

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
