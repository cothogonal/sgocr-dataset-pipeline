from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import (
    OCR_SPATIAL_QA_FINAL_ROOT,
    OCR_SPATIAL_QA_INTERMEDIATE_ROOT,
    OCR_SPATIAL_QA_RAW_ROOT,
)


@dataclass(frozen=True)
class BuildLayout:
    """Canonical artifact layout for the OCR Spatial QA dataset build."""

    raw_root: Path = OCR_SPATIAL_QA_RAW_ROOT
    intermediate_root: Path = OCR_SPATIAL_QA_INTERMEDIATE_ROOT
    final_root: Path = OCR_SPATIAL_QA_FINAL_ROOT

    def raw(self, name: str) -> Path:
        return self.raw_root / name

    def intermediate(self, name: str) -> Path:
        return self.intermediate_root / name

    def final(self, name: str) -> Path:
        return self.final_root / name

    @property
    def source_manifest(self) -> Path:
        return self.raw("source_manifest.json")

    @property
    def text_detections(self) -> Path:
        return self.intermediate("text_detections.jsonl")

    @property
    def parseq_readings(self) -> Path:
        return self.intermediate("parseq_readings.jsonl")

    @property
    def ppocr_readings(self) -> Path:
        return self.intermediate("ppocr_readings.jsonl")

    @property
    def trocr_readings(self) -> Path:
        return self.intermediate("trocr_readings.jsonl")

    @property
    def consensus_stats(self) -> Path:
        return self.intermediate("consensus_stats.json")

    @property
    def text_nodes(self) -> Path:
        return self.intermediate("text_nodes.jsonl")

    @property
    def anchor_tags(self) -> Path:
        return self.intermediate("anchor_tags.jsonl")

    @property
    def grounded_anchors(self) -> Path:
        return self.intermediate("grounded_anchors.jsonl")

    @property
    def verified_tuples(self) -> Path:
        return self.intermediate("verified_tuples.jsonl")

    @property
    def raw_qa(self) -> Path:
        return self.intermediate("raw_qa.jsonl")

    @property
    def final_dataset(self) -> Path:
        return self.final("ocr_spatial_qa_dataset.jsonl")

    @property
    def build_report(self) -> Path:
        return self.final("build_report.md")

    @property
    def audit_results(self) -> Path:
        return self.final("audit_results.jsonl")

    @property
    def audit_summary(self) -> Path:
        return self.final("audit_summary.json")
