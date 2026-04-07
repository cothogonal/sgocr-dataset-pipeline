from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStage:
    key: str
    title: str
    code_scope: tuple[str, ...]
    outputs: tuple[str, ...]


PIPELINE_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage(
        key="0",
        title="Contracts, Inventory, and Pilot Slice",
        code_scope=("sgocr.paths", "sgocr.layout", "sgocr.schemas", "sgocr.manifests"),
        outputs=("source_manifest.json",),
    ),
    PipelineStage(
        key="A",
        title="Text Detection + Orientation",
        code_scope=("sgocr.ocr_runtime",),
        outputs=("text_detections.jsonl",),
    ),
    PipelineStage(
        key="B",
        title="OCR Ensemble + Consensus",
        code_scope=("sgocr.ocr_runtime", "sgocr.consensus"),
        outputs=(
            "parseq_readings.jsonl",
            "trocr_small_readings.jsonl",
            "trocr_base_readings.jsonl",
            "consensus_stats.json",
            "text_nodes.jsonl",
        ),
    ),
    PipelineStage(
        key="C",
        title="Dynamic Anchor Discovery + Grounding",
        code_scope=("sgocr.semantic_grounding", "sgocr.full_pipeline_dev40"),
        outputs=("anchor_tags.jsonl", "grounded_anchors.jsonl", "anchors.jsonl"),
    ),
    PipelineStage(
        key="D",
        title="Tuple Construction + Geometric Verification",
        code_scope=("sgocr.semantic_grounding", "sgocr.full_pipeline_dev40"),
        outputs=("verified_tuples.jsonl", "selected_tuples.jsonl"),
    ),
    PipelineStage(
        key="E",
        title="Teacher QA Generation",
        code_scope=("sgocr.teacher.batch", "sgocr.teacher.prompting"),
        outputs=("raw_qa.jsonl",),
    ),
    PipelineStage(
        key="F",
        title="Automated Verification + Packaging",
        code_scope=("sgocr.verify", "sgocr.package"),
        outputs=("ocr_spatial_qa_dataset.jsonl", "build_report.md"),
    ),
    PipelineStage(
        key="G",
        title="Manual Audit Tool and Review",
        code_scope=("sgocr.audit.server", "sgocr.audit.static"),
        outputs=("audit_results.jsonl", "audit_summary.json"),
    ),
)

STAGE_INDEX = {stage.key: stage for stage in PIPELINE_STAGES}
