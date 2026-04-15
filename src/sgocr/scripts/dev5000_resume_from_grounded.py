from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..bootstrap import write_json, write_jsonl
from ..dev40_complete import (
    annotate_inline_frontier,
    build_question_candidates,
    enforce_type_constraints,
    row_to_final_sample,
    run_teacher_batches,
    select_candidates,
)
from ..full_pipeline_dev40 import (
    RUNTIME_MODELS,
    SEMANTIC_PROMPT_VARIANT,
    _anchor_relabel_model_name,
    build_resolvability_stats,
    build_verified_tuples,
    load_image_specs,
    load_jsonl,
    refine_anchor_labels,
)
from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT
from ..semantic_dev40_tuning import load_semantic_dev40_tuning


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resume a dev5000 semantic build from grounded anchors.")
    ap.add_argument("--source-experiment-dir", required=True)
    ap.add_argument("--resume-intermediate-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--intermediate-dir", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    return ap.parse_args()


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    args = parse_args()
    tuning = load_semantic_dev40_tuning()

    source_experiment_dir = Path(args.source_experiment_dir)
    resume_intermediate_dir = Path(args.resume_intermediate_dir)
    out_dir = Path(args.out_dir)
    intermediate_dir = Path(args.intermediate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    _log("[resume] load source image specs")
    image_specs, image_source_map = load_image_specs(source_experiment_dir)

    _log("[resume] load grounded-anchor intermediate artifacts")
    detection_rows = load_jsonl(resume_intermediate_dir / "text_detections.jsonl")
    text_nodes = load_jsonl(resume_intermediate_dir / "text_nodes.jsonl")
    resolvable_nodes = load_jsonl(resume_intermediate_dir / "text_nodes_resolvable.jsonl")
    anchor_tag_rows = load_jsonl(resume_intermediate_dir / "anchor_tags.jsonl")
    grounded_anchor_rows = load_jsonl(resume_intermediate_dir / "grounded_anchors.jsonl")
    consensus_stats = json.loads((resume_intermediate_dir / "consensus_stats.json").read_text(encoding="utf-8"))
    runtime_models = json.loads((resume_intermediate_dir / "runtime_models.json").read_text(encoding="utf-8"))

    for filename in (
        "text_detections.jsonl",
        "parseq_readings.jsonl",
        "ppocrv5_server_readings.jsonl",
        "trocr_large_readings.jsonl",
        "consensus_stats.json",
        "text_nodes.jsonl",
        "text_nodes_resolvable.jsonl",
        "resolvability_stats.json",
        "runtime_models.json",
        "anchor_tags.jsonl",
        "grounded_anchors.jsonl",
    ):
        src = resume_intermediate_dir / filename
        if src.exists():
            (intermediate_dir / filename).write_bytes(src.read_bytes())

    detection_summary = {
        "images": len(image_specs),
        "detected_boxes": len(detection_rows),
        "mean_boxes_per_image": statistics.mean(
            [sum(1 for row in detection_rows if str(row["image_id"]) == spec["image_id"]) for spec in image_specs]
        )
        if image_specs
        else 0.0,
        "median_boxes_per_image": statistics.median(
            [sum(1 for row in detection_rows if str(row["image_id"]) == spec["image_id"]) for spec in image_specs]
        )
        if image_specs
        else 0.0,
        "cache_reused": True,
    }

    best_anchor_by_node = {
        (str(row["image_id"]), str(row["node_id"])): dict(row["top_candidates"][0])
        for row in grounded_anchor_rows
        if row.get("top_candidates")
    }

    _log("[resume] build verified tuples from grounded anchors")
    verified_tuples, tuple_debug = build_verified_tuples(
        image_specs=image_specs,
        all_text_nodes=text_nodes,
        resolvable_nodes=resolvable_nodes,
        best_anchor_by_node=best_anchor_by_node,
        grounded_anchor_rows=grounded_anchor_rows,
        image_source_map=image_source_map,
    )

    relabel_model = _anchor_relabel_model_name(tuning.anchor_relabel_mode)
    if relabel_model:
        _log(f"[resume] refine anchor labels via sync Gemini model={relabel_model}")
        verified_tuples = refine_anchor_labels(verified_tuples, model_name=relabel_model)
    write_jsonl(intermediate_dir / "verified_tuples.jsonl", verified_tuples)

    _log("[resume] build candidates and selected tuples")
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    image_batches: list[dict[str, Any]] = []
    question_type_counts = Counter()
    selected_question_type_counts = Counter()
    per_image_tuples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in verified_tuples:
        per_image_tuples[str(row["image_id"])].append(row)
    for idx, spec in enumerate(image_specs, start=1):
        image_tuples = per_image_tuples.get(spec["image_id"], [])
        image_candidates: list[dict[str, Any]] = []
        for tuple_row in image_tuples:
            tuple_candidates = build_question_candidates(tuple_row, image_tuples)
            image_candidates.extend(tuple_candidates)
            question_type_counts.update(candidate["question_type"] for candidate in tuple_candidates)
        candidate_rows.extend(image_candidates)
        selected = select_candidates(image_candidates, target_count=int(args.target_per_image))
        selected = enforce_type_constraints(selected, image_candidates, target=int(args.target_per_image))
        selected_rows.extend(selected)
        selected_question_type_counts.update(candidate["question_type"] for candidate in selected)
        image_batches.append(
            {
                "image_id": spec["image_id"],
                "image_path": spec["image_path"],
                "selected_candidates": [{**candidate, "candidate_index": cidx} for cidx, candidate in enumerate(selected, start=1)],
            }
        )
        if idx % 250 == 0:
            _log(f"[resume] prepared candidates for {idx}/{len(image_specs)} images")

    write_jsonl(intermediate_dir / "candidate_tuples.jsonl", candidate_rows)
    write_jsonl(intermediate_dir / "selected_tuples.jsonl", selected_rows)

    _log(f"[resume] teacher generation on {len(image_batches)} images workers={int(args.workers)}")
    teacher_results = run_teacher_batches(
        image_batches=image_batches,
        model=str(args.model),
        max_side=int(args.max_side),
        workers=int(args.workers),
    )
    write_jsonl(out_dir / "raw_qa.jsonl", teacher_results["batch_rows"])

    _log("[resume] verifier finalize rows")
    raw_results: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    failure_counts = Counter()
    for row in teacher_results["sample_rows"]:
        raw_results.append(row)
        if row["ok"] and row["summary"]["accepted_count"] == 1:
            final_rows.append(row_to_final_sample(row, model=str(args.model), prompt_variant=SEMANTIC_PROMPT_VARIANT))
        else:
            failure_counts[row.get("failure_reason") or "validation_failed"] += 1

    _log(f"[resume] inline frontier scoring on {len(final_rows)} accepted rows")
    inline_frontier_summary = annotate_inline_frontier(
        final_rows,
        model=str(tuning.inline_frontier_model),
        max_side=int(args.max_side),
        workers=int(args.workers),
    )

    _log("[resume] write final dataset artifacts")
    write_jsonl(out_dir / "raw_results.jsonl", raw_results)
    write_jsonl(out_dir / "ocr_qa_dataset.jsonl", final_rows)
    write_jsonl(out_dir / "accepted_dataset.jsonl", final_rows)

    stage_counts = {
        "images": len(image_specs),
        "detected_boxes": len(detection_rows),
        "text_nodes": len(text_nodes),
        "resolvable_nodes": len(resolvable_nodes),
        "anchor_tag_rows": len(anchor_tag_rows),
        "grounded_nodes": len(best_anchor_by_node),
        "grounded_anchors": len(grounded_anchor_rows),
        "verified_tuples": len(verified_tuples),
        "candidate_tuples": len(candidate_rows),
        "selected_tuples": len(selected_rows),
        "raw_qa_rows": len(raw_results),
        "final_qa_rows": len(final_rows),
    }
    summary = {
        "experiment": {
            "name": out_dir.name,
            "provider": "gemini",
            "model": str(args.model),
            "prompt_variant": SEMANTIC_PROMPT_VARIANT,
            "source_experiment": source_experiment_dir.name,
            "variant": SEMANTIC_PROMPT_VARIANT,
            "workers": int(args.workers),
            "max_side": int(args.max_side),
            "target_per_image": int(args.target_per_image),
            "max_detections": 128,
            "grounding_threshold": 0.28,
            "max_tags_per_image": 14,
            "device": "cuda",
            "runtime_models": runtime_models or RUNTIME_MODELS,
            "tuning": tuning.to_metadata(),
            "cache_level": "resume_from_grounded_sync",
        },
        "same_image_universe_count": len(image_specs),
        "input_tuple_count": len(verified_tuples),
        "resolvable_tuple_count": len(verified_tuples),
        "dropped_tuple_count": 0,
        "generated_qas": len(raw_results),
        "accepted_qas": len(final_rows),
        "qa_accept_rate": (len(final_rows) / len(raw_results)) if raw_results else 0.0,
        "stage_counts": stage_counts,
        "question_type_counts": dict(question_type_counts),
        "selected_question_type_counts": dict(selected_question_type_counts),
        "disabled_question_types": [],
        "failure_counts": dict(failure_counts),
        "consensus_stats": consensus_stats,
        "resolvability_stats": build_resolvability_stats(text_nodes),
        "tuple_debug": tuple_debug,
        "detection_summary": detection_summary,
        "mean_question_words": statistics.mean(len(str(row["question"]).split()) for row in final_rows) if final_rows else 0.0,
        "images_with_final_rows": len({row["image_id"] for row in final_rows}),
        "inline_frontier": inline_frontier_summary,
    }
    write_json(out_dir / "summary.json", summary)
    _log(f"[resume] complete accepted={len(final_rows)} generated={len(raw_results)}")


if __name__ == "__main__":
    main()
