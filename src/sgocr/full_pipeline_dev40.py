from __future__ import annotations

import json
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .device import resolve_device
from .io import load_jsonl

from .bootstrap import write_json, write_jsonl
from .bootstrap_kd import bbox_xywh_to_xyxy
from .dev40_complete import (
    annotate_inline_frontier,
    build_question_candidates,
    enforce_type_constraints,
    row_to_final_sample,
    run_teacher_batches,
    sample_id_for_candidate,
    select_candidates,
)
from .nemotron_frontend import run_nemotron_ocr_stage
from .ocr_runtime import PARSeqRecognizer, PaddleOCRRecognizer, TrOCRRecognizer
from .run_quality import compute_run_quality
from .semantic_dev40_tuning import load_semantic_dev40_tuning

from .pipeline.tuple_builder import *  # noqa: F401,F403
from .pipeline.stages import *  # noqa: F401,F403

from .pipeline.stages import (
    RUNTIME_MODELS,
    SEMANTIC_PROMPT_VARIANT,
    _anchor_relabel_model_name,
    _make_stage_progress_logger,
    _ocr_runtime_signature,
    _semantic_runtime_signature,
    build_consensus_nodes,
    build_resolvability_stats,
    load_image_specs,
    run_anchor_stage,
    run_detection_stage,
    run_recognition_stage,
)
from .pipeline.tuple_builder import (
    _filter_subsumed_rows,
    _order_component_nodes,
    build_verified_tuples,
    recompute_text_node_resolvability,
)
from .pipeline.stages import (
    QwenAnchorGrounderVLLM,
    refine_anchor_labels,
)
from .pipeline.tuple_builder import _anchor_area_fraction  # noqa: F401


def build_dev40_semantic_dataset(
    *,
    source_experiment_dir: Path,
    out_dir: Path,
    intermediate_dir: Path,
    cache_intermediate_dir: Path | None = None,
    model: str = "gemini-2.5-flash",
    device: str = "auto",
    workers: int = 4,
    max_side: int = 768,
    target_per_image: int = 4,
    max_detections: int = 72,
    grounding_threshold: float = 0.36,
    max_tags_per_image: int = 8,
    cache_level: str = "verified",
) -> dict[str, Any]:
    pipeline_start = time.time()
    tuning = load_semantic_dev40_tuning()
    runtime_models = dict(RUNTIME_MODELS)
    runtime_models["ocr_frontend"] = tuning.ocr_frontend
    runtime_models["anchor_tag_discovery_backend"] = tuning.anchor_tag_discovery_backend
    runtime_models["anchor_candidate_backend"] = tuning.anchor_candidate_backend
    runtime_models["qwen_anchor_inventory_mode"] = tuning.qwen_anchor_inventory_mode
    runtime_models["qwen_anchor_inventory_pass_count"] = tuning.qwen_anchor_inventory_pass_count
    runtime_models["qwen_anchor_inventory_temperature"] = tuning.qwen_anchor_inventory_temperature
    runtime_models["qwen_anchor_inventory_consensus_iou"] = tuning.qwen_anchor_inventory_consensus_iou
    runtime_models["qwen_anchor_inventory_min_support"] = tuning.qwen_anchor_inventory_min_support
    if tuning.anchor_candidate_backend == "qwen3_vl_vllm" or tuning.anchor_tag_discovery_backend == "qwen3_vl_vllm":
        runtime_models["grounder"] = tuning.qwen_anchor_model
    elif tuning.anchor_candidate_backend == "gemma4_ollama":
        runtime_models["grounder"] = f"gemma4_ollama:{tuning.gemma_ollama_model}"
    if tuning.sam3_refine_mode != "none":
        runtime_models["sam3_refiner"] = "facebook/sam3"
    image_specs, image_source_map = load_image_specs(source_experiment_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    runtime_device = resolve_device(device)
    cached_verified_tuples: list[dict[str, Any]] | None = None
    cached_anchor_tag_rows: list[dict[str, Any]] | None = None
    cached_grounded_anchor_rows: list[dict[str, Any]] | None = None
    cached_tuple_debug: dict[str, Any] | None = None

    cache_compatible = False
    if cache_intermediate_dir and (cache_intermediate_dir / "runtime_models.json").exists():
        try:
            cached_runtime_models = json.loads((cache_intermediate_dir / "runtime_models.json").read_text(encoding="utf-8"))
            if cache_level == "ocr":
                cache_compatible = _ocr_runtime_signature(cached_runtime_models) == _ocr_runtime_signature(runtime_models)
            elif cache_level == "verified":
                cache_compatible = (
                    _ocr_runtime_signature(cached_runtime_models) == _ocr_runtime_signature(runtime_models)
                    and _semantic_runtime_signature(cached_runtime_models) == _semantic_runtime_signature(runtime_models)
                )
            else:
                cache_compatible = cached_runtime_models == runtime_models
        except Exception:
            cache_compatible = False

    if cache_level not in {"none", "ocr", "verified"}:
        raise ValueError(f"Unsupported cache_level: {cache_level}")

    if cache_compatible and cache_intermediate_dir and cache_level != "none" and (cache_intermediate_dir / "text_nodes.jsonl").exists():
        print(
            f"[stage:pipeline] cache_reuse cache_level={cache_level} cache_dir={cache_intermediate_dir}",
            flush=True,
        )
        text_nodes = load_jsonl(cache_intermediate_dir / "text_nodes.jsonl")
        consensus_stats = json.loads((cache_intermediate_dir / "consensus_stats.json").read_text(encoding="utf-8"))
        detection_rows = load_jsonl(cache_intermediate_dir / "text_detections.jsonl") if (cache_intermediate_dir / "text_detections.jsonl").exists() else []
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
        ):
            src = cache_intermediate_dir / filename
            if src.exists():
                shutil.copy2(src, intermediate_dir / filename)
        if cache_level == "verified":
            for filename in ("anchor_tags.jsonl", "grounded_anchors.jsonl", "verified_tuples.jsonl"):
                src = cache_intermediate_dir / filename
                if src.exists():
                    shutil.copy2(src, intermediate_dir / filename)
            if (cache_intermediate_dir / "verified_tuples.jsonl").exists():
                cached_verified_tuples = load_jsonl(cache_intermediate_dir / "verified_tuples.jsonl")
            if (cache_intermediate_dir / "anchor_tags.jsonl").exists():
                cached_anchor_tag_rows = load_jsonl(cache_intermediate_dir / "anchor_tags.jsonl")
            if (cache_intermediate_dir / "grounded_anchors.jsonl").exists():
                cached_grounded_anchor_rows = load_jsonl(cache_intermediate_dir / "grounded_anchors.jsonl")
            if cached_verified_tuples is not None:
                cached_tuple_debug = {
                    "word_tuples": len([row for row in cached_verified_tuples if row.get("answer_level") == "word"]),
                    "sign_tuples": len([row for row in cached_verified_tuples if row.get("answer_level") == "sign"]),
                    "dropped_no_anchor": None,
                    "cache_reused": True,
                }
    else:
        if tuning.ocr_frontend == "nemotron_v2":
            print(f"[stage:pipeline] start nemotron_ocr images={len(image_specs)}", flush=True)
            detections_by_image, detection_rows, detection_summary, text_nodes, consensus_stats = run_nemotron_ocr_stage(
                image_specs=image_specs,
                image_source_map=image_source_map,
                max_detections=max_detections,
            )
            write_jsonl(intermediate_dir / "text_detections.jsonl", detection_rows)
            write_jsonl(intermediate_dir / "text_nodes.jsonl", text_nodes)
            write_json(intermediate_dir / "consensus_stats.json", consensus_stats)
        else:
            print(f"[stage:pipeline] start classic_ocr images={len(image_specs)}", flush=True)
            detections_by_image, detection_rows, detection_summary = run_detection_stage(
                image_specs=image_specs,
                device=runtime_device,
                max_detections=max_detections,
            )
            write_jsonl(intermediate_dir / "text_detections.jsonl", detection_rows)

            parseq_rows = run_recognition_stage(
                image_specs=image_specs,
                detections_by_image=detections_by_image,
                recognizer=PARSeqRecognizer(device=runtime_device),
                model_key="parseq",
            )
            write_jsonl(intermediate_dir / "parseq_readings.jsonl", parseq_rows)

            ppocr_server_rows = run_recognition_stage(
                image_specs=image_specs,
                detections_by_image=detections_by_image,
                recognizer=PaddleOCRRecognizer("PP-OCRv5_server_rec", device=runtime_device),
                model_key="ppocrv5_server",
            )
            write_jsonl(intermediate_dir / "ppocrv5_server_readings.jsonl", ppocr_server_rows)

            trocr_large_rows = run_recognition_stage(
                image_specs=image_specs,
                detections_by_image=detections_by_image,
                recognizer=TrOCRRecognizer("microsoft/trocr-large-printed", device=runtime_device),
                model_key="trocr_large",
            )
            write_jsonl(intermediate_dir / "trocr_large_readings.jsonl", trocr_large_rows)
            if runtime_device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            text_nodes, consensus_stats = build_consensus_nodes(
                image_specs=image_specs,
                image_source_map=image_source_map,
                detections_by_image=detections_by_image,
                reading_rows=parseq_rows + ppocr_server_rows + trocr_large_rows,
            )
            write_jsonl(intermediate_dir / "text_nodes.jsonl", text_nodes)
            write_json(intermediate_dir / "consensus_stats.json", consensus_stats)

    write_json(intermediate_dir / "runtime_models.json", runtime_models)

    if tuning.ocr_frontend != "nemotron_v2":
        text_nodes = recompute_text_node_resolvability(text_nodes)
    write_jsonl(intermediate_dir / "text_nodes.jsonl", text_nodes)

    resolvable_nodes = [node for node in text_nodes if node["resolvable"]]
    print(
        f"[stage:pipeline] resolvability text_nodes={len(text_nodes)} resolvable_nodes={len(resolvable_nodes)} elapsed_s={time.time() - pipeline_start:.1f}",
        flush=True,
    )
    resolvability_stats = build_resolvability_stats(text_nodes)
    write_jsonl(intermediate_dir / "text_nodes_resolvable.jsonl", resolvable_nodes)
    write_json(intermediate_dir / "resolvability_stats.json", resolvability_stats)

    if cached_verified_tuples is not None and cached_anchor_tag_rows is not None and cached_grounded_anchor_rows is not None:
        anchor_tag_rows = cached_anchor_tag_rows
        grounded_anchor_rows = cached_grounded_anchor_rows
        verified_tuples = cached_verified_tuples
        tuple_debug = cached_tuple_debug or {}
        best_anchor_by_node = {
            (str(row["image_id"]), str(row["node_id"])): dict(row["top_candidates"][0])
            for row in grounded_anchor_rows
            if row.get("top_candidates")
        }
    else:
        print(
            f"[stage:pipeline] start anchor_stage resolvable_nodes={len(resolvable_nodes)}",
            flush=True,
        )
        anchor_tag_rows, grounded_anchor_rows, best_anchor_by_node = run_anchor_stage(
            image_specs=image_specs,
            resolvable_nodes=resolvable_nodes,
            device=runtime_device,
            max_tags_per_image=max_tags_per_image,
            grounding_threshold=grounding_threshold,
        )
        write_jsonl(intermediate_dir / "anchor_tags.jsonl", anchor_tag_rows)
        write_jsonl(intermediate_dir / "grounded_anchors.jsonl", grounded_anchor_rows)
        if runtime_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[stage:pipeline] start verified_tuple_build grounded_anchor_rows={len(grounded_anchor_rows)}",
            flush=True,
        )
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
        print(
            f"[stage:pipeline] start anchor_relabel verified_tuples={len(verified_tuples)} model={relabel_model}",
            flush=True,
        )
        verified_tuples = refine_anchor_labels(verified_tuples, model_name=relabel_model)
    write_jsonl(intermediate_dir / "verified_tuples.jsonl", verified_tuples)

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    image_batches: list[dict[str, Any]] = []
    question_type_counts = Counter()
    selected_question_type_counts = Counter()
    per_image_tuples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in verified_tuples:
        per_image_tuples[str(row["image_id"])].append(row)
    for spec in image_specs:
        image_tuples = per_image_tuples.get(spec["image_id"], [])
        image_candidates: list[dict[str, Any]] = []
        for tuple_row in image_tuples:
            tuple_candidates = build_question_candidates(tuple_row, image_tuples)
            image_candidates.extend(tuple_candidates)
            question_type_counts.update(candidate["question_type"] for candidate in tuple_candidates)
        candidate_rows.extend(image_candidates)
        selected = select_candidates(image_candidates, target_count=target_per_image)
        selected = enforce_type_constraints(selected, image_candidates, target=target_per_image)
        selected_rows.extend(selected)
        selected_question_type_counts.update(candidate["question_type"] for candidate in selected)
        image_batches.append(
            {
                "image_id": spec["image_id"],
                "image_path": spec["image_path"],
                "selected_candidates": [{**candidate, "candidate_index": idx} for idx, candidate in enumerate(selected, start=1)],
            }
        )
    print(
        f"[stage:pipeline] candidate_selection verified_tuples={len(verified_tuples)} candidate_tuples={len(candidate_rows)} selected_tuples={len(selected_rows)} elapsed_s={time.time() - pipeline_start:.1f}",
        flush=True,
    )

    write_jsonl(intermediate_dir / "candidate_tuples.jsonl", candidate_rows)
    write_jsonl(intermediate_dir / "selected_tuples.jsonl", selected_rows)

    print(
        f"[stage:pipeline] start teacher_generation image_batches={len(image_batches)} selected_tuples={len(selected_rows)}",
        flush=True,
    )
    teacher_results = run_teacher_batches(
        image_batches=image_batches,
        model=model,
        max_side=max_side,
        workers=workers,
    )
    write_jsonl(out_dir / "raw_qa.jsonl", teacher_results["batch_rows"])

    raw_results = []
    final_rows = []
    failure_counts = Counter()
    for row in teacher_results["sample_rows"]:
        raw_results.append(row)
        if row["ok"] and row["summary"]["accepted_count"] == 1:
            final_rows.append(row_to_final_sample(row, model=model, prompt_variant=SEMANTIC_PROMPT_VARIANT))
        else:
            failure_counts[row.get("failure_reason") or "validation_failed"] += 1

    rejected_by_stage: dict[str, str] = {
        str(row.get("sample_id") or ""): str(row.get("failure_reason") or "validation_failed")
        for row in raw_results
        if not (row.get("ok") and (row.get("summary") or {}).get("accepted_count") == 1)
    }

    def _mark_stage_rejections(before_rows: list[dict], after_rows: list[dict], stage: str) -> None:
        after_ids = {str(row.get("sample_id") or "") for row in after_rows}
        for row in before_rows:
            sample_id = str(row.get("sample_id") or "")
            if sample_id and sample_id not in after_ids:
                rejected_by_stage[sample_id] = stage

    inline_frontier_summary = annotate_inline_frontier(
        final_rows,
        model=str(load_semantic_dev40_tuning().inline_frontier_model),
        max_side=max_side,
        workers=workers,
    )

    # --- Inline frontier gate ---
    # Reject rows where the frontier model (shown the image) got the answer wrong.
    # This is a verification gate using already-computed data — no extra API cost.
    # "Tiny natural error": some genuinely hard questions may fail; accepted here as inherent noise.
    #
    # Error-passthrough: if the frontier eval API call failed (e.g. rate-limited 429), the row is
    # treated as "pass" — only rows with a valid model judgment of "wrong" are rejected.
    # Without this, rate limiting causes 100% rejection (all errors → correct=False → all rejected).
    if tuning.inline_frontier_gate_enabled and final_rows:
        pre_gate = len(final_rows)
        wf1_floor = float(tuning.inline_frontier_gate_word_f1_floor)

        # Determine which question types the gate applies to.
        # "all" (default) applies the gate to every row; a comma-separated list restricts it.
        _gate_types_raw = str(tuning.inline_frontier_gate_question_types or "all").strip()
        if _gate_types_raw.lower() == "all":
            _gate_question_types: frozenset[str] | None = None
        else:
            _gate_question_types = frozenset(t.strip().upper() for t in _gate_types_raw.split(",") if t.strip())

        def _frontier_eval_errored(row: dict) -> bool:
            return bool((row.get("inline_frontier") or {}).get("error"))

        def _gated(row: dict) -> bool:
            """True if this row's question type is subject to the frontier gate."""
            if _gate_question_types is None:
                return True
            return str(row.get("question_type") or "").upper() in _gate_question_types

        before_frontier_gate = list(final_rows)
        if wf1_floor < 0.0:
            # Standard binary gate: keep rows the frontier model answered correctly, or where eval
            # errored, or where the question type is excluded from the gate.
            final_rows = [
                row for row in final_rows
                if not _gated(row) or row.get("inline_frontier_correct") is True or _frontier_eval_errored(row)
            ]
        else:
            # Lenient gate: also accept rows where word-F1 meets the floor, or where eval errored.
            # word_f1 is non-zero only for REVERSE_GROUND; all other types use soft_correct.
            def _passes_lenient_gate(row: dict) -> bool:
                if not _gated(row):
                    return True
                if row.get("inline_frontier_correct") is True or _frontier_eval_errored(row):
                    return True
                word_f1 = float(
                    (row.get("inline_frontier") or {}).get("score", {}).get("word_f1") or 0.0
                )
                return word_f1 >= wf1_floor

            final_rows = [row for row in final_rows if _passes_lenient_gate(row)]
        _mark_stage_rejections(before_frontier_gate, final_rows, "inline_frontier_gate")

        frontier_gate_errored = sum(1 for row in final_rows if _gated(row) and _frontier_eval_errored(row))
        frontier_gate_rejected = pre_gate - len(final_rows)
        frontier_gate_type_skipped = sum(1 for row in final_rows if not _gated(row))
        failure_counts["inline_frontier_gate_rejected"] = frontier_gate_rejected
        print(
            f"[inline_frontier_gate] pre={pre_gate} accepted={len(final_rows)} rejected={frontier_gate_rejected}"
            f" errored_passthrough={frontier_gate_errored} type_skipped={frontier_gate_type_skipped}"
            f" wf1_floor={wf1_floor:.2f}",
            flush=True,
        )

    # --- Vision dependence gate ---
    # Run text-only eval on each row; reject rows where the answer is derivable without the image.
    # One extra Gemini Flash call per row — the empirical verification that vision is actually required.
    if tuning.vision_dependence_gate_enabled and final_rows:
        from .dev200_eval import apply_vision_dependence_gate
        pre_gate = len(final_rows)
        before_vdep_gate = list(final_rows)
        final_rows, vdep_stats = apply_vision_dependence_gate(
            final_rows,
            model=str(tuning.inline_frontier_model),
            workers=workers,
        )
        _mark_stage_rejections(before_vdep_gate, final_rows, "vision_dependence_gate")
        failure_counts["vision_dependence_gate_rejected"] = vdep_stats["rejected"]
        print(f"[vision_dependence_gate] {vdep_stats}", flush=True)

    # --- RG vision-dependence check ---
    # Single cross-model text-only call (OpenAI) per REVERSE_GROUND row.
    # Non-RG rows pass through unconditionally. Separate model family from the Gemini teacher
    # avoids self-selection bias. Error → conservative keep.
    # When rg_leakage_correction_enabled: rejected rows where the only leakage is a color/shape
    # token in the question are corrected (token stripped) and re-checked before final discard.
    if tuning.rg_vdep_check_enabled and final_rows:
        from .dev200_eval import apply_rg_vdep_check
        before_rg_vdep = list(final_rows)
        final_rows, rg_vdep_stats = apply_rg_vdep_check(
            final_rows,
            model=str(tuning.rg_vdep_model),
            workers=workers,
            correction_enabled=tuning.rg_leakage_correction_enabled,
        )
        _mark_stage_rejections(before_rg_vdep, final_rows, "rg_vdep_check")
        failure_counts["rg_vdep_rejected"] = rg_vdep_stats["rejected_rg"]
        print(f"[rg_vdep_check] {rg_vdep_stats}", flush=True)

    # --- RG leaky-label hard reject ---
    # Structurally reject REVERSE_GROUND rows where the anchor_label contains a color
    # or shape token. These rows expose the visual element's identity in the question
    # text, making them answerable without the image. This is a zero-cost structural
    # filter — no API calls — that directly targets the "leaky-label candidates" flagged
    # in ita08 diagnostics (11–14 per variant, all non-corrected by the broken vdep check).
    if tuning.rg_leaky_label_hard_reject_enabled and final_rows:
        _rg_color_tokens: frozenset[str] = frozenset({
            "red", "blue", "green", "brown", "white", "black", "gray", "grey",
            "yellow", "orange", "purple", "pink", "silver", "gold",
        })
        _rg_shape_tokens: frozenset[str] = frozenset({
            "rectangular", "circular", "square", "oval", "round",
            "triangular", "hexagonal", "cylindrical", "spherical",
            "wedge", "segment", "emblem", "badge", "bar",
        })
        pre_rg_reject = len(final_rows)

        def _has_leaky_label(row: dict) -> bool:
            if str(row.get("question_type") or "") != "REVERSE_GROUND":
                return False
            label_words = str(row.get("anchor_label") or "").lower().split()
            return any(w in _rg_color_tokens or w in _rg_shape_tokens for w in label_words)

        before_rg_leaky_reject = list(final_rows)
        final_rows = [row for row in final_rows if not _has_leaky_label(row)]
        _mark_stage_rejections(before_rg_leaky_reject, final_rows, "rg_leaky_label_hard_reject")
        rg_leaky_rejected = pre_rg_reject - len(final_rows)
        failure_counts["rg_leaky_label_rejected"] = rg_leaky_rejected
        print(
            f"[rg_leaky_label_hard_reject] pre={pre_rg_reject} rejected={rg_leaky_rejected}"
            f" remaining={len(final_rows)}",
            flush=True,
        )

    print(
        f"[stage:pipeline] finalize raw_results={len(raw_results)} final_rows={len(final_rows)} elapsed_s={time.time() - pipeline_start:.1f}",
        flush=True,
    )

    accepted_sample_ids = {str(row.get("sample_id") or "") for row in final_rows}
    rejected_rows = []
    for row in raw_results:
        sample_id = str(row.get("sample_id") or "")
        if sample_id in accepted_sample_ids:
            continue
        annotated_row = dict(row)
        annotated_row["rejection_stage"] = rejected_by_stage.get(sample_id, "post_teacher_filter")
        annotated_row["rejection_reason_final"] = rejected_by_stage.get(sample_id, "post_teacher_filter")
        rejected_rows.append(annotated_row)

    write_jsonl(out_dir / "raw_results.jsonl", raw_results)
    write_jsonl(out_dir / "ocr_qa_dataset.jsonl", final_rows)
    write_jsonl(out_dir / "accepted_dataset.jsonl", final_rows)
    write_jsonl(out_dir / "rejected_dataset.jsonl", rejected_rows)

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
        "rejected_qa_rows": len(rejected_rows),
    }
    summary = {
        "experiment": {
            "name": out_dir.name,
            "provider": "gemini",
            "model": model,
            "prompt_variant": SEMANTIC_PROMPT_VARIANT,
            "source_experiment": source_experiment_dir.name,
            "variant": SEMANTIC_PROMPT_VARIANT,
            "workers": workers,
            "max_side": max_side,
            "target_per_image": target_per_image,
            "max_detections": max_detections,
            "grounding_threshold": grounding_threshold,
            "max_tags_per_image": max_tags_per_image,
            "device": runtime_device,
            "runtime_models": runtime_models,
            "tuning": tuning.to_metadata(),
            "cache_level": cache_level,
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
        "resolvability_stats": resolvability_stats,
        "tuple_debug": tuple_debug,
        "detection_summary": detection_summary,
        "mean_question_words": statistics.mean(len(str(row["question"]).split()) for row in final_rows) if final_rows else 0.0,
        "images_with_final_rows": len({row["image_id"] for row in final_rows}),
        "inline_frontier": inline_frontier_summary,
    }
    quality_metrics = compute_run_quality(summary, final_rows)
    summary.update(
        {
            "accepted_rows": int(quality_metrics["accepted_qas"]),
            "inline_frontier_mean": float(quality_metrics["inline_frontier_mean"]),
            "inline_frontier_scored": int(quality_metrics["inline_frontier_scored"]),
            "precision_first_score": float(quality_metrics["precision_first_score"]),
            "sweep_score": float(quality_metrics["sweep_score"]),
            "q3": float(quality_metrics["q3_score"]),
            "quality_score": float(quality_metrics["quality_score"]),
        }
    )
    write_json(out_dir / "summary.json", summary)
    return summary
