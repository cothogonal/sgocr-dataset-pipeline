from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .bootstrap import (
    REGION_PHRASES,
    REGION_SYNONYMS,
    normalize_answer,
    region_key_for_bbox,
    write_json,
    write_jsonl,
)
from .bootstrap_kd import (
    bbox_xywh_to_xyxy,
    bbox_xyxy_to_xywh,
)
from .semantic_grounding import extract_anchor_color
from .semantic_dev40_tuning import load_semantic_dev40_tuning


def normalize_candidate_result(
    candidate: dict[str, Any],
    item: dict[str, Any] | None,
    result: dict[str, Any],
    *,
    batch_error: str | None = None,
) -> dict[str, Any]:
    tuple_row = candidate["tuple"]
    validations, summary, failure_reason = validate_candidate_output(candidate, item)
    if batch_error and failure_reason is None:
        failure_reason = "teacher_batch_failed"
    question = str((item or {}).get("question") or "").strip()
    answer = str((item or {}).get("answer") or "").strip()
    return {
        "sample_id": sample_id_for_candidate(candidate),
        "ok": item is not None and not batch_error,
        "image_id": tuple_row["image_id"],
        "ann_id": tuple_row["ann_id"],
        "tuple": {
            **tuple_row,
            "question_type": candidate["question_type"],
            "answer_source": candidate["answer_source"],
            "answer_type": candidate["answer_type"],
            "yesno_polarity": candidate.get("yesno_polarity"),
            "yesno_distractor_source": candidate.get("yesno_distractor_source"),
            "text_property_type": candidate.get("text_property_type"),
            "anchor_property_type": candidate.get("anchor_property_type"),
            "expected_answer": candidate.get("expected_answer"),
            "queried_text": candidate.get("queried_text"),
            "query_text_reference": candidate.get("query_text_reference"),
            "query_anchor_label": candidate.get("query_anchor_label"),
            "query_anchor_synonyms": list(candidate.get("query_anchor_synonyms") or []),
            "query_anchor_box": candidate.get("query_anchor_box"),
            "query_anchor_local_phrase": candidate.get("query_anchor_local_phrase"),
            "query_anchor_local_synonyms": list(candidate.get("query_anchor_local_synonyms") or []),
            "query_location_phrase": candidate.get("query_location_phrase"),
            "query_location_synonyms": list(candidate.get("query_location_synonyms") or []),
            "query_specific_location_phrase": candidate.get("query_specific_location_phrase"),
            "query_specific_location_synonyms": list(candidate.get("query_specific_location_synonyms") or []),
            "query_relation": candidate.get("query_relation"),
            "query_location_required": bool(candidate.get("query_location_required")),
            "query_group_mode": candidate.get("query_group_mode"),
            "query_group_count": candidate.get("query_group_count"),
            "reverse_ground_scope_preference": candidate.get("reverse_ground_scope_preference"),
            "grounded_exclusion_score": candidate.get("grounded_exclusion_score"),
            "grounded_exclusion_source_tuple_id": candidate.get("grounded_exclusion_source_tuple_id"),
            "candidate_id": candidate["candidate_id"],
            "candidate_index": candidate["candidate_index"],
        },
        "result": result,
        "items": [{"question": question, "answer": answer}] if item is not None else [],
        "validations": [validations],
        "summary": summary,
        "failure_reason": failure_reason,
        "filter_stage": {"reason": failure_reason} if failure_reason else None,
    }


def validate_candidate_output(candidate: dict[str, Any], item: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from .question_builder import (
        TEXT_APPEARANCE_TERMS,
        TEXT_PROPERTY_TYPE_TERMS,
        TEXT_PROPERTY_VISUAL_TYPES,
        ANCHOR_PROPERTY_TYPE_TERMS,
        _GENERIC_TEXT_PROPERTY_ANCHOR_TOKENS,
        _anchor_label_has_generic_tokens,
        _is_high_prior_text_property_answer,
        _unique_anchor_can_skip_global_location,
        ambiguity_requires_explicit_specific,
        candidate_anchor_label,
        candidate_anchor_local_synonyms,
        candidate_anchor_phrases,
        candidate_cheap_ambiguity_proxy_score,
        candidate_disambiguation_cues,
        candidate_has_answer_leakage,
        candidate_location_synonyms,
        candidate_requires_anchor_disambiguation,
        candidate_scene_repeat_group_mode,
        candidate_specific_location_synonyms,
        location_ambiguity_score,
    )

    tuple_row = candidate["tuple"]
    tuning = load_semantic_dev40_tuning()
    if item is None:
        validation = {
            "answer_ok": False,
            "anchor_ok": False,
            "length_ok": False,
            "duplicate_ok": True,
            "accepted": False,
            "question_type_ok": False,
            "mechanical_ok": False,
        }
        return validation, {"generated_count": 0, "accepted_count": 0}, "missing_candidate"

    question = str(item.get("question") or "").strip()
    answer = str(item.get("answer") or "").strip()
    normalized_question = normalize_answer(question)
    normalized_answer = normalize_answer(answer)
    anchor_ok = any(normalize_answer(anchor) in normalized_question or normalize_answer(anchor) in normalized_answer for anchor in candidate_anchor_phrases(candidate))
    location_ok = any(normalize_answer(loc) in normalized_question or normalize_answer(loc) in normalized_answer for loc in candidate_location_synonyms(candidate))
    specific_location_ok = any(normalize_answer(loc) in normalized_question or normalize_answer(loc) in normalized_answer for loc in candidate_specific_location_synonyms(candidate))
    anchor_local_ok = any(normalize_answer(loc) in normalized_question or normalize_answer(loc) in normalized_answer for loc in candidate_anchor_local_synonyms(candidate))
    length_ok = 5 <= len(question.split()) <= 35
    question_type_ok = str(item.get("question_type") or "").strip().upper() == candidate["question_type"]
    answer_ok = False
    mechanical_ok = False
    failure_reason = None
    ambiguity_score = location_ambiguity_score(tuple_row)
    ambiguity_hard = ambiguity_score >= tuning.ambiguity_hard_reject_score
    ambiguity_reverse = ambiguity_score >= tuning.ambiguity_reverse_reject_score
    disambiguated_ok = bool(specific_location_ok or anchor_local_ok)
    explicit_specific_required = bool(ambiguity_hard and ambiguity_requires_explicit_specific(tuple_row))
    reverse_disambiguated_ok = bool(specific_location_ok or (anchor_local_ok and location_ok))
    unique_skip_location = _unique_anchor_can_skip_global_location(candidate)
    strong_specific_grounding_ok = bool(candidate_specific_location_synonyms(candidate) and specific_location_ok)
    disambiguation_required = candidate_requires_anchor_disambiguation(candidate)
    cheap_proxy_score = candidate_cheap_ambiguity_proxy_score(candidate)
    cheap_proxy_hard = bool(tuning.cheap_ambiguity_proxy_enabled and cheap_proxy_score >= tuning.cheap_ambiguity_proxy_reject_score)
    grouped_scene_read = bool(candidate_scene_repeat_group_mode(candidate))
    base_anchor_norm = normalize_answer(candidate_anchor_label(candidate))
    disambiguation_ok = any(
        (norm := normalize_answer(cue))
        and norm != base_anchor_norm
        and (norm in normalized_question or norm in normalized_answer)
        for cue in candidate_disambiguation_cues(candidate)
    )

    if tuning.answer_leakage_filter_enabled and candidate_has_answer_leakage(candidate, question):
        return {
            "answer_ok": False,
            "anchor_ok": anchor_ok,
            "length_ok": length_ok,
            "duplicate_ok": True,
            "accepted": False,
            "question_type_ok": question_type_ok,
            "mechanical_ok": False,
            "cheap_ambiguity_proxy_score": candidate_cheap_ambiguity_proxy_score(candidate),
            "group_mode": candidate_scene_repeat_group_mode(candidate),
            "failure_reason": "answer_leakage_trivial",
        }

    if candidate["question_type"] == "DIRECT_READ":
        answer_ok = normalized_answer == normalize_answer(str(candidate["expected_answer"]))
        mechanical_ok = answer_ok
        if not answer_ok:
            failure_reason = "direct_read_answer_mismatch"
        elif candidate.get("query_location_required") and not unique_skip_location and candidate_specific_location_synonyms(candidate) and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "direct_read_specific_location_missing"
        elif candidate.get("query_location_required") and not unique_skip_location and not location_ok:
            mechanical_ok = False
            failure_reason = "direct_read_location_missing"
        elif explicit_specific_required and not unique_skip_location and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "direct_read_specific_location_missing"
        elif ambiguity_hard and not disambiguated_ok and not strong_specific_grounding_ok:
            mechanical_ok = False
            failure_reason = "ambiguous_grounding"
        elif (
            tuning.dr_ambiguity_reject_score >= 0
            and ambiguity_score >= tuning.dr_ambiguity_reject_score
            and not strong_specific_grounding_ok
        ):
            mechanical_ok = False
            failure_reason = "dr_high_ambiguity"
        elif disambiguation_required and not disambiguation_ok:
            mechanical_ok = False
            failure_reason = "anchor_instance_ambiguous"
        elif cheap_proxy_hard and not grouped_scene_read and not (strong_specific_grounding_ok or disambiguation_ok):
            mechanical_ok = False
            failure_reason = "cheap_proxy_ambiguous"
    elif candidate["question_type"] == "YES_NO":
        answer_ok = normalized_answer == normalize_answer(str(candidate["expected_answer"]))
        queried_ok = normalize_answer(str(candidate["queried_text"])) in normalized_question
        location_required = bool(candidate.get("query_location_required"))
        location_match_ok = specific_location_ok if candidate_specific_location_synonyms(candidate) else (location_ok or anchor_local_ok)
        mechanical_ok = bool(answer_ok and queried_ok and (unique_skip_location or not location_required or location_match_ok))
        if not queried_ok:
            failure_reason = "yesno_query_missing"
        elif not answer_ok:
            failure_reason = "yesno_answer_mismatch"
        elif location_required and not unique_skip_location and candidate_specific_location_synonyms(candidate) and not specific_location_ok:
            failure_reason = "yesno_specific_location_missing"
        elif location_required and not unique_skip_location and not location_match_ok:
            failure_reason = "yesno_location_missing"
        elif explicit_specific_required and not unique_skip_location and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "yesno_specific_location_missing"
        elif ambiguity_hard and not disambiguated_ok:
            mechanical_ok = False
            failure_reason = "ambiguous_grounding"
        elif disambiguation_required and not disambiguation_ok:
            mechanical_ok = False
            failure_reason = "anchor_instance_ambiguous"
        elif cheap_proxy_hard and not grouped_scene_read and not (strong_specific_grounding_ok or disambiguation_ok):
            mechanical_ok = False
            failure_reason = "cheap_proxy_ambiguous"
    elif candidate["question_type"] == "TEXT_PROPERTY":
        property_type = str(candidate["text_property_type"] or "")
        property_terms = TEXT_PROPERTY_TYPE_TERMS.get(property_type, ())
        property_ok = any(normalize_answer(term) in normalized_question for term in property_terms)
        location_required = bool(candidate.get("query_location_required"))
        location_match_ok = specific_location_ok if candidate_specific_location_synonyms(candidate) else (location_ok or anchor_local_ok)
        if property_type in {"word_count", "first_word", "last_word"}:
            answer_ok = normalized_answer == normalize_answer(str(candidate["expected_answer"]))
            mechanical_ok = bool(answer_ok and property_ok and (not location_required or location_match_ok))
        else:
            answer_ok = 1 <= len(answer.split()) <= 4 and len(answer.strip()) >= 2
            reference_text = normalize_answer(str(candidate.get("query_text_reference") or tuple_row["answer"]))
            text_focus_ok = reference_text in normalized_question or any(term in normalized_question.split() for term in TEXT_APPEARANCE_TERMS)
            mechanical_ok = bool(answer_ok and property_ok and text_focus_ok and (not location_required or location_match_ok))
        if not property_ok:
            failure_reason = "text_property_query_missing"
        elif not answer_ok:
            failure_reason = "text_property_answer_mismatch"
        elif property_type not in {"word_count", "first_word", "last_word"} and not (reference_text in normalized_question or any(term in normalized_question.split() for term in TEXT_APPEARANCE_TERMS)):
            failure_reason = "text_property_text_reference_missing"
        elif location_required and candidate_specific_location_synonyms(candidate) and not specific_location_ok:
            failure_reason = "text_property_specific_location_missing"
        elif location_required and not location_match_ok:
            failure_reason = "text_property_location_missing"
        elif (
            tuning.tp_visual_high_prior_answer_filter_enabled
            and property_type in {"text_color", "text_curvature"}
            and _is_high_prior_text_property_answer(answer, property_type)
            and _anchor_label_has_generic_tokens(candidate_anchor_label(candidate), _GENERIC_TEXT_PROPERTY_ANCHOR_TOKENS)
        ):
            mechanical_ok = False
            failure_reason = "text_property_prior_leaky"
        elif explicit_specific_required and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "text_property_specific_location_missing"
        elif ambiguity_hard and not disambiguated_ok:
            mechanical_ok = False
            failure_reason = "ambiguous_grounding"
        elif disambiguation_required and not disambiguation_ok:
            mechanical_ok = False
            failure_reason = "anchor_instance_ambiguous"
        elif cheap_proxy_hard and not grouped_scene_read and not (strong_specific_grounding_ok or disambiguation_ok):
            mechanical_ok = False
            failure_reason = "cheap_proxy_ambiguous"
    elif candidate["question_type"] == "REVERSE_GROUND":
        query_ok = normalize_answer(str(tuple_row["answer"])) in normalized_question
        rg_style = tuning.reverse_ground_answer_style
        min_words = 1 if rg_style == "frontier" else 2
        answer_ok = min_words <= len(answer.split()) <= 10 and (anchor_ok or location_ok or specific_location_ok or anchor_local_ok)
        if not answer_ok and rg_style == "frontier":
            answer_has_content = len(answer.split()) >= min_words and len(answer.strip()) >= 2
            answer_ok = answer_has_content and anchor_ok
        mechanical_ok = bool(query_ok and answer_ok)
        if not query_ok:
            failure_reason = "reverse_ground_query_missing"
        elif not answer_ok:
            failure_reason = "reverse_ground_answer_invalid"
        elif ambiguity_reverse and ambiguity_requires_explicit_specific(tuple_row) and not reverse_disambiguated_ok:
            mechanical_ok = False
            failure_reason = "reverse_ground_ambiguous"
        elif ambiguity_reverse and not (specific_location_ok or anchor_local_ok):
            mechanical_ok = False
            failure_reason = "reverse_ground_ambiguous"
        elif disambiguation_required and not disambiguation_ok:
            mechanical_ok = False
            failure_reason = "anchor_instance_ambiguous"
        elif cheap_proxy_hard and not grouped_scene_read and not reverse_disambiguated_ok:
            mechanical_ok = False
            failure_reason = "cheap_proxy_ambiguous"
    elif candidate["question_type"] == "ANCHOR_PROPERTY":
        property_terms = ANCHOR_PROPERTY_TYPE_TERMS.get(str(candidate.get("anchor_property_type") or "anchor_color"), ())
        property_ok = any(normalize_answer(term) in normalized_question for term in property_terms)
        reference_text = normalize_answer(str(candidate.get("query_text_reference") or tuple_row["answer"]))
        reference_text_ok = reference_text in normalized_question or any(normalize_answer(word) in normalized_question for word in tuple_row.get("child_words") or [])
        location_required = bool(candidate.get("query_location_required"))
        location_match_ok = specific_location_ok if candidate_specific_location_synonyms(candidate) else (location_ok or anchor_local_ok)
        if candidate.get("answer_source") == "mechanical_color" and candidate.get("expected_answer"):
            answer_ok = normalized_answer == normalize_answer(str(candidate["expected_answer"]))
        else:
            answer_ok = 1 <= len(answer.split()) <= 4 and len(answer.strip()) >= 2
        mechanical_ok = bool(answer_ok and anchor_ok and reference_text_ok and property_ok and (not location_required or location_match_ok))
        if not answer_ok:
            failure_reason = "anchor_property_answer_invalid"
        elif not reference_text_ok:
            failure_reason = "anchor_property_text_reference_missing"
        elif not property_ok:
            failure_reason = "anchor_property_signal_missing"
        elif location_required and candidate_specific_location_synonyms(candidate) and not specific_location_ok:
            failure_reason = "anchor_property_specific_location_missing"
        elif location_required and not location_match_ok:
            failure_reason = "anchor_property_location_missing"
        elif explicit_specific_required and not specific_location_ok:
            mechanical_ok = False
            failure_reason = "anchor_property_specific_location_missing"
        elif ambiguity_hard and not disambiguated_ok:
            mechanical_ok = False
            failure_reason = "ambiguous_grounding"
        elif disambiguation_required and not disambiguation_ok:
            mechanical_ok = False
            failure_reason = "anchor_instance_ambiguous"
        elif cheap_proxy_hard and not grouped_scene_read and not (strong_specific_grounding_ok or disambiguation_ok):
            mechanical_ok = False
            failure_reason = "cheap_proxy_ambiguous"

    grounding_ok = bool(
        anchor_ok
        or anchor_local_ok
        or (candidate["question_type"] == "REVERSE_GROUND" and (location_ok or specific_location_ok))
        or (candidate["question_type"] in {"DIRECT_READ", "YES_NO", "TEXT_PROPERTY"} and strong_specific_grounding_ok)
    )
    accepted = bool(grounding_ok and length_ok and question_type_ok and mechanical_ok)
    validation = {
        "answer_ok": answer_ok,
        "anchor_ok": anchor_ok,
        "length_ok": length_ok,
        "duplicate_ok": True,
        "accepted": accepted,
        "question_type_ok": question_type_ok,
        "mechanical_ok": mechanical_ok,
        "cheap_ambiguity_proxy_score": cheap_proxy_score,
        "group_mode": candidate_scene_repeat_group_mode(candidate),
    }
    if accepted:
        failure_reason = None
    elif failure_reason is None:
        if not grounding_ok:
            failure_reason = "anchor_missing"
        elif not question_type_ok:
            failure_reason = "question_type_mismatch"
        else:
            failure_reason = "validation_failed"
    return validation, {"generated_count": 1, "accepted_count": int(accepted)}, failure_reason


def row_to_final_sample(row: dict[str, Any], *, model: str, prompt_variant: str) -> dict[str, Any]:
    from .question_builder import (
        candidate_anchor_region_phrase,
        candidate_anchor_region_synonyms,
        location_ambiguity_score,
    )

    tuple_row = row["tuple"]
    item = row["items"][0]
    tags = build_tags(tuple_row)
    grounding = {
        "text_polygon": tuple_row["text_polygon"],
        "text_bbox": tuple_row["text_bbox"],
        "text_node_ids": tuple_row["text_node_ids"],
        "anchor_box": tuple_row["anchor_box"],
        "anchor_score": tuple_row["anchor_score"],
        "anchor_label_source": tuple_row.get("anchor_label_source"),
        "anchor_synonyms": tuple_row["anchor_synonyms"],
        "anchor_local_phrase": tuple_row.get("anchor_local_phrase"),
        "anchor_local_synonyms": tuple_row.get("anchor_local_synonyms"),
        "anchor_color": tuple_row.get("anchor_color") or extract_anchor_color(str(tuple_row.get("anchor_label") or "")),
        "anchor_region_phrase": candidate_anchor_region_phrase({"tuple": tuple_row}),
        "anchor_region_synonyms": candidate_anchor_region_synonyms({"tuple": tuple_row}),
        "relation": tuple_row["relation"],
        "ref_label": tuple_row["ref_label"],
        "ref_box": tuple_row["ref_box"],
        "ocr_confidence": tuple_row["ocr_confidence"],
        "region_key": tuple_row["region_key"],
        "location_phrase": tuple_row.get("location_phrase"),
        "specific_location_phrase": tuple_row.get("specific_location_phrase"),
        "specific_location_synonyms": tuple_row.get("specific_location_synonyms"),
        "query_anchor_label": tuple_row.get("query_anchor_label"),
        "query_anchor_synonyms": tuple_row.get("query_anchor_synonyms"),
        "query_anchor_color": tuple_row.get("query_anchor_color"),
        "query_anchor_box": tuple_row.get("query_anchor_box"),
        "query_anchor_local_phrase": tuple_row.get("query_anchor_local_phrase"),
        "query_anchor_local_synonyms": tuple_row.get("query_anchor_local_synonyms"),
        "query_anchor_region_phrase": candidate_anchor_region_phrase({"tuple": tuple_row, "query_anchor_box": tuple_row.get("query_anchor_box")}),
        "query_anchor_region_synonyms": candidate_anchor_region_synonyms({"tuple": tuple_row, "query_anchor_box": tuple_row.get("query_anchor_box")}),
        "query_location_phrase": tuple_row.get("query_location_phrase"),
        "query_location_synonyms": tuple_row.get("query_location_synonyms"),
        "query_specific_location_phrase": tuple_row.get("query_specific_location_phrase"),
        "query_specific_location_synonyms": tuple_row.get("query_specific_location_synonyms"),
        "query_relation": tuple_row.get("query_relation"),
        "query_text_reference": tuple_row.get("query_text_reference"),
        "reverse_ground_scope_preference": tuple_row.get("reverse_ground_scope_preference"),
        "query_grounding_matches_text": bool(
            normalize_answer(str(tuple_row.get("query_anchor_label") or tuple_row["anchor_label"]))
            == normalize_answer(str(tuple_row["anchor_label"]))
            and normalize_answer(str(tuple_row.get("query_location_phrase") or location_phrase_for_tuple(tuple_row)))
            == normalize_answer(location_phrase_for_tuple(tuple_row))
        ),
    }
    if tuple_row.get("anchor_source") is not None:
        grounding["anchor_source"] = tuple_row["anchor_source"]
    if tuple_row.get("semantic_debug") is not None:
        grounding["semantic_debug"] = tuple_row["semantic_debug"]
    kd_metadata = dict(tuple_row["kd_metadata"])
    kd_metadata["teacher_answer_logprobs"] = None
    if row.get("answer_probe") is not None:
        kd_metadata["answer_probe"] = row["answer_probe"]
    return {
        "sample_id": row["sample_id"],
        "image_id": tuple_row["image_id"],
        "image_path": tuple_row["image_path"],
        "image_width": tuple_row["image_width"],
        "image_height": tuple_row["image_height"],
        "ann_id": tuple_row["ann_id"],
        "question": item["question"],
        "answer": item["answer"],
        "answer_level": tuple_row["answer_level"],
        "anchor_label": tuple_row["anchor_label"],
        "anchor_box": tuple_row["anchor_box"],
        "relation": tuple_row["relation"],
        "ref_label": tuple_row["ref_label"],
        "ref_box": tuple_row["ref_box"],
        "text_polygon": tuple_row["text_polygon"],
        "text_bbox": tuple_row["text_bbox"],
        "ocr_confidence": tuple_row["ocr_confidence"],
        "consensus_tier": tuple_row["consensus_tier"],
        "unique": tuple_row["unique"],
        "dataset_source": tuple_row["dataset_source"],
        "resolvable": tuple_row["resolvable"],
        "resolvability": tuple_row["resolvability"],
        "question_type": tuple_row["question_type"],
        "quality_tier": "unscored",
        "tags": tags,
        "grounding": grounding,
        "kd_metadata": kd_metadata,
        "teacher_provider": "gemini",
        "teacher_model": model,
        "prompt_variant": prompt_variant,
    }


def build_tags(tuple_row: dict[str, Any]) -> dict[str, Any]:
    from .question_builder import (
        ambiguity_level_from_score,
        location_ambiguity_score,
    )

    question_type = str(tuple_row["question_type"])
    answer_source = str(tuple_row["answer_source"])
    area_frac = float(tuple_row["area_fraction"])
    ambiguity_score = location_ambiguity_score(tuple_row)
    tags = {
        "question_type": question_type,
        "answer_type": answer_type_for(tuple_row),
        "answer_source": answer_source,
        "consensus_tier": str(tuple_row["consensus_tier"]),
        "ocr_confidence_bin": confidence_bin(float(tuple_row["ocr_confidence"])),
        "text_length_bin": text_length_bin(str(tuple_row["answer"])),
        "text_case": text_case(str(tuple_row["answer"])),
        "answer_level": str(tuple_row["answer_level"]),
        "relation": str(tuple_row["relation"]),
        "anchor_category": str(tuple_row["anchor_category"]),
        "anchor_label": str(tuple_row["anchor_label"]),
        "has_reference_object": bool(tuple_row["ref_label"]),
        "unique": bool(tuple_row["unique"]),
        "text_density": text_density_bin(int(tuple_row["kd_metadata"]["text_density"])),
        "text_size_bin": text_size_bin(area_frac),
        "image_source": str(tuple_row["dataset_source"]),
        "anchor_score_bin": anchor_score_bin(float(tuple_row["anchor_score"])),
        "answer_verified_by": "mechanical" if answer_source == "mechanical" else "teacher_only",
        "yesno_polarity": tuple_row.get("yesno_polarity"),
        "yesno_distractor_source": tuple_row.get("yesno_distractor_source"),
        "text_property_type": tuple_row.get("text_property_type"),
        "anchor_property_type": tuple_row.get("anchor_property_type"),
        "ambiguity_level": ambiguity_level_from_score(ambiguity_score),
        "quality_tier": "unscored",
    }
    tags["difficulty"] = estimate_difficulty(tags)
    return tags


def answer_type_for(tuple_row: dict[str, Any]) -> str:
    from .question_builder import TEXT_PROPERTY_VISUAL_TYPES

    if tuple_row["question_type"] == "YES_NO":
        return "yes" if normalize_answer(str(tuple_row["expected_answer"])) == "yes" else "no"
    if tuple_row["question_type"] == "REVERSE_GROUND":
        return "spatial_phrase"
    if tuple_row["question_type"] == "ANCHOR_PROPERTY":
        return "attribute"
    if tuple_row["question_type"] == "TEXT_PROPERTY" and tuple_row.get("text_property_type") in TEXT_PROPERTY_VISUAL_TYPES:
        return "attribute"
    if tuple_row.get("text_property_type") == "word_count":
        return "number"
    return "text_string"


def confidence_bin(value: float) -> str:
    if value > 0.95:
        return "high_>0.95"
    if value >= 0.85:
        return "medium_0.85-0.95"
    return "low_0.80-0.85"


def text_length_bin(text: str) -> str:
    length = len(text.strip())
    if length <= 1:
        return "single_char"
    if length <= 4:
        return "short_2-4"
    if length <= 10:
        return "medium_5-10"
    return "long_11+"


def text_case(text: str) -> str:
    stripped = "".join(ch for ch in text if not ch.isspace())
    if stripped.isdigit():
        return "numeric"
    if stripped.isalnum() and not stripped.isalpha():
        return "alphanumeric"
    if stripped.isupper():
        return "upper"
    if stripped.islower():
        return "lower"
    return "mixed"


def text_density_bin(count: int) -> str:
    if count <= 2:
        return "sparse_1-2"
    if count <= 5:
        return "moderate_3-5"
    return "dense_6+"


def text_size_bin(area_fraction: float) -> str:
    if area_fraction > 0.10:
        return "large_>10pct"
    if area_fraction >= 0.03:
        return "medium_3-10pct"
    if area_fraction >= 0.01:
        return "small_1-3pct"
    return "tiny_<1pct"


def anchor_score_bin(score: float) -> str:
    if score > 0.75:
        return "high_>0.75"
    if score >= 0.50:
        return "medium_0.50-0.75"
    return "low_<0.50"


def estimate_difficulty(tags: dict[str, Any]) -> str:
    score = 0
    if tags["text_size_bin"] == "small_1-3pct":
        score += 2
    elif tags["text_size_bin"] == "medium_3-10pct":
        score += 1
    elif tags["text_size_bin"] == "tiny_<1pct":
        score += 3
    if tags["text_density"] == "dense_6+":
        score += 2
    elif tags["text_density"] == "moderate_3-5":
        score += 1
    if not tags["unique"]:
        score += 1
    if tags["text_length_bin"] == "long_11+":
        score += 1
    if tags["ocr_confidence_bin"] == "low_0.80-0.85":
        score += 1
    if tags["question_type"] in ("REVERSE_GROUND", "TEXT_PROPERTY", "ANCHOR_PROPERTY"):
        score += 1
    if tags.get("ambiguity_level") == "high":
        score += 2
    elif tags.get("ambiguity_level") == "medium":
        score += 1
    if score <= 1:
        return "easy"
    if score <= 3:
        return "medium"
    return "hard"


def sample_id_for_candidate(candidate: dict[str, Any]) -> str:
    return sanitize_id(str(candidate["candidate_id"]))


def location_phrase_for_tuple(tuple_row: dict[str, Any]) -> str:
    if tuple_row.get("location_phrase"):
        return str(tuple_row["location_phrase"])
    return REGION_PHRASES.get(str(tuple_row.get("region_key") or ""), "area of the image")


def location_synonyms_for_tuple(tuple_row: dict[str, Any]) -> list[str]:
    if tuple_row.get("location_synonyms"):
        return list(tuple_row["location_synonyms"])
    key = str(tuple_row.get("region_key") or "")
    if key in REGION_SYNONYMS:
        return list(REGION_SYNONYMS[key])
    return [location_phrase_for_tuple(tuple_row)]


def anchor_local_phrase_for_tuple(tuple_row: dict[str, Any]) -> str:
    phrase = str(tuple_row.get("anchor_local_phrase") or "").strip()
    return phrase


def anchor_local_synonyms_for_tuple(tuple_row: dict[str, Any]) -> list[str]:
    raw = tuple_row.get("anchor_local_synonyms") or []
    if not isinstance(raw, list):
        raw = [raw]
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if not text:
            continue
        norm = normalize_answer(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def specific_location_phrase_for_tuple(tuple_row: dict[str, Any]) -> str:
    phrase = str(tuple_row.get("specific_location_phrase") or "").strip()
    if not phrase:
        return ""
    if normalize_answer(phrase) == normalize_answer(location_phrase_for_tuple(tuple_row)):
        return ""
    return phrase


def specific_location_synonyms_for_tuple(tuple_row: dict[str, Any]) -> list[str]:
    raw = tuple_row.get("specific_location_synonyms") or []
    if not isinstance(raw, list):
        raw = [raw]
    coarse = normalize_answer(location_phrase_for_tuple(tuple_row))
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        text = str(item or "").strip()
        if not text:
            continue
        norm = normalize_answer(text)
        if not norm or norm == coarse or norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def sanitize_id(text: str) -> str:
    return text.replace("::", "__")


def group_node_id(group: list[dict[str, Any]]) -> str:
    return "group_" + "_".join(str(node["node_id"]) for node in group)


def union_bbox(boxes: list[list[float]]) -> list[float]:
    xs1 = [box[0] for box in boxes]
    ys1 = [box[1] for box in boxes]
    xs2 = [box[2] for box in boxes]
    ys2 = [box[3] for box in boxes]
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


def bbox_to_polygon(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def overlap_1d(left_a: float, right_a: float, left_b: float, right_b: float) -> float:
    return max(0.0, min(right_a, right_b) - max(left_a, left_b))


def box_height(box: list[float]) -> float:
    return max(float(box[3]) - float(box[1]), 1e-6)


def _anchor_for_region(region_key: str, image_anchors: list[dict[str, Any]]) -> dict[str, Any]:
    return next(
        (anchor for anchor in image_anchors if str(anchor.get("region_key")) == region_key),
        {
            "label": REGION_PHRASES.get(region_key, "region of the image"),
            "box": image_anchors[0]["box"] if image_anchors else [0.0, 0.0, 1.0, 1.0],
            "score": 1.0,
        },
    )


def is_blocked_answer(text: str) -> bool:
    from .question_builder import URL_LIKE_RE

    stripped = str(text or "").strip()
    if not stripped:
        return True
    if URL_LIKE_RE.search(stripped):
        return True
    digits = sum(ch.isdigit() for ch in stripped)
    if digits >= 7 and any(ch in stripped for ch in "-()"):
        return True
    return False


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _maybe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
