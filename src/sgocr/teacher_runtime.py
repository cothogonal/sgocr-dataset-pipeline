from __future__ import annotations

import json
import math
import re
import statistics
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from .bootstrap import (
    normalize_answer,
    write_json,
    write_jsonl,
)
from .gemini_batch import GeminiBatchRequest, batch_generate_json, image_part_from_payload
from .semantic_dev40_tuning import load_semantic_dev40_tuning
from .secrets import GEMINI, get_secret
from .teacher.http_clients import encode_image

from .question_builder import (
    _unique_anchor_can_skip_global_location,
    candidate_anchor_color,
    candidate_anchor_label,
    candidate_anchor_label_competitors,
    candidate_anchor_local_phrase,
    candidate_anchor_local_synonyms,
    candidate_anchor_phrases,
    candidate_anchor_region_phrase,
    candidate_anchor_region_synonyms,
    candidate_cheap_ambiguity_proxy_score,
    candidate_disambiguation_cues,
    candidate_location_phrase,
    candidate_location_synonyms,
    candidate_requires_anchor_disambiguation,
    candidate_scene_repeat_group_mode,
    candidate_specific_location_phrase,
    candidate_specific_location_synonyms,
    preferred_location_phrase_for_candidate,
)
from .dataset_assembly import (
    normalize_candidate_result,
    sample_id_for_candidate,
)


def run_teacher_batches(
    *,
    image_batches: list[dict[str, Any]],
    model: str,
    max_side: int,
    workers: int,
) -> dict[str, list[dict[str, Any]]]:
    tuning = load_semantic_dev40_tuning()
    batch_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    if tuning.gemini_api_mode == "batch":
        return _run_teacher_batches_via_gemini_batch(
            image_batches=image_batches,
            model=model,
            max_side=max_side,
            tuning=tuning,
        )

    def run_one(batch: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        image_path = batch["image_path"]
        selected_candidates = list(batch["selected_candidates"])
        indexed_candidates = [{**candidate, "candidate_index": index} for index, candidate in enumerate(selected_candidates, start=1)]
        if not image_path or not indexed_candidates:
            batch_row = {
                "image_id": batch["image_id"],
                "image_path": image_path,
                "selected_count": len(indexed_candidates),
                "ok": True,
                "items": [],
            }
            return batch_row, []

        try:
            payload = encode_image(Path(image_path), max_side=max_side)
            result = call_gemini_batched(model=model, image_payload=payload, selected_candidates=indexed_candidates)
            items = list(result.get("parsed", {}).get("items") or [])
            by_index = {int(entry.get("candidate_index")): entry for entry in items if entry.get("candidate_index") is not None}
            normalized_rows = []
            for position, candidate in enumerate(indexed_candidates, start=1):
                item = by_index.get(position)
                if item is None and position - 1 < len(items):
                    item = items[position - 1]
                normalized_rows.append(normalize_candidate_result(candidate, item, result))
            annotate_answer_probe_rows(
                model=model,
                image_payload=payload,
                indexed_candidates=indexed_candidates,
                normalized_rows=normalized_rows,
                tuning=tuning,
            )
            apply_answer_probe_policy(normalized_rows, tuning=tuning)
            batch_row = {
                "image_id": batch["image_id"],
                "image_path": image_path,
                "selected_count": len(indexed_candidates),
                "ok": True,
                "usage": result.get("usage"),
                "raw_text": result.get("raw_text"),
                "items": items,
            }
            return batch_row, normalized_rows
        except Exception as exc:  # pragma: no cover - exercised only on remote API failure
            result = {"provider": "gemini", "model": model, "raw_text": "", "parsed": {"items": []}, "usage": {}}
            rows = [normalize_candidate_result(candidate, None, result, batch_error=str(exc)) for candidate in indexed_candidates]
            batch_row = {
                "image_id": batch["image_id"],
                "image_path": image_path,
                "selected_count": len(indexed_candidates),
                "ok": False,
                "error": str(exc),
                "items": [],
            }
            return batch_row, rows

    total_batches = len(image_batches)
    completed_batches = 0

    def _report_teacher_progress(batch_row: dict) -> None:
        nonlocal completed_batches
        completed_batches += 1
        ok_flag = "ok" if batch_row.get("ok") else "err"
        print(
            f"[stage:teacher] progress {completed_batches}/{total_batches}"
            f" selected={batch_row.get('selected_count', 0)} [{ok_flag}]"
            f" image={batch_row.get('image_id', '?')}",
            flush=True,
        )

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, batch) for batch in image_batches]
            for future in as_completed(futures):
                batch_row, rows = future.result()
                batch_rows.append(batch_row)
                sample_rows.extend(rows)
                _report_teacher_progress(batch_row)
    else:
        for batch in image_batches:
            batch_row, rows = run_one(batch)
            batch_rows.append(batch_row)
            sample_rows.extend(rows)
            _report_teacher_progress(batch_row)

    batch_rows.sort(key=lambda row: row["image_id"])
    sample_rows.sort(key=lambda row: row["sample_id"])
    return {"batch_rows": batch_rows, "sample_rows": sample_rows}


def _run_teacher_batches_via_gemini_batch(
    *,
    image_batches: list[dict[str, Any]],
    model: str,
    max_side: int,
    tuning: Any,
) -> dict[str, list[dict[str, Any]]]:
    request_batches: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for batch in image_batches:
        image_path = batch["image_path"]
        selected_candidates = list(batch["selected_candidates"])
        indexed_candidates = [{**candidate, "candidate_index": index} for index, candidate in enumerate(selected_candidates, start=1)]
        if not image_path or not indexed_candidates:
            batch_rows.append(
                {
                    "image_id": batch["image_id"],
                    "image_path": image_path,
                    "selected_count": len(indexed_candidates),
                    "ok": True,
                    "items": [],
                }
            )
            continue
        payload = encode_image(Path(image_path), max_side=max_side)
        request_batches.append(
            {
                "batch": batch,
                "indexed_candidates": indexed_candidates,
                "request": GeminiBatchRequest(
                    key=str(batch["image_id"]),
                    contents=[
                        build_batched_prompt(indexed_candidates),
                        image_part_from_payload(payload),
                    ],
                    generation_config={
                        "temperature": 0.1,
                        "response_mime_type": "application/json",
                        "response_json_schema": batched_response_schema(len(indexed_candidates)),
                    },
                    metadata={"image_id": str(batch["image_id"])},
                ),
            }
        )

    result_map = batch_generate_json(
        model=model,
        requests=[entry["request"] for entry in request_batches],
        display_name_prefix="sgocr-teacher",
        chunk_size=int(tuning.gemini_batch_chunk_size),
        poll_interval_s=int(tuning.gemini_batch_poll_seconds),
        timeout_s=int(tuning.gemini_batch_timeout_seconds),
    )

    for entry in request_batches:
        batch = entry["batch"]
        indexed_candidates = list(entry["indexed_candidates"])
        outcome = result_map.get(str(batch["image_id"]))
        if outcome is None or outcome.error:
            error_text = str((outcome.error if outcome else "missing batch response") or "missing batch response")
            result = {"provider": "gemini", "model": model, "raw_text": "", "parsed": {"items": []}, "usage": {}}
            rows = [normalize_candidate_result(candidate, None, result, batch_error=error_text) for candidate in indexed_candidates]
            batch_rows.append(
                {
                    "image_id": batch["image_id"],
                    "image_path": batch["image_path"],
                    "selected_count": len(indexed_candidates),
                    "ok": False,
                    "error": error_text,
                    "items": [],
                }
            )
            sample_rows.extend(rows)
            continue
        try:
            parsed = json.loads(outcome.raw_text)
        except Exception as exc:
            error_text = f"batch_parse_failed: {exc}"
            result = {"provider": "gemini", "model": model, "raw_text": outcome.raw_text, "parsed": {"items": []}, "usage": outcome.usage}
            rows = [normalize_candidate_result(candidate, None, result, batch_error=error_text) for candidate in indexed_candidates]
            batch_rows.append(
                {
                    "image_id": batch["image_id"],
                    "image_path": batch["image_path"],
                    "selected_count": len(indexed_candidates),
                    "ok": False,
                    "error": error_text,
                    "raw_text": outcome.raw_text,
                    "items": [],
                }
            )
            sample_rows.extend(rows)
            continue
        result = {
            "provider": "gemini",
            "model": model,
            "raw_text": outcome.raw_text,
            "parsed": parsed,
            "usage": outcome.usage,
        }
        items = list(result.get("parsed", {}).get("items") or [])
        by_index = {int(item.get("candidate_index")): item for item in items if item.get("candidate_index") is not None}
        normalized_rows = []
        for position, candidate in enumerate(indexed_candidates, start=1):
            item = by_index.get(position)
            if item is None and position - 1 < len(items):
                item = items[position - 1]
            normalized_rows.append(normalize_candidate_result(candidate, item, result))
        payload = encode_image(Path(batch["image_path"]), max_side=max_side)
        annotate_answer_probe_rows(
            model=model,
            image_payload=payload,
            indexed_candidates=indexed_candidates,
            normalized_rows=normalized_rows,
            tuning=tuning,
        )
        apply_answer_probe_policy(normalized_rows, tuning=tuning)
        batch_rows.append(
            {
                "image_id": batch["image_id"],
                "image_path": batch["image_path"],
                "selected_count": len(indexed_candidates),
                "ok": True,
                "usage": result.get("usage"),
                "raw_text": result.get("raw_text"),
                "items": items,
            }
        )
        sample_rows.extend(normalized_rows)

    batch_rows.sort(key=lambda row: row["image_id"])
    sample_rows.sort(key=lambda row: row["sample_id"])
    return {"batch_rows": batch_rows, "sample_rows": sample_rows}


def call_gemini_batched(*, model: str, image_payload: Any, selected_candidates: list[dict[str, Any]], timeout_s: int = 120) -> dict[str, Any]:
    prompt = build_batched_prompt(selected_candidates)
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": image_payload.mime_type,
                            "data": image_payload.image_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
            "responseJsonSchema": batched_response_schema(len(selected_candidates)),
        },
    }
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "x-goog-api-key": get_secret(GEMINI),
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    raw_text = extract_gemini_text(payload)
    return {
        "provider": "gemini",
        "model": model,
        "raw_text": raw_text,
        "parsed": json.loads(raw_text),
        "usage": payload.get("usageMetadata", {}),
    }


def annotate_inline_frontier(
    rows: list[dict[str, Any]],
    *,
    model: str,
    max_side: int,
    workers: int,
) -> dict[str, Any]:
    tuning = load_semantic_dev40_tuning()
    if not rows or not tuning.inline_frontier_enabled:
        return {
            "enabled": bool(tuning.inline_frontier_enabled),
            "model": model,
            "scored_rows": 0,
            "mean_inline_frontier_correct": 0.0,
            "precision_first_score": 0.0,
            "errors": 0,
        }
    if tuning.gemini_api_mode == "batch":
        return _annotate_inline_frontier_via_gemini_batch(
            rows,
            model=model,
            max_side=max_side,
            tuning=tuning,
        )

    rows_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_image.setdefault(str(row["image_id"]), []).append(row)

    def run_one(image_id: str, image_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None]:
        image_path = str(image_rows[0]["image_path"])
        indexed_rows = [{**row, "_inline_index": idx + 1} for idx, row in enumerate(image_rows)]
        try:
            payload = encode_image(Path(image_path), max_side=max_side)
            result = call_gemini_inline_frontier_batched(model=model, image_payload=payload, rows=indexed_rows)
            items = list(result.get("parsed", {}).get("items") or [])
            by_index = {int(item.get("candidate_index")): item for item in items if item.get("candidate_index") is not None}
            for position, row in enumerate(indexed_rows, start=1):
                item = by_index.get(position)
                prediction = str((item or {}).get("answer") or "").strip()
                score = _inline_score_prediction(row, prediction) if prediction else {
                    "prediction_raw": "",
                    "prediction_norm": "",
                    "gold_norm": normalize_answer(str(row.get("answer") or "")),
                    "exact_correct": False,
                    "partial_correct": False,
                    "semantic_correct": False,
                    "word_f1": 0.0,
                    "soft_correct": False,
                }
                row["inline_frontier"] = {
                    "provider": "gemini",
                    "model": model,
                    "correct": bool(score["soft_correct"]),
                    "score": score,
                }
                row["inline_frontier_correct"] = bool(score["soft_correct"])
                row["quality_tier"] = "tier_a" if row["inline_frontier_correct"] else "tier_b"
                row["tags"]["quality_tier"] = row["quality_tier"]
            return indexed_rows, None
        except Exception as exc:  # pragma: no cover - remote API failure only
            for row in indexed_rows:
                row["inline_frontier"] = {
                    "provider": "gemini",
                    "model": model,
                    "correct": False,
                    "error": str(exc),
                }
                row["inline_frontier_correct"] = False
                row["quality_tier"] = "unscored"
                row["tags"]["quality_tier"] = "unscored"
            return indexed_rows, str(exc)

    completed_rows: list[dict[str, Any]] = []
    errors = 0
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, image_id, image_rows) for image_id, image_rows in rows_by_image.items()]
            for future in as_completed(futures):
                image_rows, error = future.result()
                completed_rows.extend(image_rows)
                if error:
                    errors += 1
    else:
        for image_id, image_rows in rows_by_image.items():
            scored_rows, error = run_one(image_id, image_rows)
            completed_rows.extend(scored_rows)
            if error:
                errors += 1

    by_sample_id = {str(row["sample_id"]): row for row in completed_rows}
    for idx, row in enumerate(rows):
        rows[idx] = by_sample_id.get(str(row["sample_id"]), row)

    scored_rows = [row for row in rows if "inline_frontier_correct" in row]
    mean_correct = sum(1.0 for row in scored_rows if row.get("inline_frontier_correct")) / max(len(scored_rows), 1)
    return {
        "enabled": True,
        "model": model,
        "scored_rows": len(scored_rows),
        "mean_inline_frontier_correct": round(mean_correct, 4),
        "precision_first_score": round(mean_correct * math.sqrt(len(scored_rows)), 4),
        "errors": errors,
    }


def _annotate_inline_frontier_via_gemini_batch(
    rows: list[dict[str, Any]],
    *,
    model: str,
    max_side: int,
    tuning: Any,
) -> dict[str, Any]:
    rows_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_image.setdefault(str(row["image_id"]), []).append(row)

    request_entries: list[dict[str, Any]] = []
    for image_id, image_rows in rows_by_image.items():
        indexed_rows = [{**row, "_inline_index": idx + 1} for idx, row in enumerate(image_rows)]
        payload = encode_image(Path(str(image_rows[0]["image_path"])), max_side=max_side)
        request_entries.append(
            {
                "image_id": image_id,
                "rows": indexed_rows,
                "request": GeminiBatchRequest(
                    key=str(image_id),
                    contents=[
                        build_inline_frontier_prompt(indexed_rows),
                        image_part_from_payload(payload),
                    ],
                    generation_config={
                        **_inline_gemini_generation_config(model, candidate_count=len(indexed_rows), retry_attempt=1),
                        "response_mime_type": "application/json",
                        "response_json_schema": inline_frontier_response_schema(len(indexed_rows)),
                    },
                    metadata={"image_id": str(image_id), "request_kind": "inline_frontier"},
                ),
            }
        )

    result_map = batch_generate_json(
        model=model,
        requests=[entry["request"] for entry in request_entries],
        display_name_prefix="sgocr-inline-frontier",
        chunk_size=int(tuning.gemini_batch_chunk_size),
        poll_interval_s=int(tuning.gemini_batch_poll_seconds),
        timeout_s=int(tuning.gemini_batch_timeout_seconds),
    )

    completed_rows: list[dict[str, Any]] = []
    errors = 0
    for entry in request_entries:
        indexed_rows = list(entry["rows"])
        outcome = result_map.get(str(entry["image_id"]))
        if outcome is None or outcome.error:
            errors += 1
            error_text = str((outcome.error if outcome else "missing batch response") or "missing batch response")
            for row in indexed_rows:
                row["inline_frontier"] = {
                    "provider": "gemini",
                    "model": model,
                    "correct": False,
                    "error": error_text,
                }
                row["inline_frontier_correct"] = False
                row["quality_tier"] = "unscored"
                row["tags"]["quality_tier"] = "unscored"
            completed_rows.extend(indexed_rows)
            continue
        try:
            parsed = json.loads(outcome.raw_text)
            items = list(parsed.get("items") or [])
            by_index = {int(item.get("candidate_index")): item for item in items if item.get("candidate_index") is not None}
            for position, row in enumerate(indexed_rows, start=1):
                item = by_index.get(position)
                prediction = str((item or {}).get("answer") or "").strip()
                score = _inline_score_prediction(row, prediction) if prediction else {
                    "prediction_raw": "",
                    "prediction_norm": "",
                    "gold_norm": normalize_answer(str(row.get("answer") or "")),
                    "exact_correct": False,
                    "partial_correct": False,
                    "semantic_correct": False,
                    "word_f1": 0.0,
                    "soft_correct": False,
                }
                row["inline_frontier"] = {
                    "provider": "gemini",
                    "model": model,
                    "correct": bool(score["soft_correct"]),
                    "score": score,
                }
                row["inline_frontier_correct"] = bool(score["soft_correct"])
                row["quality_tier"] = "tier_a" if row["inline_frontier_correct"] else "tier_b"
                row["tags"]["quality_tier"] = row["quality_tier"]
        except Exception as exc:
            errors += 1
            for row in indexed_rows:
                row["inline_frontier"] = {
                    "provider": "gemini",
                    "model": model,
                    "correct": False,
                    "error": str(exc),
                }
                row["inline_frontier_correct"] = False
                row["quality_tier"] = "unscored"
                row["tags"]["quality_tier"] = "unscored"
        completed_rows.extend(indexed_rows)

    by_sample_id = {str(row["sample_id"]): row for row in completed_rows}
    for idx, row in enumerate(rows):
        rows[idx] = by_sample_id.get(str(row["sample_id"]), row)

    scored_rows = [row for row in rows if "inline_frontier_correct" in row]
    mean_correct = sum(1.0 for row in scored_rows if row.get("inline_frontier_correct")) / max(len(scored_rows), 1)
    return {
        "enabled": True,
        "model": model,
        "scored_rows": len(scored_rows),
        "mean_inline_frontier_correct": round(mean_correct, 4),
        "precision_first_score": round(mean_correct * math.sqrt(len(scored_rows)), 4),
        "errors": errors,
    }


def _gemini_post_with_http_retry(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout_s: int = 120,
    max_http_attempts: int = 4,
) -> "requests.Response":
    """POST to the Gemini API with exponential backoff on transient HTTP errors.

    Retries on 429 (rate limit) and 5xx (server errors). Waits 2**attempt seconds
    before each retry (1s, 2s, 4s for attempts 1-3). Raises immediately on 4xx
    client errors other than 429.
    """
    last_exc: Exception | None = None
    for attempt in range(max_http_attempts):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout_s)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < max_http_attempts - 1:
                    time.sleep(2.0 ** attempt)
                    last_exc = requests.exceptions.HTTPError(
                        f"HTTP {response.status_code}", response=response
                    )
                    continue
                response.raise_for_status()
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout as exc:
            if attempt < max_http_attempts - 1:
                time.sleep(2.0 ** attempt)
                last_exc = exc
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Gemini POST failed after all retry attempts")


def call_gemini_inline_frontier_batched(*, model: str, image_payload: Any, rows: list[dict[str, Any]], timeout_s: int = 120) -> dict[str, Any]:
    last_error: Exception | None = None
    candidate_count = len(rows)
    for attempt in range(2):
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": build_inline_frontier_prompt(rows)},
                        {
                            "inline_data": {
                                "mime_type": image_payload.mime_type,
                                "data": image_payload.image_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                **_inline_gemini_generation_config(model, candidate_count=candidate_count, retry_attempt=attempt),
                "responseMimeType": "application/json",
                "responseJsonSchema": inline_frontier_response_schema(candidate_count),
            },
        }
        response = _gemini_post_with_http_retry(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={
                "x-goog-api-key": get_secret(GEMINI),
                "Content-Type": "application/json",
            },
            body=body,
            timeout_s=timeout_s,
        )
        payload = response.json()
        raw_text = extract_gemini_text(payload)
        try:
            parsed = json.loads(raw_text)
            return {
                "provider": "gemini",
                "model": model,
                "raw_text": raw_text,
                "parsed": parsed,
                "usage": payload.get("usageMetadata", {}),
            }
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Inline frontier Gemini call failed without a parseable response")


def build_inline_frontier_prompt(rows: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, row in enumerate(rows, start=1):
        blocks.append(
            "\n".join(
                [
                    f"--- Candidate {idx} ---",
                    f"candidate_index: {idx}",
                    _inline_benchmark_prompt(row),
                ]
            )
        )
    return (
        "Answer each visual question using the image.\n"
        "Return a JSON object with an `items` array in the same order.\n"
        "Each item must contain `candidate_index` and the shortest literal `answer` only.\n"
        "Do not include explanations.\n\n"
        + "\n\n".join(blocks)
    )


_INLINE_RG_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "in", "on", "at", "of", "to", "is", "it", "be", "as",
    "and", "or", "for", "with", "by", "from", "this", "that", "are", "was",
    "not", "but", "if", "its", "into", "do", "so", "area", "image", "text",
    "near", "around",
})


def _inline_normalize_text_answer(answer: str) -> str:
    text = unicodedata.normalize("NFKC", str(answer or "").strip().lower())
    text = re.sub(r"[\s]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _inline_normalize_vqa_answer(answer: str) -> str:
    text = _inline_normalize_text_answer(answer)
    if text in {"yes", "yeah", "y"}:
        return "yes"
    if text in {"no", "nope", "n"}:
        return "no"
    return text


def _inline_normalize_answer_by_type(answer: str, answer_type: str) -> str:
    kind = str(answer_type or "").strip().lower()
    if kind in {"yes", "no", "number"}:
        return _inline_normalize_vqa_answer(answer)
    return _inline_normalize_text_answer(answer)


def _inline_strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _inline_content_word_f1(gold_norm: str, pred_norm: str) -> float:
    def content_words(s: str) -> list[str]:
        return [w for w in s.split() if w not in _INLINE_RG_STOPWORDS and len(w) > 1]

    g = content_words(gold_norm)
    p = content_words(pred_norm)
    if not g or not p:
        return 0.0
    common = sum((Counter(g) & Counter(p)).values())
    precision = common / len(p)
    recall = common / len(g)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _inline_partial_correct_direct_read(gold_norm: str, pred_norm: str) -> bool:
    gold_clean = gold_norm.rstrip("][(.,;:!?)'\"").strip()
    if not gold_clean:
        return False
    g = _inline_strip_accents(gold_clean)
    p = _inline_strip_accents(pred_norm)
    if len(gold_clean) < 3:
        return bool(g) and g in p
    g_words = g.split()
    p_words = p.split()
    if not g_words or not p_words:
        return False

    def _stem(w: str) -> str:
        return w[:-1] if len(w) > 3 and w.endswith("s") else w

    g_set = {_stem(w) for w in g_words}
    p_set = {_stem(w) for w in p_words}
    if g_set.issubset(p_set):
        return True
    if len(p_words) <= 3 and p_set.issubset(g_set):
        return True
    if len(g_words) == 1 and len(g) >= 4 and len(p_words) >= 2 and g == "".join(p_words):
        return True
    if len(p_words) == 1 and len(p) >= 4 and len(g_words) >= 2 and p == "".join(g_words):
        return True
    return False


def _inline_score_prediction(row: dict[str, Any], prediction: str) -> dict[str, Any]:
    tags = dict(row.get("tags") or {})
    question_type = str(tags.get("question_type") or row.get("question_type") or "").strip().upper()
    gold = str(row.get("answer") or "")
    answer_type = str(tags.get("answer_type") or "")
    pred_norm = _inline_normalize_answer_by_type(prediction, answer_type)
    gold_norm = _inline_normalize_answer_by_type(gold, answer_type)
    exact = bool(pred_norm == gold_norm or _inline_strip_accents(pred_norm) == _inline_strip_accents(gold_norm))
    partial = bool(question_type == "DIRECT_READ" and not exact and _inline_partial_correct_direct_read(gold_norm, pred_norm))
    semantic = False
    word_f1_val = 0.0
    if question_type == "REVERSE_GROUND":
        word_f1_val = _inline_content_word_f1(gold_norm, pred_norm)
        semantic = word_f1_val >= 0.5
    return {
        "prediction_raw": prediction,
        "prediction_norm": pred_norm,
        "gold_norm": gold_norm,
        "exact_correct": exact,
        "partial_correct": partial,
        "semantic_correct": semantic,
        "word_f1": round(word_f1_val, 4),
        "soft_correct": bool(exact or partial or semantic),
    }


def _inline_benchmark_prompt(row: dict[str, Any]) -> str:
    tags = dict(row.get("tags") or {})
    question_type = str(tags.get("question_type") or row.get("question_type") or "").strip()
    if question_type == "REVERSE_GROUND":
        anchor_label = str(tags.get("anchor_label") or row.get("anchor_label") or "")
        anchor_hint = f" (anchor object: {anchor_label})" if anchor_label else ""
        return (
            "Answer the visual question using the image.\n"
            "This is a location question, so answer with a short plain-text location phrase only.\n"
            "Do not return coordinates, JSON, or explanations.\n"
            "Answer length: 1 to 6 words.\n"
            f"Question type: REVERSE_GROUND{anchor_hint}\n"
            f"Question: {row.get('question', '')}"
        )
    return (
        "Answer the visual question using the image.\n"
        "Give only the shortest literal answer.\n"
        "Do not explain.\n"
        f"Question type: {question_type or 'unknown'}\n"
        f"Question: {row.get('question', '')}"
    )


def _inline_gemini_generation_config(model: str, *, candidate_count: int, retry_attempt: int = 0) -> dict[str, Any]:
    max_output_tokens = max(96, 28 + 24 * max(int(candidate_count), 1) + 32 * max(int(retry_attempt), 0))
    lower = model.lower()
    if "gemini-3" in lower:
        level = "minimal" if "flash" in lower else "low"
        return {"temperature": 0.0, "thinkingConfig": {"thinkingLevel": level}, "maxOutputTokens": max_output_tokens}
    if "gemini-2.5" in lower and "flash" in lower:
        return {"temperature": 0.0, "thinkingConfig": {"thinkingBudget": 0}, "maxOutputTokens": max_output_tokens}
    return {"temperature": 0.0, "maxOutputTokens": max_output_tokens}


def inline_frontier_response_schema(candidate_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {"type": "integer"},
                        "answer": {"type": "string"},
                    },
                    "required": ["candidate_index", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def build_batched_prompt(selected_candidates: list[dict[str, Any]]) -> str:
    tuning = load_semantic_dev40_tuning()
    blocks = []
    for candidate in selected_candidates:
        tuple_row = candidate["tuple"]
        qtype = candidate["question_type"]
        unique_skip_location = _unique_anchor_can_skip_global_location(candidate)
        location_phrase = candidate_location_phrase(candidate)
        preferred_location_phrase = preferred_location_phrase_for_candidate(candidate)
        specific_location_phrase = candidate_specific_location_phrase(candidate)
        anchor_region_phrase = candidate_anchor_region_phrase(candidate)
        anchor_color = candidate_anchor_color(candidate)
        disambiguation_cues = candidate_disambiguation_cues(candidate)
        same_anchor_competitors = candidate_anchor_label_competitors(candidate)
        anchor_disambiguation_required = candidate_requires_anchor_disambiguation(candidate)
        group_mode = candidate_scene_repeat_group_mode(candidate)
        cheap_proxy_score = candidate_cheap_ambiguity_proxy_score(candidate)
        lines = [
            f"--- Candidate {candidate['candidate_index']} ---",
            f"candidate_index: {candidate['candidate_index']}",
            f"question_type: {qtype}",
            f'text: "{tuple_row["answer"]}"',
            f'anchor_label: "{candidate_anchor_label(candidate)}"',
            f'anchor_color: "{anchor_color}"',
            f'group_mode: "{group_mode}"',
            f"group_anchor_count: {int(candidate.get('query_group_count') or 0)}",
            f"allowed_anchor_phrases: {json.dumps(candidate_anchor_phrases(candidate), ensure_ascii=False)}",
            f"anchor_disambiguation_required: {'yes' if anchor_disambiguation_required else 'no'}",
            f"same_anchor_label_competitors: {same_anchor_competitors}",
            f"cheap_ambiguity_proxy_score: {cheap_proxy_score}",
            f"preferred_disambiguation_cues: {json.dumps(disambiguation_cues[:6], ensure_ascii=False)}",
            f'anchor_local_location: "{candidate_anchor_local_phrase(candidate)}"',
            f"allowed_anchor_local_phrases: {json.dumps(candidate_anchor_local_synonyms(candidate), ensure_ascii=False)}",
            f'anchor_region: "{anchor_region_phrase}"',
            f"allowed_anchor_region_phrases: {json.dumps(candidate_anchor_region_synonyms(candidate), ensure_ascii=False)}",
            f'coarse_location: "{location_phrase}"',
            f'preferred_location_phrase: "{preferred_location_phrase}"',
            f"allowed_location_phrases: {json.dumps(candidate_location_synonyms(candidate), ensure_ascii=False)}",
            f'specific_location: "{specific_location_phrase}"',
            f"allowed_specific_location_phrases: {json.dumps(candidate_specific_location_synonyms(candidate), ensure_ascii=False)}",
            f"answer_level: {tuple_row['answer_level']}",
            f"unique: {'yes' if tuple_row['unique'] else 'no'}",
            f'relation: "{candidate.get("query_relation") or tuple_row["relation"]}"',
            f"competing_tuples: {int(tuple_row.get('kd_metadata', {}).get('competing_tuples') or 0)}",
            f'local_layout_detail: "{tuple_row.get("semantic_debug", {}).get("location_detail") or tuple_row.get("kd_metadata", {}).get("layout_detail") or ""}"',
        ]
        if qtype == "DIRECT_READ":
            lines.append(f'Write 1 natural question asking what the text says at this location. The answer must be exactly "{tuple_row["answer"]}".')
            if group_mode == "scene_repeat_same_text":
                lines.append("This text appears on multiple matching anchors in the image. Ask about the repeated set as a whole instead of singling out one instance.")
                lines.append("Use a plural anchor phrase such as one of the allowed anchor phrases. Do not use anchor-local or image-global location wording unless it is absolutely necessary.")
            if unique_skip_location:
                lines.append("The target text is unique in this image. Do not mention any image-level location phrase; identify it using the anchor phrase only.")
            if tuning.location_wording_mode == "lite":
                lines.append("Use the preferred location phrase directly. Keep the wording low-entropy, literal, and brief.")
            elif tuning.location_wording_mode == "varied":
                lines.append("Use a varied spatial phrase instead of repeating the same canned wording when possible, but keep the question literal and brief.")
            elif tuning.location_wording_mode == "rich_local":
                lines.append("Use one clean local spatial phrase tied to the anchor, such as upper left, lower right, above, below, left side, right side, center, top edge, or bottom edge, when that phrasing is visibly warranted.")
                lines.append("Prefer anchor-relative wording over broad image-global wording. Use at most one coarse image phrase if it is truly needed for disambiguation.")
            elif tuning.location_wording_mode == "finalv0":
                lines.append("Use minimal localization first. Prefer a short anchor phrase alone when it is unique, or add exactly one extra cue such as color, anchor-local position, or one coarse image phrase when needed.")
                lines.append("Avoid duplicated tiers such as `upper-left text in the upper-left area of the image`. If you need two tiers, make them different, such as anchor-local plus global, or color plus anchor-local.")
                lines.append("Vary the wording naturally across examples: rotate between `left side of`, `upper part of`, `near the lower edge of`, `on the right side of`, or `in the upper right of the image` when visually warranted.")
            else:
                lines.append("Use a small amount of spatial wording variation when possible, but keep the question literal and brief.")
            if anchor_disambiguation_required:
                lines.append("This anchor type appears multiple times in the image. Use the anchor phrase plus one additional disambiguation cue. Prefer color first, then a local anchor phrase, then one coarse image phrase.")
            if candidate.get("query_location_required") and not unique_skip_location:
                lines.append(f'Because nearby text boxes share this region, make the question more specific by mentioning both an allowed anchor phrase and the preferred specific location phrase "{preferred_location_phrase}".')
                lines.append("Do not stack multiple near-synonymous global phrases together; use one clean specific phrase instead of layered wording.")
        elif qtype == "YES_NO":
            lines.append(f'Write 1 yes/no question asking whether the text at this location says "{candidate["queried_text"]}".')
            lines.append(f'The answer must be exactly "{candidate["expected_answer"]}".')
            if group_mode == "scene_repeat_same_text":
                lines.append("This text appears on multiple matching anchors. Ask about that repeated set as a group and use a plural anchor phrase.")
                lines.append("Do not single out one instance with local position wording.")
            if unique_skip_location:
                lines.append("The target text is unique in this image. Do not mention any image-level location phrase; identify it using the anchor phrase only.")
            if tuning.location_wording_mode == "rich_local":
                lines.append("When location wording is useful, prefer one clean anchor-relative spatial phrase over a broad image-global phrase.")
            elif tuning.location_wording_mode == "finalv0":
                lines.append("Keep location wording minimal. Use the anchor phrase alone when possible; otherwise add one explicit disambiguation cue without repeating the same region wording twice.")
            if anchor_disambiguation_required:
                lines.append("The anchor type repeats in the image. Mention one additional disambiguation cue such as color, local anchor position, or a single global image phrase.")
            if candidate.get("yesno_polarity") == "negative":
                lines.append("This is a grounded exclusion check: the queried location is intentionally false for this image. Ask whether the text appears at this provided location, not whether it appears anywhere else in the image.")
                if not unique_skip_location:
                    lines.append(f'Mention the preferred location phrase "{preferred_location_phrase}" so the question is clearly about the false grounded location.')
            elif candidate.get("query_location_required") and not unique_skip_location:
                lines.append(f'Because the anchor has multiple text regions, mention the preferred location phrase "{preferred_location_phrase}" to disambiguate the target text.')
        elif qtype == "REVERSE_GROUND":
            scope_preference = str(candidate.get("reverse_ground_scope_preference") or "global")
            is_unique = bool(tuple_row.get("unique"))
            rg_style = tuning.reverse_ground_answer_style
            lines.append(f'Write 1 question asking where the text "{tuple_row["answer"]}" appears.')
            if rg_style in {"frontier", "frontier_rich"}:
                lines.append("The answer must be a concise plain-text location phrase of 1 to 6 words.")
                lines.append("GUARDRAILS: Do NOT output coordinates, JSON, arrays, or structured data. Do NOT say 'I cannot determine' — always provide a best-effort location.")
                anchor_lbl = candidate_anchor_label(candidate)
                example_anchor = anchor_lbl or "sign"
                if rg_style == "frontier_rich":
                    lines.append(
                        f'Example: Q: "Where does the text \'OPEN\' appear?" → A: "on the upper part of the {example_anchor}"'
                    )
                    lines.append(
                        "Prefer a short human referring phrase that combines an object name with one clean local spatial cue when that cue is visible."
                    )
                else:
                    lines.append(
                        f'Example: Q: "Where does the text \'OPEN\' appear?" → A: "on the {example_anchor}"'
                    )
                if scope_preference == "mixed":
                    local_phrase = candidate_anchor_local_phrase(candidate)
                    region_part = f" in the {anchor_region_phrase}" if anchor_region_phrase else ""
                    local_part = f"{local_phrase} of the " if local_phrase else "on the "
                    lines.append(
                        "This is a MIXED-scope answer: combine a brief anchor-local phrase with a coarse image-position. "
                        f'For example: "{local_part}{example_anchor}{region_part}" or '
                        f'"the {anchor_region_phrase} {example_anchor}" — pick whichever is most concise and unambiguous.'
                        if anchor_region_phrase else
                        "This is a MIXED-scope answer: use both the specific anchor position and where on the anchor the text sits. "
                        f'For example: "upper left of the {example_anchor}" or "bottom right corner of the {example_anchor}".'
                    )
                elif is_unique and anchor_lbl:
                    lines.append(
                        f'This anchor ({anchor_lbl}) is the only one of its kind visible. '
                        f'Prefer the short form "on the {anchor_lbl}" or just "{anchor_lbl}" — '
                        f"add an image-level qualifier only if the anchor type genuinely repeats."
                    )
                else:
                    lines.append(
                        "Avoid stacking redundant qualifiers. "
                        "Do not combine an anchor phrase AND a full image-level position unless the anchor appears multiple times."
                    )
                if rg_style == "frontier_rich":
                    lines.append(
                        "When possible, use one visible attribute or part cue, such as color or object-part wording, as long as it is directly visible and concise."
                    )
                    lines.append(
                        "Good patterns: `on the red bus side`, `above the player helmet`, `on the left side of the storefront sign`."
                    )
                if tuning.location_wording_mode == "finalv0":
                    lines.append(
                        "Prefer minimal answer phrases. Start with the anchor phrase alone if it is unique. If the anchor repeats, add exactly one stronger cue such as color or local anchor position, and only add a global phrase when that is still necessary."
                    )
                    lines.append(
                        "Good multi-tier patterns: `on the red sign near the top right`, `on the left side of the blue bus`, `at the lower edge of the white poster`."
                    )
                if anchor_region_phrase or candidate_anchor_local_phrase(candidate):
                    lines.append(
                        "Preferred phrasing options (use one, not all): "
                        + ", ".join(filter(None, [
                            f'"on the {anchor_lbl}"' if anchor_lbl else None,
                            f'"the {anchor_region_phrase} {anchor_lbl}"' if anchor_region_phrase and anchor_lbl else None,
                            f'"{candidate_anchor_local_phrase(candidate)} of the {anchor_lbl}"' if candidate_anchor_local_phrase(candidate) and anchor_lbl else None,
                        ]))
                    )
                if candidate_anchor_local_synonyms(candidate) and scope_preference != "global":
                    lines.append(f'Local anchor phrase options: {json.dumps(candidate_anchor_local_synonyms(candidate), ensure_ascii=False)}')
                if anchor_lbl:
                    lines.append(f'Allowed anchor phrases: {json.dumps(candidate_anchor_phrases(candidate), ensure_ascii=False)}')
            else:
                lines.append("The answer must be a short visible location phrase of 2 to 10 words and should mention an allowed anchor phrase, the specific location, or both.")
                if candidate_anchor_local_synonyms(candidate):
                    global_examples = [
                        f"on the {candidate_anchor_label(candidate)} in the {anchor_region_phrase}" if anchor_region_phrase else f"{candidate_anchor_label(candidate)} at the {preferred_location_phrase}",
                        f"the {anchor_region_phrase} {candidate_anchor_label(candidate)}" if anchor_region_phrase else f"the {preferred_location_phrase} on the {candidate_anchor_label(candidate)}",
                    ]
                    mixed_examples = [
                        f"{candidate_anchor_local_phrase(candidate)} in the {anchor_region_phrase}" if anchor_region_phrase else f"{candidate_anchor_local_phrase(candidate)} near the {preferred_location_phrase}",
                        f"{candidate_anchor_local_phrase(candidate)} on the {candidate_anchor_label(candidate)} in the {anchor_region_phrase}" if anchor_region_phrase else f"the {preferred_location_phrase}, {candidate_anchor_local_phrase(candidate)}",
                    ]
                    lines.append(f'reverse_ground_scope_preference: "{scope_preference}"')
                    lines.append("If the local anchor phrase is clean, you may answer with an anchor-scoped phrase instead of a global image phrase.")
                    lines.append(f'Local examples: {json.dumps(candidate_anchor_local_synonyms(candidate), ensure_ascii=False)}')
                    lines.append(f'Global examples: {json.dumps(global_examples, ensure_ascii=False)}')
                    lines.append(f'Mixed examples: {json.dumps(mixed_examples, ensure_ascii=False)}')
                    lines.append("Follow the provided scope preference: `local` prefers anchor-local phrasing, `global` prefers anchor+image phrasing, and `mixed` may combine both when concise.")
                    if scope_preference == "mixed":
                        lines.append("When `mixed`, prefer a concise answer that combines one clean anchor-local phrase with one coarse image-location phrase, for example `on the bottom left of the car in the center of the image`.")
                else:
                    if anchor_region_phrase:
                        lines.append(f'Use only the provided allowed anchor/location phrases plus simple connectors, for example "on the {candidate_anchor_label(candidate)} in the {anchor_region_phrase}" or "the {anchor_region_phrase} {candidate_anchor_label(candidate)}".')
                    else:
                        lines.append(f'Use only the provided allowed anchor/location phrases plus simple connectors, for example "{candidate_anchor_label(candidate)} at the {preferred_location_phrase}" or "the {preferred_location_phrase} on the {candidate_anchor_label(candidate)}".')
        elif qtype == "TEXT_PROPERTY":
            prop = candidate["text_property_type"]
            use_text_reference = not (
                prop in {"text_color", "text_orientation", "text_curvature"}
                and tuning.tp_visual_avoid_text_reference_with_specific_location_enabled
                and candidate.get("query_location_required")
            )
            if prop in {"word_count", "first_word", "last_word"}:
                lines.append(f"Write 1 question about the text property `{prop}` for this text at this location.")
                lines.append(f'The answer must be exactly "{candidate["expected_answer"]}".')
            elif prop == "text_color":
                if use_text_reference:
                    lines.append(
                        f'Write 1 question about the visible color of the text that says "{candidate.get("query_text_reference") or tuple_row["answer"]}" at this location.'
                    )
                else:
                    lines.append("Write 1 question about the visible color of the target text at this location.")
                lines.append('The answer must be a short color phrase of 1 to 3 words, such as "white" or "bright red".')
            elif prop == "text_orientation":
                if use_text_reference:
                    lines.append(
                        f'Write 1 question about the visible orientation of the text that says "{candidate.get("query_text_reference") or tuple_row["answer"]}" at this location.'
                    )
                else:
                    lines.append("Write 1 question about the visible orientation of the target text at this location.")
                lines.append("The answer must be a short orientation phrase of 1 to 3 words, such as horizontal, vertical, or diagonal.")
            elif prop == "text_curvature":
                if use_text_reference:
                    lines.append(
                        f'Write 1 question about whether the text that says "{candidate.get("query_text_reference") or tuple_row["answer"]}" looks straight, curved, or arched.'
                    )
                else:
                    lines.append("Write 1 question about whether the target text looks straight, curved, or arched.")
                lines.append("The answer must be a short shape phrase of 1 to 3 words.")
            if candidate.get("query_location_required"):
                lines.append(f'Because nearby text boxes share this region, mention the preferred specific location phrase "{preferred_location_phrase}" so the property question targets the correct text.')
            if tuning.location_wording_mode == "rich_local":
                lines.append("If a location phrase is needed, prefer a concise anchor-relative phrase over a broad image-global phrase.")
            elif tuning.location_wording_mode == "finalv0":
                lines.append("If a location phrase is needed, use a single minimal cue and avoid layered duplicate global phrasing.")
            if anchor_disambiguation_required:
                lines.append("This anchor repeats in the image. Use one additional disambiguation cue, preferably color or a local anchor phrase.")
        elif qtype == "ANCHOR_PROPERTY":
            property_type = str(candidate.get("anchor_property_type") or "anchor_color")
            reference_text = str(candidate.get("query_text_reference") or tuple_row["answer"])
            if property_type == "anchor_color":
                lines.append(f'Write 1 question about the visible color of the {candidate_anchor_label(candidate)} that has "{reference_text}" on it.')
                if candidate.get("answer_source") == "mechanical_color" and candidate.get("expected_answer"):
                    lines.append(f'The answer must be exactly "{candidate["expected_answer"]}".')
                else:
                    lines.append('The answer must be a short color phrase of 1 to 3 words, such as "green" or "dark blue".')
            elif property_type == "anchor_material":
                lines.append(f'Write 1 question about the visible material or surface type of the {candidate_anchor_label(candidate)} that has "{reference_text}" on it.')
                lines.append('The answer must be a short material phrase of 1 to 4 words, such as "metal" or "painted wood".')
            elif property_type == "anchor_shape":
                lines.append(f'Write 1 question about the visible shape of the {candidate_anchor_label(candidate)} that has "{reference_text}" on it.')
                lines.append('The answer must be a short shape phrase of 1 to 3 words, such as "round" or "rectangular".')
            if candidate.get("query_location_required"):
                lines.append(f'Mention the preferred specific location phrase "{preferred_location_phrase}" so the question clearly targets the correct text region.')
            lines.append("Keep it OCR-adjacent: use the text as the reference for which object to describe, but ask about the object or surface itself.")
            lines.append("If the anchor_label above contains text-content words (handwritten, printed, dates, initials, etc.), use only the plain object noun in the question — do not repeat those terms.")
            if anchor_disambiguation_required:
                lines.append("The anchor repeats in the image. Use one added disambiguation cue, preferably color or a local anchor phrase.")
        blocks.append("\n".join(lines))

    location_instruction = (
        "Prefer stable, low-entropy location wording. Reuse the preferred location phrase rather than inventing extra variants.\n"
        if tuning.location_wording_mode == "lite"
        else "Vary the location wording across candidates when possible instead of repeating the same phrase every time.\n"
        if tuning.location_wording_mode == "varied"
        else "Use minimal localization first. Prefer concise anchor-relative phrases, and only add a global phrase when it resolves ambiguity. If two tiers are needed, make them non-redundant.\n"
        if tuning.location_wording_mode == "finalv0"
        else "Use some location wording variation across candidates, but keep the phrasing stable and literal.\n"
    )
    strictness_instruction = "Prefer short literal questions and short literal answers over expressive phrasing.\n" if tuning.teacher_strictness == "very_strict" else ""
    return (
        "You are generating OCR spatial QA training data for a single image.\n\n"
        "Use only the verified facts below.\n"
        "Every question must make the location identifiable using one of the allowed anchor phrases.\n"
        + location_instruction
        + "Do not invent new text, objects, materials, activities, scene context, or semantic interpretations.\n"
        + "Do not paraphrase the anchor into a richer description than the allowed anchor phrases.\n"
        + "If the allowed anchor phrase is generic like sign, poster, label, board, screen, bottle, or box, use that exact noun instead of inventing a more specific object description.\n"
        + "If the anchor_label contains words that reference text content — such as 'handwritten', 'printed', 'written', 'engraved', 'dates', 'initials', 'signatures', 'numbers', 'letters', 'text', 'inscription', or similar — strip those terms and use only the plain visual object noun (e.g., 'stack of books' not 'stack of books with handwritten dates', 'plaque' not 'plaque engraved letters'). Never copy text-content descriptors from the anchor_label into the generated question.\n"
        + "Do not use the image to guess subject matter such as medical notes, menus, therapy, music, sports, or brands unless that wording already appears in the verified text itself.\n"
        + strictness_instruction
        + "Do not ask questions that require reading two separate text regions, comparing multiple text regions, or using world knowledge.\n"
        + "Do not ask about URLs, email addresses, or phone numbers.\n"
        + "Keep every question between 5 and 35 words.\n"
        + "If location disambiguation is needed, prefer one clean specific phrase or one clean anchor-local plus anchor-region phrase; avoid redundant stacking like `upper-left part of the upper-left section`.\n"
        + "For DIRECT_READ, YES_NO, and mechanical TEXT_PROPERTY questions, the answer must exactly match the required value.\n"
        + "For REVERSE_GROUND, the answer must be a short visible location phrase that uses only the provided anchor/location wording.\n"
        + "When an anchor-local phrase is provided, you may use it, but keep the answer short and literal.\n"
        + "When `group_mode` is `scene_repeat_same_text`, ask about the repeated anchors as a group using plural anchor wording, not a single singled-out instance.\n"
        + "For visual TEXT_PROPERTY questions, the answer must be a short visible attribute phrase of 1 to 4 words and the question must stay about the text itself.\n"
        + "For ANCHOR_PROPERTY, the answer must be a short visible attribute phrase of 1 to 4 words and the question must use the text as the reference for which object or surface to describe.\n"
        + "Return one JSON item per candidate in the same order.\n\n"
        + "\n\n".join(blocks)
    )


def batched_response_schema(candidate_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {"type": "integer"},
                        "question_type": {"type": "string"},
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                    },
                    "required": ["candidate_index", "question_type", "question", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def answer_probe_response_schema(candidate_count: int, probe_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": candidate_count,
                "maxItems": candidate_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_index": {"type": "integer"},
                        "answers": {
                            "type": "array",
                            "minItems": probe_count,
                            "maxItems": probe_count,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["candidate_index", "answers"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def build_answer_probe_prompt(selected_candidates: list[dict[str, Any]], *, probe_count: int) -> str:
    blocks = []
    for candidate in selected_candidates:
        tuple_row = candidate["tuple"]
        item_question = str((candidate.get("generated_item") or {}).get("question") or "").strip()
        if not item_question:
            continue
        blocks.append(
            "\n".join(
                [
                    f"--- Candidate {candidate['candidate_index']} ---",
                    f"candidate_index: {candidate['candidate_index']}",
                    f"question_type: {candidate['question_type']}",
                    f'question: "{item_question}"',
                    f'anchor_label: "{candidate_anchor_label(candidate)}"',
                    f'text_gold: "{tuple_row.get("answer") or ""}"',
                    "Give several independent short answers to the same visual question.",
                    "Each answer must be literal and brief. Do not explain.",
                ]
            )
        )
    return (
        "You are probing answer ambiguity for OCR spatial QA.\n"
        f"For each candidate below, answer the same question {probe_count} times independently using the image.\n"
        "Return JSON only.\n\n"
        + "\n\n".join(blocks)
    )


def call_gemini_answer_probe_batched(
    *,
    model: str,
    image_payload: Any,
    selected_candidates: list[dict[str, Any]],
    probe_count: int,
    temperature: float,
    timeout_s: int = 120,
) -> dict[str, Any]:
    body = {
        "contents": [
            {
                "parts": [
                    {"text": build_answer_probe_prompt(selected_candidates, probe_count=probe_count)},
                    {
                        "inline_data": {
                            "mime_type": image_payload.mime_type,
                            "data": image_payload.image_b64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": float(temperature),
            "responseMimeType": "application/json",
            "responseJsonSchema": answer_probe_response_schema(len(selected_candidates), int(probe_count)),
        },
    }
    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={
            "x-goog-api-key": get_secret(GEMINI),
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    raw_text = extract_gemini_text(payload)
    return {
        "provider": "gemini",
        "model": model,
        "raw_text": raw_text,
        "parsed": json.loads(raw_text),
        "usage": payload.get("usageMetadata", {}),
    }


def _normalize_probe_answer(answer: str, tuple_row: dict[str, Any]) -> str:
    return normalize_answer(answer)


def annotate_answer_probe_rows(
    *,
    model: str,
    image_payload: Any,
    indexed_candidates: list[dict[str, Any]],
    normalized_rows: list[dict[str, Any]],
    tuning: Any,
) -> None:
    probe_count = int(tuning.teacher_answer_probe_count)
    if probe_count <= 1:
        return
    eligible_rows = []
    for candidate, row in zip(indexed_candidates, normalized_rows):
        if not row.get("ok"):
            continue
        if candidate["question_type"] not in {"DIRECT_READ", "REVERSE_GROUND", "TEXT_PROPERTY"}:
            continue
        item = (row.get("items") or [{}])[0]
        if not str(item.get("question") or "").strip():
            continue
        eligible_rows.append((candidate, row, item))
    if not eligible_rows:
        return
    probe_candidates = []
    for candidate, _, item in eligible_rows:
        probe_candidates.append({**candidate, "generated_item": item})
    try:
        result = call_gemini_answer_probe_batched(
            model=model,
            image_payload=image_payload,
            selected_candidates=probe_candidates,
            probe_count=probe_count,
            temperature=float(tuning.teacher_answer_probe_temperature),
        )
        items = list((result.get("parsed") or {}).get("items") or [])
        by_index = {int(item.get("candidate_index")): item for item in items if item.get("candidate_index") is not None}
    except Exception as exc:
        by_index = {}
        result = {"error": str(exc), "usage": {}}
    for candidate, row, _ in eligible_rows:
        item = by_index.get(int(candidate["candidate_index"])) or {}
        answers = [str(x or "").strip() for x in (item.get("answers") or [])]
        normalized = [_normalize_probe_answer(answer, candidate["tuple"]) for answer in answers if answer]
        counts = Counter(normalized)
        distinct_count = len(counts)
        plurality_answer = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0] if counts else ""
        plurality_count = int(max(counts.values()) if counts else 0)
        row["answer_probe"] = {
            "enabled": True,
            "count": probe_count,
            "temperature": float(tuning.teacher_answer_probe_temperature),
            "answers": answers,
            "normalized_answers": normalized,
            "distinct_count": distinct_count,
            "plurality_answer": plurality_answer,
            "plurality_count": plurality_count,
            "ambiguous": distinct_count >= 2,
            "usage": result.get("usage", {}),
            "error": result.get("error"),
        }


def apply_answer_probe_policy(normalized_rows: list[dict[str, Any]], *, tuning: Any) -> None:
    if int(tuning.teacher_answer_probe_count) <= 1:
        return
    for row in normalized_rows:
        probe = row.get("answer_probe") or {}
        if not probe or not probe.get("enabled") or not probe.get("ambiguous"):
            continue
        question_type = str(((row.get("tuple") or {}).get("question_type") or "")).upper()
        plurality_count = int(probe.get("plurality_count") or 0)
        distinct_count = int(probe.get("distinct_count") or 0)
        strict_probe_failure = False
        if question_type == "DIRECT_READ":
            strict_probe_failure = distinct_count >= 3 or plurality_count <= 1
        elif question_type == "REVERSE_GROUND":
            strict_probe_failure = distinct_count >= 3 and plurality_count <= 1
        if not strict_probe_failure:
            continue
        validations = list(row.get("validations") or [])
        validation = dict(validations[0] if validations else {})
        validation["accepted"] = False
        validation["mechanical_ok"] = False
        validation["answer_probe_ambiguous"] = True
        row["validations"] = [validation]
        row["summary"] = {"generated_count": 1, "accepted_count": 0}
        row["failure_reason"] = "answer_probe_ambiguous"
        row["filter_stage"] = {"reason": "answer_probe_ambiguous"}


def extract_gemini_text(payload: dict[str, Any]) -> str:
    for candidate in payload.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                return str(text)
    raise RuntimeError("Gemini response did not contain text")
