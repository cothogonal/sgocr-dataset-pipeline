from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..verify import validate_generated_items
from .http_clients import ImagePayload, call_gemini, call_openai_chat, encode_image
from .prompts import PROMPT_VARIANTS, PromptVariant


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    provider: str
    model: str
    prompt_variant: str
    limit: int
    max_side: int = 768
    workers: int = 1


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["density_bucket"], row["area_bucket"], row["region_key"], row["image_id"]))[:limit]


def _client_for(provider: str) -> Callable[[str, dict[str, Any], ImagePayload, PromptVariant], dict[str, Any]]:
    if provider == "gemini":
        return call_gemini
    if provider == "openai":
        return call_openai_chat
    raise ValueError(f"Unsupported provider: {provider}")


def run_experiment(spec: ExperimentSpec, *, tuples_path: Path, out_dir: Path) -> dict[str, Any]:
    rows = _select_rows(_load_jsonl(tuples_path), spec.limit)
    variant = PROMPT_VARIANTS[spec.prompt_variant]
    client = _client_for(spec.provider)

    image_cache: dict[str, ImagePayload] = {}

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        image_path = str(row["image_path"])
        payload = image_cache.get(image_path)
        if payload is None:
            payload = encode_image(Path(image_path), max_side=spec.max_side)
            image_cache[image_path] = payload
        try:
            result = client(spec.model, row, payload, variant)
            items = list((result.get("parsed") or {}).get("items") or [])
            validations, summary = validate_generated_items(row, items)
            return {
                "ok": True,
                "image_id": row["image_id"],
                "ann_id": row["ann_id"],
                "tuple": row,
                "result": result,
                "items": items,
                "validations": [v.__dict__ for v in validations],
                "summary": summary,
            }
        except Exception as exc:
            return {
                "ok": False,
                "image_id": row["image_id"],
                "ann_id": row["ann_id"],
                "tuple": row,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    if spec.workers > 1 and spec.provider == "gemini":
        results = []
        with ThreadPoolExecutor(max_workers=spec.workers) as ex:
            futures = [ex.submit(run_one, row) for row in rows]
            for fut in as_completed(futures):
                results.append(fut.result())
        results.sort(key=lambda row: (row["image_id"], row["ann_id"]))
    else:
        results = [run_one(row) for row in rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "raw_results.jsonl", results)

    success_rows = [row for row in results if row.get("ok")]
    failed_rows = [row for row in results if not row.get("ok")]
    accepted_dataset_rows = []
    accepted_qas = sum(int(item["accepted"]) for row in success_rows for item in row["validations"])
    generated_qas = sum(len(row["items"]) for row in success_rows)
    prompt_tokens = []
    completion_tokens = []
    total_tokens = []
    question_lengths = []
    for row in success_rows:
        usage = row["result"].get("usage") or {}
        if "prompt_tokens" in usage:
            prompt_tokens.append(int(usage.get("prompt_tokens") or 0))
            completion_tokens.append(int(usage.get("completion_tokens") or 0))
            total_tokens.append(int(usage.get("total_tokens") or 0))
        else:
            prompt_tokens.append(int(usage.get("promptTokenCount") or 0))
            completion_tokens.append(int(usage.get("candidatesTokenCount") or 0))
            total_tokens.append(int(usage.get("totalTokenCount") or 0))
        for item in row["items"]:
            question_lengths.append(len(str(item["question"]).split()))
        for item, val in zip(row["items"], row["validations"]):
            if not val["accepted"]:
                continue
            tup = row["tuple"]
            accepted_dataset_rows.append(
                {
                    "image_id": tup["image_id"],
                    "question": item["question"],
                    "answer": item["answer"],
                    "answer_level": tup["answer_level"],
                    "anchor_label": tup["anchor_label"],
                    "relation": tup["relation"],
                    "ref_label": None,
                    "text_polygon": tup["text_polygon"],
                    "text_bbox": tup["text_bbox"],
                    "unique": tup["unique"],
                    "dataset_source": tup["source_dataset"],
                    "teacher_provider": spec.provider,
                    "teacher_model": spec.model,
                    "prompt_variant": spec.prompt_variant,
                    "bootstrap_region_key": tup["region_key"],
                }
            )

    summary = {
        "experiment": spec.__dict__,
        "tuple_count": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(failed_rows),
        "generated_qas": generated_qas,
        "accepted_qas": accepted_qas,
        "qa_accept_rate": (accepted_qas / generated_qas) if generated_qas else 0.0,
        "tuple_full_accept_rate": (
            sum(1 for row in success_rows if row["summary"]["accepted_count"] == len(row["items"])) / len(rows)
            if rows
            else 0.0
        ),
        "mean_prompt_tokens": statistics.mean(prompt_tokens) if prompt_tokens else 0.0,
        "mean_completion_tokens": statistics.mean(completion_tokens) if completion_tokens else 0.0,
        "mean_total_tokens": statistics.mean(total_tokens) if total_tokens else 0.0,
        "mean_question_words": statistics.mean(question_lengths) if question_lengths else 0.0,
    }
    _write_json(out_dir / "summary.json", summary)
    _write_jsonl(out_dir / "accepted_dataset.jsonl", accepted_dataset_rows)
    return summary
