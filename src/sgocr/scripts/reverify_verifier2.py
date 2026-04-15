from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..dev40_complete import row_to_final_sample, validate_candidate_output
from ..run_quality import compute_run_quality


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def _accepted_rows_with_inline(experiment_dir: Path) -> dict[str, dict[str, Any]]:
    accepted_path = experiment_dir / "accepted_dataset.jsonl"
    if not accepted_path.exists():
        return {}
    return {str(row["sample_id"]): row for row in _load_jsonl(accepted_path)}


def _candidate_from_raw_row(row: dict[str, Any]) -> dict[str, Any]:
    tuple_row = dict(row.get("tuple") or {})
    return {
        "tuple": tuple_row,
        "question_type": tuple_row.get("question_type"),
        "answer_source": tuple_row.get("answer_source"),
        "answer_type": tuple_row.get("answer_type"),
        "yesno_polarity": tuple_row.get("yesno_polarity"),
        "yesno_distractor_source": tuple_row.get("yesno_distractor_source"),
        "text_property_type": tuple_row.get("text_property_type"),
        "anchor_property_type": tuple_row.get("anchor_property_type"),
        "expected_answer": tuple_row.get("expected_answer"),
        "queried_text": tuple_row.get("queried_text"),
        "query_text_reference": tuple_row.get("query_text_reference"),
        "query_anchor_label": tuple_row.get("query_anchor_label"),
        "query_anchor_synonyms": list(tuple_row.get("query_anchor_synonyms") or []),
        "query_anchor_box": tuple_row.get("query_anchor_box"),
        "query_anchor_local_phrase": tuple_row.get("query_anchor_local_phrase"),
        "query_anchor_local_synonyms": list(tuple_row.get("query_anchor_local_synonyms") or []),
        "query_location_phrase": tuple_row.get("query_location_phrase"),
        "query_location_synonyms": list(tuple_row.get("query_location_synonyms") or []),
        "query_specific_location_phrase": tuple_row.get("query_specific_location_phrase"),
        "query_specific_location_synonyms": list(tuple_row.get("query_specific_location_synonyms") or []),
        "query_relation": tuple_row.get("query_relation"),
        "query_location_required": bool(tuple_row.get("query_location_required")),
        "reverse_ground_scope_preference": tuple_row.get("reverse_ground_scope_preference"),
        "grounded_exclusion_score": tuple_row.get("grounded_exclusion_score"),
        "grounded_exclusion_source_tuple_id": tuple_row.get("grounded_exclusion_source_tuple_id"),
        "candidate_id": tuple_row.get("candidate_id") or row.get("sample_id"),
        "candidate_index": tuple_row.get("candidate_index") or 1,
    }


def _reverify_experiment(experiment_dir: Path) -> dict[str, Any]:
    summary_path = experiment_dir / "summary.json"
    raw_results_path = experiment_dir / "raw_results.jsonl"
    if not summary_path.exists() or not raw_results_path.exists():
        raise FileNotFoundError(f"Missing summary or raw_results in {experiment_dir}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_rows = _load_jsonl(raw_results_path)
    prior_accepted = _accepted_rows_with_inline(experiment_dir)
    updated_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()

    model = str(((summary.get("experiment") or {}).get("model")) or "gemini-2.5-flash")
    prompt_variant = str(((summary.get("experiment") or {}).get("prompt_variant")) or "semantic_dev40_v8")

    rescued_count = 0
    rescued_reasons: Counter[str] = Counter()

    for row in raw_rows:
        candidate = _candidate_from_raw_row(row)
        item = None
        items = list(row.get("items") or [])
        if items:
            item = {
                **items[0],
                "question_type": candidate.get("question_type"),
            }
        validations, mini_summary, failure_reason = validate_candidate_output(candidate, item)
        old_reason = str(row.get("failure_reason") or "")
        new_row = dict(row)
        new_row["validations"] = [validations]
        new_row["summary"] = mini_summary
        new_row["failure_reason"] = failure_reason
        new_row["filter_stage"] = {"reason": failure_reason} if failure_reason else None
        updated_rows.append(new_row)
        if validations.get("accepted"):
            final_row = row_to_final_sample(new_row, model=model, prompt_variant=prompt_variant)
            prior = prior_accepted.get(str(final_row["sample_id"]))
            if prior is not None:
                for key in ("inline_frontier", "inline_frontier_correct", "quality_tier"):
                    if key in prior:
                        final_row[key] = prior[key]
                if prior.get("tags"):
                    final_row["tags"]["quality_tier"] = prior["tags"].get("quality_tier", final_row["tags"].get("quality_tier"))
                    if "quality_tier" in prior:
                        final_row["quality_tier"] = prior["quality_tier"]
            final_rows.append(final_row)
            if old_reason and not failure_reason:
                rescued_count += 1
                rescued_reasons[old_reason] += 1
        else:
            failure_counts[failure_reason or "validation_failed"] += 1

    new_summary = dict(summary)
    new_summary["accepted_qas"] = len(final_rows)
    new_summary["generated_qas"] = len(updated_rows)
    new_summary["qa_accept_rate"] = (len(final_rows) / len(updated_rows)) if updated_rows else 0.0
    new_summary["images_with_final_rows"] = len({row["image_id"] for row in final_rows})
    new_summary["failure_counts"] = dict(sorted(failure_counts.items()))

    question_type_counts: Counter[str] = Counter()
    for row in updated_rows:
        qtype = str((row.get("tuple") or {}).get("question_type") or "")
        if qtype:
            question_type_counts[qtype] += 1
    selected_question_type_counts: Counter[str] = Counter(row["question_type"] for row in final_rows)
    new_summary["question_type_counts"] = dict(question_type_counts)
    new_summary["selected_question_type_counts"] = dict(selected_question_type_counts)
    new_summary["mean_question_words"] = (
        sum(len(str(row["question"]).split()) for row in final_rows) / len(final_rows) if final_rows else 0.0
    )

    quality_metrics = compute_run_quality(new_summary, final_rows)
    reverify_summary = {
        "policy_name": "verifier2_relaxed_specific_location_v1",
        "experiment_dir": str(experiment_dir),
        "accepted_qas": int(quality_metrics["accepted_qas"]),
        "generated_qas": int(quality_metrics["generated_qas"]),
        "qa_accept_rate": float(quality_metrics["qa_accept_rate"]),
        "images_with_final_rows": int(quality_metrics["images_with_final_rows"]),
        "failure_counts": dict(sorted(failure_counts.items())),
        "question_types": quality_metrics["question_types"],
        "quality_score": float(quality_metrics["quality_score"]),
        "q3_score": float(quality_metrics["q3_score"]),
        "inline_frontier_mean_lower_bound": float(quality_metrics["inline_frontier_mean"]),
        "inline_frontier_scored_rows_lower_bound": int(quality_metrics["inline_frontier_scored"]),
        "precision_first_score_lower_bound": float(quality_metrics["precision_first_score"]),
        "sweep_score_lower_bound": float(quality_metrics["sweep_score"]),
        "rescued_count": rescued_count,
        "rescued_reasons": dict(sorted(rescued_reasons.items())),
    }

    _write_jsonl(experiment_dir / "accepted_dataset_verifier2.jsonl", final_rows)
    _write_jsonl(experiment_dir / "raw_results_verifier2.jsonl", updated_rows)
    _write_json(experiment_dir / "reverify_verifier2_summary.json", reverify_summary)
    return reverify_summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Posthoc re-verify existing SGOCR experiment dirs with the current verifier2 policy.")
    ap.add_argument("experiment_dirs", nargs="+", help="One or more final experiment directories.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    for raw in args.experiment_dirs:
        experiment_dir = Path(raw)
        summary = _reverify_experiment(experiment_dir)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
