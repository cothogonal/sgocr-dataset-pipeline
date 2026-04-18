from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from ..bootstrap import write_json, write_jsonl
from ..paths import OCR_SPATIAL_QA_FINAL_ROOT
from ..run_quality import compute_anchor_coverage, compute_answer_distribution, compute_run_quality
from .dev200_eval import (
    _call_gemini_text_only_eval,
    _call_model,
    _call_openai_text_only_eval,
    _load_jsonl,
    _normalize_answer_by_type,
    _parse_model_spec,
    _score_prediction,
    compute_frontier_agreement,
    compute_frontier_ambiguity_agreement,
    run_frontier_ambiguity_eval,
    run_frontier_benchmark,
)


DEFAULT_RUNS = [
    "sgocr_dam01_20260417_081018_coverage_dam01_r72_t10",
    "sgocr_dam01_20260417_081018_balanced_ita15_r48",
    "sgocr_mixed_ita20_20260417_033706_g_t8_no_yn_tp_visual",
    "sgocr_mixed_ita21_20260417_060334_g_t8_no_yn_no_rg_tp_vis_cement",
]

DEFAULT_CODEX_MODEL = "openai:gpt-5.3-codex"
DEFAULT_GEMINI_MODEL = "gemini:gemini-2.5-flash"
_GENERIC_ANCHOR_TOKENS = frozenset({
    "sign",
    "label",
    "panel",
    "display",
    "screen",
    "wall",
    "surface",
    "object",
    "area",
    "region",
    "document",
    "box",
    "poster",
    "board",
})
_LOCATION_TOKENS = (
    "upper-left", "upper left", "upper-right", "upper right",
    "lower-left", "lower left", "lower-right", "lower right",
    "middle-left", "middle left", "middle-right", "middle right",
    "top-center", "top center", "bottom-center", "bottom center",
    "center of the image", "area of the image", "in the image",
)


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="Run a final mixed150 decision panel across four candidate dataset variants."
    )
    ap.add_argument("--run", action="append", dest="runs", default=[])
    ap.add_argument("--codex-model", default=DEFAULT_CODEX_MODEL)
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument("--vdep-workers", type=int, default=6)
    ap.add_argument("--benchmark-workers", type=int, default=8)
    ap.add_argument("--ambiguity-workers", type=int, default=8)
    ap.add_argument("--out-name", default=f"sgocr_mixed150_final_decision_{stamp}")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) >= 2 else 0.0


def _load_summary(experiment_dir: Path) -> dict[str, Any]:
    return json.loads((experiment_dir / "summary.json").read_text(encoding="utf-8"))


def _normalized_answer_in_question(row: dict[str, Any]) -> bool:
    answer = _normalize_answer_by_type(
        str(row.get("answer") or ""),
        str((row.get("tags") or {}).get("answer_type") or ""),
    )
    question = _normalize_answer_by_type(
        str(row.get("question") or ""),
        "text_string",
    )
    if not answer or len(answer) < 2:
        return False
    if answer in {"yes", "no"}:
        return False
    return answer in question


def _quoted_answer_in_question(row: dict[str, Any]) -> bool:
    answer = str(row.get("answer") or "").strip()
    question = str(row.get("question") or "")
    if not answer or len(answer) < 2:
        return False
    quoted_forms = (
        f"'{answer}'",
        f'"{answer}"',
        f"“{answer}”",
        f"‘{answer}’",
    )
    return any(token in question for token in quoted_forms)


def _is_generic_anchor(row: dict[str, Any]) -> bool:
    anchor = str(row.get("anchor_label") or (row.get("tags") or {}).get("anchor_label") or "").strip().lower()
    if not anchor:
        return True
    words = {part for part in anchor.replace("-", " ").split() if part}
    return bool(words & _GENERIC_ANCHOR_TOKENS)


def _has_templatey_location(row: dict[str, Any]) -> bool:
    question = str(row.get("question") or "").lower()
    return any(token in question for token in _LOCATION_TOKENS)


def _join_row_fields(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    joined: dict[str, dict[str, Any]] = {}
    for row in rows:
        tags = dict(row.get("tags") or {})
        joined[str(row.get("sample_id") or "")] = {
            "sample_id": str(row.get("sample_id") or ""),
            "image_id": str(row.get("image_id") or ""),
            "dataset_source": str(row.get("dataset_source") or tags.get("image_source") or ""),
            "question_type": str(tags.get("question_type") or row.get("question_type") or ""),
            "answer_type": str(tags.get("answer_type") or ""),
            "difficulty": str(tags.get("difficulty") or ""),
            "ambiguity_level": str(tags.get("ambiguity_level") or ""),
            "anchor_label": str(row.get("anchor_label") or tags.get("anchor_label") or ""),
            "question": str(row.get("question") or ""),
            "answer": str(row.get("answer") or ""),
            "teacher_model": str(row.get("teacher_model") or ""),
            "teacher_provider": str(row.get("teacher_provider") or ""),
            "prompt_variant": str(row.get("prompt_variant") or ""),
            "inline_frontier_correct": row.get("inline_frontier_correct"),
            "normalized_answer_in_question": _normalized_answer_in_question(row),
            "quoted_answer_in_question": _quoted_answer_in_question(row),
            "generic_anchor": _is_generic_anchor(row),
            "templatey_location": _has_templatey_location(row),
        }
    return joined


def _group_rows(
    rows: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _aggregate_dataset_groups(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped = _group_rows(rows, lambda row: str(row.get(key) or ""))
    result: list[dict[str, Any]] = []
    for group_name, subset in grouped.items():
        if not group_name:
            continue
        qtypes = Counter(str(row.get("question_type") or "") for row in subset)
        result.append(
            {
                "group": group_name,
                "rows": len(subset),
                "images": len({str(row.get("image_id") or "") for row in subset}),
                "mean_rows_per_image": round(_safe_div(len(subset), len({str(row.get("image_id") or "") for row in subset})), 4),
                "answer_in_question_rate": round(_safe_div(sum(1 for row in subset if row.get("normalized_answer_in_question")), len(subset)), 4),
                "quoted_answer_rate": round(_safe_div(sum(1 for row in subset if row.get("quoted_answer_in_question")), len(subset)), 4),
                "generic_anchor_rate": round(_safe_div(sum(1 for row in subset if row.get("generic_anchor")), len(subset)), 4),
                "templatey_location_rate": round(_safe_div(sum(1 for row in subset if row.get("templatey_location")), len(subset)), 4),
                "inline_frontier_rate": round(_safe_div(sum(1 for row in subset if row.get("inline_frontier_correct") is True), sum(1 for row in subset if row.get("inline_frontier_correct") is not None)), 4),
                "question_types": dict(sorted(qtypes.items())),
            }
        )
    result.sort(key=lambda row: (-row["rows"], row["group"]))
    return result


def _aggregate_vdep_groups(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped = _group_rows(rows, lambda row: str(row.get(key) or ""))
    result: list[dict[str, Any]] = []
    for group_name, subset in grouped.items():
        if not group_name:
            continue
        result.append(
            {
                "group": group_name,
                "n": len(subset),
                "image_accuracy": round(_mean([float(row.get("image_soft") or 0.0) for row in subset]), 4),
                "text_only_accuracy": round(_mean([float(row.get("text_only_soft") or 0.0) for row in subset]), 4),
                "vision_delta_mean": round(_mean([float(row.get("vision_delta") or 0.0) for row in subset]), 4),
                "vision_necessary_rate": round(_safe_div(sum(1 for row in subset if row.get("vision_necessary")), len(subset)), 4),
                "text_leaky_rate": round(_safe_div(sum(1 for row in subset if row.get("text_leaky")), len(subset)), 4),
            }
        )
    result.sort(key=lambda row: (-row["n"], row["group"]))
    return result


def _aggregate_benchmark_groups(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped = _group_rows(rows, lambda row: str(row.get(key) or ""))
    result: list[dict[str, Any]] = []
    for group_name, subset in grouped.items():
        if not group_name:
            continue
        valid = [row for row in subset if not row.get("error_type")]
        result.append(
            {
                "group": group_name,
                "n": len(subset),
                "valid_rows": len(valid),
                "exact_accuracy": round(_safe_div(sum(1 for row in valid if row.get("exact_correct")), len(valid)), 4),
                "soft_accuracy": round(_safe_div(sum(1 for row in valid if row.get("soft_correct")), len(valid)), 4),
                "error_rate": round(_safe_div(sum(1 for row in subset if row.get("error_type")), len(subset)), 4),
            }
        )
    result.sort(key=lambda row: (-row["n"], row["group"]))
    return result


def _aggregate_ambiguity_groups(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped = _group_rows(rows, lambda row: str(row.get(key) or ""))
    result: list[dict[str, Any]] = []
    for group_name, subset in grouped.items():
        if not group_name:
            continue
        result.append(
            {
                "group": group_name,
                "n": len(subset),
                "ambiguous_rate": round(_safe_div(sum(1 for row in subset if row.get("ambiguous")), len(subset)), 4),
                "mean_confidence": round(_mean([float(row.get("confidence") or 0.0) for row in subset]), 4),
                "parse_ok_rate": round(_safe_div(sum(1 for row in subset if row.get("parse_ok")), len(subset)), 4),
            }
        )
    result.sort(key=lambda row: (-row["n"], row["group"]))
    return result


def _pairwise_rate_by_group(
    rows: list[dict[str, Any]],
    *,
    model_field: str,
    value_field: str,
    group_field: str,
) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[str(row.get("sample_id") or "")].append(row)
    grouped_hits: dict[str, list[bool]] = defaultdict(list)
    for sample_rows in by_sample.values():
        if len(sample_rows) < 2:
            continue
        sample_rows = sorted(sample_rows, key=lambda row: str(row.get(model_field) or ""))
        values = [row.get(value_field) for row in sample_rows]
        if any(value is None for value in values):
            continue
        group_name = str(sample_rows[0].get(group_field) or "")
        grouped_hits[group_name].append(len(set(values)) == 1)
    result: list[dict[str, Any]] = []
    for group_name, flags in sorted(grouped_hits.items()):
        if not group_name:
            continue
        result.append(
            {
                "group": group_name,
                "n": len(flags),
                "agreement_rate": round(_safe_div(sum(1 for flag in flags if flag), len(flags)), 4),
            }
        )
    result.sort(key=lambda row: (-row["n"], row["group"]))
    return result


def _extract_notable_tuning(summary: dict[str, Any]) -> dict[str, Any]:
    tuning = dict(((summary.get("experiment") or {}).get("tuning") or {}))
    interesting_keys = [
        "max_yesno_per_image",
        "max_negative_yesno_per_image",
        "tp_visual_only_enabled",
        "tp_per_image_hard_cap",
        "rg_vdep_check_enabled",
        "rg_vdep_model",
        "rg_per_image_hard_cap",
        "rg_structural_anchor_filter_enabled",
        "rg_scrub_color_anchor_phrases_enabled",
        "qwen_dam01_prompt_enabled",
        "property_candidate_selection_bonus",
        "per_image_anchor_diversity_bonus",
        "anchor_type_soft_cap_count",
        "anchor_type_soft_cap_penalty",
        "direct_read_selection_bonus",
        "dr_generic_anchor_penalty",
        "dr_same_anchor_repeat_penalty",
    ]
    return {key: tuning[key] for key in interesting_keys if key in tuning}


def run_provider_image_dependence_eval(
    *,
    experiment_dir: Path,
    model_spec_text: str,
    rows: list[dict[str, Any]],
    metadata_by_sample: dict[str, dict[str, Any]],
    out_dir: Path,
    workers: int,
    force: bool,
) -> dict[str, Any]:
    summary_path = out_dir / "summary.json"
    rows_path = out_dir / "rows.jsonl"
    if not force and summary_path.exists() and rows_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))

    spec = _parse_model_spec(model_spec_text)
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_one(row: dict[str, Any]) -> dict[str, Any]:
        image_result = {"answer": ""}
        text_result = {"answer": ""}
        try:
            image_result = _call_model(spec, row)
            image_scored = _score_prediction(row, str(image_result.get("answer") or ""))
        except Exception as exc:
            image_scored = {"soft_correct": False, "error": str(exc)}
        try:
            if spec.provider == "openai":
                text_result = _call_openai_text_only_eval(spec.model, row)
            else:
                text_result = _call_gemini_text_only_eval(spec.model, row)
            text_scored = _score_prediction(row, str(text_result.get("answer") or ""))
        except Exception as exc:
            text_scored = {"soft_correct": False, "error": str(exc)}

        image_soft = int(bool(image_scored.get("soft_correct")))
        text_only_soft = int(bool(text_scored.get("soft_correct")))
        meta = dict(metadata_by_sample.get(str(row.get("sample_id") or ""), {}))
        return {
            "sample_id": row.get("sample_id"),
            "image_id": row.get("image_id"),
            "requested_model": model_spec_text,
            "provider": spec.provider,
            "model": spec.model,
            "image_soft": image_soft,
            "text_only_soft": text_only_soft,
            "vision_delta": image_soft - text_only_soft,
            "vision_necessary": image_soft > text_only_soft,
            "text_leaky": text_only_soft == 1,
            "image_prediction": image_result.get("answer") if not image_scored.get("error") else "",
            "text_prediction": text_result.get("answer") if not text_scored.get("error") else "",
            "gold_answer": row.get("answer"),
            **meta,
        }

    if int(workers) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as ex:
            futures = [ex.submit(run_one, row) for row in rows]
            result_rows = [future.result() for future in concurrent.futures.as_completed(futures)]
    else:
        result_rows = [run_one(row) for row in rows]
    result_rows.sort(key=lambda row: str(row.get("sample_id") or ""))

    n = len(result_rows)
    summary = {
        "experiment_dir": str(experiment_dir),
        "dataset_path": str(experiment_dir / "ocr_qa_dataset.jsonl"),
        "requested_model": model_spec_text,
        "provider": spec.provider,
        "model": spec.model,
        "rows": n,
        "image_accuracy": round(_safe_div(sum(row["image_soft"] for row in result_rows), n), 4),
        "text_only_accuracy": round(_safe_div(sum(row["text_only_soft"] for row in result_rows), n), 4),
        "vision_delta_mean": round(_mean([float(row["vision_delta"]) for row in result_rows]), 4),
        "vision_delta_stdev": round(_stdev([float(row["vision_delta"]) for row in result_rows]), 4),
        "vision_necessary_rate": round(_safe_div(sum(1 for row in result_rows if row["vision_necessary"]), n), 4),
        "text_leaky_rate": round(_safe_div(sum(1 for row in result_rows if row["text_leaky"]), n), 4),
        "by_question_type": _aggregate_vdep_groups(result_rows, "question_type"),
        "by_dataset_source": _aggregate_vdep_groups(result_rows, "dataset_source"),
        "by_difficulty": _aggregate_vdep_groups(result_rows, "difficulty"),
        "by_ambiguity_level": _aggregate_vdep_groups(result_rows, "ambiguity_level"),
        "by_answer_type": _aggregate_vdep_groups(result_rows, "answer_type"),
    }
    write_json(summary_path, summary)
    write_jsonl(rows_path, result_rows)
    return summary


def _load_prediction_rows(path: Path, metadata_by_sample: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _load_jsonl(path)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        meta = dict(metadata_by_sample.get(str(row.get("sample_id") or ""), {}))
        enriched.append({**row, **{k: v for k, v in meta.items() if k not in row}})
    return enriched


def _safe_model_metric(summary: dict[str, Any], requested_model: str) -> dict[str, Any]:
    target = str(requested_model or "").split(":", 1)[-1]
    for item in summary.get("models") or []:
        candidate_requested = str(item.get("requested_model") or "")
        candidate_model = str(item.get("model") or "")
        if candidate_requested == requested_model or candidate_model == requested_model:
            return dict(item)
        if candidate_requested == target or candidate_model == target:
            return dict(item)
    return {}


def _model_matches(row: dict[str, Any], requested_model: str) -> bool:
    target = str(requested_model or "").split(":", 1)[-1]
    for key in ("requested_model", "model"):
        value = str(row.get(key) or "")
        if value == requested_model or value == target:
            return True
    return False


def _build_overall_row(
    *,
    name: str,
    codex_model: str,
    gemini_model: str,
    dataset_metrics: dict[str, Any],
    answer_distribution: dict[str, Any],
    anchor_coverage: dict[str, Any],
    codex_vdep: dict[str, Any],
    benchmark_summary: dict[str, Any],
    ambiguity_summary: dict[str, Any],
    answer_agreement: dict[str, Any],
    ambiguity_agreement: dict[str, Any],
) -> dict[str, Any]:
    codex_bench = _safe_model_metric(benchmark_summary, codex_model)
    gemini_bench = _safe_model_metric(benchmark_summary, gemini_model)
    codex_amb = _safe_model_metric(ambiguity_summary, codex_model)
    gemini_amb = _safe_model_metric(ambiguity_summary, gemini_model)
    pairwise_answer = (answer_agreement.get("pairwise") or [{}])[0]
    pairwise_ambiguity = (ambiguity_agreement.get("pairwise") or [{}])[0]
    type_diversity = float(dataset_metrics.get("type_diversity") or 0.0)
    answer_entropy = min(float(answer_distribution.get("answer_entropy") or 0.0) / 4.8, 1.0)
    anchor_div = min(float(anchor_coverage.get("mean_unique_anchors") or 0.0) / 2.0, 1.0)
    decision_score = (
        0.26 * float(codex_vdep.get("vision_necessary_rate") or 0.0)
        + 0.20 * (1.0 - float(codex_vdep.get("text_leaky_rate") or 0.0))
        + 0.16 * float(codex_bench.get("soft_accuracy") or 0.0)
        + 0.10 * float(gemini_bench.get("soft_accuracy") or 0.0)
        + 0.10 * type_diversity
        + 0.06 * answer_entropy
        + 0.05 * anchor_div
        + 0.04 * (1.0 - float(codex_amb.get("ambiguous_rate") or 0.0))
        + 0.03 * float(pairwise_answer.get("agreement_rate") or 0.0)
    )
    return {
        "run": name,
        "rows": int(dataset_metrics.get("accepted_qas") or 0),
        "sweep_score": float(dataset_metrics.get("sweep_score") or 0.0),
        "q4_score": float(dataset_metrics.get("q4_score") or 0.0),
        "type_diversity": round(type_diversity, 4),
        "answer_entropy": float(answer_distribution.get("answer_entropy") or 0.0),
        "mean_unique_anchors": float(anchor_coverage.get("mean_unique_anchors") or 0.0),
        "codex_vdep": float(codex_vdep.get("vision_necessary_rate") or 0.0),
        "codex_leaky": float(codex_vdep.get("text_leaky_rate") or 0.0),
        "codex_soft": float(codex_bench.get("soft_accuracy") or 0.0),
        "gemini_soft": float(gemini_bench.get("soft_accuracy") or 0.0),
        "codex_ambiguous": float(codex_amb.get("ambiguous_rate") or 0.0),
        "gemini_ambiguous": float(gemini_amb.get("ambiguous_rate") or 0.0),
        "answer_agreement": float(pairwise_answer.get("agreement_rate") or 0.0),
        "ambiguity_agreement": float(pairwise_ambiguity.get("agreement_rate") or 0.0),
        "decision_score": round(decision_score * 100.0, 2),
    }


def _render_group_table(
    title: str,
    rows_by_run: dict[str, list[dict[str, Any]]],
    columns: list[tuple[str, str]],
) -> list[str]:
    lines = [f"## {title}", "", f"| Run | {' | '.join(header for header, _ in columns)} |", f"|---|{'---|' * len(columns)}"]
    for run_name, rows in rows_by_run.items():
        for row in rows:
            values = []
            for _, key in columns:
                value = row.get(key)
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            lines.append(f"| `{run_name}` | {' | '.join(values)} |")
        lines.append("")
    return lines


def _render_report(payload: dict[str, Any]) -> str:
    overall = list(payload.get("overall_ranking") or [])
    lines = [
        "# Mixed150 Final Decision Panel",
        "",
        f"- Generated: `{payload['updated_at']}`",
        f"- Codex model: `{payload['codex_model']}`",
        f"- Gemini model: `{payload['gemini_model']}`",
        f"- Runs: `{', '.join(row['run'] for row in overall)}`",
        "",
        "## Overall Ranking",
        "",
        "| Run | Decision | Rows | Sweep | Q4 | Type div | Codex vdep | Codex leaky | Codex soft | Gemini soft | Codex ambig | Ans agree |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| `{row['run']}` | {row['decision_score']:.2f} | {row['rows']} | {row['sweep_score']:.4f} | "
            f"{row['q4_score']:.2f} | {row['type_diversity']:.4f} | {row['codex_vdep']:.4f} | {row['codex_leaky']:.4f} | "
            f"{row['codex_soft']:.4f} | {row['gemini_soft']:.4f} | {row['codex_ambiguous']:.4f} | {row['answer_agreement']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            payload.get("recommendation_text") or "",
            "",
        ]
    )

    lines.extend(
        _render_group_table(
            "Codex Vdep By Question Type",
            {name: run["codex_vdep"]["by_question_type"] for name, run in payload["runs"].items()},
            [
                ("Group", "group"),
                ("N", "n"),
                ("VisionNec", "vision_necessary_rate"),
                ("TextLeaky", "text_leaky_rate"),
                ("ImgAcc", "image_accuracy"),
                ("TxtAcc", "text_only_accuracy"),
            ],
        )
    )
    lines.extend(
        _render_group_table(
            "Codex Vdep By Source",
            {name: run["codex_vdep"]["by_dataset_source"] for name, run in payload["runs"].items()},
            [
                ("Group", "group"),
                ("N", "n"),
                ("VisionNec", "vision_necessary_rate"),
                ("TextLeaky", "text_leaky_rate"),
            ],
        )
    )
    lines.extend(
        _render_group_table(
            "Codex Answer Soft By Question Type",
            {name: run["codex_benchmark_by_question_type"] for name, run in payload["runs"].items()},
            [
                ("Group", "group"),
                ("N", "n"),
                ("Soft", "soft_accuracy"),
                ("Exact", "exact_accuracy"),
                ("Err", "error_rate"),
            ],
        )
    )
    lines.extend(
        _render_group_table(
            "Codex Ambiguity By Question Type",
            {name: run["codex_ambiguity_by_question_type"] for name, run in payload["runs"].items()},
            [
                ("Group", "group"),
                ("N", "n"),
                ("Ambig", "ambiguous_rate"),
                ("Conf", "mean_confidence"),
                ("ParseOK", "parse_ok_rate"),
            ],
        )
    )
    lines.extend(
        _render_group_table(
            "Local Dataset Audit By Question Type",
            {name: run["dataset_audit_by_question_type"] for name, run in payload["runs"].items()},
            [
                ("Group", "group"),
                ("Rows", "rows"),
                ("AnsInQ", "answer_in_question_rate"),
                ("QuotedAns", "quoted_answer_rate"),
                ("GenericAnch", "generic_anchor_rate"),
                ("TemplateLoc", "templatey_location_rate"),
                ("InlineFrontier", "inline_frontier_rate"),
            ],
        )
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    final_root = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150"
    run_names = list(args.runs or DEFAULT_RUNS)
    run_dirs = {name: final_root / name for name in run_names}
    missing = [name for name, path in run_dirs.items() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing experiment dirs: {missing}")

    panel_dir = final_root / args.out_name
    panel_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "bundle_id": args.out_name,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "codex_model": str(args.codex_model),
        "gemini_model": str(args.gemini_model),
        "runs": {},
    }

    for name, experiment_dir in run_dirs.items():
        rows = _load_jsonl(experiment_dir / "ocr_qa_dataset.jsonl")
        summary = _load_summary(experiment_dir)
        dataset_metrics = compute_run_quality(summary, rows)
        answer_distribution = compute_answer_distribution(rows)
        anchor_coverage = compute_anchor_coverage(rows)
        metadata_by_sample = _join_row_fields(rows)
        audit_rows = list(metadata_by_sample.values())

        vdep_dir = experiment_dir / "evals" / "final_decision_vdep_codex_v1"
        codex_vdep = run_provider_image_dependence_eval(
            experiment_dir=experiment_dir,
            model_spec_text=str(args.codex_model),
            rows=rows,
            metadata_by_sample=metadata_by_sample,
            out_dir=vdep_dir,
            workers=int(args.vdep_workers),
            force=bool(args.force),
        )

        benchmark_dir = experiment_dir / "evals" / "final_decision_frontier_benchmark_v1"
        if bool(args.force) and benchmark_dir.exists():
            # Simple overwrite semantics: downstream helpers replace files in-place.
            pass
        benchmark_summary = run_frontier_benchmark(
            experiment_dir=experiment_dir,
            dataset_path=experiment_dir / "ocr_qa_dataset.jsonl",
            model_specs=[str(args.codex_model), str(args.gemini_model)],
            out_dir=benchmark_dir,
            workers=int(args.benchmark_workers),
            write_index=False,
        ) if bool(args.force) or not (benchmark_dir / "summary.json").exists() or not (benchmark_dir / "predictions.jsonl").exists() else json.loads((benchmark_dir / "summary.json").read_text(encoding="utf-8"))
        answer_agreement = compute_frontier_agreement(benchmark_dir=benchmark_dir)
        benchmark_rows = _load_prediction_rows(benchmark_dir / "predictions.jsonl", metadata_by_sample)

        ambiguity_dir = experiment_dir / "evals" / "final_decision_frontier_ambiguity_v1"
        ambiguity_summary = run_frontier_ambiguity_eval(
            experiment_dir=experiment_dir,
            dataset_path=experiment_dir / "ocr_qa_dataset.jsonl",
            model_specs=[str(args.codex_model), str(args.gemini_model)],
            out_dir=ambiguity_dir,
            workers=int(args.ambiguity_workers),
        ) if bool(args.force) or not (ambiguity_dir / "ambiguity_summary.json").exists() or not (ambiguity_dir / "ambiguity_predictions.jsonl").exists() else json.loads((ambiguity_dir / "ambiguity_summary.json").read_text(encoding="utf-8"))
        ambiguity_agreement = compute_frontier_ambiguity_agreement(benchmark_dir=ambiguity_dir)
        ambiguity_rows = _load_prediction_rows(ambiguity_dir / "ambiguity_predictions.jsonl", metadata_by_sample)

        payload["runs"][name] = {
            "experiment_dir": str(experiment_dir),
            "teacher_model": str(rows[0].get("teacher_model") or "") if rows else "",
            "teacher_provider": str(rows[0].get("teacher_provider") or "") if rows else "",
            "prompt_variant": str(rows[0].get("prompt_variant") or "") if rows else "",
            "dataset_metrics": dataset_metrics,
            "answer_distribution": answer_distribution,
            "anchor_coverage": anchor_coverage,
            "codex_vdep": codex_vdep,
            "benchmark_summary": benchmark_summary,
            "ambiguity_summary": ambiguity_summary,
            "answer_agreement": answer_agreement,
            "ambiguity_agreement": ambiguity_agreement,
            "dataset_audit_by_question_type": _aggregate_dataset_groups(audit_rows, "question_type"),
            "dataset_audit_by_source": _aggregate_dataset_groups(audit_rows, "dataset_source"),
            "dataset_audit_by_difficulty": _aggregate_dataset_groups(audit_rows, "difficulty"),
            "dataset_audit_by_ambiguity": _aggregate_dataset_groups(audit_rows, "ambiguity_level"),
            "codex_benchmark_by_question_type": _aggregate_benchmark_groups(
                [row for row in benchmark_rows if _model_matches(row, str(args.codex_model))],
                "question_type",
            ),
            "codex_benchmark_by_source": _aggregate_benchmark_groups(
                [row for row in benchmark_rows if _model_matches(row, str(args.codex_model))],
                "dataset_source",
            ),
            "codex_ambiguity_by_question_type": _aggregate_ambiguity_groups(
                [row for row in ambiguity_rows if _model_matches(row, str(args.codex_model))],
                "question_type",
            ),
            "codex_ambiguity_by_source": _aggregate_ambiguity_groups(
                [row for row in ambiguity_rows if _model_matches(row, str(args.codex_model))],
                "dataset_source",
            ),
            "pairwise_answer_agreement_by_question_type": _pairwise_rate_by_group(
                benchmark_rows,
                model_field="requested_model",
                value_field="prediction_norm",
                group_field="question_type",
            ),
            "pairwise_ambiguity_agreement_by_question_type": _pairwise_rate_by_group(
                ambiguity_rows,
                model_field="requested_model",
                value_field="ambiguous",
                group_field="question_type",
            ),
            "notable_tuning": _extract_notable_tuning(summary),
        }

    overall = [
        _build_overall_row(
            name=name,
            codex_model=str(args.codex_model),
            gemini_model=str(args.gemini_model),
            dataset_metrics=run["dataset_metrics"],
            answer_distribution=run["answer_distribution"],
            anchor_coverage=run["anchor_coverage"],
            codex_vdep=run["codex_vdep"],
            benchmark_summary=run["benchmark_summary"],
            ambiguity_summary=run["ambiguity_summary"],
            answer_agreement=run["answer_agreement"],
            ambiguity_agreement=run["ambiguity_agreement"],
        )
        for name, run in payload["runs"].items()
    ]
    overall.sort(key=lambda row: (-row["decision_score"], -row["codex_vdep"], row["codex_leaky"], row["run"]))
    payload["overall_ranking"] = overall

    winner = overall[0]
    winner_run = payload["runs"][winner["run"]]
    payload["recommendation_text"] = (
        f"Ship `{winner['run']}` for the next dataset cut. It has the best composite trade-off in this panel: "
        f"Codex vision-dependence `{winner['codex_vdep']:.4f}`, Codex text leakage `{winner['codex_leaky']:.4f}`, "
        f"type diversity `{winner['type_diversity']:.4f}`, and Codex frontier soft accuracy `{winner['codex_soft']:.4f}`. "
        f"Teacher=`{winner_run['teacher_provider']}:{winner_run['teacher_model']}`, prompt variant=`{winner_run['prompt_variant']}`. "
        f"Notable tuning knobs captured from the run summary: `{json.dumps(winner_run['notable_tuning'], sort_keys=True)}`."
    )

    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_json = panel_dir / "report.json"
    report_md = panel_dir / "report.md"
    write_json(report_json, payload)
    report_md.write_text(_render_report(payload), encoding="utf-8")
    print(str(report_md))


if __name__ == "__main__":
    main()
