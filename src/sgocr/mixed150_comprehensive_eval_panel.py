from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .bootstrap import write_json
from .dev200_eval import (
    compute_frontier_agreement,
    compute_frontier_ambiguity_agreement,
    run_frontier_ambiguity_eval,
    run_frontier_benchmark,
)


DEFAULT_MODELS = [
    "openai:gpt-5.3-codex",
    "gemini:gemini-3-flash-preview",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run answer benchmark + agreement + ambiguity eval panel over mixed150 accepted datasets."
    )
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--run", action="append", dest="runs", default=[])
    ap.add_argument("--model", action="append", dest="models", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    return ap.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_model_metric(summary: dict[str, Any], requested_model: str) -> dict[str, Any]:
    for item in summary.get("models") or []:
        if str(item.get("requested_model") or item.get("model") or "") == requested_model:
            return dict(item)
    return {}


def _collect_answer_metrics(*, summary: dict[str, Any], agreement: dict[str, Any]) -> dict[str, Any]:
    gpt = _safe_model_metric(summary, "gpt-5.3-codex")
    gemini = _safe_model_metric(summary, "gemini-3-flash-preview")
    pairwise = (agreement.get("pairwise") or [{}])[0]
    return {
        "rows": int(summary.get("rows") or 0),
        "gpt_soft": float(gpt.get("soft_accuracy") or 0.0),
        "gpt_exact": float(gpt.get("exact_accuracy") or 0.0),
        "gemini_soft": float(gemini.get("soft_accuracy") or 0.0),
        "gemini_exact": float(gemini.get("exact_accuracy") or 0.0),
        "answer_agreement": float(pairwise.get("agreement_rate") or 0.0),
        "answer_unanimous": float(agreement.get("overall_unanimous_rate") or 0.0),
    }


def _collect_ambiguity_metrics(*, summary: dict[str, Any], agreement: dict[str, Any]) -> dict[str, Any]:
    gpt = _safe_model_metric(summary, "gpt-5.3-codex")
    gemini = _safe_model_metric(summary, "gemini-3-flash-preview")
    pairwise = (agreement.get("pairwise") or [{}])[0]
    return {
        "rows": int(summary.get("rows") or 0),
        "gpt_ambiguous_rate": float(gpt.get("ambiguous_rate") or 0.0),
        "gpt_ambiguity_conf": float(gpt.get("mean_confidence") or 0.0),
        "gemini_ambiguous_rate": float(gemini.get("ambiguous_rate") or 0.0),
        "gemini_ambiguity_conf": float(gemini.get("mean_confidence") or 0.0),
        "ambiguity_agreement": float(pairwise.get("agreement_rate") or 0.0),
        "ambiguity_unanimous": float(agreement.get("overall_unanimous_rate") or 0.0),
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Mixed150 Comprehensive Frontier Panel",
        "",
        f"- Source sweep: `{payload['bundle_id']}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Models: `{', '.join(payload['models'])}`",
        "",
        "## Results",
        "",
        "| Run | Status | Rows | GPT soft | Gemini soft | Ans agree | GPT ambig | Gemini ambig | Ambig agree |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, run_payload in payload["results"].items():
        if str(run_payload.get("status") or "") != "ok":
            lines.append(f"| `{run_name}` | `{run_payload.get('status', 'pending')}` | - | - | - | - | - | - | - |")
            continue
        answer = run_payload.get("answer_metrics") or {}
        amb = run_payload.get("ambiguity_metrics") or {}
        lines.append(
            f"| `{run_name}` | `ok` | {answer.get('rows', 0)} | "
            f"{answer.get('gpt_soft', 0.0):.4f} | {answer.get('gemini_soft', 0.0):.4f} | {answer.get('answer_agreement', 0.0):.4f} | "
            f"{amb.get('gpt_ambiguous_rate', 0.0):.4f} | {amb.get('gemini_ambiguous_rate', 0.0):.4f} | {amb.get('ambiguity_agreement', 0.0):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `GPT/Gemini soft` are answer soft-accuracy metrics from the standard frontier benchmark.",
            "- `Ans agree` is exact agreement between frontier model answers.",
            "- `GPT/Gemini ambig` are the rates at which each frontier model marked the tuple ambiguous.",
            "- `Ambig agree` is exact agreement between frontier model ambiguity judgments.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_panel(report_json: Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    out_json = report_json.with_name(f"{payload['bundle_id']}_comprehensive_eval_panel.json")
    out_md = report_json.with_name(f"{payload['bundle_id']}_comprehensive_eval_panel.md")
    write_json(out_json, payload)
    out_md.write_text(_render_report(payload), encoding="utf-8")
    return out_json, out_md


def main() -> None:
    args = parse_args()
    report_json = Path(args.report_json)
    sweep = _load_json(report_json)
    bundle_id = str(sweep.get("bundle_id") or report_json.stem.removesuffix("_report"))
    models = list(args.models or DEFAULT_MODELS)
    selected_runs = {str(x) for x in (args.runs or []) if str(x).strip()}

    results: dict[str, Any] = {}
    for run_name, run_info in (sweep.get("results") or {}).items():
        if selected_runs and run_name not in selected_runs:
            continue
        if str(run_info.get("status") or "") not in {"ok", "cached"}:
            continue
        results[run_name] = {
            "status": "pending",
            "experiment_dir": str(run_info["out_dir"]),
        }

    panel = {
        "bundle_id": bundle_id,
        "models": models,
        "source_report_json": str(report_json),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    _, out_md = _write_panel(report_json, panel)

    for run_name, run_payload in results.items():
        experiment_dir = Path(run_payload["experiment_dir"])
        dataset_path = experiment_dir / "accepted_dataset.jsonl"
        results[run_name]["status"] = "answer_benchmark"
        panel["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_panel(report_json, panel)
        answer_dir = experiment_dir / "evals" / "frontier_benchmark_finalv0"
        answer_summary = run_frontier_benchmark(
            experiment_dir=experiment_dir,
            dataset_path=dataset_path,
            model_specs=models,
            limit=int(args.limit),
            out_dir=answer_dir,
            workers=int(args.workers),
            write_index=False,
        )

        results[run_name]["status"] = "answer_agreement"
        panel["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_panel(report_json, panel)
        answer_agreement = compute_frontier_agreement(benchmark_dir=answer_dir)

        results[run_name]["status"] = "ambiguity_eval"
        panel["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_panel(report_json, panel)
        ambiguity_dir = experiment_dir / "evals" / "frontier_ambiguity_finalv0"
        ambiguity_summary = run_frontier_ambiguity_eval(
            experiment_dir=experiment_dir,
            dataset_path=dataset_path,
            model_specs=models,
            limit=int(args.limit),
            out_dir=ambiguity_dir,
            workers=int(args.workers),
        )

        results[run_name]["status"] = "ambiguity_agreement"
        panel["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_panel(report_json, panel)
        ambiguity_agreement = compute_frontier_ambiguity_agreement(benchmark_dir=ambiguity_dir)

        results[run_name] = {
            "status": "ok",
            "experiment_dir": str(experiment_dir),
            "dataset_path": str(dataset_path),
            "answer_dir": str(answer_dir),
            "ambiguity_dir": str(ambiguity_dir),
            "answer_metrics": _collect_answer_metrics(summary=answer_summary, agreement=answer_agreement),
            "ambiguity_metrics": _collect_ambiguity_metrics(summary=ambiguity_summary, agreement=ambiguity_agreement),
        }
        panel["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_panel(report_json, panel)

    print(str(out_md))


if __name__ == "__main__":
    main()
