from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .bootstrap import write_json
from .dev200_eval import compute_frontier_agreement, run_frontier_benchmark


DEFAULT_MODELS = [
    "openai:gpt-5.3-codex",
    "gemini:gemini-3-flash-preview",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run frontier benchmark + agreement over old/rescued mixed150 accepted datasets."
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


def _collect_eval_metrics(*, summary: dict[str, Any], agreement: dict[str, Any]) -> dict[str, Any]:
    gpt = _safe_model_metric(summary, "gpt-5.3-codex")
    gemini = _safe_model_metric(summary, "gemini-3-flash-preview")
    pairwise = (agreement.get("pairwise") or [{}])[0]
    return {
        "rows": int(summary.get("rows") or 0),
        "gpt_soft": float(gpt.get("soft_accuracy") or 0.0),
        "gpt_exact": float(gpt.get("exact_accuracy") or 0.0),
        "gemini_soft": float(gemini.get("soft_accuracy") or 0.0),
        "gemini_exact": float(gemini.get("exact_accuracy") or 0.0),
        "agreement": float(pairwise.get("agreement_rate") or 0.0),
        "unanimous": float(agreement.get("overall_unanimous_rate") or 0.0),
    }


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Mixed150 Frontier Panel",
        "",
        f"- Source sweep: `{payload['bundle_id']}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Models: `{', '.join(payload['models'])}`",
        "",
        "## Results",
        "",
        "| Run | Variant | Rows | GPT soft | Gemini soft | Agreement | Unanimous |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run_name, run_payload in payload["results"].items():
        for variant in ("old", "rescued"):
            item = run_payload.get(variant) or {}
            if item.get("status") != "ok":
                lines.append(f"| `{run_name}` | `{variant}` | - | - | - | - | - |")
                continue
            m = item["metrics"]
            lines.append(
                f"| `{run_name}` | `{variant}` | {m['rows']} | {m['gpt_soft']:.4f} | {m['gemini_soft']:.4f} | {m['agreement']:.4f} | {m['unanimous']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| Run | Rescued minus old rows | GPT soft delta | Gemini soft delta | Agreement delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for run_name, run_payload in payload["results"].items():
        old = (run_payload.get("old") or {}).get("metrics") or {}
        rescued = (run_payload.get("rescued") or {}).get("metrics") or {}
        if not old or not rescued:
            lines.append(f"| `{run_name}` | - | - | - | - |")
            continue
        lines.append(
            f"| `{run_name}` | {int(rescued['rows']) - int(old['rows'])} | "
            f"{float(rescued['gpt_soft']) - float(old['gpt_soft']):+.4f} | "
            f"{float(rescued['gemini_soft']) - float(old['gemini_soft']):+.4f} | "
            f"{float(rescued['agreement']) - float(old['agreement']):+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report_json = Path(args.report_json)
    sweep = _load_json(report_json)
    bundle_id = str(sweep.get("bundle_id") or report_json.stem.removesuffix("_report"))
    models = list(args.models or DEFAULT_MODELS)

    selected_runs = set(str(x) for x in (args.runs or []) if str(x).strip())
    results: dict[str, Any] = {}
    for run_name, run_info in (sweep.get("results") or {}).items():
        if selected_runs and run_name not in selected_runs:
            continue
        if str(run_info.get("status") or "") not in {"ok", "cached"}:
            continue
        experiment_dir = Path(str(run_info["out_dir"]))
        results[run_name] = {
            "experiment_dir": str(experiment_dir),
            "old": {"status": "missing"},
            "rescued": {"status": "missing"},
        }
        for variant, dataset_name, eval_dir_name in (
            ("old", "accepted_dataset.jsonl", "frontier_benchmark_old"),
            ("rescued", "accepted_dataset_verifier2.jsonl", "frontier_benchmark_rescued"),
        ):
            dataset_path = experiment_dir / dataset_name
            if not dataset_path.exists():
                continue
            out_dir = experiment_dir / "evals" / eval_dir_name
            summary = run_frontier_benchmark(
                experiment_dir=experiment_dir,
                dataset_path=dataset_path,
                model_specs=models,
                limit=int(args.limit),
                out_dir=out_dir,
                workers=int(args.workers),
                write_index=False,
            )
            agreement = compute_frontier_agreement(benchmark_dir=out_dir)
            results[run_name][variant] = {
                "status": "ok",
                "dataset_path": str(dataset_path),
                "benchmark_dir": str(out_dir),
                "metrics": _collect_eval_metrics(summary=summary, agreement=agreement),
            }

    panel = {
        "bundle_id": bundle_id,
        "models": models,
        "source_report_json": str(report_json),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    out_json = report_json.with_name(f"{bundle_id}_frontier_panel.json")
    out_md = report_json.with_name(f"{bundle_id}_frontier_panel.md")
    write_json(out_json, panel)
    out_md.write_text(_render_report(panel), encoding="utf-8")
    print(str(out_md))


if __name__ == "__main__":
    main()
