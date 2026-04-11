from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .mixed_ocr_frontend_canary import _q01_baseline, _run_variant
from .nemotron_frontend import nemotron_frontend_diagnostic
from .paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(description="Run a focused Nemotron wording/relabel sweep on an existing mixed OCR source universe.")
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_nemo_refine_{stamp}")
    ap.add_argument(
        "--source-name",
        default="chartqa50_textocr50_cocotext50_source_20260410_110952",
    )
    ap.add_argument(
        "--baseline-nemo-name",
        default="sgocr_mixed_frontend_canary_20260410_110952_nemo",
    )
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def _load_baseline_summary(root_final: Path, baseline_name: str) -> dict[str, Any] | None:
    summary_path = root_final / baseline_name / "summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _baseline_metrics(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not summary:
        return None
    inline = summary.get("inline_frontier") or {}
    return {
        "accepted": int(summary.get("accepted_qas") or 0),
        "images_with_rows": int(summary.get("images_with_final_rows") or 0),
        "inline_mean": float(inline.get("mean_inline_frontier_correct") or 0.0),
        "sweep": float(inline.get("precision_first_score") or 0.0),
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Mixed Nemotron Refine Sweep",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Source universe: `{payload['source_dir']}`",
        f"- Baseline Nemo: `{payload['baseline_nemo_name']}`",
        "",
        "## Variants",
        "",
        "- `nemo_rich_local`: richer local spatial wording in questions and richer reverse-ground answers",
        "- `nemo_rich_local_antigeneric`: richer wording plus anti-generic relabeling",
        "- `nemo_antigeneric_only`: anti-generic relabeling only",
        "",
        "## Nemotron Diagnostic",
        "",
        "```json",
        json.dumps(payload["nemotron_diagnostic"], indent=2, sort_keys=True),
        "```",
        "",
        "## Results",
        "",
        "| Variant | Status | Accepted | Images w/ rows | Inline mean | Sweep |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    baseline = payload.get("baseline_metrics")
    if baseline:
        lines.append(
            f"| `baseline_nemo` | ref | {baseline['accepted']} | {baseline['images_with_rows']} | {baseline['inline_mean']:.4f} | {baseline['sweep']:.4f} |"
        )
    for name in ("nemo_rich_local", "nemo_rich_local_antigeneric", "nemo_antigeneric_only"):
        result = payload["results"].get(name) or {}
        if result.get("status") not in {"ok", "cached"}:
            note = (result.get("process") or {}).get("stderr_tail") or (result.get("process") or {}).get("stdout_tail") or ""
            lines.append(f"| `{name}` | {result.get('status', 'missing')} | - | - | - | - |")
            if note:
                lines.append("")
                lines.append(f"`{name}` failure tail: `{note[:240].replace(chr(10), ' ')}`")
            continue
        metrics = result["metrics"]
        lines.append(
            f"| `{name}` | ok | {metrics['accepted_qas']} | {metrics['images_with_final_rows']} | {metrics['inline_frontier_mean']:.4f} | {metrics['sweep_score']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150"
    source_dir = root_final / args.source_name
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))

    baseline_env, baseline_cli = _q01_baseline()
    common_nemo_env = {
        **baseline_env,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
    }
    variants = {
        "nemo_rich_local": {
            "env": {
                **common_nemo_env,
                "SGOCR_LOCATION_WORDING_MODE": "rich_local",
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier_rich",
            },
        },
        "nemo_rich_local_antigeneric": {
            "env": {
                **common_nemo_env,
                "SGOCR_LOCATION_WORDING_MODE": "rich_local",
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier_rich",
                "SGOCR_ANCHOR_RELABEL_GENERIC_MODE": "anti_generic",
            },
        },
        "nemo_antigeneric_only": {
            "env": {
                **common_nemo_env,
                "SGOCR_ANCHOR_RELABEL_GENERIC_MODE": "anti_generic",
            },
        },
    }

    results: dict[str, Any] = {}
    for name, spec in variants.items():
        out_dir = root_final / f"{args.bundle_id}_{name}"
        intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}"
        results[name] = _run_variant(
            source_dir=source_dir,
            out_dir=out_dir,
            intermediate_dir=intermediate_dir,
            cache_intermediate_dir=None,
            cache_level="none",
            model=str(args.model),
            workers=int(args.workers),
            max_side=int(args.max_side),
            device=str(args.device),
            env_overrides=spec["env"],
            cli_overrides=dict(baseline_cli),
        )
        results[name]["out_dir"] = str(out_dir)
        results[name]["intermediate_dir"] = str(intermediate_dir)

    baseline_summary = _load_baseline_summary(root_final, str(args.baseline_nemo_name))
    payload = {
        "bundle_id": args.bundle_id,
        "source_dir": str(source_dir),
        "source_manifest": source_manifest,
        "baseline_nemo_name": str(args.baseline_nemo_name),
        "baseline_metrics": _baseline_metrics(baseline_summary),
        "results": results,
        "nemotron_diagnostic": nemotron_frontend_diagnostic(),
    }
    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(render_report(payload), encoding="utf-8")
    print(str(report_md))


if __name__ == "__main__":
    main()
