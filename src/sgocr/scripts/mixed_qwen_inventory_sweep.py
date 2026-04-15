from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .mixed_ocr_frontend_canary import _q01_baseline, _run_variant
from ..nemotron_frontend import nemotron_frontend_diagnostic
from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description="Run an overnight mixed150 sweep covering Qwen local discovery and constrained one-shot inventory modes."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_qwen_inventory_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--baseline-nemo-name", default="sgocr_mixed_frontend_canary_20260410_110952_nemo")
    ap.add_argument("--baseline-qwen-anchor-name", default="sgocr_mixed_qwen_anchor_20260410_220228_nemo_qwen_anchor")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def _load_summary(root_final: Path, experiment_name: str) -> dict[str, Any] | None:
    summary_path = root_final / experiment_name / "summary.json"
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
        "# Mixed Qwen Inventory Sweep",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Source universe: `{payload['source_dir']}`",
        f"- Updated: `{payload['updated_at']}`",
        "",
        "## Goal",
        "",
        "- Compare the current Nemotron + Qwen selected-tag control against the new Qwen local-discovery and one-shot constrained full-image inventory paths.",
        "- Keep the downstream semantic pipeline fixed so the comparison stays about anchor discovery and grounding.",
        "",
        "## Variants",
        "",
        "- `control_florence_selected`: Florence local tag discovery + Qwen selected-tag grounding",
        "- `qwen_local_constrained_selected`: Qwen local crop discovery in constrained vocab mode + Qwen selected-tag grounding",
        "- `qwen_local_open_selected`: Qwen local crop discovery in open/raw mode + Qwen selected-tag grounding",
        "- `qwen_local_constrained_global_inventory`: Qwen local crop discovery in constrained vocab mode + Qwen one-shot constrained full-image inventory grounding",
        "",
        "## Reference Baselines",
        "",
        "| Reference | Accepted | Images w/ rows | Inline mean | Sweep |",
        "|---|---:|---:|---:|---:|",
    ]
    baseline_nemo = payload.get("baseline_nemo_metrics")
    if baseline_nemo:
        lines.append(
            f"| `baseline_nemo` | {baseline_nemo['accepted']} | {baseline_nemo['images_with_rows']} | {baseline_nemo['inline_mean']:.4f} | {baseline_nemo['sweep']:.4f} |"
        )
    baseline_qwen = payload.get("baseline_qwen_anchor_metrics")
    if baseline_qwen:
        lines.append(
            f"| `baseline_qwen_anchor` | {baseline_qwen['accepted']} | {baseline_qwen['images_with_rows']} | {baseline_qwen['inline_mean']:.4f} | {baseline_qwen['sweep']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Nemotron Diagnostic",
            "",
            "```json",
            json.dumps(payload["nemotron_diagnostic"], indent=2, sort_keys=True),
            "```",
            "",
            "## Results",
            "",
            "| Variant | Status | Accepted | Images w/ rows | Inline mean | Sweep | Notes |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name in (
        "control_florence_selected",
        "qwen_local_constrained_selected",
        "qwen_local_open_selected",
        "qwen_local_constrained_global_inventory",
    ):
        result = payload["results"].get(name) or {}
        status = result.get("status", "missing")
        if status not in {"ok", "cached"}:
            note = result.get("note") or ""
            process = result.get("process") or {}
            if not note and process:
                note = (process.get("stderr_tail") or process.get("stdout_tail") or "").replace("\n", " ")[:180]
            lines.append(f"| `{name}` | {status} | - | - | - | - | {note} |")
            continue
        metrics = result["metrics"]
        lines.append(
            f"| `{name}` | ok | {metrics['accepted_qas']} | {metrics['images_with_final_rows']} | "
            f"{metrics['inline_frontier_mean']:.4f} | {metrics['sweep_score']:.4f} | |"
        )
    return "\n".join(lines) + "\n"


def _write_reports(*, payload: dict[str, Any], report_json: Path, report_md: Path) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(render_report(payload), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150"
    source_dir = root_final / args.source_name
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))

    baseline_env, baseline_cli = _q01_baseline()
    common_qwen_env = {
        **baseline_env,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_MODEL": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": "0.90",
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": "2",
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": "2048",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    }
    variants = {
        "control_florence_selected": {
            "env": {
                **common_qwen_env,
                "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "selected_tags",
            },
            "note": "current Qwen-grounded control",
        },
        "qwen_local_constrained_selected": {
            "env": {
                **common_qwen_env,
                "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
                "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "constrained",
                "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "selected_tags",
            },
            "note": "new constrained local-Qwen discovery path",
        },
        "qwen_local_open_selected": {
            "env": {
                **common_qwen_env,
                "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
                "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "open",
                "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "selected_tags",
            },
            "note": "new open local-Qwen discovery path",
        },
        "qwen_local_constrained_global_inventory": {
            "env": {
                **common_qwen_env,
                "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
                "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "constrained",
                "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "global_inventory",
            },
            "note": "new one-shot constrained full-image inventory + join path",
        },
    }

    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    payload: dict[str, Any] = {
        "bundle_id": args.bundle_id,
        "source_dir": str(source_dir),
        "source_manifest": source_manifest,
        "baseline_nemo_name": str(args.baseline_nemo_name),
        "baseline_nemo_metrics": _baseline_metrics(_load_summary(root_final, str(args.baseline_nemo_name))),
        "baseline_qwen_anchor_name": str(args.baseline_qwen_anchor_name),
        "baseline_qwen_anchor_metrics": _baseline_metrics(_load_summary(root_final, str(args.baseline_qwen_anchor_name))),
        "nemotron_diagnostic": nemotron_frontend_diagnostic(),
        "results": {name: {"status": "pending", "note": spec["note"]} for name, spec in variants.items()},
    }
    _write_reports(payload=payload, report_json=report_json, report_md=report_md)

    for name, spec in variants.items():
        payload["results"][name] = {"status": "running", "note": spec["note"]}
        _write_reports(payload=payload, report_json=report_json, report_md=report_md)
        out_dir = root_final / f"{args.bundle_id}_{name}"
        intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}"
        result = _run_variant(
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
        result["out_dir"] = str(out_dir)
        result["intermediate_dir"] = str(intermediate_dir)
        result.setdefault("note", spec["note"])
        payload["results"][name] = result
        _write_reports(payload=payload, report_json=report_json, report_md=report_md)

    print(str(report_md))


if __name__ == "__main__":
    main()
