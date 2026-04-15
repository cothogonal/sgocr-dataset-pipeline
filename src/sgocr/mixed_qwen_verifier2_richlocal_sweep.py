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
    ap = argparse.ArgumentParser(
        description="Run a mixed150 verifier2-era sweep focused on no-relabel constrained/open Qwen plus rich-local wording."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_qwen_v2_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-panel-json", default="sgocr_mixed_qwen_raw_20260411_130217_frontier_panel.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--qwen-batch-size", type=int, default=4)
    ap.add_argument("--qwen-gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--qwen-max-model-len", type=int, default=2048)
    return ap.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _render_report(payload: dict[str, Any]) -> str:
    perf = payload["performance_profile"]
    lines = [
        "# Mixed Qwen Verifier2 Rich-Local Sweep",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Source universe: `{payload['source_dir']}`",
        f"- Updated: `{payload['updated_at']}`",
        "",
        "## Goal",
        "",
        "- Keep verifier2-era no-relabel Qwen anchor runs.",
        "- Drop global-inventory exploration as the primary direction.",
        "- Test whether richer local wording improves frontier quality on the constrained/open Qwen paths that survived the rescued-set eval panel.",
        "",
        "## Performance Profile",
        "",
        f"- semantic workers: `{perf['workers']}`",
        f"- Qwen batch size: `{perf['qwen_batch_size']}`",
        f"- Qwen GPU memory utilization: `{perf['qwen_gpu_mem_util']}`",
        f"- Qwen max model len: `{perf['qwen_max_model_len']}`",
        "- relabel mode: `none` on every run",
        "- verifier policy: `verifier2` (current patched policy in codebase)",
        "",
        "## Variants",
        "",
        "- `qwen_local_constrained_selected_v2`: constrained local discovery + selected-tag grounding + no relabel",
        "- `qwen_local_constrained_selected_richlocal_v2`: same as above + `rich_local` wording + `frontier_rich` reverse-ground answers",
        "- `qwen_local_open_selected_v2`: open/raw local discovery + selected-tag grounding + no relabel",
        "- `qwen_local_open_selected_richlocal_v2`: same as above + `rich_local` wording + `frontier_rich` reverse-ground answers",
        "",
        "## Reference Panel",
        "",
        "```json",
        json.dumps(payload.get("reference_panel") or {}, indent=2, sort_keys=True),
        "```",
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
    for name in (
        "qwen_local_constrained_selected_v2",
        "qwen_local_constrained_selected_richlocal_v2",
        "qwen_local_open_selected_v2",
        "qwen_local_open_selected_richlocal_v2",
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
    report_md.write_text(_render_report(payload), encoding="utf-8")


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
        "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_MODEL": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": f"{float(args.qwen_gpu_mem_util):.2f}",
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": str(int(args.qwen_batch_size)),
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": str(int(args.qwen_max_model_len)),
        "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "selected_tags",
        "SGOCR_ANCHOR_RELABEL_MODE": "none",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    }
    variants = {
        "qwen_local_constrained_selected_v2": {
            "env": {
                **common_qwen_env,
                "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "constrained",
            },
            "note": "constrained local qwen + selected grounding + verifier2",
        },
        "qwen_local_constrained_selected_richlocal_v2": {
            "env": {
                **common_qwen_env,
                "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "constrained",
                "SGOCR_LOCATION_WORDING_MODE": "rich_local",
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier_rich",
            },
            "note": "constrained local qwen + rich_local wording",
        },
        "qwen_local_open_selected_v2": {
            "env": {
                **common_qwen_env,
                "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "open",
            },
            "note": "open local qwen + selected grounding + verifier2",
        },
        "qwen_local_open_selected_richlocal_v2": {
            "env": {
                **common_qwen_env,
                "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "open",
                "SGOCR_LOCATION_WORDING_MODE": "rich_local",
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier_rich",
            },
            "note": "open local qwen + rich_local wording",
        },
    }

    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    payload: dict[str, Any] = {
        "bundle_id": args.bundle_id,
        "source_dir": str(source_dir),
        "source_manifest": source_manifest,
        "reference_panel_json": str(root_final / args.reference_panel_json),
        "reference_panel": _load_json(root_final / args.reference_panel_json),
        "performance_profile": {
            "workers": int(args.workers),
            "qwen_batch_size": int(args.qwen_batch_size),
            "qwen_gpu_mem_util": float(args.qwen_gpu_mem_util),
            "qwen_max_model_len": int(args.qwen_max_model_len),
        },
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
