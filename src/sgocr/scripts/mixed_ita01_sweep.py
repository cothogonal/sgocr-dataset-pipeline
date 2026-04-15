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
    ap = argparse.ArgumentParser(description="Run ita01: current frozen champ with independent-text-anchor Qwen inventory.")
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita01_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--qwen-batch-size", type=int, default=4)
    ap.add_argument("--qwen-gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--qwen-max-model-len", type=int, default=2048)
    ap.add_argument("--qwen-pass-count", type=int, default=1)
    ap.add_argument("--qwen-temperature", type=float, default=0.0)
    ap.add_argument("--qwen-consensus-iou", type=float, default=0.55)
    ap.add_argument("--qwen-min-support", type=int, default=1)
    return ap.parse_args()


def _render_report(payload: dict[str, Any]) -> str:
    perf = payload["performance_profile"]
    result = payload["results"].get("ita01_open_color_sibling_independent_anchor") or {}
    lines = [
        "# Mixed ITA01 Sweep",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Source universe: `{payload['source_dir']}`",
        f"- Updated: `{payload['updated_at']}`",
        "",
        "## Goal",
        "",
        "- Keep the frozen `open_selected_color_sibling_v0` champ settings.",
        "- Replace selected-tag anchor generation with one-shot independent full-image Qwen anchor inventory.",
        "- Join raw Qwen anchor boxes and labels onto Nemotron text nodes downstream in code.",
        "",
        "## Performance Profile",
        "",
        f"- semantic workers: `{perf['workers']}`",
        f"- Qwen batch size: `{perf['qwen_batch_size']}`",
        f"- Qwen GPU memory utilization: `{perf['qwen_gpu_mem_util']}`",
        f"- Qwen max model len: `{perf['qwen_max_model_len']}`",
        f"- Qwen inventory pass count: `{perf['qwen_pass_count']}`",
        f"- Qwen inventory temperature: `{perf['qwen_temperature']}`",
        f"- Qwen inventory consensus IoU: `{perf['qwen_consensus_iou']}`",
        f"- Qwen inventory min support: `{perf['qwen_min_support']}`",
        "- OCR frontend: `nemotron_v2`",
        "- anchor relabel: `none`",
        "- sibling disambiguation: `on`",
        "- anchor reference color: `on`",
        "",
        "## Variant",
        "",
        "- `ita01_open_color_sibling_independent_anchor`: frozen champ with independent full-image raw Qwen anchor inventory and code-side text-anchor join",
        "",
        "## Nemotron Diagnostic",
        "",
        "```json",
        json.dumps(payload["nemotron_diagnostic"], indent=2, sort_keys=True),
        "```",
        "",
        "## Result",
        "",
        "| Variant | Status | Accepted | Images w/ rows | Inline mean | Sweep | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    status = result.get("status", "missing")
    if status not in {"ok", "cached"}:
        note = result.get("note") or ""
        process = result.get("process") or {}
        if not note and process:
            note = (process.get("stderr_tail") or process.get("stdout_tail") or "").replace("\n", " ")[:180]
        lines.append("| `ita01_open_color_sibling_independent_anchor` | "
                     f"{status} | - | - | - | - | {note} |")
    else:
        metrics = result["metrics"]
        lines.append(
            "| `ita01_open_color_sibling_independent_anchor` | ok | "
            f"{metrics['accepted_qas']} | {metrics['images_with_final_rows']} | "
            f"{metrics['inline_frontier_mean']:.4f} | {metrics['sweep_score']:.4f} | |"
        )
    return "\n".join(lines) + "\n"


def _write_ita_reports(*, payload: dict[str, Any], report_json: Path, report_md: Path) -> None:
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
    env = {
        **baseline_env,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
        "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "florence",
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "independent_raw",
        "SGOCR_QWEN_ANCHOR_INVENTORY_PASS_COUNT": str(int(args.qwen_pass_count)),
        "SGOCR_QWEN_ANCHOR_INVENTORY_TEMPERATURE": str(float(args.qwen_temperature)),
        "SGOCR_QWEN_ANCHOR_INVENTORY_CONSENSUS_IOU": str(float(args.qwen_consensus_iou)),
        "SGOCR_QWEN_ANCHOR_INVENTORY_MIN_SUPPORT": str(int(args.qwen_min_support)),
        "SGOCR_ANCHOR_RELABEL_MODE": "none",
        "SGOCR_QWEN_ANCHOR_MODEL": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": f"{float(args.qwen_gpu_mem_util):.2f}",
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": str(int(args.qwen_batch_size)),
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": str(int(args.qwen_max_model_len)),
        "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
        "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
        "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    }

    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    payload: dict[str, Any] = {
        "bundle_id": args.bundle_id,
        "source_dir": str(source_dir),
        "source_manifest": source_manifest,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "performance_profile": {
            "workers": int(args.workers),
            "qwen_batch_size": int(args.qwen_batch_size),
            "qwen_gpu_mem_util": float(args.qwen_gpu_mem_util),
            "qwen_max_model_len": int(args.qwen_max_model_len),
            "qwen_pass_count": int(args.qwen_pass_count),
            "qwen_temperature": float(args.qwen_temperature),
            "qwen_consensus_iou": float(args.qwen_consensus_iou),
            "qwen_min_support": int(args.qwen_min_support),
        },
        "nemotron_diagnostic": nemotron_frontend_diagnostic(),
        "results": {
            "ita01_open_color_sibling_independent_anchor": {
                "status": "pending",
                "note": "frozen champ with independent raw qwen anchor inventory",
            }
        },
    }
    _write_ita_reports(payload=payload, report_json=report_json, report_md=report_md)

    name = "ita01_open_color_sibling_independent_anchor"
    payload["results"][name] = {"status": "running", "note": "frozen champ with independent raw qwen anchor inventory"}
    _write_ita_reports(payload=payload, report_json=report_json, report_md=report_md)
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
        env_overrides=env,
        cli_overrides=dict(baseline_cli),
    )
    result["out_dir"] = str(out_dir)
    result["intermediate_dir"] = str(intermediate_dir)
    result.setdefault("note", "frozen champ with independent raw qwen anchor inventory")
    payload["results"][name] = result
    _write_ita_reports(payload=payload, report_json=report_json, report_md=report_md)
    print(str(report_md))


if __name__ == "__main__":
    main()
