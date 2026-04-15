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
        description="Run a mixed150 finalv0 sweep centered on the open-selected Qwen frontier."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_finalv0_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-panel-json", default="sgocr_mixed_qwen_raw_20260411_130217_frontier_panel.json")
    ap.add_argument("--reference-sweep-json", default="sgocr_mixed_qwen_v2_20260411_153004_report.json")
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
        "# Mixed FinalV0 Sweep",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Source universe: `{payload['source_dir']}`",
        f"- Updated: `{payload['updated_at']}`",
        "",
        "## Goal",
        "",
        "- Treat `open-selected` as the sole frontier.",
        "- Add better sibling-aware disambiguation, color-bearing anchor references, less repetitive localization, and a 3-answer ambiguity probe.",
        "- Measure which subset of those changes actually improves the open-selected path on the shared mix150 universe.",
        "",
        "## Performance Profile",
        "",
        f"- semantic workers: `{perf['workers']}`",
        f"- Qwen batch size: `{perf['qwen_batch_size']}`",
        f"- Qwen GPU memory utilization: `{perf['qwen_gpu_mem_util']}`",
        f"- Qwen max model len: `{perf['qwen_max_model_len']}`",
        "- OCR frontend: `nemotron_v2` on every run",
        "- anchor candidate backend: `qwen3_vl_vllm` on every run",
        "- relabel mode: `none` on every run",
        "- verifier: `verifier2` policy currently in codebase",
        "",
        "## Variants",
        "",
        "- `open_selected_control_v0`: current open-selected frontier control",
        "- `open_selected_color_sibling_v0`: add color-specific open-Qwen naming plus sibling-aware disambiguation",
        "- `open_selected_color_sibling_localize_v0`: above plus `finalv0` localization and `frontier_rich` reverse-ground phrasing",
        "- `open_selected_color_sibling_localize_probe_v0`: above plus 3-answer ambiguity probe",
        "",
        "## Reference Panel",
        "",
        "```json",
        json.dumps(payload.get("reference_panel") or {}, indent=2, sort_keys=True),
        "```",
        "",
        "## Reference Sweep",
        "",
        "```json",
        json.dumps(payload.get("reference_sweep") or {}, indent=2, sort_keys=True),
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
        "open_selected_control_v0",
        "open_selected_color_sibling_v0",
        "open_selected_color_sibling_localize_v0",
        "open_selected_color_sibling_localize_probe_v0",
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
    common_open_env = {
        **baseline_env,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
        "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE": "open",
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "selected_tags",
        "SGOCR_ANCHOR_RELABEL_MODE": "none",
        "SGOCR_QWEN_ANCHOR_MODEL": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": f"{float(args.qwen_gpu_mem_util):.2f}",
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": str(int(args.qwen_batch_size)),
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": str(int(args.qwen_max_model_len)),
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    }
    variants = {
        "open_selected_control_v0": {
            "env": {
                **common_open_env,
            },
            "note": "current open-selected frontier control",
        },
        "open_selected_color_sibling_v0": {
            "env": {
                **common_open_env,
                "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
                "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
                "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
            },
            "note": "color-specific local open qwen + sibling-aware disambiguation",
        },
        "open_selected_color_sibling_localize_v0": {
            "env": {
                **common_open_env,
                "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
                "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
                "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
                "SGOCR_LOCATION_WORDING_MODE": "finalv0",
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier_rich",
            },
            "note": "add finalv0 localization and richer reverse-ground phrasing",
        },
        "open_selected_color_sibling_localize_probe_v0": {
            "env": {
                **common_open_env,
                "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
                "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
                "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
                "SGOCR_LOCATION_WORDING_MODE": "finalv0",
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier_rich",
                "SGOCR_TEACHER_ANSWER_PROBE_COUNT": "3",
                "SGOCR_TEACHER_ANSWER_PROBE_TEMPERATURE": "0.35",
            },
            "note": "full finalv0 stack plus 3-answer ambiguity probe",
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
        "reference_sweep_json": str(root_final / args.reference_sweep_json),
        "reference_sweep": _load_json(root_final / args.reference_sweep_json),
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
