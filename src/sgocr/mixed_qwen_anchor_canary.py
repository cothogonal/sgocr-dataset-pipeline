from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .mixed_ocr_frontend_canary import _q01_baseline, _run_variant
from .paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(description="Run a mixed150 Nemotron baseline with Qwen3-VL vLLM anchor grounding.")
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_qwen_anchor_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--baseline-nemo-name", default="sgocr_mixed_frontend_canary_20260410_110952_nemo")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    return ap.parse_args()


def render_report(payload: dict[str, Any]) -> str:
    result = payload["result"]
    lines = [
        "# Mixed Qwen Anchor Canary",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Source universe: `{payload['source_dir']}`",
        "- OCR frontend: `nemotron_v2`",
        "- Anchor candidate backend: `qwen3_vl_vllm`",
        '- Official grounding prompt style: `Locate every instance that belongs to the following categories: \"...\". Report bbox coordinates in JSON format like this: {"bbox_2d": [x1, y1, x2, y2], "label": "category"}.`',
        "",
        "## Result",
        "",
        "| Variant | Status | Accepted | Images w/ rows | Inline mean | Sweep |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if result.get("status") in {"ok", "cached"}:
        metrics = result["metrics"]
        lines.append(
            f"| `nemo_qwen_anchor` | ok | {metrics['accepted_qas']} | {metrics['images_with_final_rows']} | "
            f"{metrics['inline_frontier_mean']:.4f} | {metrics['sweep_score']:.4f} |"
        )
    else:
        note = (result.get("process") or {}).get("stderr_tail") or (result.get("process") or {}).get("stdout_tail") or ""
        lines.append(f"| `nemo_qwen_anchor` | {result.get('status','failed')} | - | - | - | - |")
        if note:
            lines.extend(["", f"Failure tail: `{note[:400].replace(chr(10), ' ')}`"])
    return "\n".join(lines) + "\n"


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
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_MODEL": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": "0.90",
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": "2",
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": "2048",
    }
    out_dir = root_final / f"{args.bundle_id}_nemo_qwen_anchor"
    intermediate_dir = root_intermediate / f"{args.bundle_id}_nemo_qwen_anchor"
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
    payload = {
        "bundle_id": args.bundle_id,
        "source_dir": str(source_dir),
        "source_manifest": source_manifest,
        "result": result,
    }
    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(render_report(payload), encoding="utf-8")
    print(str(report_md))


if __name__ == "__main__":
    main()
