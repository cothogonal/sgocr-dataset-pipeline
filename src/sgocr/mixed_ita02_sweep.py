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
    ap = argparse.ArgumentParser(description="Run ita02: independent-anchor mix150 sweep with scene-aware and verifier updates.")
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita02_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita01_20260412_081132_report.json")
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
    ap.add_argument("--group-min-instances", type=int, default=3)
    ap.add_argument("--cheap-proxy-reject-score", type=int, default=6)
    return ap.parse_args()


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _render_report(payload: dict[str, Any]) -> str:
    perf = payload["performance_profile"]
    lines = [
        "# Mixed ITA02 Sweep",
        "",
        f"- Bundle: `{payload['bundle_id']}`",
        f"- Source universe: `{payload['source_dir']}`",
        f"- Updated: `{payload['updated_at']}`",
        "",
        "## Goal",
        "",
        "- Keep the independent-anchor architecture from `ita01`.",
        "- Reduce over-specific single-instance reads in repeated same-anchor scenery.",
        "- Suppress anchor-local spatial wording unless there is competing text on the same anchor.",
        "- Add a cheap verifier-side ambiguity proxy instead of relying on expensive frontier ambiguity agreement.",
        "",
        "## Performance Profile",
        "",
        f"- semantic workers: `{perf['workers']}`",
        f"- Qwen batch size: `{perf['qwen_batch_size']}`",
        f"- Qwen GPU memory utilization: `{perf['qwen_gpu_mem_util']}`",
        f"- Qwen max model len: `{perf['qwen_max_model_len']}`",
        f"- Qwen inventory pass count: `{perf['qwen_pass_count']}`",
        f"- repeated-anchor group min instances: `{perf['group_min_instances']}`",
        f"- cheap ambiguity proxy reject score: `{perf['cheap_proxy_reject_score']}`",
        "- OCR frontend: `nemotron_v2`",
        "- anchor candidate backend: `qwen3_vl_vllm`",
        "- anchor inventory mode: `independent_raw`",
        "- anchor relabel: `none`",
        "",
        "## Reference ITA01",
        "",
        "```json",
        json.dumps(payload.get("reference_report") or {}, indent=2, sort_keys=True),
        "```",
        "",
        "## Nemotron Diagnostic",
        "",
        "```json",
        json.dumps(payload["nemotron_diagnostic"], indent=2, sort_keys=True),
        "```",
        "",
        "## Variants",
        "",
        "- `ita02_control`: current independent-anchor control",
        "- `ita02_localminimal`: suppress anchor-local phrases unless the anchor has competing text",
        "- `ita02_sceneaware`: above plus repeated-anchor grouped reads",
        "- `ita02_sceneaware_proxy`: above plus cheap verifier-side ambiguity proxy",
        "",
        "## Results",
        "",
        "| Variant | Status | Accepted | Images w/ rows | Inline mean | Sweep | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("ita02_control", "ita02_localminimal", "ita02_sceneaware", "ita02_sceneaware_proxy"):
        result = payload["results"].get(name) or {}
        status = result.get("status", "missing")
        if status not in {"ok", "cached"}:
            note = result.get("note") or ""
            process = result.get("process") or {}
            if not note and process:
                note = (process.get("stderr_tail") or process.get("stdout_tail") or "").replace("\n", " ")[:200]
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
    common_env = {
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
    variants = {
        "ita02_control": {
            "env": {
                **common_env,
            },
            "note": "independent-anchor control",
        },
        "ita02_localminimal": {
            "env": {
                **common_env,
                "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
            },
            "note": "suppress anchor-local phrasing unless same-anchor competing text exists",
        },
        "ita02_sceneaware": {
            "env": {
                **common_env,
                "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
                "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
                "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
            },
            "note": "scene-aware grouped reads plus local-anchor suppression",
        },
        "ita02_sceneaware_proxy": {
            "env": {
                **common_env,
                "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
                "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
                "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
                "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "1",
                "SGOCR_CHEAP_AMBIGUITY_PROXY_REJECT_SCORE": str(int(args.cheap_proxy_reject_score)),
            },
            "note": "scene-aware grouped reads plus cheap ambiguity proxy",
        },
    }

    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    payload: dict[str, Any] = {
        "bundle_id": args.bundle_id,
        "source_dir": str(source_dir),
        "source_manifest": source_manifest,
        "reference_report_json": str(root_final / args.reference_report_json),
        "reference_report": _load_json(root_final / args.reference_report_json),
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
            "group_min_instances": int(args.group_min_instances),
            "cheap_proxy_reject_score": int(args.cheap_proxy_reject_score),
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
