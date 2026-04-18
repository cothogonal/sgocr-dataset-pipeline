from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .dev200_eval import run_image_dependence_eval
from .mixed_ocr_frontend_canary import _build_variant_cmd, _q01_baseline
from ..nemotron_frontend import nemotron_frontend_diagnostic
from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from ..run_quality import compute_anchor_coverage, compute_answer_distribution, compute_run_quality

# ita20 goal: push vision_necessary_rate above 40% by eliminating residual leaky question types.
#
# Analysis of ita19 results — key findings from image-dependence by type:
#
#   g_t8_no_yn (ita19 winner — best text_leaky at 16.6%):
#     DIRECT_READ  (n=140): vision_nec=40.0%, text_leaky=0.71%   ← excellent, our champion
#     REVERSE_GROUND(n=14): vision_nec=7.1%,  text_leaky=57.1%   ← terrible, low count
#     TEXT_PROPERTY (n=69): vision_nec=31.9%, text_leaky=40.6%   ← drags average down
#     Overall:              vision_nec=35.4%, text_leaky=16.6%
#
#   TEXT_PROPERTY breaks down into two sub-kinds:
#     Mechanical (word_count/first_word/last_word): answer from OCR text alone — always leaky
#     Visual (text_color/text_curvature): requires actual image inspection — genuinely visual
#   The high TP text_leaky rate (40.6%) is dominated by mechanical TP questions.
#
#   REVERSE_GROUND contributes only 14 QAs at 7.1% vision_nec — a pure drag.
#   In prior sweeps, the rg_per_image_hard_cap already capped at min(1,rg_cap);
#   a bug prevented setting it to 0. Fixed in ita20.
#
# Engineering changes (ita20):
#   1. Fix: rg_per_image_hard_cap=0 now actually eliminates REVERSE_GROUND
#      (prior code had `max(1, rg_cap)` preventing elimination).
#   2. New flag: SGOCR_TP_VISUAL_ONLY_ENABLED=1 — restricts TEXT_PROPERTY to
#      text_color and text_curvature only; word_count/first_word/last_word are
#      pre-filtered from the candidate pool so DIRECT_READ fills those slots instead.
#
# Expected impact (based on ita19 no_yn data):
#   Removing RG (14 QAs @ 7.1%): 35.4% → ~37% vision_nec if DRfills gap
#   Restricting TP to visual-only: fewer but higher-quality TP; more DR in mix
#   Combined: estimated 38-42% vision_nec
#
# Variants:
#   g_t8_no_yn_cement       — ita19 no_yn carryforward (cement: text_leaky=16.6%)
#   g_t8_no_yn_no_rg        — no_yn + rg_cap=0 (push: eliminate RG's 7.1% vision_nec drag)
#   g_t8_no_yn_tp_visual    — no_yn + tp_visual_only (push: TP restricted to color/curvature)
#   g_t8_no_yn_no_rg_tp_vis — no_yn + rg=0 + tp_visual (push: most aggressive pure DR+visual)
#
# All variants use Gemma4 Ollama backend (antidoc prompt, flash-lite teacher).
# All cache at cache_level="verified" from ita19 no_yn (rg cap + tp filter run at selection
# time, after the cached verified-tuples stage, so no anchor stage re-run needed).
#
# SERVICE NOTE:
#   All variants require Ollama running (Gemma4 backend). vLLM must be stopped.
#   Start Ollama before running: ollama serve &; sleep 5
#   curl -s http://localhost:11434/api/tags | head -c 200  # verify up

VARIANT_ORDER = [
    "g_t8_no_yn_cement",
    "g_t8_no_yn_no_rg",
    "g_t8_no_yn_tp_visual",
    "g_t8_no_yn_no_rg_tp_vis",
]

VARIANT_DESCRIPTIONS: dict[str, str] = {
    "g_t8_no_yn_cement": (
        "Gemma4 antidoc + flash-lite + no_yn + target=8 "
        "(ita19 no_yn carryforward — cement: text_leaky=16.6%, vision_nec=35.4%)"
    ),
    "g_t8_no_yn_no_rg": (
        "Gemma4 antidoc + flash-lite + no_yn + rg_cap=0 "
        "(push: eliminate REVERSE_GROUND which drags at 7.1% vision_nec, 57.1% text_leaky)"
    ),
    "g_t8_no_yn_tp_visual": (
        "Gemma4 antidoc + flash-lite + no_yn + tp_visual_only "
        "(push: TEXT_PROPERTY restricted to text_color/text_curvature; "
        "word_count/first_word/last_word excluded so DR fills those slots instead)"
    ),
    "g_t8_no_yn_no_rg_tp_vis": (
        "Gemma4 antidoc + flash-lite + no_yn + rg_cap=0 + tp_visual_only "
        "(push: most aggressive — pure DR + visual-TP dataset; target 40%+ vision_nec)"
    ),
}

# OCR cache: reuse nemotron_v2 text_detections from ita15_t8 (unchanged across all sweeps)
_ITA15_T8_INTERMEDIATE = (
    str(OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150" /
        "sgocr_mixed_ita15_20260415_204008_ita15_t8")
)

# ita19 no_yn winner cache: all ita20 variants share the same anchor+grounding config
# (Gemma4 antidoc, same flags) so we can reuse at cache_level="verified".
# rg_cap=0 and tp_visual_only both apply at candidate-selection time (post-verified-tuples),
# so they run fresh on top of the cached tuples — no anchor re-run needed.
_ITA19_NO_YN_INTERMEDIATE = (
    OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150" /
    "sgocr_mixed_ita19_20260417_011023_g_t8_no_yn"
)

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "174_sgocr_mixed_ita20_launch_2026-04-17.md"


def _run_variant_streaming(
    *,
    source_dir: Path,
    out_dir: Path,
    intermediate_dir: Path,
    cache_intermediate_dir: Path | None,
    cache_level: str,
    model: str,
    workers: int,
    max_side: int,
    device: str,
    env_overrides: dict[str, str],
    cli_overrides: dict[str, str],
) -> dict[str, Any]:
    """Run a pipeline variant streaming stdout+stderr live while capturing tails."""
    if (out_dir / "summary.json").exists() and (out_dir / "ocr_qa_dataset.jsonl").exists():
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in (out_dir / "ocr_qa_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        metrics = compute_run_quality(summary, rows)
        return {"status": "cached", "metrics": metrics, "summary": summary}

    cmd, env = _build_variant_cmd(
        source_dir=source_dir,
        out_dir=out_dir,
        intermediate_dir=intermediate_dir,
        cache_intermediate_dir=cache_intermediate_dir,
        cache_level=cache_level,
        model=model,
        workers=workers,
        max_side=max_side,
        device=device,
        env_overrides=env_overrides,
        cli_overrides=cli_overrides,
    )

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def _drain(pipe: Any, buf: io.StringIO, dest: Any) -> None:
        for line in pipe:
            dest.write(line)
            dest.flush()
            buf.write(line)
        pipe.close()

    t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_buf, sys.stdout), daemon=True)
    t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_buf, sys.stderr), daemon=True)
    t_out.start()
    t_err.start()
    proc.wait()
    t_out.join()
    t_err.join()

    stdout_text = stdout_buf.getvalue()
    stderr_text = stderr_buf.getvalue()
    payload = {
        "cmd": [str(part) for part in cmd],
        "returncode": int(proc.returncode),
        "stdout_tail": stdout_text[-4000:],
        "stderr_tail": stderr_text[-4000:],
    }
    if proc.returncode != 0:
        return {"status": "failed", "process": payload}
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (out_dir / "ocr_qa_dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = compute_run_quality(summary, rows)
    return {"status": "ok", "process": payload, "metrics": metrics, "summary": summary}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        image_id = str(row.get("image_id") or "")
        if "chartqa" in image_id.lower():
            counts["chartqa"] += 1
        elif "textocr" in image_id.lower():
            counts["textocr"] += 1
        elif "coco" in image_id.lower():
            counts["coco_text"] += 1
        else:
            counts["other"] += 1
    return dict(counts)


def _rg_leakage_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rg_rows = [r for r in rows if str(r.get("question_type") or "") == "REVERSE_GROUND"]
    corrected = sum(1 for r in rg_rows if r.get("rg_correction_applied"))
    groundback_failed = sum(1 for r in rg_rows if r.get("groundback_failed"))
    return {
        "rg_count": len(rg_rows),
        "rg_corrected": corrected,
        "rg_groundback_failed": groundback_failed,
    }


def _render_sweep_report(payload: dict[str, Any]) -> str:
    bundle_id = payload["bundle_id"]
    results = payload.get("results", {})
    report_path = str(OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150" / f"{bundle_id}_report.md")
    log_path = str(REPO_ROOT / "logs" / bundle_id / "run.log")

    lines = [
        f"# {bundle_id}",
        "",
        f"- Date: `{payload['created_at'][:10]}`",
        f"- Bundle: `{bundle_id}`",
        f"- Updated: `{payload['updated_at']}`",
        f"- Live report: [{bundle_id}_report.md]({report_path})",
        f"- Log: [run.log]({log_path})",
        "",
        "## Goal",
        "",
        "**ita20: push vision_necessary_rate above 40% by eliminating residual leaky question types.**",
        "",
        "Changes from ita19:",
        "- Fix: SGOCR_RG_PER_IMAGE_HARD_CAP=0 now eliminates REVERSE_GROUND (prior max(1,...) bug fixed).",
        "- New: SGOCR_TP_VISUAL_ONLY_ENABLED=1 restricts TEXT_PROPERTY to text_color/text_curvature only.",
        "- In ita19 no_yn winner: RG was 7.1% vision_nec (14 QAs) and TP was 31.9% vision_nec (69 QAs).",
        "- Mechanical TP (word_count/first_word/last_word) is answerable from OCR text alone — always leaky.",
        "- With both flags, pipeline fills with DIRECT_READ (40.0% vision_nec, 0.7% text_leaky).",
        "- All variants build on ita19 no_yn (max_yesno=0 + Gemma4 antidoc + flash-lite).",
        "",
        "## Gemma4 Backend",
        "",
        "| Setting | Value |",
        "|---|---|",
        "| Model | `gemma4:e4b-it-q4_K_M` |",
        "| Serving | Ollama (`http://localhost:11434`) |",
        "| num_ctx | 4096 |",
        "| Prompt variant | antidoc (ITA16_ANTIDOC) |",
        "| Text-ref filter | enabled |",
        "",
        "## Variant Summary",
        "",
        "| # | Variant | YN | RG | TP | Target | Accepted | Images | Inline mean | Sweep | DR | RG# | YN# | TP# | AP |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    _yn_labels = {v: "0" for v in VARIANT_ORDER}
    _rg_labels = {
        "g_t8_no_yn_cement": "cap3",
        "g_t8_no_yn_no_rg": "0",
        "g_t8_no_yn_tp_visual": "cap3",
        "g_t8_no_yn_no_rg_tp_vis": "0",
    }
    _tp_labels = {
        "g_t8_no_yn_cement": "all",
        "g_t8_no_yn_no_rg": "all",
        "g_t8_no_yn_tp_visual": "visual",
        "g_t8_no_yn_no_rg_tp_vis": "visual",
    }
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        status = result.get("status", "pending")
        yn = _yn_labels.get(name, "—")
        rg = _rg_labels.get(name, "—")
        tp = _tp_labels.get(name, "—")
        if status == "pending":
            lines.append(f"| — | `{name}` | {yn} | {rg} | {tp} | 8 | — | — | — | — | — | — | — | — | — |")
            continue
        if status in {"running", "image_dependence"}:
            lines.append(f"| ⏳ | `{name}` | {yn} | {rg} | {tp} | 8 | running | — | — | — | — | — | — | — | — |")
            continue
        if status == "failed":
            lines.append(f"| ✗ | `{name}` | {yn} | {rg} | {tp} | 8 | FAILED | — | — | — | — | — | — | — | — |")
            continue
        m = result.get("metrics") or {}
        qt = m.get("question_types") or {}
        lines.append(
            f"| ✓ | `{name}` | {yn} | {rg} | {tp} | 8 |"
            f" {m.get('accepted_qas', 0)} |"
            f" {m.get('images_with_final_rows', 0)} |"
            f" {m.get('inline_frontier_mean', 0.0):.4f} |"
            f" {m.get('sweep_score', 0.0):.4f} |"
            f" {qt.get('DIRECT_READ', 0)} |"
            f" {qt.get('REVERSE_GROUND', 0)} |"
            f" {qt.get('YES_NO', 0)} |"
            f" {qt.get('TEXT_PROPERTY', 0)} |"
            f" {qt.get('ANCHOR_PROPERTY', 0)} |"
        )

    lines += [
        "",
        "## Image-Dependence Scores",
        "",
        "| Variant | Img acc | Text-only acc | Δ mean | Vision-nec % | Text-leaky % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        idep = result.get("image_dependence") or {}
        if not idep:
            lines.append(f"| `{name}` | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{name}` |"
            f" {idep.get('image_accuracy', 0.0):.4f} |"
            f" {idep.get('text_only_accuracy', 0.0):.4f} |"
            f" {idep.get('vision_delta_mean', 0.0):+.4f} |"
            f" {idep.get('vision_necessary_rate', 0.0)*100:.1f}% |"
            f" {idep.get('text_leaky_rate', 0.0)*100:.1f}% |"
        )

    return "\n".join(lines) + "\n"


def _render_live_doc(payload: dict[str, Any]) -> str:
    return _render_sweep_report(payload)


def _write_all(*, payload: dict[str, Any], report_json: Path, report_md: Path, live_doc: Path) -> None:
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    report_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_md.write_text(_render_sweep_report(payload), encoding="utf-8")
    live_doc.write_text(_render_live_doc(payload), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ap = argparse.ArgumentParser(
        description=(
            "ita20: push vision_necessary_rate above 40% by eliminating RG and mechanical TP.\n"
            "\n"
            "SERVICE NOTE: All variants require Ollama running (Gemma4 backend).\n"
            "Start Ollama before running: ollama serve &; sleep 5"
        )
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita20_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita19_20260417_011023_report.json")
    ap.add_argument("--model", default="gemini-3.1-flash-lite-preview")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    # Gemma Ollama settings
    ap.add_argument("--gemma-model", default="gemma4:e4b-it-q4_K_M")
    ap.add_argument("--gemma-base-url", default="http://localhost:11434")
    ap.add_argument("--gemma-num-ctx", type=int, default=4096)
    # Common
    ap.add_argument("--group-min-instances", type=int, default=3)
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.20)
    ap.add_argument("--rg-candidate-oversample-boost", type=float, default=1.5)
    ap.add_argument("--property-candidate-selection-bonus", type=float, default=0.5)
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview")
    ap.add_argument("--image-dependence-workers", type=int, default=4)
    ap.add_argument("--skip-image-dependence", action="store_true")
    ap.add_argument("--skip-qwen", action="store_true", help="No-op: no Qwen variants in ita20")
    ap.add_argument("--skip-gemma", action="store_true", help="Skip all g_* Gemma variants")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150"
    source_dir = root_final / args.source_name
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))

    baseline_env, baseline_cli = _q01_baseline()

    _TYPED_GATE_QT = "DIRECT_READ,YES_NO,TEXT_PROPERTY,ANCHOR_PROPERTY"

    _gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    common_env = {
        **baseline_env,
        "GEMINI_API_KEY": _gemini_api_key,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
        "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "florence",
        "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "independent_raw",
        "SGOCR_ANCHOR_RELABEL_MODE": "none",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
        "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
        "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
        "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
        "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
        "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
        "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "0",
        "SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD": "0.92",
        "SGOCR_QWEN_ITA15_PROMPT_ENABLED": "1",
        "SGOCR_SAM3_REFINE_MODE": "none",
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED": "1",
        "SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": _TYPED_GATE_QT,
        "SGOCR_RG_LEAKY_LABEL_HARD_REJECT_ENABLED": "1",
        "SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": str(float(args.rg_candidate_oversample_boost)),
        "SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": str(float(args.property_candidate_selection_bonus)),
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
        "SGOCR_ANCHOR_PROPERTY_SPECIFIC_THRESHOLD": "3",
        # ita19 no_yn baseline
        "SGOCR_MAX_YESNO_PER_IMAGE": "0",
    }

    common_cli = {**baseline_cli}
    _ocr_cache = Path(_ITA15_T8_INTERMEDIATE)

    # Gemma-specific env additions (all ita20 variants use Gemma4 antidoc)
    gemma_base_env = {
        **common_env,
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "gemma4_ollama",
        "SGOCR_GEMMA_OLLAMA_MODEL": args.gemma_model,
        "SGOCR_GEMMA_OLLAMA_BASE_URL": args.gemma_base_url,
        "SGOCR_GEMMA_OLLAMA_NUM_CTX": str(int(args.gemma_num_ctx)),
        "SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED": "1",
        "SGOCR_GEMMA_ITA16_PROMPT_VARIANT": "antidoc",
        "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
    }

    # All variants reuse ita19 no_yn's verified-tuples cache.
    # rg_cap and tp_visual_only apply at candidate-selection time (post-verified-tuples),
    # so no anchor re-run is needed — cache_level="verified" is correct for all.
    _gemma_cache = _ITA19_NO_YN_INTERMEDIATE

    variants: dict[str, dict[str, Any]] = {
        "g_t8_no_yn_cement": {
            "env": {**gemma_base_env},
            "cli": {**common_cli, "--target-per-image": "8"},
            "model": "gemini-3.1-flash-lite-preview",
            "note": VARIANT_DESCRIPTIONS["g_t8_no_yn_cement"],
            "cache_dir": _gemma_cache,
            "cache_level": "verified",
        },
        "g_t8_no_yn_no_rg": {
            "env": {
                **gemma_base_env,
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "0",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "model": "gemini-3.1-flash-lite-preview",
            "note": VARIANT_DESCRIPTIONS["g_t8_no_yn_no_rg"],
            "cache_dir": _gemma_cache,
            "cache_level": "verified",
        },
        "g_t8_no_yn_tp_visual": {
            "env": {
                **gemma_base_env,
                "SGOCR_TP_VISUAL_ONLY_ENABLED": "1",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "model": "gemini-3.1-flash-lite-preview",
            "note": VARIANT_DESCRIPTIONS["g_t8_no_yn_tp_visual"],
            "cache_dir": _gemma_cache,
            "cache_level": "verified",
        },
        "g_t8_no_yn_no_rg_tp_vis": {
            "env": {
                **gemma_base_env,
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "0",
                "SGOCR_TP_VISUAL_ONLY_ENABLED": "1",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "model": "gemini-3.1-flash-lite-preview",
            "note": VARIANT_DESCRIPTIONS["g_t8_no_yn_no_rg_tp_vis"],
            "cache_dir": _gemma_cache,
            "cache_level": "verified",
        },
    }

    report_json = root_final / f"{args.bundle_id}_report.json"
    report_md = root_final / f"{args.bundle_id}_report.md"
    LIVE_DOC_DIR.mkdir(parents=True, exist_ok=True)
    live_doc = LIVE_DOC_DIR / LIVE_DOC_NAME

    now = datetime.now().isoformat(timespec="seconds")
    existing_payload = _load_json(report_json)
    if existing_payload:
        payload = existing_payload
        payload["updated_at"] = now
        for name, spec in variants.items():
            if name not in payload["results"]:
                payload["results"][name] = {"status": "pending", "note": spec["note"]}
        completed = [n for n, r in payload["results"].items() if r.get("status") in {"ok", "cached"}]
        if completed:
            print(f"[resume] skipping completed variants: {completed}", flush=True)
    else:
        payload = {
            "bundle_id": args.bundle_id,
            "source_dir": str(source_dir),
            "source_manifest": source_manifest,
            "reference_report_json": str(root_final / args.reference_report_json),
            "reference_report": _load_json(root_final / args.reference_report_json),
            "created_at": now,
            "updated_at": now,
            "performance_profile": {
                "workers": int(args.workers),
                "gemma_model": str(args.gemma_model),
                "gemma_num_ctx": int(args.gemma_num_ctx),
                "group_min_instances": int(args.group_min_instances),
                "spatial_min_centroid_offset": float(args.spatial_min_centroid_offset),
                "rg_candidate_oversample_boost": float(args.rg_candidate_oversample_boost),
                "property_candidate_selection_bonus": float(args.property_candidate_selection_bonus),
                "image_dependence_model": str(args.image_dependence_model),
            },
            "nemotron_diagnostic": nemotron_frontend_diagnostic(),
            "results": {name: {"status": "pending", "note": spec["note"]} for name, spec in variants.items()},
        }
    _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

    for name, spec in variants.items():
        is_gemma_variant = name.startswith("g_")
        if is_gemma_variant and args.skip_gemma:
            print(f"[skip] {name} — --skip-gemma", flush=True)
            continue
        if payload["results"].get(name, {}).get("status") in {"ok", "cached"}:
            print(f"[skip] {name} already completed — skipping", flush=True)
            continue

        payload["results"][name] = {"status": "running", "note": spec["note"]}
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

        out_dir = root_final / f"{args.bundle_id}_{name}"
        intermediate_dir = root_intermediate / f"{args.bundle_id}_{name}"

        variant_cache_dir = spec.get("cache_dir")
        variant_cache_level = spec.get("cache_level", "none")
        if variant_cache_dir and not variant_cache_dir.exists():
            print(
                f"[warn] cache dir missing for {name}: {variant_cache_dir} — falling back to ocr cache",
                flush=True,
            )
            variant_cache_dir = _ocr_cache
            variant_cache_level = "ocr"

        result = _run_variant_streaming(
            source_dir=source_dir,
            out_dir=out_dir,
            intermediate_dir=intermediate_dir,
            cache_intermediate_dir=variant_cache_dir,
            cache_level=variant_cache_level,
            model=str(spec.get("model") or args.model),
            workers=int(args.workers),
            max_side=int(args.max_side),
            device=str(args.device),
            env_overrides=spec["env"],
            cli_overrides=spec["cli"],
        )
        result["out_dir"] = str(out_dir)
        result["intermediate_dir"] = str(intermediate_dir)
        result.setdefault("note", spec["note"])

        if result.get("status") in {"ok", "cached"}:
            rows = _load_jsonl(out_dir / "ocr_qa_dataset.jsonl")
            result["source_breakdown"] = _source_breakdown(rows)
            result["answer_distribution"] = compute_answer_distribution(rows)
            result["anchor_coverage"] = compute_anchor_coverage(rows)
            result["rg_leakage_stats"] = _rg_leakage_stats(rows)

            if not args.skip_image_dependence and rows:
                payload["results"][name] = {**result, "status": "image_dependence"}
                _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)
                try:
                    idep_dir = out_dir / "evals" / "image_dependence_ita20"
                    idep_summary = run_image_dependence_eval(
                        experiment_dir=out_dir,
                        model=str(args.image_dependence_model),
                        out_dir=idep_dir,
                        workers=int(args.image_dependence_workers),
                    )
                    result["image_dependence"] = idep_summary
                except Exception as exc:
                    result["image_dependence_error"] = str(exc)

        payload["results"][name] = result
        _write_all(payload=payload, report_json=report_json, report_md=report_md, live_doc=live_doc)

    print(str(live_doc))


if __name__ == "__main__":
    main()
