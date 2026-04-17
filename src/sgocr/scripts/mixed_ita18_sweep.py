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

# ita18 goal: first full sweep with all ita17-post fixes applied; Qwen3-VL as primary anchor backend.
#
#   Changes from ita17 (all applied in codebase, not yet swept):
#
#   Fix 1 — Prompt: removed "medium-to-large" size preference from both Gemma ITA15 and Qwen ITA15
#     prompts. Replaced with "every visible object or surface that has text on or near it."
#     Strengthened anti-OCR: "never quote, paraphrase, or reference." Targets background bbox
#     regression ("light blue sky", "pavement") seen in ita16/ita17.
#
#   Fix 2 — Filter hardening: extended is_text_ref_anchor_label() to catch "printed text",
#     "visible text", "written text" substrings and "text area" prefix. Added
#     _BACKGROUND_TRAILING_WORDS check ("sky", "ceiling", "pavement", "sidewalk") to
#     is_degenerate_anchor_label(). Added "text" to _TEXT_REF_TRAILING_WORDS (catches
#     "footer area text").
#
#   Fix 3 — Dropped text_orientation from TEXT_PROPERTY_VISUAL_TYPES. Was generating near-uniform
#     "horizontal" answers, killing training signal diversity. Rotation is now text_color /
#     text_curvature only.
#
#   Fix 4 — anchor_property_specific_threshold=3 (SGOCR_ANCHOR_PROPERTY_SPECIFIC_THRESHOLD).
#     Previously ANCHOR_PROPERTY shared TEXT_PROPERTY's threshold=4; Gemma's lower-specificity
#     labels couldn't pass, producing zero ANCHOR_PROPERTY QAs. Default is now 3 in code.
#
#   Fix 5 — _order_component_nodes: replaced orientation-class sort with reading-order
#     row-bucketed algorithm (y-centroid proximity via median height, x-sort within rows).
#     Fixes scrambled word order on vertical signs (e.g. ASHFIELD COUNCIL KEEP DRIVEWAY CLEAR).
#     All 6 synthetic test cases pass (sgocr/src/sgocr/scripts/merge_order_test.py).
#
#   Fix 6 — Teacher prompt label sanitization: build_batched_prompt() now instructs the teacher
#     to strip text-content descriptors from the anchor label in generated questions
#     (e.g. use "stack of books" not "stack of books with handwritten dates").
#
#   OCR cache: reuse nemotron_v2 text_detections from ita15_t8 (unchanged across all sweeps).
#   Anchor stage: always re-runs (new prompts + new backends = cache_level="ocr").
#
#   SERVICE NOTE:
#     Qwen variants (q_*) require vLLM — Ollama must NOT be running (GPU OOM).
#     Gemma variants (g_*) require Ollama — vLLM must be stopped.
#     Run with --skip-gemma to run only Qwen variants (stop Ollama first).
#     Run with --skip-qwen to run only Gemma variants (start Ollama first).
#
#   Variants:
#   q_t8_filtered           — Qwen3-VL vLLM + text-ref filter + target=8 (new primary baseline)
#   q_t8_no_filter          — Qwen3-VL vLLM, no filter + target=8 (ablation: new prompt vs filter)
#   q_t12_filtered          — Qwen3-VL vLLM + text-ref filter + target=12 (push: higher yield)
#   q_t8_flash_lite         — q_t8_filtered config but teacher=gemini-3.1-flash-lite (cost pilot)
#   g_t8_antidoc_filtered   — Gemma4 antidoc + filter + target=8 (ita17 winner carryforward)
#   g_t8_antidoc_flash_lite — g_t8_antidoc_filtered config but teacher=gemini-3.1-flash-lite

VARIANT_ORDER = [
    # Qwen variants first — requires Ollama stopped, vLLM running
    "q_t8_filtered",
    "q_t8_no_filter",
    "q_t12_filtered",
    "q_t8_flash_lite",
    # Gemma carryforward — requires vLLM stopped, Ollama running
    "g_t8_antidoc_filtered",
    "g_t8_antidoc_flash_lite",
]

VARIANT_DESCRIPTIONS: dict[str, str] = {
    "q_t8_filtered": (
        "Qwen3-VL vLLM + text-ref filter + target=8 "
        "(new primary baseline: updated ITA15 prompt, all fixes, filter on)"
    ),
    "q_t8_no_filter": (
        "Qwen3-VL vLLM + NO text-ref filter + target=8 "
        "(ablation: does new anchor prompt alone prevent text-ref labels without filter?)"
    ),
    "q_t12_filtered": (
        "Qwen3-VL vLLM + text-ref filter + target=12 "
        "(push: higher yield with Qwen backend after prompt + filter hardening)"
    ),
    "q_t8_flash_lite": (
        "Qwen3-VL vLLM + text-ref filter + target=8 + teacher=gemini-3.1-flash-lite-preview "
        "(Flash Lite cost pilot: same anchor config as q_t8_filtered, cheaper teacher)"
    ),
    "g_t8_antidoc_filtered": (
        "Gemma4 antidoc prompt + text-ref filter + target=8 "
        "(ita17 winner carryforward: direct Qwen vs Gemma quality comparison)"
    ),
    "g_t8_antidoc_flash_lite": (
        "Gemma4 antidoc prompt + text-ref filter + target=8 + teacher=gemini-3.1-flash-lite-preview "
        "(Flash Lite cost pilot: same anchor config as g_t8_antidoc_filtered, cheaper teacher)"
    ),
}

# OCR cache: reuse nemotron_v2 text_detections from ita15_t8 (unchanged)
_ITA15_T8_INTERMEDIATE = (
    str(OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150" /
        "sgocr_mixed_ita15_20260415_204008_ita15_t8")
)

# q_t12_filtered produced a full verified cache (anchor_tags, grounded_anchors, verified_tuples, etc.)
# q_t8_filtered and q_t8_flash_lite share the same anchor config (same model, filter ON, same prompt),
# so they can reuse q_t12_filtered's intermediate at cache_level="verified" — skipping Qwen entirely.
# q_t8_no_filter CANNOT reuse it (filter OFF → different anchor results).
_ITA18_Q_T12_INTERMEDIATE = (
    OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150" /
    "sgocr_mixed_ita18_20260416_193245_q_t12_filtered"
)

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "172_sgocr_mixed_ita18_launch_2026-04-17.md"


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
    """Like _run_variant but streams stdout+stderr live to the terminal while also capturing tails."""
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
        "**ita18: first full sweep with all post-ita17 fixes; Qwen3-VL as primary anchor backend.**",
        "",
        "Changes from ita17:",
        "- Anchor prompt: removed 'medium-to-large' preference; now 'every visible object or surface",
        "  that has text on or near it'. Strengthened anti-OCR phrasing in both Gemma + Qwen prompts.",
        "- Filter: extended text-ref detection (substrings, 'text area' prefix, background trailing",
        "  words). Added 'text' to trailing-word set (catches 'footer area text').",
        "- Fix 3: `text_orientation` removed from TEXT_PROPERTY_VISUAL_TYPES — reduces 'horizontal'",
        "  dominance, improves training signal diversity.",
        "- Fix 4: `SGOCR_ANCHOR_PROPERTY_SPECIFIC_THRESHOLD=3` (split from TEXT_PROPERTY threshold=4)",
        "  — restores ANCHOR_PROPERTY question generation for Gemma labels.",
        "- Fix 5: `_order_component_nodes` uses reading-order row-bucketed sort (y-centroid +",
        "  median height tolerance) — fixes scrambled word order on vertical signs.",
        "- Fix 6: Teacher prompt instructs model to strip text-content descriptors from anchor label",
        "  before using it in the generated question.",
        "- OCR cache reused from ita15_t8 (nemotron_v2 text_detections unchanged).",
        "",
        "## Qwen3-VL Backend",
        "",
        "| Setting | Value |",
        "|---|---|",
        "| Model | `Qwen/Qwen3-VL-8B-Instruct-FP8` |",
        "| Serving | vLLM (local GPU) |",
        "| GPU memory utilization | 0.85 |",
        "| Batch size | 3 |",
        "| Max model len | 2048 |",
        "| Prompt | INDEPENDENT_QWEN_INVENTORY_PROMPT_ITA15 (updated) |",
        "",
        "## Gemma4 Backend (carryforward)",
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
        "| # | Variant | Backend | Teacher | Accepted | Images | Inline mean | Sweep | DR | RG | YN | TP | AP |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    _flash_lite_variants = {"q_t8_flash_lite", "g_t8_antidoc_flash_lite"}
    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        status = result.get("status", "pending")
        backend = "qwen" if name.startswith("q_") else "gemma"
        teacher = "flash-lite" if name in _flash_lite_variants else "flash"
        if status == "pending":
            lines.append(f"| — | `{name}` | {backend} | {teacher} | — | — | — | — | — | — | — | — | — |")
            continue
        if status in {"running", "image_dependence"}:
            lines.append(f"| ⏳ | `{name}` | {backend} | {teacher} | running | — | — | — | — | — | — | — | — |")
            continue
        if status == "failed":
            lines.append(f"| ✗ | `{name}` | {backend} | {teacher} | FAILED | — | — | — | — | — | — | — | — |")
            continue
        m = result.get("metrics") or {}
        qt = m.get("question_types") or {}
        lines.append(
            f"| ✓ | `{name}` | {backend} | {teacher} |"
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
            "ita18: Qwen3-VL primary anchor backend + all post-ita17 fixes (4 variants).\n"
            "\n"
            "SERVICE NOTE: Qwen variants (q_*) require Ollama to be STOPPED.\n"
            "Gemma variants (g_*) require Ollama to be RUNNING.\n"
            "Use --skip-gemma to run only Qwen variants, or --skip-qwen to run only Gemma."
        )
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita18_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita17_20260416_031540_report.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    # Qwen vLLM settings
    ap.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    ap.add_argument("--qwen-gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--qwen-batch-size", type=int, default=3)
    ap.add_argument("--qwen-max-model-len", type=int, default=2048)
    # Gemma Ollama settings (carryforward)
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
    ap.add_argument("--skip-qwen", action="store_true", help="Skip all q_* Qwen variants (run Gemma only)")
    ap.add_argument("--skip-gemma", action="store_true", help="Skip all g_* Gemma variants (run Qwen only)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150"
    source_dir = root_final / args.source_name
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))

    baseline_env, baseline_cli = _q01_baseline()

    _TYPED_GATE_QT = "DIRECT_READ,YES_NO,TEXT_PROPERTY,ANCHOR_PROPERTY"

    # Base env shared by all variants: post-ita17 settings.
    # New in ita18: SGOCR_ANCHOR_PROPERTY_SPECIFIC_THRESHOLD=3 (fix 4).
    # GEMINI_API_KEY is passed explicitly so it reaches the subprocess even if the
    # parent shell environment is inconsistent (e.g. set after Claude Code started).
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
        # Fix 4: split ANCHOR_PROPERTY specificity gate (was sharing TEXT_PROPERTY threshold=4)
        "SGOCR_ANCHOR_PROPERTY_SPECIFIC_THRESHOLD": "3",
    }

    common_cli = {**baseline_cli}
    _ocr_cache = Path(_ITA15_T8_INTERMEDIATE)

    # Qwen-specific env additions
    qwen_base_env = {
        **common_env,
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_MODEL": args.qwen_model,
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": str(float(args.qwen_gpu_memory_utilization)),
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": str(int(args.qwen_batch_size)),
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": str(int(args.qwen_max_model_len)),
    }

    # Gemma-specific env additions
    gemma_base_env = {
        **common_env,
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "gemma4_ollama",
        "SGOCR_GEMMA_OLLAMA_MODEL": args.gemma_model,
        "SGOCR_GEMMA_OLLAMA_BASE_URL": args.gemma_base_url,
        "SGOCR_GEMMA_OLLAMA_NUM_CTX": str(int(args.gemma_num_ctx)),
        "SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED": "1",
    }

    variants: dict[str, dict[str, Any]] = {
        "q_t8_filtered": {
            "env": {
                **qwen_base_env,
                "SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED": "1",
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "note": VARIANT_DESCRIPTIONS["q_t8_filtered"],
            "cache_dir": _ITA18_Q_T12_INTERMEDIATE,
            "cache_level": "verified",
        },
        "q_t8_no_filter": {
            "env": {
                **qwen_base_env,
                "SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED": "0",
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "note": VARIANT_DESCRIPTIONS["q_t8_no_filter"],
            "cache_dir": _ocr_cache,
            "cache_level": "ocr",
        },
        "q_t12_filtered": {
            "env": {
                **qwen_base_env,
                "SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED": "1",
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "4",
            },
            "cli": {**common_cli, "--target-per-image": "12"},
            "note": VARIANT_DESCRIPTIONS["q_t12_filtered"],
            "cache_dir": _ocr_cache,
            "cache_level": "ocr",
        },
        "q_t8_flash_lite": {
            "env": {
                **qwen_base_env,
                "SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED": "1",
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "model": "gemini-3.1-flash-lite-preview",
            "note": VARIANT_DESCRIPTIONS["q_t8_flash_lite"],
            "cache_dir": _ITA18_Q_T12_INTERMEDIATE,
            "cache_level": "verified",
        },
        "g_t8_antidoc_filtered": {
            "env": {
                **gemma_base_env,
                "SGOCR_GEMMA_ITA16_PROMPT_VARIANT": "antidoc",
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "note": VARIANT_DESCRIPTIONS["g_t8_antidoc_filtered"],
            "cache_dir": _ocr_cache,
            "cache_level": "ocr",
        },
        "g_t8_antidoc_flash_lite": {
            "env": {
                **gemma_base_env,
                "SGOCR_GEMMA_ITA16_PROMPT_VARIANT": "antidoc",
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "model": "gemini-3.1-flash-lite-preview",
            "note": VARIANT_DESCRIPTIONS["g_t8_antidoc_flash_lite"],
            "cache_dir": _ocr_cache,
            "cache_level": "ocr",
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
                "qwen_model": str(args.qwen_model),
                "qwen_gpu_memory_utilization": float(args.qwen_gpu_memory_utilization),
                "qwen_batch_size": int(args.qwen_batch_size),
                "qwen_max_model_len": int(args.qwen_max_model_len),
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
        is_qwen_variant = name.startswith("q_")
        is_gemma_variant = name.startswith("g_")
        if is_qwen_variant and args.skip_qwen:
            print(f"[skip] {name} — --skip-qwen", flush=True)
            continue
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
                f"[warn] cache dir missing for {name}: {variant_cache_dir} — falling back to cache_level=none",
                flush=True,
            )
            variant_cache_dir = None
            variant_cache_level = "none"

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
                    idep_dir = out_dir / "evals" / "image_dependence_ita18"
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
