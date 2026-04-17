from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .dev200_eval import run_image_dependence_eval
from .mixed_ocr_frontend_canary import _q01_baseline, _run_variant
from ..nemotron_frontend import nemotron_frontend_diagnostic
from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, REPO_ROOT
from ..run_quality import compute_anchor_coverage, compute_answer_distribution

# ita17 goal: validate text-reference anchor label filter on Gemma4-ollama backend.
#
#   Context:
#     ita16 (g_t5_base) showed 8.3% of final anchors had text-referencing labels
#     (e.g. "x-axis region date labels", "y-axis region labels", "text block black text").
#     These labels reference visible text content rather than describing visual objects,
#     leaking potential answers through the anchor name itself.
#     New filter: SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED=1
#       → is_text_ref_anchor_label() removes anchors whose last token is in
#         {"labels", "caption", "footnote", "footnotes", "watermark"}, or that
#         start with "text block".
#
#   OCR cache: reuse nemotron_v2 text_detections from ita15_t8 intermediate
#   Anchor stage: always re-runs (Gemma HTTP calls + new filter = cache_level="ocr")
#
#   Variants:
#   g_t8_filtered       — g_t8_base + text-ref filter (cement frontier + fix)
#   g_t8_antidoc_filtered — antidoc prompt + text-ref filter (strongest anti-text-leak)
#   g_t12_filtered      — target=12 + text-ref filter (push: higher QA yield at quality)
#   g_t8_compact_filtered — compact prompt + text-ref filter (push: minimal prompt + filter)

VARIANT_ORDER = [
    "g_t8_filtered",
    "g_t8_antidoc_filtered",
    "g_t12_filtered",
    "g_t8_compact_filtered",
]

VARIANT_DESCRIPTIONS: dict[str, str] = {
    "g_t8_filtered": (
        "Gemma4 ita15 prompt + text-ref filter + target=8 + rg_cap=3 "
        "(ita16 g_t8_base + new filter: cements frontier with fix applied)"
    ),
    "g_t8_antidoc_filtered": (
        "Gemma4 antidoc prompt + text-ref filter + target=8 + rg_cap=3 "
        "(strongest anti-text-leak config: anti-doc + trailing-label filter)"
    ),
    "g_t12_filtered": (
        "Gemma4 ita15 prompt + text-ref filter + target=12 + rg_cap=4 "
        "(push: higher yield after filter removes low-quality anchors)"
    ),
    "g_t8_compact_filtered": (
        "Gemma4 compact prompt + text-ref filter + target=8 + rg_cap=3 "
        "(push: minimal prompt naturally avoids compound labels; filter as backup)"
    ),
}

# OCR cache: reuse nemotron_v2 text_detections from ita15_t8
_ITA15_T8_INTERMEDIATE = (
    str(OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150" /
        "sgocr_mixed_ita15_20260415_204008_ita15_t8")
)

LIVE_DOC_DIR = REPO_ROOT / "tasks" / "mm_bridge" / "docs"
LIVE_DOC_NAME = "171_sgocr_mixed_ita17_launch_2026-04-16.md"


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
        "**ita17: text-reference anchor label filter validation on Gemma4-ollama.**",
        "",
        "Changes from ita16:",
        "- `SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED=1`: new filter removes anchors whose",
        "  label ends with a text-role word ('labels', 'caption', 'footnote', 'watermark')",
        "  or starts with 'text block'. Targets compound labels like 'x-axis region date labels'",
        "  that reference visible text content rather than describing visual objects.",
        "- OCR cache reused from ita15_t8 (nemotron_v2 text_detections unchanged).",
        "- Two target levels: t8 (cement frontier) and t12 (push for higher yield).",
        "",
        "## Gemma4 Backend",
        "",
        "| Setting | Value |",
        "|---|---|",
        "| Model | `gemma4:e4b-it-q4_K_M` |",
        "| Serving | Ollama (`http://localhost:11434`) |",
        "| Coordinate format | 0-1 normalized floats (auto-detected) |",
        "| num_ctx | 4096 |",
        "| Text-ref filter | `SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED=1` (new) |",
        "",
        "## Variant Summary",
        "",
        "| # | Variant | Accepted | Images | Inline mean | Sweep | DR | RG | YN | TP | AP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name in VARIANT_ORDER:
        result = results.get(name) or {}
        status = result.get("status", "pending")
        if status == "pending":
            lines.append(f"| — | `{name}` | — | — | — | — | — | — | — | — | — |")
            continue
        if status in {"running", "image_dependence"}:
            lines.append(f"| ⏳ | `{name}` | running | — | — | — | — | — | — | — | — |")
            continue
        if status == "failed":
            lines.append(f"| ✗ | `{name}` | FAILED | — | — | — | — | — | — | — | — |")
            continue
        m = result.get("metrics") or {}
        qt = m.get("question_types") or {}
        lines.append(
            f"| ✓ | `{name}` |"
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
        description="ita17: text-reference anchor label filter sweep (4 Gemma4 variants)."
    )
    ap.add_argument("--bundle-id", default=f"sgocr_mixed_ita17_{stamp}")
    ap.add_argument("--source-name", default="chartqa50_textocr50_cocotext50_source_20260410_110952")
    ap.add_argument("--reference-report-json", default="sgocr_mixed_ita16_20260416_003058_report.json")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gemma-model", default="gemma4:e4b-it-q4_K_M")
    ap.add_argument("--gemma-base-url", default="http://localhost:11434")
    ap.add_argument("--gemma-num-ctx", type=int, default=4096)
    ap.add_argument("--group-min-instances", type=int, default=3)
    ap.add_argument("--spatial-min-centroid-offset", type=float, default=0.20)
    ap.add_argument("--rg-candidate-oversample-boost", type=float, default=1.5)
    ap.add_argument("--property-candidate-selection-bonus", type=float, default=0.5)
    ap.add_argument("--image-dependence-model", default="gemini-3-flash-preview")
    ap.add_argument("--image-dependence-workers", type=int, default=4)
    ap.add_argument("--skip-image-dependence", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root_final = OCR_SPATIAL_QA_FINAL_ROOT / "mixed_dev150"
    root_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "mixed_dev150"
    source_dir = root_final / args.source_name
    source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))

    baseline_env, baseline_cli = _q01_baseline()

    _TYPED_GATE_QT = "DIRECT_READ,YES_NO,TEXT_PROPERTY,ANCHOR_PROPERTY"

    # All Gemma variants share this base env — same post-OCR settings as ita16.
    # New addition: SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED=1
    common_env = {
        **baseline_env,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
        "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "florence",
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "gemma4_ollama",
        "SGOCR_GEMMA_OLLAMA_MODEL": args.gemma_model,
        "SGOCR_GEMMA_OLLAMA_BASE_URL": args.gemma_base_url,
        "SGOCR_GEMMA_OLLAMA_NUM_CTX": str(int(args.gemma_num_ctx)),
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
        # New in ita17: filter anchors whose labels reference visible text content
        "SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED": "1",
        "SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": _TYPED_GATE_QT,
        "SGOCR_RG_LEAKY_LABEL_HARD_REJECT_ENABLED": "1",
        "SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": str(float(args.rg_candidate_oversample_boost)),
        "SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": str(float(args.property_candidate_selection_bonus)),
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
    }

    common_cli = {**baseline_cli}
    _ita15_t8_cache = Path(_ITA15_T8_INTERMEDIATE)

    variants: dict[str, dict[str, Any]] = {
        "g_t8_filtered": {
            "env": {
                **common_env,
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
                # ita15 Gemma prompt (default via SGOCR_QWEN_ITA15_PROMPT_ENABLED=1)
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "note": VARIANT_DESCRIPTIONS["g_t8_filtered"],
            "cache_dir": _ita15_t8_cache,
            "cache_level": "ocr",
        },
        "g_t8_antidoc_filtered": {
            "env": {
                **common_env,
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
                "SGOCR_GEMMA_ITA16_PROMPT_VARIANT": "antidoc",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "note": VARIANT_DESCRIPTIONS["g_t8_antidoc_filtered"],
            "cache_dir": _ita15_t8_cache,
            "cache_level": "ocr",
        },
        "g_t12_filtered": {
            "env": {
                **common_env,
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "4",
                # ita15 Gemma prompt (default)
            },
            "cli": {**common_cli, "--target-per-image": "12"},
            "note": VARIANT_DESCRIPTIONS["g_t12_filtered"],
            "cache_dir": _ita15_t8_cache,
            "cache_level": "ocr",
        },
        "g_t8_compact_filtered": {
            "env": {
                **common_env,
                "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
                "SGOCR_GEMMA_ITA16_PROMPT_VARIANT": "compact",
            },
            "cli": {**common_cli, "--target-per-image": "8"},
            "note": VARIANT_DESCRIPTIONS["g_t8_compact_filtered"],
            "cache_dir": _ita15_t8_cache,
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

        result = _run_variant(
            source_dir=source_dir,
            out_dir=out_dir,
            intermediate_dir=intermediate_dir,
            cache_intermediate_dir=variant_cache_dir,
            cache_level=variant_cache_level,
            model=str(args.model),
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
                    idep_dir = out_dir / "evals" / "image_dependence_ita17"
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
