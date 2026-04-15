"""Dev200 J-series sweep: quality-gate refinement on h03 Pareto frontier.

Context:
  - h03 (resolvability 1.42 + aggressive anchors, frontier RG) = Q3=80.24, 106/200
  - i-series showed resolvability loosening backfires: 1.35 → 105 images, Q3=77
  - i02 best i-series (ambiguity gate 9/8 on 1.35): Q3=78.59 — ambigate helps but not enough
  - RG soft improved dramatically from teacher prompt guardrail fix (+25pp GPT, +34pp Gemini)
  - rev_mixed stuck at 1–2: teacher can't generate non-trivial location for generic anchors
  - DR soft ceiling ~46%: grounding failures dominate, not normalization gaps

Goals:
  1. Test ambiguity gate 9/8 on pure h03 (without resolvability loosening) — isolate gate effect
  2. Test DR threshold 5 + very_strict teacher — harder discrimination, higher quality signal
  3. Test softer anchor penalty 0.02 (vs 0.04) — more diverse anchor types, less miss pressure
  4. Combo: penalty 0.02 + ambiguity 9/8 + DR 5 + very_strict (stacked quality)

Run matrix:
  j01  h03 base + ambiguity 9/8                              (reuses h03 OCR cache)
  j02  h03 base + DR threshold 5 + very_strict teacher       (reuses h03 OCR cache)
  j03  h03 base + anchor penalty 0.02                        (full pipeline — penalty changes grounding)
  j04  penalty 0.02 + ambiguity 9/8 + DR 5 + very_strict    (reuses j03 OCR cache)
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, LOGS_ROOT, REPO_ROOT
from ..run_quality import compute_run_quality
from .semantic_dev40_sweep_v2 import AblationSpec, append_timeline, gpu_snapshot, now_stamp, write_text


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "dev200_source_universe"

# Prior best runs for comparison
H03_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_h_20260407_055403_h03_aggressive_anchors"
H03_INTERM = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev200_h_20260407_055403_h03_aggressive_anchors"
I02_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_i_20260407_102647_i02_resolvability135_ambiguity98"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="J-series dev200 sweep: quality-gate refinement on h03 Pareto frontier.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_j_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(bundle_id: str) -> list[AblationSpec]:
    # ----------------------------------------------------------------
    # Base: exact h03 settings (the Pareto frontier)
    #   - consensus 0.72
    #   - resolvability 1.42/0.56
    #   - aggressive anchors: penalty 0.04, expansion=aggressive
    #   - frontier RG answer style
    #   - 14 tags, grounding threshold 0.28
    #   - DR threshold 4, ambiguity 7/6
    # ----------------------------------------------------------------
    base_env = {
        # Merge
        "SGOCR_TEXT_MERGE_ENABLED": "1",
        "SGOCR_TEXT_MERGE_ANCHOR_OVERLAP_MIN": "0.08",
        "SGOCR_TEXT_MERGE_X_OVERLAP_MIN": "0.42",
        "SGOCR_TEXT_MERGE_Y_OVERLAP_MIN": "0.32",
        "SGOCR_TEXT_MERGE_GAP_RATIO_MAX": "2.0",
        "SGOCR_TEXT_MERGE_HEIGHT_RATIO_MAX": "2.2",
        "SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES": "3",
        # Detection
        "SGOCR_DETECTOR_BOX_THRESH": "0.35",
        "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
        "SGOCR_OCR_CROP_PAD_RATIO": "0.10",
        "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.08",
        # Consensus — 0.72 from f/h series
        "SGOCR_STRONG_CONSENSUS_FLOOR": "0.72",
        "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.79",
        "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
        # Resolvability — h03 sweet spot
        "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.42",
        "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.56",
        # Anchor — h03 aggressive settings
        "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
        "SGOCR_ANCHOR_RELABEL_MODE": "flash",
        "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
        "SGOCR_GENERIC_ANCHOR_PENALTY": "0.04",
        "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.34",
        "SGOCR_FLORENCE_REGION_BONUS": "0.06",
        # Wording
        "SGOCR_LOCATION_WORDING_MODE": "varied",
        "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
        # RG answer style: frontier
        "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier",
        # Verifier thresholds — h03 baseline
        "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
        "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
        "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
        "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "4",
        "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.35",
        # Ambiguity — h03 baseline
        "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
        "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
        # Local bias: extreme (h03 best)
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.65",
        "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.35",
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.50",
        "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.30",
        # SAM3
        "SGOCR_SAM3_REFINE_MODE": "top3",
        "SGOCR_SAM3_APPLY_MODE": "targeted",
        "SGOCR_SAM3_TOPK_PROMPTS": "3",
        "SGOCR_SAM3_CONFIDENCE_THRESHOLD": "0.35",
        "SGOCR_SAM3_BOX_THRESHOLD": "0.30",
        "SGOCR_SAM3_RELEVANCE_BONUS": "0.10",
        "SGOCR_SAM3_TARGET_SUPPORT_MAX": "1",
        "SGOCR_SAM3_TARGET_CLUSTER_MIN": "2",
        "SGOCR_SAM3_TARGET_AREA_START": "0.34",
        # Teacher strictness — h03 baseline
        "SGOCR_TEACHER_STRICTNESS": "strict",
    }

    # j03 runs full pipeline; j04 reuses its OCR cache.
    j03_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / f"{bundle_id}_j03_anchor_penalty02"

    return [
        # ============================================================
        # j01: h03 + tighter ambiguity gate (9/8)
        #   Isolates the ambiguity gate effect without touching resolvability.
        #   i02 used 9/8 on 1.35 resolvability; this tests 9/8 on h03's 1.42
        #   to determine if the gate alone can push Q3 past 80.24.
        #   Reuses h03 OCR cache — only downstream stages re-run.
        # ============================================================
        AblationSpec(
            "j01_ambigate98",
            "h03 + ambiguity gate 9/8 — isolate gate effect on Pareto frontier",
            "ocr",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "9",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "8",
            },
            cache_intermediate_dir=H03_INTERM,
        ),

        # ============================================================
        # j02: h03 + DR threshold 5 + very_strict teacher
        #   Raises the bar for DIRECT_READ selection (need 5+ candidates
        #   instead of 4) and switches teacher to very_strict mode.
        #   Hypothesis: fewer DR rows but each one is harder — discriminates
        #   reading ability more precisely, raises frontier eval difficulty
        #   and improves soft eval signal quality.
        #   Reuses h03 OCR cache.
        # ============================================================
        AblationSpec(
            "j02_dr5_verystrict",
            "h03 + DR threshold 5 + very_strict teacher — harder discrimination",
            "ocr",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "5",
                "SGOCR_TEACHER_STRICTNESS": "very_strict",
            },
            cache_intermediate_dir=H03_INTERM,
        ),

        # ============================================================
        # j03: h03 + anchor penalty 0.02 (softer diversity pressure)
        #   Reduces generic anchor penalty from 0.04 → 0.02. More diverse
        #   anchor types (sign walls, display panels) will score higher
        #   relative to specific anchors. May increase anchor diversity
        #   and unlock anchor types that were systematically excluded.
        #   Full pipeline run (penalty changes affect grounding scoring).
        # ============================================================
        AblationSpec(
            "j03_anchor_penalty02",
            "h03 + anchor penalty 0.02 — softer diversity pressure, more anchor types",
            "none",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.02",
            },
        ),

        # ============================================================
        # j04: penalty 0.02 + ambiguity 9/8 + DR 5 + very_strict
        #   Stacks all three quality improvements on the softer anchor base.
        #   Reuses j03 OCR cache.
        #   Hypothesis: combined gates produce the cleanest, most informative
        #   dataset even if yield is lower — optimizing for data quality
        #   over quantity.
        # ============================================================
        AblationSpec(
            "j04_combo_quality",
            "penalty 0.02 + ambiguity 9/8 + DR 5 + very_strict — stacked quality combo",
            "ocr",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.02",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "9",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "8",
                "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "5",
                "SGOCR_TEACHER_STRICTNESS": "very_strict",
            },
            cache_intermediate_dir=j03_intermediate,
        ),
    ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_metrics(experiment_dir: Path) -> dict[str, Any]:
    summary = read_json(experiment_dir / "summary.json")
    rows = load_rows(experiment_dir / "ocr_qa_dataset.jsonl")
    return compute_run_quality(summary, rows)


def render_report(bundle_id: str, specs: list[AblationSpec], results: dict[str, dict[str, Any]], docs_path: Path) -> str:
    lines = [
        "# SGOCR Dev200 J-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Design",
        "",
        "4-run sweep on the h03 Pareto frontier (Q3=80.24, 106/200).",
        "Goal: improve dataset quality above h03 without compromising coverage.",
        "All j01/j02 reuse h03 OCR cache; j03 runs full pipeline; j04 reuses j03 OCR.",
        "",
        "Hypotheses:",
        "1. **Ambiguity gate 9/8** (j01): tighter ambiguity filter on h03 — isolate gate effect without resolvability change.",
        "2. **DR threshold 5 + very_strict** (j02): harder DIRECT_READ discrimination + stricter teacher answer acceptance.",
        "3. **Anchor penalty 0.02** (j03): softer generic anchor penalty → more diverse anchor types, less miss pressure.",
        "4. **Stacked combo** (j04): penalty 0.02 + ambigate 9/8 + DR 5 + very_strict — maximum quality stack.",
        "",
        "## Prior Best Runs",
        "",
        "| Run | Q3 | Accepted | Accept% | Images/200 | Rev local | Rev mixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, prior_dir in [("h03_aggressive_anchors", H03_DEV200), ("i02_resolvability135_ambiguity98", I02_DEV200)]:
        try:
            m = collect_metrics(prior_dir)
            lines.append(f"| `{label}` | {m['q3_score']} | {m['accepted_qas']} | {m['qa_accept_rate']:.3f} | {m['images_with_final_rows']}/200 | {m['reverse_local']} | {m['reverse_mixed']} |")
        except Exception:
            lines.append(f"| `{label}` | - | - | - | - | - | - |")

    lines.extend([
        "",
        "## J-Series Scoreboard",
        "",
        "| Run | Status | Accepted | Accept% | Images/200 | Q3 | Neg | Rev local | Rev mixed | Ambig | AnchorMiss | Diversity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for spec in specs:
        row = results.get(spec.name)
        if not row:
            lines.append(f"| `{spec.name}` | pending | - | - | - | - | - | - | - | - | - | - |")
            continue
        if row.get("status") not in {"ok", "cached"}:
            lines.append(f"| `{spec.name}` | {row.get('status')} | - | - | - | - | - | - | - | - | - | - |")
            continue
        m = row["metrics"]
        lines.append(
            f"| `{spec.name}` | {row.get('status')}"
            f" | {m['accepted_qas']}"
            f" | {m['qa_accept_rate']:.3f}"
            f" | {m['images_with_final_rows']}/200"
            f" | {m['q3_score']}"
            f" | {m['yesno_negative']}"
            f" | {m['reverse_local']}"
            f" | {m['reverse_mixed']}"
            f" | {m['ambiguity_high']}"
            f" | {m['anchor_missing']}"
            f" | {m['type_diversity']:.3f} |"
        )

    complete = [row["metrics"] | {"name": name} for name, row in results.items() if row.get("status") in {"ok", "cached"}]
    if complete:
        ranked = sorted(complete, key=lambda r: (-r["q3_score"], -r["accepted_qas"], r["name"]))
        lines.extend(["", "## Rankings", ""])
        for idx, row in enumerate(ranked, 1):
            lines.append(
                f"{idx}. `{row['name']}`: Q3=`{row['q3_score']}`"
                f", accepted=`{row['accepted_qas']}`"
                f", rate=`{row['qa_accept_rate']:.3f}`"
                f", images=`{row['images_with_final_rows']}/200`"
                f", rev-local=`{row['reverse_local']}`"
                f", rev-mixed=`{row['reverse_mixed']}`"
                f", anchor-miss=`{row['anchor_missing']}`"
            )

    lines.extend([
        "",
        "## Notes",
        "",
        "- j01/j02 reuse h03 OCR cache; j03 is a full pipeline run; j04 reuses j03 OCR cache.",
        "- h03 base: consensus 0.72, resolvability 1.42/0.56, penalty 0.04, ambigate 7/6, DR thresh 4, strict teacher.",
        "- SGOCR_TEACHER_STRICTNESS=very_strict tested for first time in j02/j04.",
        "- Anchor penalty 0.02 (vs default 0.075, h03 0.04) is the least restrictive setting tested.",
        f"- Live report: `{docs_path}`",
    ])
    return "\n".join(lines) + "\n"


def run_ablation(
    *,
    bundle_dir: Path,
    bundle_id: str,
    spec: AblationSpec,
    args: argparse.Namespace,
    docs_path: Path,
    report_path: Path,
    results: dict[str, dict[str, Any]],
    specs: list[AblationSpec],
) -> None:
    experiment_name = f"{bundle_id}_{spec.name}"
    out_dir = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / experiment_name
    intermediate_dir = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / experiment_name
    log_path = bundle_dir / f"{spec.name}.log"
    timeline_path = bundle_dir / "timeline.log"

    if (out_dir / "summary.json").exists():
        metrics = collect_metrics(out_dir)
        results[spec.name] = {"status": "cached", "metrics": metrics, "experiment_name": experiment_name}
        append_timeline(timeline_path, f"END {spec.name} cached accepted={metrics['accepted_qas']} q3={metrics['q3_score']}")
        write_text(report_path, render_report(bundle_id, specs, results, docs_path))
        write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
        return

    cache_intermediate_dir = spec.cache_intermediate_dir or intermediate_dir
    model_name = spec.model_override or args.model
    cmd = [
        sys.executable, "-m", "sgocr.dev200_harness", "build-dev40-semantic",
        "--source-experiment-dir", str(SOURCE_EXPERIMENT),
        "--out-dir", str(out_dir),
        "--intermediate-dir", str(intermediate_dir),
        "--cache-intermediate-dir", str(cache_intermediate_dir),
        "--cache-level", spec.cache_level,
        "--model", model_name,
        "--workers", str(args.workers),
        "--max-side", str(args.max_side),
        "--target-per-image", spec.cli.get("--target-per-image", str(args.target_per_image)),
        "--max-detections", spec.cli.get("--max-detections", "128"),
        "--grounding-threshold", spec.cli.get("--grounding-threshold", "0.28"),
        "--max-tags-per-image", spec.cli.get("--max-tags-per-image", "14"),
    ]
    env = os.environ.copy()
    env.update(spec.env)

    append_timeline(
        timeline_path,
        f"START {spec.name} cache={spec.cache_level} model={model_name}"
        f" consensus={spec.env.get('SGOCR_STRONG_CONSENSUS_FLOOR','?')}"
        f" penalty={spec.env.get('SGOCR_GENERIC_ANCHOR_PENALTY','?')}"
        f" ambi={spec.env.get('SGOCR_AMBIGUITY_HARD_REJECT_SCORE','?')}"
        f" dr_thresh={spec.env.get('SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD','?')}"
        f" strictness={spec.env.get('SGOCR_TEACHER_STRICTNESS','?')}"
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("CMD: " + " ".join(shlex.quote(p) for p in cmd) + "\n")
        if spec.env:
            handle.write("ENV_OVERRIDES: " + json.dumps(spec.env, sort_keys=True) + "\n")
        handle.flush()
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, env=env, text=True)
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            append_timeline(timeline_path, f"WATCH {spec.name} pid={proc.pid} gpu=[{gpu_snapshot()}]")
            write_text(report_path, render_report(bundle_id, specs, results, docs_path))
            write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
            time.sleep(max(5, int(args.watch_seconds)))

    if proc.returncode != 0:
        results[spec.name] = {"status": f"failed:{proc.returncode}", "experiment_name": experiment_name}
        append_timeline(timeline_path, f"END {spec.name} FAILED code={proc.returncode}")
    else:
        metrics = collect_metrics(out_dir)
        results[spec.name] = {"status": "ok", "metrics": metrics, "experiment_name": experiment_name}
        append_timeline(timeline_path, f"END {spec.name} ok accepted={metrics['accepted_qas']} images={metrics['images_with_final_rows']} q3={metrics['q3_score']}")

    write_text(report_path, render_report(bundle_id, specs, results, docs_path))
    write_text(docs_path, render_report(bundle_id, specs, results, docs_path))


def main() -> None:
    args = parse_args()
    bundle_id = args.bundle_id
    specs = build_specs(bundle_id)[: args.limit]
    bundle_dir = LOGS_ROOT / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = bundle_dir / "timeline.log"
    report_path = bundle_dir / "report.md"
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "129_sgocr_dev200_j_sweep_2026-04-07.md"
    results: dict[str, dict[str, Any]] = {}
    append_timeline(timeline_path, f"BUNDLE {bundle_id} start specs={len(specs)} images=200")
    write_text(report_path, render_report(bundle_id, specs, results, docs_path))
    write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
    for spec in specs:
        run_ablation(
            bundle_dir=bundle_dir,
            bundle_id=bundle_id,
            spec=spec,
            args=args,
            docs_path=docs_path,
            report_path=report_path,
            results=results,
            specs=specs,
        )
    n_ok = len([r for r in results.values() if r.get("status") in ("ok", "cached")])
    append_timeline(timeline_path, f"BUNDLE {bundle_id} complete specs={len(specs)} ok={n_ok}")


if __name__ == "__main__":
    main()
