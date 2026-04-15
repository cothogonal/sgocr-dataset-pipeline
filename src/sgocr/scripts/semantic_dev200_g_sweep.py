"""Dev200 G-series sweep: resolvability relaxation + extreme RG bias + coverage push.

Key finding from f-series: 258/470 nodes (55%) are dropped at resolvability
(min_width_patches=1.55). Coverage is stuck at 96-100/200 images because the
resolvability filter is too aggressive. This series attacks that bottleneck.

4 runs:
  g01  resolvability 1.30/0.45, 30 tags, threshold 0.18, aggressive anchors  (full pipeline)
  g02  g01 OCR + extreme mixed bias (0.80/0.55) to fix REVERSE_GROUND near-zero scores
  g03  g01 OCR + tighter ambiguity gate (9/8) to test quality vs coverage trade-off
  g04  consensus 0.65, resolvability 1.20/0.40, threshold 0.15, 30 tags   (wildcard / full pipeline)
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

# F-series best run for reference
F02_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_f_20260406_235248_f02_consensus72_frontier14"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="G-series dev200 sweep: resolvability relaxation + extreme RG bias.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_g_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(bundle_id: str) -> list[AblationSpec]:
    # ----------------------------------------------------------------
    # Shared base: f02 winner foundation — consensus 0.72, frontier RG
    # style, proven merge/detection/anchor settings. Only resolvability
    # and RG bias are varied in this series.
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
        # Consensus — 0.72 from f02 winner
        "SGOCR_STRONG_CONSENSUS_FLOOR": "0.72",
        "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.79",
        "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
        # Resolvability — KEY CHANGE: relaxed from f-series 1.55/0.62
        # f-series dropped 258/470 nodes (55%) here; this is the coverage bottleneck
        "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.30",
        "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.45",
        # Anchor — aggressive expansion to recover more anchors on wider node set
        "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
        "SGOCR_ANCHOR_RELABEL_MODE": "flash",
        "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
        "SGOCR_GENERIC_ANCHOR_PENALTY": "0.10",
        "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.34",
        "SGOCR_FLORENCE_REGION_BONUS": "0.06",
        # Wording
        "SGOCR_LOCATION_WORDING_MODE": "varied",
        "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
        # RG answer style — frontier in all g-series runs
        "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier",
        # Verifier
        "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
        "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
        "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
        "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "4",
        "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.35",
        # Ambiguity — unchanged from f-series baseline
        "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
        "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
        # Local/mixed bias — f-series baseline for g01; g02 pushes extreme
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
    }

    # g01 runs the full pipeline; g02/g03 reuse its OCR cache.
    g01_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / f"{bundle_id}_g01_resolv130_tags30"

    return [
        # ============================================================
        # g01: relaxed resolvability (1.30/0.45), 30 tags, threshold 0.18
        #   Full pipeline — establishes new OCR cache for g02/g03.
        #   Hypothesis: 55% node dropout at resolvability is the main
        #   coverage bottleneck; relaxing to 1.30/0.45 should recover
        #   many of the 258 dropped nodes and push images/200 toward 130+.
        # ============================================================
        AblationSpec(
            "g01_resolv130_tags30",
            "resolvability 1.30/0.45 + 30 tags + threshold 0.18 — attack coverage bottleneck",
            "none",
            cli={
                "--grounding-threshold": "0.18",
                "--max-tags-per-image": "30",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env=dict(base_env),
        ),

        # ============================================================
        # g02: extreme mixed bias (0.80/0.55), same coverage settings
        #   Reuses g01 OCR cache. Hypothesis: REVERSE_GROUND soft=0.059
        #   for gemini3flash because mixed-directional QA pairs are too
        #   rare (bias 0.50/0.30). Pushing to 0.80/0.55 forces many more
        #   mixed questions, which may be genuinely easier to answer since
        #   they avoid strict directional precision.
        # ============================================================
        AblationSpec(
            "g02_extreme_mixed_bias",
            "extreme mixed bias (0.80/0.55) — force more mixed RG questions to fix near-zero soft accuracy",
            "ocr",
            cli={
                "--grounding-threshold": "0.18",
                "--max-tags-per-image": "30",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.80",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.55",
            },
            cache_intermediate_dir=g01_intermediate,
        ),

        # ============================================================
        # g03: tighter ambiguity gate (9/8), same coverage settings
        #   Reuses g01 OCR cache. Hypothesis: current 7/6 gate lets
        #   borderline QA pairs through; raising to 9/8 may reduce noise
        #   and improve eval soft accuracy even if accepted count drops.
        #   Tests whether quality > quantity is the right trade-off.
        # ============================================================
        AblationSpec(
            "g03_ambiguity_gate_98",
            "tighter ambiguity gate (9/8) — quality vs coverage trade-off with relaxed resolvability",
            "ocr",
            cli={
                "--grounding-threshold": "0.18",
                "--max-tags-per-image": "30",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "9",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "8",
            },
            cache_intermediate_dir=g01_intermediate,
        ),

        # ============================================================
        # g04: wildcard — consensus 0.65, resolvability 1.20/0.40, threshold 0.15
        #   Full pipeline, most aggressive coverage push yet. Combines
        #   the lowest consensus tested (0.65), most lenient resolvability
        #   (1.20/0.40), and lowest grounding threshold (0.15) with extreme
        #   mixed bias. Probes the absolute coverage ceiling before quality
        #   collapses. Expected: highest images/200 but possibly lower rate.
        # ============================================================
        AblationSpec(
            "g04_wildcard_max_coverage",
            "wildcard: consensus 0.65 + resolvability 1.20/0.40 + threshold 0.15 — absolute coverage ceiling probe",
            "none",
            cli={
                "--grounding-threshold": "0.15",
                "--max-tags-per-image": "30",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_STRONG_CONSENSUS_FLOOR": "0.65",
                "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.72",
                "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.20",
                "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.40",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.80",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.55",
            },
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
        "# SGOCR Dev200 G-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Design",
        "",
        "4-run sweep attacking the coverage bottleneck discovered in f-series:",
        "- **Root cause**: f02 best run dropped 258/470 nodes (55%) at resolvability filter (min_width=1.55).",
        "- **Coverage stuck at 96-100/200 images** despite lowering consensus to 0.70.",
        "- **REVERSE_GROUND near-zero**: 0.000 exact / 0.059 soft for gemini3flash — mixed bias too weak.",
        "",
        "Hypotheses:",
        "1. **Resolvability relaxation** (1.55→1.30/0.45): recover the 55% node dropout → images/200 > 130.",
        "2. **Extreme mixed bias** (0.50→0.80, 0.30→0.55): more mixed RG questions may be easier for frontier models.",
        "3. **Tighter ambiguity gate** (7/6→9/8): test if quality improves with stricter filtering at relaxed resolvability.",
        "4. **Wildcard** (consensus 0.65, resolvability 1.20/0.40): probe absolute coverage ceiling.",
        "",
        "## F-Series Reference (best run)",
        "",
        "| Run | Accepted | Accept% | Images/200 | Q3 | Rev local | Rev mixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    try:
        m = collect_metrics(F02_DEV200)
        lines.append(f"| `f02_consensus72_frontier14` | {m['accepted_qas']} | {m['qa_accept_rate']:.3f} | {m['images_with_final_rows']}/200 | {m['q3_score']} | {m['reverse_local']} | {m['reverse_mixed']} |")
    except Exception:
        lines.append("| `f02_consensus72_frontier14` | 341 | 0.770 | 96/200 | 80.53 | - | - |")

    lines.extend([
        "",
        "## G-Series Scoreboard",
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
        "- g01/g04 are full pipeline runs (cache_level=none); g02/g03 reuse g01 OCR cache.",
        "- Resolvability relaxed from f-series 1.55/0.62 → 1.30/0.45 (g01-g03) and 1.20/0.40 (g04).",
        "- Extreme mixed bias (0.80/0.55) in g02/g04 targets REVERSE_GROUND near-zero eval scores.",
        "- Ambiguity gate raised to 9/8 in g03 tests quality vs coverage with the new wider node set.",
        "- Anchor expansion mode = aggressive in all g-series runs (f-series was supportive).",
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

    append_timeline(timeline_path, f"START {spec.name} cache={spec.cache_level} model={model_name} consensus={spec.env.get('SGOCR_STRONG_CONSENSUS_FLOOR','?')} resolv_w={spec.env.get('SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES','?')} tags={spec.cli.get('--max-tags-per-image','?')} mixed_bias={spec.env.get('SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS','?')}")
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "126_sgocr_dev200_g_sweep_2026-04-07.md"
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
