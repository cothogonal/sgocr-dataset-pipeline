"""Dev200 H-series sweep: resolvability bisection + confounder isolation.

Key findings from g-series:
- g01 changed 4 things simultaneously vs f02: resolvability (1.55→1.30), tags (14→30),
  threshold (0.28→0.18), and anchor mode (supportive→aggressive). Q3 dropped 7 pts (80→73).
- g03 (ambiguity gate 9/8) partially recovered quality (Q3 67→73) within g-series.
- g04 wildcard confirmed coverage ceiling: 125/200 at 1.20/0.40 but Q3=61.61 (too low).
- REVERSE_GROUND soft accuracy remains broken across all series (0.032-0.353 range).
- Recommended: bisect resolvability at 1.42/0.53 and isolate which g-series changes hurt quality.

4 runs probing:
  h01  resolvability 1.42/0.53, supportive anchors, ambiguity 9/8, 14 tags, thresh 0.28
       (full pipeline — clean bisection with f02's quality-preserving settings)
  h02  h01 OCR + 22 tags + threshold 0.22  (f03 tag/threshold combo at bisect resolvability)
  h03  h01 OCR + aggressive anchors + generic penalty 0.04  (isolate anchor mode as quality lever)
  h04  h01 OCR + very tight ambiguity gate 10/9               (wildcard: approach f02 Q3 with coverage gain)
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

from .paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, LOGS_ROOT, REPO_ROOT
from .run_quality import compute_run_quality
from .semantic_dev40_sweep_v2 import AblationSpec, append_timeline, gpu_snapshot, now_stamp, write_text


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "dev200_source_universe"

# G-series best run for reference
G03_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_g_20260407_020133_g03_ambiguity_gate_98"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="H-series dev200 sweep: resolvability bisection + confounder isolation.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_h_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(bundle_id: str) -> list[AblationSpec]:
    # ----------------------------------------------------------------
    # Shared base: f02 quality foundation PLUS g03's ambiguity gate.
    # The single new structural change is resolvability bisected to
    # 1.42/0.53 (midpoint between f02's 1.55/0.62 and g03's 1.30/0.45).
    # Anchor mode kept at supportive (f02 baseline) so that h01 isolates
    # resolvability alone, and h03 can test anchor mode independently.
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
        # Resolvability — KEY CHANGE: bisected between f02 (1.55/0.62) and g01 (1.30/0.45)
        # Midpoint: 1.42/0.53. Expect ~105/200 images and Q3 ~77 if the Pareto is smooth.
        "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.42",
        "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.53",
        # Anchor — supportive (f02 baseline). g-series used aggressive for all runs,
        # which may have contributed to the Q3 drop. h01 holds this at supportive
        # so h03 can test aggressive in isolation.
        "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
        "SGOCR_ANCHOR_RELABEL_MODE": "flash",
        "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
        "SGOCR_GENERIC_ANCHOR_PENALTY": "0.10",
        "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.34",
        "SGOCR_FLORENCE_REGION_BONUS": "0.06",
        # Wording
        "SGOCR_LOCATION_WORDING_MODE": "varied",
        "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
        # RG answer style — frontier in all h-series runs (g03/f02 proven better than standard)
        "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier",
        # Verifier
        "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
        "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
        "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
        "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "4",
        "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.35",
        # Ambiguity gate — raised to 9/8 (g03's improvement over 7/6 baseline).
        # g03 showed this recovers Q3 by ~6 pts within g-series without hurting coverage.
        "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "9",
        "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "8",
        # Local bias — f02 baseline (mixed bias not varied; g02 showed extreme bias had no effect)
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

    # h01 runs the full pipeline; h02/h03/h04 reuse its OCR cache.
    h01_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / f"{bundle_id}_h01_resolv142_sweet_spot"

    return [
        # ============================================================
        # h01: resolvability 1.42/0.53, supportive anchors, ambiguity 9/8, 14 tags
        #   Full pipeline — cleanest possible bisection test.
        #   ONLY changes from f02: resolvability to 1.42 + ambiguity gate 9/8.
        #   Anchor mode = supportive (f02), tags = 14 (f02), threshold = 0.28 (f02).
        #   If this recovers ~105 images at Q3 ~77, we have the sweet spot.
        #   Establishes OCR cache for h02/h03/h04.
        # ============================================================
        AblationSpec(
            "h01_resolv142_sweet_spot",
            "resolvability 1.42/0.53 + ambiguity 9/8 + supportive anchors — clean bisection with f02 quality settings",
            "none",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env=dict(base_env),
        ),

        # ============================================================
        # h02: h01 OCR + 22 tags + threshold 0.22
        #   Reuses h01 OCR cache. Tests whether the f03 tag/threshold
        #   combo (22 tags, threshold 0.22) that previously outperformed
        #   14 tags in f-series also lifts quality at bisect resolvability.
        #   22 tags = more anchor diversity; 0.22 = slightly more permissive
        #   grounding. Both are downstream of OCR so safe to cache.
        # ============================================================
        AblationSpec(
            "h02_resolv142_tags22",
            "h01 OCR + 22 tags + threshold 0.22 — f03 tag/threshold combo at bisect resolvability",
            "ocr",
            cli={
                "--grounding-threshold": "0.22",
                "--max-tags-per-image": "22",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env=dict(base_env),
            cache_intermediate_dir=h01_intermediate,
        ),

        # ============================================================
        # h03: h01 OCR + aggressive anchor expansion + generic penalty 0.04
        #   Reuses h01 OCR cache. Isolates anchor mode as a quality lever.
        #   g-series used aggressive expansion throughout — if h03 drops
        #   Q3 vs h01, we know anchor mode (not resolvability) caused g's
        #   quality regression. penalty 0.04 (vs 0.10) makes generic anchors
        #   more competitive, tested briefly in e-series with mixed results.
        # ============================================================
        AblationSpec(
            "h03_aggressive_anchors",
            "h01 OCR + aggressive anchor expansion + penalty 0.04 — isolate anchor mode quality impact",
            "ocr",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.04",
            },
            cache_intermediate_dir=h01_intermediate,
        ),

        # ============================================================
        # h04: h01 OCR + very tight ambiguity gate 10/9    [wildcard]
        #   Reuses h01 OCR cache. Probes whether an even stricter quality
        #   gate approaches f02's Q3=80.53 while maintaining the coverage
        #   gain from resolvability 1.42. If yes, defines the new Pareto
        #   frontier: better quality AND more coverage than f02.
        #   Combines 22 tags (h02 variant) for maximum information gain.
        # ============================================================
        AblationSpec(
            "h04_tight_gate_1009",
            "wildcard: ambiguity gate 10/9 + 22 tags — push toward f02 Q3 quality at bisect coverage",
            "ocr",
            cli={
                "--grounding-threshold": "0.22",
                "--max-tags-per-image": "22",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "10",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "9",
            },
            cache_intermediate_dir=h01_intermediate,
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
        "# SGOCR Dev200 H-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Design",
        "",
        "4-run sweep bisecting the quality/coverage Pareto frontier discovered in f/g-series:",
        "- **f02 best**: resolvability 1.55/0.62 → Q3=80.53, 96/200 images",
        "- **g03 best**: resolvability 1.30/0.45 → Q3=73.42, 115/200 images",
        "- **h-series target**: resolvability 1.42/0.53 (midpoint) → Q3 ~77, ~105/200 images",
        "",
        "Hypotheses:",
        "1. **Resolvability bisection** (h01): sweet spot at 1.42/0.53 recovers ~9 images vs f02 at Q3 ~77.",
        "2. **Tag/threshold combo** (h02): 22 tags + threshold 0.22 improves Q3 at bisect resolvability.",
        "3. **Anchor mode isolation** (h03): aggressive expansion explains g-series quality drop vs f02.",
        "4. **Tighter gate wildcard** (h04): ambiguity 10/9 pushes Q3 above f02 baseline with coverage gain.",
        "",
        "## G-Series Reference (best run)",
        "",
        "| Run | Accepted | Accept% | Images/200 | Q3 | Rev local | Rev mixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    try:
        m = collect_metrics(G03_DEV200)
        lines.append(f"| `g03_ambiguity_gate_98` | {m['accepted_qas']} | {m['qa_accept_rate']:.3f} | {m['images_with_final_rows']}/200 | {m['q3_score']} | {m['reverse_local']} | {m['reverse_mixed']} |")
    except Exception:
        lines.append("| `g03_ambiguity_gate_98` | 404 | 0.786 | 115/200 | 73.42 | - | - |")

    lines.extend([
        "",
        "## H-Series Scoreboard",
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
        "- h01 is a full pipeline run (cache_level=none); h02/h03/h04 reuse h01 OCR cache.",
        "- Resolvability bisected at 1.42/0.53 (between f02's 1.55/0.62 and g01's 1.30/0.45).",
        "- h03 tests aggressive anchor expansion in isolation — g-series confounded this with resolvability change.",
        "- Ambiguity gate 9/8 in base (g03's proven improvement); h04 pushes to 10/9 as wildcard.",
        "- Anchor mode = supportive in h01/h02/h04; aggressive only in h03 for isolation.",
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

    append_timeline(timeline_path, f"START {spec.name} cache={spec.cache_level} model={model_name} resolv_w={spec.env.get('SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES','?')} ambig={spec.env.get('SGOCR_AMBIGUITY_HARD_REJECT_SCORE','?')}/{spec.env.get('SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE','?')} anchor={spec.env.get('SGOCR_ANCHOR_PROMPT_EXPANSION_MODE','?')} tags={spec.cli.get('--max-tags-per-image','?')}")
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "127_sgocr_dev200_h_sweep_2026-04-07.md"
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
