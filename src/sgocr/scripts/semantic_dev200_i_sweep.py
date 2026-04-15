"""Dev200 I-series sweep: coverage bisection + quality guards.

Context:
  - h03 (resolvability 1.42 + aggressive anchors) is the best coverage run: 106/200 images, Q3=80.24
  - f02 (resolvability 1.55 + frontier RG) is the best Q3 run: Q3=80.53, 96/200 images
  - Gemini-3-flash RG is now scorable (0.559 soft) with the fixed benchmark prompt
  - rev_mixed bottleneck confirmed at teacher generation, not sampling
  - DR failures are ~67% wrong-region reads; scoring fixes add ~8pp (now at 45%)

Goals:
  1. Push coverage past 110/200 while keeping Q3 ≥ 80 (bisect below 1.42)
  2. Test tighter ambiguity gate at expanded coverage (does it recover Q3?)
  3. Test 30 tags + lower grounding threshold at 1.35 (breadth vs depth)
  4. Wildcard: consensus 0.68 at 1.42 resolvability — max node coverage push

Run matrix:
  i01  resolvability 1.35/0.54 + aggressive anchors, frontier14  (full pipeline)
  i02  resolvability 1.35/0.54 + ambiguity gate 9/8               (reuses i01 OCR cache)
  i03  resolvability 1.35/0.54 + 30 tags + thresh 0.18            (reuses i01 OCR cache)
  i04  consensus 0.68 + resolvability 1.42/0.56 + aggressive      (full pipeline — wildcard)
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
F02_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_f_20260406_235248_f02_consensus72_frontier14"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="I-series dev200 sweep: coverage bisection + quality guards.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_i_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(bundle_id: str) -> list[AblationSpec]:
    # ----------------------------------------------------------------
    # Base: h03 winner foundation
    #   - consensus 0.72 (recovered ~11 images over e-series)
    #   - resolvability 1.42/0.56 (h03 bisect — recovered ~10 more images)
    #   - aggressive anchors: penalty 0.04, expansion=aggressive
    #   - frontier RG answer style
    #   - 14 tags, grounding threshold 0.28
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
        # Consensus — 0.72 from f-series (best coverage/quality trade-off)
        "SGOCR_STRONG_CONSENSUS_FLOOR": "0.72",
        "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.79",
        "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
        # Resolvability — h03 bisect point (1.42/0.56); i01/i02/i03 push to 1.35/0.54
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
        # RG answer style: frontier throughout
        "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier",
        # Verifier
        "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
        "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
        "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
        "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "4",
        "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.35",
        # Ambiguity — h03 baseline 7/6
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
    }

    # i01 runs the full pipeline; i02/i03 reuse its OCR cache.
    i01_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / f"{bundle_id}_i01_resolvability135_aggressive"

    return [
        # ============================================================
        # i01: resolvability 1.35/0.54 + h03 aggressive anchors
        #   Bisects below h03's 1.42. Expected: 110-115/200 images.
        #   Full pipeline run — establishes new OCR cache for i02/i03.
        #   Quality guard: aggressive anchor penalty (0.04) keeps accept_rate up.
        # ============================================================
        AblationSpec(
            "i01_resolvability135_aggressive",
            "resolvability 1.35/0.54 + aggressive anchors — coverage push",
            "none",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.35",
                "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.54",
            },
        ),

        # ============================================================
        # i02: resolvability 1.35 + tighter ambiguity gate (9/8)
        #   At lower resolvability, more marginal text nodes enter the
        #   pipeline. A tighter ambiguity gate (7→9, 6→8) should filter
        #   out the weakest new additions and keep accept_rate high.
        #   Reuses i01 OCR cache.
        # ============================================================
        AblationSpec(
            "i02_resolvability135_ambiguity98",
            "resolvability 1.35 + ambiguity gate 9/8 — quality guard at expanded coverage",
            "ocr",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.35",
                "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.54",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "9",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "8",
            },
            cache_intermediate_dir=i01_intermediate,
        ),

        # ============================================================
        # i03: resolvability 1.35 + 30 tags + grounding threshold 0.18
        #   Tests whether more anchor breadth (30 tags vs 14) + lower
        #   grounding score threshold produces more diverse, high-quality
        #   anchors for the extra images unlocked at 1.35 resolvability.
        #   Reuses i01 OCR cache.
        # ============================================================
        AblationSpec(
            "i03_resolvability135_tags30_thresh018",
            "resolvability 1.35 + 30 tags + threshold 0.18 — anchor breadth at lower coverage floor",
            "ocr",
            cli={
                "--grounding-threshold": "0.18",
                "--max-tags-per-image": "30",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.35",
                "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.54",
            },
            cache_intermediate_dir=i01_intermediate,
        ),

        # ============================================================
        # i04: consensus 0.68 + resolvability 1.42 + aggressive anchors
        #   Wildcard: h03 settings but consensus pushed further down
        #   (0.72→0.68). Recovers more text nodes at the merge step.
        #   Full pipeline run — measures the node-formation floor.
        # ============================================================
        AblationSpec(
            "i04_consensus68_resolvability142",
            "consensus 0.68 + resolvability 1.42 — max node-formation push",
            "none",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_STRONG_CONSENSUS_FLOOR": "0.68",
                "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.75",
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
        "# SGOCR Dev200 I-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Design",
        "",
        "4-run sweep building on h03 (best coverage: 106/200, Q3=80.24) and f02 (best Q3: 80.53, 96/200).",
        "Goal: push coverage past 110/200 while keeping Q3 ≥ 80.",
        "",
        "Hypotheses:",
        "1. **Coverage bisection** (i01): resolvability 1.35/0.54 recovers 5-10 more images beyond h03's 1.42.",
        "2. **Quality guard** (i02): ambiguity gate 9/8 filters marginal new nodes at 1.35 — tests quality vs coverage trade-off.",
        "3. **Anchor breadth** (i03): 30 tags + threshold 0.18 finds better anchors for the new 1.35 images.",
        "4. **Node floor** (i04): consensus 0.68 at h03 resolvability — how far can we push node-formation recovery?",
        "",
        "## Prior Best Runs",
        "",
        "| Run | Q3 | Accepted | Accept% | Images/200 | Rev local | Rev mixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, prior_dir in [("h03_aggressive_anchors", H03_DEV200), ("f02_consensus72_frontier14", F02_DEV200)]:
        try:
            m = collect_metrics(prior_dir)
            lines.append(f"| `{label}` | {m['q3_score']} | {m['accepted_qas']} | {m['qa_accept_rate']:.3f} | {m['images_with_final_rows']}/200 | {m['reverse_local']} | {m['reverse_mixed']} |")
        except Exception:
            lines.append(f"| `{label}` | - | - | - | - | - | - |")

    lines.extend([
        "",
        "## I-Series Scoreboard",
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
        "- i01/i04 are full pipeline runs (cache_level=none); i02/i03 reuse i01 OCR cache.",
        "- h03 base: consensus 0.72, resolvability 1.42/0.56, aggressive anchors (penalty=0.04), frontier RG.",
        "- i01-i03 bisect resolvability to 1.35/0.54; i04 bisects consensus to 0.68 instead.",
        "- Eval with fixed gemini-3-flash RG prompt (coordinates suppressed) expected RG soft ~0.50-0.60.",
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

    append_timeline(timeline_path, f"START {spec.name} cache={spec.cache_level} model={model_name} consensus={spec.env.get('SGOCR_STRONG_CONSENSUS_FLOOR','?')} resolvability={spec.env.get('SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES','?')} rg_style={spec.env.get('SGOCR_REVERSE_GROUND_ANSWER_STYLE','standard')} tags={spec.cli.get('--max-tags-per-image','?')}")
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "128_sgocr_dev200_i_sweep_2026-04-07.md"
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
