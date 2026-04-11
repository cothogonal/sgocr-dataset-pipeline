"""Dev200 F-series sweep: consensus recovery + frontier RG answer style.

4 runs probing two orthogonal changes:
  1. Lower strong_consensus_floor (0.76→0.72 / 0.70) to recover images lost at node-formation.
  2. SGOCR_REVERSE_GROUND_ANSWER_STYLE=frontier — shorter, redundancy-resistant gold answers
     that better mimic the terse natural-language answers frontier eval models produce.

Run matrix:
  f01  consensus 0.72, standard style, 14 tags  (full pipeline — establishes new OCR cache)
  f02  consensus 0.72, frontier style, 14 tags  (reuses f01 OCR cache)
  f03  consensus 0.72, frontier style, 22 tags  (reuses f01 OCR cache — d03 tag count won Q3)
  f04  consensus 0.70, frontier style, 22 tags  (full pipeline — push coverage floor lower)
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

# Prior best dev200 run for comparison
E02_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_20260406_200346_e02_d03_dev200"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="F-series dev200 sweep: consensus + frontier RG style.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_f_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(bundle_id: str) -> list[AblationSpec]:
    # ----------------------------------------------------------------
    # Shared base: proven c06/d03 foundation (varied wording, extreme
    # local bias, Flash teacher, merge enabled, SAM3 targeted).
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
        # Consensus — lowered from 0.76 (prior best) to 0.72 to recover node-formation drops.
        # standard_consensus_floor also nudged down proportionally.
        "SGOCR_STRONG_CONSENSUS_FLOOR": "0.72",
        "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.79",
        "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
        # Resolvability (unchanged from c06 winner)
        "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.55",
        "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.62",
        # Anchor
        "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
        "SGOCR_ANCHOR_RELABEL_MODE": "flash",
        "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
        "SGOCR_GENERIC_ANCHOR_PENALTY": "0.10",
        "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.34",
        "SGOCR_FLORENCE_REGION_BONUS": "0.06",
        # Wording
        "SGOCR_LOCATION_WORDING_MODE": "varied",
        "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
        # RG answer style — standard in f01, overridden to frontier in f02-f04
        "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "standard",
        # Verifier
        "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
        "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
        "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
        "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "4",
        "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.35",
        # Ambiguity
        "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
        "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
        # Local bias: extreme
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

    # f01 runs the full pipeline; f02/f03 reuse its OCR cache.
    f01_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / f"{bundle_id}_f01_consensus72_standard"

    return [
        # ============================================================
        # f01: consensus 0.72, standard RG style, 14 tags
        #   Isolation run: does lower consensus floor alone improve
        #   image coverage without hurting QA quality?
        #   Full pipeline run — establishes the new OCR cache for f02/f03.
        # ============================================================
        AblationSpec(
            "f01_consensus72_standard",
            "consensus 0.72 (standard RG style, 14 tags) — coverage baseline",
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
        # f02: consensus 0.72, frontier RG style, 14 tags
        #   Adds frontier RG answer style on top of f01's coverage gain.
        #   Gold answers become shorter and more natural — hopefully
        #   closing the gap between our template and frontier model output.
        #   Reuses f01 OCR cache (same consensus threshold).
        # ============================================================
        AblationSpec(
            "f02_consensus72_frontier14",
            "consensus 0.72 + frontier RG answers (14 tags) — redundancy-resistant",
            "ocr",
            cli={
                "--grounding-threshold": "0.28",
                "--max-tags-per-image": "14",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier",
            },
            cache_intermediate_dir=f01_intermediate,
        ),

        # ============================================================
        # f03: consensus 0.72, frontier RG style, 22 tags
        #   22 tags gave best Q3 in d03 and d-series. Combines the
        #   two most impactful changes (lower consensus + frontier style)
        #   with higher tag count for anchor coverage breadth.
        #   Reuses f01 OCR cache.
        # ============================================================
        AblationSpec(
            "f03_consensus72_frontier22",
            "consensus 0.72 + frontier RG answers (22 tags) — full stack best guess",
            "ocr",
            cli={
                "--grounding-threshold": "0.22",
                "--max-tags-per-image": "22",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier",
            },
            cache_intermediate_dir=f01_intermediate,
        ),

        # ============================================================
        # f04: consensus 0.70, frontier RG style, 22 tags
        #   Full pipeline run at even lower consensus (0.70) to see
        #   how far we can push node-formation recovery before QA
        #   quality degrades. 22 tags + frontier style stacked.
        # ============================================================
        AblationSpec(
            "f04_consensus70_frontier22",
            "consensus 0.70 (push lower) + frontier RG answers (22 tags) — max coverage test",
            "none",
            cli={
                "--grounding-threshold": "0.22",
                "--max-tags-per-image": "22",
                "--target-per-image": "5",
                "--max-detections": "128",
            },
            env={
                **base_env,
                "SGOCR_STRONG_CONSENSUS_FLOOR": "0.70",
                "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.77",
                "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier",
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
        "# SGOCR Dev200 F-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Design",
        "",
        "4-run sweep testing two orthogonal improvements over the e-series winner (e02, Q3=75.8):",
        "1. **Consensus recovery**: `strong_consensus_floor` 0.76→0.72/0.70 to recover the ~58-image node-formation gap.",
        "2. **Frontier RG style**: shorter, redundancy-resistant REVERSE_GROUND gold answers that better match natural model output.",
        "",
        "## E-Series Reference",
        "",
        "| Run | Accepted | Accept% | Images/200 | Q3 | Rev local | Rev mixed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    try:
        m = collect_metrics(E02_DEV200)
        lines.append(f"| `e02_d03_dev200` | {m['accepted_qas']} | {m['qa_accept_rate']:.3f} | {m['images_with_final_rows']}/200 | {m['q3_score']} | {m['reverse_local']} | {m['reverse_mixed']} |")
    except Exception:
        lines.append("| `e02_d03_dev200` | - | - | - | - | - | - |")

    lines.extend([
        "",
        "## F-Series Scoreboard",
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
        "- f01/f04 are full pipeline runs (cache_level=none); f02/f03 reuse f01 OCR cache.",
        "- frontier RG style: shorter gold answers, redundancy-resistant, unique-anchor shortening.",
        "- consensus 0.72 vs 0.70 isolates how low we can push before QA degrades.",
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

    append_timeline(timeline_path, f"START {spec.name} cache={spec.cache_level} model={model_name} consensus={spec.env.get('SGOCR_STRONG_CONSENSUS_FLOOR','?')} rg_style={spec.env.get('SGOCR_REVERSE_GROUND_ANSWER_STYLE','standard')} tags={spec.cli.get('--max-tags-per-image','?')}")
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "125_sgocr_dev200_f_sweep_2026-04-07.md"
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
