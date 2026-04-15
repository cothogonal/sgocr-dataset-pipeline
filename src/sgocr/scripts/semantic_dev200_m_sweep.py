"""Dev200 M-series sweep: precision-first quality optimization after L-series eval readout.

L-series conclusions that drive this sweep:
  - Q3 and frontier-eval quality rank in opposite order; Q3 is no longer the objective.
  - l04 is the eval Pareto frontier: best both-correct / soft accuracy despite poor Q3.
  - DR ambiguity reject is the dominant quality lever and should stay on.
  - Mixed-bias 0.65 is reliable and should stay on.
  - SAM3 broad hurts anchor quality under mixed-bias and should be avoided.

M-series objective:
  precision_first_score = mean_inline_frontier_correct * sqrt(accepted_rows)

All four runs reuse h03 OCR cache and use the newly landed code changes:
  - color-preserving anchor labels / relabel prompts
  - unique-anchor location suppression for DR / YES_NO
  - merge left-to-right ordering fix
  - inline frontier scoring on accepted rows
  - anchor monotony soft-cap support
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
H03_INTERM = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev200_h_20260407_055403_h03_aggressive_anchors"
L03_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_l_20260407_191137_l03_mixed65_only"
L04_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_l_20260407_191137_l04_full_quality_stack"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="M-series dev200 sweep: precision-first quality objective with inline frontier scoring.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_m_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(_bundle_id: str) -> list[AblationSpec]:
    # l04 quality stack + targeted SAM3 is the new default foundation.
    # The only question is how hard to push selection pressure and row count
    # under the precision-first objective.
    quality_base_env = {
        "SGOCR_TEXT_MERGE_ENABLED": "1",
        "SGOCR_TEXT_MERGE_ANCHOR_OVERLAP_MIN": "0.08",
        "SGOCR_TEXT_MERGE_X_OVERLAP_MIN": "0.42",
        "SGOCR_TEXT_MERGE_Y_OVERLAP_MIN": "0.32",
        "SGOCR_TEXT_MERGE_GAP_RATIO_MAX": "2.0",
        "SGOCR_TEXT_MERGE_HEIGHT_RATIO_MAX": "2.2",
        "SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES": "3",
        "SGOCR_DETECTOR_BOX_THRESH": "0.35",
        "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
        "SGOCR_OCR_CROP_PAD_RATIO": "0.10",
        "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.08",
        "SGOCR_STRONG_CONSENSUS_FLOOR": "0.72",
        "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.79",
        "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
        "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.42",
        "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.56",
        "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
        "SGOCR_ANCHOR_RELABEL_MODE": "flash",
        "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
        "SGOCR_GENERIC_ANCHOR_PENALTY": "0.04",
        "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.34",
        "SGOCR_FLORENCE_REGION_BONUS": "0.06",
        "SGOCR_LOCATION_WORDING_MODE": "varied",
        "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
        "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier",
        "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
        "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
        "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
        "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "4",
        "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.35",
        "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "6",
        "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
        "SGOCR_DR_AMBIGUITY_REJECT_SCORE": "5",
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.65",
        "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.35",
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.65",
        "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.35",
        "SGOCR_SAM3_REFINE_MODE": "top3",
        "SGOCR_SAM3_APPLY_MODE": "targeted",
        "SGOCR_SAM3_TOPK_PROMPTS": "3",
        "SGOCR_SAM3_CONFIDENCE_THRESHOLD": "0.35",
        "SGOCR_SAM3_BOX_THRESHOLD": "0.30",
        "SGOCR_SAM3_RELEVANCE_BONUS": "0.10",
        "SGOCR_SAM3_TARGET_SUPPORT_MAX": "1",
        "SGOCR_SAM3_TARGET_CLUSTER_MIN": "2",
        "SGOCR_SAM3_TARGET_AREA_START": "0.34",
        "SGOCR_TEACHER_STRICTNESS": "strict",
        "SGOCR_INLINE_FRONTIER_ENABLED": "1",
        "SGOCR_INLINE_FRONTIER_MODEL": "gemini-3-flash-preview",
    }

    return [
        AblationSpec(
            "m01_quality_base_nocap",
            "l04 quality stack + new code path, diversity cap disabled for clean baseline",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**quality_base_env, "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "0"},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "m02_quality_base_cap",
            "m01 + anchor monotony soft-cap (sign/wall/board after two picks)",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={
                **quality_base_env,
                "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "2",
                "SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY": "0.15",
            },
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "m03_relaxed_global_cap",
            "m02 but relax global hard ambiguity gate 6 -> 7 to recover rows under the new objective",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={
                **quality_base_env,
                "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "2",
                "SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY": "0.15",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
            },
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "m04_quality_cap_target4",
            "m02 but target only 4 rows/image; tests whether lower-yield higher-purity selection wins precision-first score",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "4", "--max-detections": "128"},
            env={
                **quality_base_env,
                "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "2",
                "SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY": "0.15",
            },
            cache_intermediate_dir=H03_INTERM,
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
        "# SGOCR Dev200 M-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        "- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Design",
        "",
        "4-run sweep rebuilt after the L-series eval report invalidated Q3 as the quality objective.",
        "All runs reuse h03 OCR cache and use targeted SAM3 only. SAM3 broad is intentionally excluded.",
        "Primary objective: `precision_first_score = mean_inline_frontier_correct * sqrt(accepted_rows)`.",
        "",
        "L-series feedback incorporated:",
        "1. `l04` is the eval Pareto frontier, so the quality stack is now the baseline.",
        "2. DR ambiguity reject stays on in every run.",
        "3. Mixed-bias 0.65 stays on in every run.",
        "4. SAM3 broad is removed entirely.",
        "5. New code-path improvements are active in every run: color-bearing anchors, unique-anchor location suppression, inline frontier scoring.",
        "",
        "M-series ablations:",
        "1. **m01**: l04-quality baseline, diversity cap disabled",
        "2. **m02**: m01 + anchor monotony soft-cap",
        "3. **m03**: m02 + relaxed global hard ambiguity gate (6→7)",
        "4. **m04**: m02 + target-per-image 4 instead of 5",
        "",
        "## Prior Eval Frontier",
        "",
        "| Run | Notes | Accepted | Q3 | Both-correct |",
        "|---|---|---:|---:|---:|",
        "| `l04_full_quality_stack` | eval-best L run | 269 | 70.14 | 55.4% |",
        "| `l03_mixed65_only` | Q3-best L run | 362 | 76.86 | 47.9% |",
        "",
        "## M-Series Scoreboard",
        "",
        "| Run | Status | Sweep | Inline mean | Accepted | Images/200 | Q3 | Ambig | Anchor miss | Diversity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in specs:
        row = results.get(spec.name)
        if not row:
            lines.append(f"| `{spec.name}` | pending | - | - | - | - | - | - | - | - |")
            continue
        if row.get("status") not in {"ok", "cached"}:
            lines.append(f"| `{spec.name}` | {row.get('status')} | - | - | - | - | - | - | - | - |")
            continue
        m = row["metrics"]
        lines.append(
            f"| `{spec.name}` | {row.get('status')}"
            f" | {m['sweep_score']:.4f}"
            f" | {m['inline_frontier_mean']:.4f}"
            f" | {m['accepted_qas']}"
            f" | {m['images_with_final_rows']}/200"
            f" | {m['q3_score']}"
            f" | {m['ambiguity_high']}"
            f" | {m['anchor_missing']}"
            f" | {m['type_diversity']:.3f} |"
        )

    complete = [row["metrics"] | {"name": name} for name, row in results.items() if row.get("status") in {"ok", "cached"}]
    if complete:
        ranked = sorted(complete, key=lambda r: (-r["sweep_score"], -r["inline_frontier_mean"], -r["accepted_qas"], r["name"]))
        lines.extend(["", "## Rankings", ""])
        for idx, row in enumerate(ranked, 1):
            lines.append(
                f"{idx}. `{row['name']}`: sweep=`{row['sweep_score']:.4f}`"
                f", inline=`{row['inline_frontier_mean']:.4f}`"
                f", accepted=`{row['accepted_qas']}`"
                f", images=`{row['images_with_final_rows']}/200`"
                f", q3=`{row['q3_score']}`"
            )

    lines.extend([
        "",
        "## Notes",
        "",
        "- All M-series runs reuse h03 OCR cache (`cache_level=ocr`).",
        "- Ranking is by `sweep_score`, not Q3.",
        "- Inline frontier scoring is enabled in every run via `gemini-3-flash-preview`.",
        "- Color-bearing anchors and unique-anchor phrasing changes are code-level defaults, not per-run env toggles.",
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
        append_timeline(timeline_path, f"END {spec.name} cached accepted={metrics['accepted_qas']} sweep={metrics['sweep_score']:.4f}")
        write_text(report_path, render_report(bundle_id, specs, results, docs_path))
        write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
        return

    cache_intermediate_dir = spec.cache_intermediate_dir or intermediate_dir
    model_name = spec.model_override or args.model
    target = spec.cli.get("--target-per-image", str(args.target_per_image))
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
        "--target-per-image", target,
        "--max-detections", spec.cli.get("--max-detections", "128"),
        "--grounding-threshold", spec.cli.get("--grounding-threshold", "0.28"),
        "--max-tags-per-image", spec.cli.get("--max-tags-per-image", "14"),
    ]
    env = os.environ.copy()
    env.update(spec.env)

    append_timeline(
        timeline_path,
        f"START {spec.name} cache={spec.cache_level} target={target}"
        f" hard_ambi={spec.env.get('SGOCR_AMBIGUITY_HARD_REJECT_SCORE','?')}"
        f" dr_ambi={spec.env.get('SGOCR_DR_AMBIGUITY_REJECT_SCORE','?')}"
        f" cap_count={spec.env.get('SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT','default')}"
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
        append_timeline(
            timeline_path,
            f"END {spec.name} ok accepted={metrics['accepted_qas']} images={metrics['images_with_final_rows']}"
            f" inline={metrics['inline_frontier_mean']:.4f} sweep={metrics['sweep_score']:.4f}"
        )

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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "134_sgocr_dev200_m_sweep_2026-04-07.md"
    results: dict[str, dict[str, Any]] = {}
    append_timeline(timeline_path, f"BUNDLE {bundle_id} start specs={len(specs)} images=200 cache=h03_ocr objective=precision_first_score")
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
