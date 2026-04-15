"""Dev200 O-series sweep: finalize the precision-first line after the completed N-series readout.

O-series is built from the now-valid N-series evidence:
  - n02 is the current leader: sweep 11.5118, inline 0.6831, accepted 284.
  - n03 has the best inline mean (0.6895) but gives back enough rows to finish second.
  - n04 shows tighter grounding threshold is not the way forward.
  - Standard frontier evals on n02 are real now:
      gemini-3-flash soft 0.6312
      gpt-5.3-codex soft 0.6056
      both-soft overlap 0.4965
  - The remaining content debt is still generic anchor dominance, especially text_container
    anchors like sign wall and display panel.

This sweep keeps the stabilized pipeline fixed and spends all four runs on one question:
  can we push anti-generic anchor pressure harder without giving back the n02 gains in
  reverse-ground quality and total accepted rows?
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
from ..secrets import GEMINI, missing_secret_env_vars
from .semantic_dev40_sweep_v2 import AblationSpec, append_timeline, gpu_snapshot, now_stamp, write_text


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "dev200_source_universe"
H03_INTERM = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev200_h_20260407_055403_h03_aggressive_anchors"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="O-series dev200 sweep: precision-first row recovery vs anchor specificity.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_o_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(_bundle_id: str) -> list[AblationSpec]:
    precision_base_env = {
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
        "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
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
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "2",
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY": "0.15",
    }

    return [
        AblationSpec(
            "o01_global7_baseline",
            "n02 carried forward unchanged; current valid precision-first leader",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**precision_base_env},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "o02_global7_count1",
            "n03 idea retested on the winning global7 base: push soft-cap count from 2 to 1",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**precision_base_env, "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "1"},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "o03_global7_generic06",
            "baseline + stronger generic-anchor penalty to directly suppress sign wall / display panel style anchors",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**precision_base_env, "SGOCR_GENERIC_ANCHOR_PENALTY": "0.06"},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "o04_global7_count1_generic06",
            "wildcard: combine the count1 pressure with generic penalty 0.06 on the n02 foundation",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={
                **precision_base_env,
                "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "1",
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.06",
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
    metrics = compute_run_quality(summary, rows)
    metrics["objective_ready"] = metrics["inline_frontier_scored"] > 0
    metrics["sweep_score_source"] = "precision_first_score" if metrics["objective_ready"] else "q3_fallback"
    return metrics


def render_report(bundle_id: str, specs: list[AblationSpec], results: dict[str, dict[str, Any]], docs_path: Path) -> str:
    lines = [
        "# SGOCR Dev200 O-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        "- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## What I Found",
        "",
        "- The completed N-series made the relaxed global ambiguity gate real rather than hypothetical: `n02_quality_cap_global7_retry` is the current best valid run at sweep=`11.5118`, inline=`0.6831`, accepted=`284`.",
        "- `n03_quality_cap_count1` has the best inline mean (`0.6895`) but gives back enough rows to finish behind `n02`.",
        "- `n04_quality_cap_thresh30` lost both external quality and precision-first score, so tighter grounding threshold is not the right finalization path.",
        "- Standard frontier evals on `n02` landed cleanly: Gemini soft=`0.6312`, GPT soft=`0.6056`, both-soft overlap=`0.4965`.",
        "- Reverse-ground is now the strongest externally validated open strength in the bundle rather than the weakness: soft=`0.6780` on Gemini, `0.6271` on GPT.",
        "- The remaining debt is generic text-container anchoring. `sign wall` and `display panel` still dominate accepted rows even after the current soft cap.",
        "",
        "## Why These Runs Exist",
        "",
        "O-series keeps the stabilized N-series pipeline fixed and spends the whole sweep on anchor specificity. The question now is not whether to keep global gate 7; that already won. The question is whether stronger anti-generic-anchor pressure can beat `n02` without giving back reverse-ground quality or enough rows to lose the sqrt(rows) objective.",
        "",
        "## Hypotheses Tested",
        "",
        "1. **o01**: `n02` is the real baseline and should be reproducible.",
        "2. **o02**: count1 may become a winner once it rides on the better global7 base rather than the older gate6 base.",
        "3. **o03**: a direct generic-anchor penalty increase may suppress `sign wall` style anchors more efficiently than changing the cap threshold.",
        "4. **o04**: the count1 + generic-penalty combination may finally move text-container dominance enough to outweigh any row loss.",
        "",
        "## Prior Evidence",
        "",
        "| Run | Objective-valid | Sweep | Inline mean | Accepted | Images/200 | Q3 | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        "| `n01_quality_cap_baseline` | yes | 11.3765 | 0.6848 | 276 | 89/200 | 71.01 | gate6 baseline rerun |",
        "| `n02_quality_cap_global7_retry` | yes | **11.5118** | 0.6831 | **284** | 87/200 | **72.11** | new leader; global7 wins |",
        "| `n03_quality_cap_count1` | yes | 11.4761 | **0.6895** | 277 | 85/200 | 68.45 | best mean, not best total |",
        "| `n04_quality_cap_thresh30` | yes | 10.7745 | 0.6486 | 276 | 88/200 | 70.34 | threshold tightening loses |",
        "",
        "## Scoreboard",
        "",
        "| Run | Status | Objective-valid | Sweep | Inline mean | Accepted | Images/200 | Q3 | High ambig | Anchor miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in specs:
        row = results.get(spec.name)
        if not row:
            lines.append(f"| `{spec.name}` | pending | - | - | - | - | - | - | - | - |")
            continue
        if row.get("status") not in {"ok", "cached", "invalid_objective"}:
            lines.append(f"| `{spec.name}` | {row.get('status')} | - | - | - | - | - | - | - | - |")
            continue
        m = row["metrics"]
        lines.append(
            f"| `{spec.name}` | {row.get('status')}"
            f" | {'yes' if m['objective_ready'] else 'no'}"
            f" | {m['sweep_score']:.4f}"
            f" | {m['inline_frontier_mean']:.4f}"
            f" | {m['accepted_qas']}"
            f" | {m['images_with_final_rows']}/200"
            f" | {m['q3_score']:.2f}"
            f" | {m['ambiguity_high']}"
            f" | {m['anchor_missing']} |"
        )

    valid_rows = [
        row["metrics"] | {"name": name}
        for name, row in results.items()
        if row.get("status") in {"ok", "cached"} and row["metrics"]["objective_ready"]
    ]
    invalid_rows = [
        row["metrics"] | {"name": name}
        for name, row in results.items()
        if row.get("status") == "invalid_objective"
    ]
    if valid_rows:
        ranked = sorted(valid_rows, key=lambda r: (-r["sweep_score"], -r["inline_frontier_mean"], -r["accepted_qas"], r["name"]))
        lines.extend(["", "## Rankings", ""])
        for idx, row in enumerate(ranked, 1):
            lines.append(
                f"{idx}. `{row['name']}`: sweep=`{row['sweep_score']:.4f}`"
                f", inline=`{row['inline_frontier_mean']:.4f}`"
                f", accepted=`{row['accepted_qas']}`"
                f", images=`{row['images_with_final_rows']}/200`"
                f", q3=`{row['q3_score']:.2f}`"
            )
    if invalid_rows:
        lines.extend(["", "## Objective-Invalid Runs", ""])
        for row in invalid_rows:
            lines.append(
                f"- `{row['name']}` produced `inline_frontier_scored=0`, so its displayed sweep falls back to Q3 and is excluded from ranking."
            )

    lines.extend([
        "",
        "## Eval Results",
        "",
        "- Baseline eval target: `n02_quality_cap_global7_retry`.",
        "- `gemini-3-flash-preview`: exact=`0.4681`, soft=`0.6312`, n=`282` valid rows.",
        "- `gpt-5.3-codex`: exact=`0.4859`, soft=`0.6056`, n=`284` valid rows.",
        "- Overlap set (`282` rows): both-soft=`0.4965`, exact-answer agreement=`0.5496`.",
        "- Strongest type on both models is `REVERSE_GROUND` soft accuracy: Gemini=`0.6780`, GPT=`0.6271`.",
        "- Weakest type is still `DIRECT_READ`: Gemini=`0.6027`, GPT=`0.5342`.",
        "",
        "## Key Takeaways",
        "",
        "- The pipeline and objective are finally aligned enough to trust the leaderboard: the best inline run (`n02`) is also externally decent on both frontier models.",
        "- Global ambiguity gate `7` is now part of the stable foundation, not an experiment branch.",
        "- Tighter grounding threshold is out. The whole remaining search space is anchor specificity pressure.",
        "- Cute little specificity goblins still need supervision: `sign wall` and `display panel` are still too common in accepted rows.",
        "",
        "## Recommended Next",
        "",
        "- Keep `hard_ambi=7`, `ground=0.28`, and the current stabilized scoring path fixed in O-series.",
        "- Spend all four O runs on anti-generic-anchor pressure, not on new threshold churn.",
        "- Treat `DIRECT_READ` quality under text-container anchors as the main thing to improve without regressing `REVERSE_GROUND`.",
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
        status = "cached" if metrics["objective_ready"] else "invalid_objective"
        results[spec.name] = {"status": status, "metrics": metrics, "experiment_name": experiment_name}
        append_timeline(
            timeline_path,
            f"END {spec.name} cached objective={'ok' if metrics['objective_ready'] else 'invalid'}"
            f" inline_scored={metrics['inline_frontier_scored']} sweep={metrics['sweep_score']:.4f}",
        )
        write_text(report_path, render_report(bundle_id, specs, results, docs_path))
        write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
        return

    model_name = spec.model_override or args.model
    target = spec.cli.get("--target-per-image", str(args.target_per_image))
    cmd = [
        sys.executable, "-m", "sgocr.dev200_harness", "build-dev40-semantic",
        "--source-experiment-dir", str(SOURCE_EXPERIMENT),
        "--out-dir", str(out_dir),
        "--intermediate-dir", str(intermediate_dir),
        "--cache-intermediate-dir", str(spec.cache_intermediate_dir or H03_INTERM),
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
        f" cap_count={spec.env.get('SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT','?')}"
        f" generic_penalty={spec.env.get('SGOCR_GENERIC_ANCHOR_PENALTY','?')}"
        f" ground={spec.cli.get('--grounding-threshold','?')}",
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
        status = "ok" if metrics["objective_ready"] else "invalid_objective"
        results[spec.name] = {"status": status, "metrics": metrics, "experiment_name": experiment_name}
        append_timeline(
            timeline_path,
            f"END {spec.name} {status} accepted={metrics['accepted_qas']} images={metrics['images_with_final_rows']}"
            f" inline_scored={metrics['inline_frontier_scored']} inline={metrics['inline_frontier_mean']:.4f}"
            f" sweep={metrics['sweep_score']:.4f}",
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "136_sgocr_dev200_o_sweep_2026-04-08.md"
    results: dict[str, dict[str, Any]] = {}
    missing_env_vars = missing_secret_env_vars([GEMINI])
    if missing_env_vars:
        missing_str = ",".join(missing_env_vars)
        append_timeline(timeline_path, f"BUNDLE {bundle_id} blocked missing_secrets={missing_str}")
        write_text(report_path, render_report(bundle_id, specs, results, docs_path))
        write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
        sys.stderr.write(
            f"O-series launch blocked before start: missing required env var(s): {missing_str}\n"
        )
        raise SystemExit(2)
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
