"""Dev200 N-series sweep: precision-first continuation with objective-validity guards.

N-series is built from the actual M-series readout:
  - m02 is the best valid precision-first run so far (11.4144 at n=280).
  - m01 shows the diversity cap helps quality, but m02 edges it on inline score.
  - m04 confirms target-per-image 4 cuts too much yield.
  - m03 looked promising on Q3 but is invalid for precision-first because it
    completed without any inline frontier summary or row annotations.

The key repair in this sweep is operational rather than architectural:
  runs that do not produce inline frontier scoring are marked objective-invalid
  and excluded from the ranked precision-first leaderboard.
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="N-series dev200 sweep: continue precision-first search with inline-score validity guards.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_n_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "2",
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY": "0.15",
    }

    return [
        AblationSpec(
            "n01_quality_cap_baseline",
            "m02 carried forward unchanged; best valid precision-first baseline so far",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**precision_base_env},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "n02_quality_cap_global7_retry",
            "m03 retried under objective-validity guard; same config, but eligible only if inline scoring lands",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**precision_base_env, "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7"},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "n03_quality_cap_count1",
            "baseline with stronger diversity pressure: allow only one dominant anchor type before penalty",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**precision_base_env, "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "1"},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "n04_quality_cap_thresh30",
            "baseline with slightly tighter grounding threshold to trade a little yield for cleaner anchors",
            "ocr",
            cli={"--grounding-threshold": "0.30", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**precision_base_env},
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
        "# SGOCR Dev200 N-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        "- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## What I Found",
        "",
        "- The newest healthy bundle is `sgocr_dev200_m_20260407_231300`; it finished cleanly, but `m03_relaxed_global_cap` is not a valid precision-first result because it produced no inline frontier summary and no row-level inline annotations.",
        "- Among valid precision-first runs, `m02_quality_base_cap` is the current leader: sweep=`11.4144`, inline=`0.6821`, accepted=`280`.",
        "- The anchor diversity soft-cap looks real rather than decorative: `m02` beat `m01` on inline score by improving DIRECT_READ and TEXT_PROPERTY quality while giving up only 5 rows.",
        "- `m04` improved some reverse-ground quality but the target-per-image 4 cut still costs too much yield under the sqrt(rows) objective.",
        "",
        "## Why These Runs Exist",
        "",
        "This sweep keeps the M-series baseline but repairs the operational weak spot: any run without inline frontier evidence is marked objective-invalid and cannot win on a Q3 fallback. The four N runs are one clean baseline, one repaired retry of the promising invalid M variant, one diversity-pressure probe, and one tighter-grounding wildcard.",
        "",
        "## Hypotheses Tested",
        "",
        "1. **n01**: `m02` is still the best overall trade under a valid precision-first objective.",
        "2. **n02**: relaxing the global ambiguity gate to 7 may recover enough rows to win, but only if inline scoring actually lands this time.",
        "3. **n03**: a stricter anchor-type soft cap may cut sign/wall/board monotony further and raise inline correctness without the severe row loss seen from target-per-image 4.",
        "4. **n04**: a slightly tighter grounding threshold may improve anchor specificity more efficiently than lowering target count.",
        "",
        "## Prior Evidence",
        "",
        "| Run | Objective-valid | Sweep | Inline mean | Accepted | Images/200 | Q3 | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        "| `m01_quality_base_nocap` | yes | 11.2546 | 0.6667 | 285 | 89/200 | 71.93 | no diversity cap |",
        "| `m02_quality_base_cap` | yes | **11.4144** | **0.6821** | 280 | 87/200 | 68.62 | current valid leader |",
        "| `m03_relaxed_global_cap` | no | 72.1500 | 0.0000 | 285 | 88/200 | 72.15 | Q3 fallback only; do not trust |",
        "| `m04_quality_cap_target4` | yes | 10.7291 | 0.6693 | 257 | 89/200 | 70.46 | cleaner but too row-starved |",
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
        "- Pending until the best finished run from the newest healthy bundle is frontier-evaluated.",
        "",
        "## Key Takeaways",
        "",
        "- Color-bearing reverse-ground answers are now showing up in practice, so that code path is paying rent.",
        "- The soft diversity cap helped overall precision, but the best next question is whether a stronger version or a cleaner grounding filter can improve specificity without replaying the `target-per-image=4` row collapse.",
        "- Precision-first sweeps need objective-validity checks. Cute little moonbeams do not get to crown Q3 impostors.",
        "",
        "## Recommended Next",
        "",
        "- If `n02` lands valid inline scores and beats `n01`, keep the relaxed global gate alive.",
        "- If `n03` wins, continue working the anchor diversity axis.",
        "- If `n04` wins, spend the next series on threshold/tag-budget cleanup rather than more prompt churn.",
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "135_sgocr_dev200_n_sweep_2026-04-08.md"
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
