"""Dev200 Q-series sweep: final local tuning around the P-series winner.

Q-series is built from the completed P-series evidence:
  - p02 is the new overall leader: sweep 11.2949, inline 0.6750, accepted 280.
  - p02 also wins the external evals cleanly:
      gemini-3-flash soft 0.6464
      gpt-5.3-codex soft 0.6679
      both-soft overlap 0.5607
  - p03 looked promising on inline mean and Q3, but lost externally:
      gemini-3-flash soft 0.6381
      gpt-5.3-codex soft 0.6567
      both-soft overlap 0.5336
  - The only clear p03 win is TEXT_PROPERTY, while p02 remains better on
    DIRECT_READ, REVERSE_GROUND, and total overlap quality.

This is no longer a branch-finding sweep. It is a finalization sweep.
The Q question is whether we can improve the last two local debts on top of p02:
  1. shave down generic text-container anchors a bit harder
  2. recover the narrow TEXT_PROPERTY gain without carrying p03's broader losses
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
from .secrets import GEMINI, missing_secret_env_vars
from .semantic_dev40_sweep_v2 import AblationSpec, append_timeline, gpu_snapshot, now_stamp, write_text


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "dev200_source_universe"
H03_INTERM = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev200_h_20260407_055403_h03_aggressive_anchors"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Q-series dev200 sweep: final local tuning around the p02 winner.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_q_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(_bundle_id: str) -> list[AblationSpec]:
    p02_base_env = {
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
        "SGOCR_GENERIC_ANCHOR_PENALTY": "0.06",
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
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "1",
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY": "0.12",
    }

    return [
        AblationSpec(
            "q01_p02_baseline",
            "carry forward p02 unchanged; current best objective and external-eval trade",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**p02_base_env},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "q02_p02_property5",
            "isolate the only clear p03 win by raising property specificity alone while keeping the p02 DR/YES_NO setup",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**p02_base_env, "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "5"},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "q03_p02_generic07",
            "push anti-generic anchor pressure one click harder on top of p02 to suppress sign wall / display panel without changing question specificity",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={**p02_base_env, "SGOCR_GENERIC_ANCHOR_PENALTY": "0.07"},
            cache_intermediate_dir=H03_INTERM,
        ),
        AblationSpec(
            "q04_p02_generic07_property5",
            "combine stronger anti-generic anchor pressure with the isolated property-specificity change as the final local wildcard",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14", "--target-per-image": "5", "--max-detections": "128"},
            env={
                **p02_base_env,
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.07",
                "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "5",
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
        "# SGOCR Dev200 Q-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        "- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## What I Found",
        "",
        "- `p02_o04_softcap012` is the new stable leader: sweep=`11.2949`, inline=`0.6750`, accepted=`280`.",
        "- `p02` is also the best externally validated run so far: Gemini soft=`0.6464`, GPT soft=`0.6679`, both-soft overlap=`0.5607`.",
        "- `p03_o04_specificity_plus` is not the next baseline. It loses overall external quality (`0.6381` Gemini, `0.6567` GPT, `0.5336` both-soft) even though it improves `TEXT_PROPERTY`.",
        "- The pipeline / scoring / eval story is now aligned enough to finalize locally rather than reopen the search tree.",
        "- The remaining debts are narrow: generic text-container anchors are still too common, and `TEXT_PROPERTY` is still the one lane where the runner-up teaches us something useful.",
        "",
        "## Why These Runs Exist",
        "",
        "Q-series is a finalization sweep around `p02`, not a new branch. It isolates the only two plausible local improvements left: stronger anti-generic-anchor pressure and a narrower property-specificity increase. If neither helps, `p02` is the correct near-final baseline.",
        "",
        "## Hypotheses Tested",
        "",
        "1. **q01**: `p02` is already the right near-final setting and should reproduce.",
        "2. **q02**: the only useful part of `p03` may be the `TEXT_PROPERTY` cleanup, which should be testable by raising property specificity alone.",
        "3. **q03**: `p02` still tolerates too many generic text-container anchors; a small penalty increase may improve final external quality without giving back too many rows.",
        "4. **q04**: the best last-step setting may be the combination of stronger generic-anchor pressure plus isolated property cleanup.",
        "",
        "## Prior Evidence",
        "",
        "| Run | Objective-valid | Sweep | Inline mean | Accepted | Images/200 | Q3 | Eval note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        "| `o04_global7_count1_generic06` | yes | 11.1555 | 0.6655 | 281 | 88/200 | 68.62 | first externally strong finalization line |",
        "| `p02_o04_softcap012` | yes | **11.2949** | 0.6750 | **280** | **88/200** | 65.50 | best external story: Gm `0.6464`, GPT `0.6679`, both-soft `0.5607` |",
        "| `p03_o04_specificity_plus` | yes | 11.2396 | **0.6866** | 268 | 84/200 | **67.68** | only clear win is `TEXT_PROPERTY`; loses overall external quality |",
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

    lines.extend([
        "",
        "## Eval Results",
        "",
        "- Pending until the best finished Q run is frontier-evaluated.",
        "",
        "## Key Takeaways",
        "",
        "- The stable foundation is now explicit: `hard_ambi=7`, `ground=0.28`, `count1`, `generic06`, `softcap_penalty=0.12`.",
        "- The sweep objective and external evals are aligned enough to trust for local search, but close neighbors should still be frontier-checked before promotion.",
        "- Q is intentionally narrow. If these four runs do not beat `p02`, the pipeline is close to frozen for this phase.",
        "",
        "## Recommended Next",
        "",
        "- Keep the current pipeline, scoring path, and eval harness fixed while comparing these four local variants.",
        "- Promote a Q winner only if it preserves or improves the external lead of `p02`, not just the inline proxy.",
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
        f" cap_penalty={spec.env.get('SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY','?')}"
        f" generic_penalty={spec.env.get('SGOCR_GENERIC_ANCHOR_PENALTY','?')}"
        f" property_specific={spec.env.get('SGOCR_PROPERTY_SPECIFIC_THRESHOLD','?')}",
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "138_sgocr_dev200_q_sweep_2026-04-08.md"
    results: dict[str, dict[str, Any]] = {}
    missing_env_vars = missing_secret_env_vars([GEMINI])
    if missing_env_vars:
        missing_str = ",".join(missing_env_vars)
        append_timeline(timeline_path, f"BUNDLE {bundle_id} blocked missing_secrets={missing_str}")
        write_text(report_path, render_report(bundle_id, specs, results, docs_path))
        write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
        sys.stderr.write(f"Q-series launch blocked before start: missing required env var(s): {missing_str}\n")
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
