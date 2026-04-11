"""Dev200 K-series sweep: structural improvements targeting untapped Q3 components.

Context:
  - h03 is the local Pareto optimum: Q3=80.24, accept_rate=0.807, 106/200 images
  - coverage (20 pts) and yield_eff (10 pts) are already MAXED — adding more images
    or more rows per image gives zero additional Q3 pts
  - Remaining Q3 gain targets:
      * precision: 24.20/30 — each +0.01 accept_rate = +0.30 pts
      * rev_mixed frontier: 3.0/7.5 pts — biggest single uncapped lever (+4.5 pts max)
      * anchor_miss penalty: 1.76 pts lost — reducing to near-zero worth ~+1.8 pts
  - i/j series: all threshold/gate tuning. None beat h03. Confirms h03 is a local
    threshold optimum — further gains need structural changes.

Baseline code fix (applied to all k-series):
  - frontier mode was missing a mixed-scope instruction; added explicit "MIXED-scope"
    teacher guidance that mirrors the standard-mode mixed example. This should unlock
    rev_mixed from the stuck 1-2 baseline.

K-series structural levers (all untested across f-j):
  k01  anchor_relabel=pro       novel: gemini-2.5-pro anchor relabeling (was flash)
  k02  target-per-image=4       novel: fewer candidates → only strongest reach teacher
  k03  SAM3 apply_mode=all      novel: broaden SAM3 anchor refinement beyond targeted
  k04  grounded_exclusion=2.0   novel: permissive grounding threshold + larger area window

All four reuse h03 OCR cache (structural changes are all downstream of OCR).
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

H03_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_h_20260407_055403_h03_aggressive_anchors"
H03_INTERM = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev200_h_20260407_055403_h03_aggressive_anchors"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="K-series dev200 sweep: structural improvements on h03 Pareto frontier.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_k_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(_bundle_id: str) -> list[AblationSpec]:
    # ----------------------------------------------------------------
    # Base: exact h03 settings
    # ----------------------------------------------------------------
    base_env = {
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
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.65",
        "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.35",
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.50",
        "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.30",
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
    }

    return [
        # ============================================================
        # k01: gemini-2.5-pro anchor relabeling
        #   h03 uses flash for borderline anchor relabeling. Pro is
        #   higher quality and may produce more precise anchor labels
        #   for generic cases (wall, panel, display) — directly reducing
        #   anchor_miss failures (penalty = 15 * miss_rate).
        #   h03 lost 1.76 pts to anchor_miss; halving it = +0.88 pts.
        # ============================================================
        AblationSpec(
            "k01_relabel_pro",
            "h03 + anchor_relabel=pro — higher-quality anchor label refinement",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14",
                 "--target-per-image": "5", "--max-detections": "128"},
            env={**base_env, "SGOCR_ANCHOR_RELABEL_MODE": "pro"},
            cache_intermediate_dir=H03_INTERM,
        ),

        # ============================================================
        # k02: reduced target-per-image=4
        #   All prior series use 5 teacher candidates per image. Reducing
        #   to 4 pushes the teacher toward only the strongest candidates
        #   (greedy top-4 anchor selection). Expected effect: accept_rate
        #   rises because marginal 5th candidates dilute the pool.
        #   yield_eff is already capped at 10 pts (3.5/image), so fewer
        #   candidates only costs if images lose ALL candidates — unlikely.
        #   Precision gain: +0.01 accept_rate = +0.30 Q3 pts.
        # ============================================================
        AblationSpec(
            "k02_target4",
            "h03 + target-per-image=4 — tighter candidate selection → higher accept_rate",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14",
                 "--target-per-image": "4", "--max-detections": "128"},
            env={**base_env},
            cache_intermediate_dir=H03_INTERM,
        ),

        # ============================================================
        # k03: SAM3 apply_mode=all + cluster_min=1
        #   h03 uses apply_mode=targeted (SAM3 only on anchors meeting a
        #   cluster+support criterion) with cluster_min=2 (needs 2+ nearby
        #   anchor instances). Changing to all + cluster_min=1 broadens
        #   SAM3 to refine isolated anchor candidates too, improving
        #   grounding box quality across a wider set of images.
        #   Targeted anchor_miss reduction without changing OCR or consensus.
        # ============================================================
        AblationSpec(
            "k03_sam3_broad",
            "h03 + SAM3 apply_mode=all + cluster_min=1 — broader anchor refinement coverage",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14",
                 "--target-per-image": "5", "--max-detections": "128"},
            env={
                **base_env,
                "SGOCR_SAM3_APPLY_MODE": "all",
                "SGOCR_SAM3_TARGET_CLUSTER_MIN": "1",
            },
            cache_intermediate_dir=H03_INTERM,
        ),

        # ============================================================
        # k04: permissive grounding exclusion + larger anchor area window
        #   h03's grounded_exclusion_min_strength=2.35 rejects anchors
        #   with weak spatial grounding evidence. Relaxing to 2.0 lets
        #   marginally-grounded anchors compete. Also opens the oversized
        #   anchor area threshold from 0.34 → 0.28 (allows larger visible
        #   objects as anchors on dense-text images). SAM3 area threshold
        #   aligned at 0.28. Likely increases anchor_miss slightly but
        #   may unlock new anchor types on the ~94 uncovered images.
        # ============================================================
        AblationSpec(
            "k04_permissive_grounding",
            "h03 + grounded_exclusion=2.0 + oversized_area=0.28 — wider anchor acceptance window",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14",
                 "--target-per-image": "5", "--max-detections": "128"},
            env={
                **base_env,
                "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.0",
                "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.28",
                "SGOCR_SAM3_TARGET_AREA_START": "0.28",
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
        "# SGOCR Dev200 K-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Design",
        "",
        "4-run structural sweep on h03 Pareto frontier (Q3=80.24, 106/200).",
        "All runs reuse h03 OCR cache — only downstream stages (grounding, SAM3, teacher, verifier) re-run.",
        "",
        "Baseline fix applied to all k-series: frontier teacher now includes an explicit mixed-scope",
        "instruction (was missing — likely root cause of rev_mixed stuck at 1-2).",
        "",
        "Q3 component analysis for h03 (maxed components marked *):",
        "  precision=24.20/30  coverage=20.0*  yield_eff=10.0*  diversity=6.29  frontier=22.0  penalty=-2.27",
        "Uncapped levers: precision (+0.30/0.01 rate), rev_mixed (3.0→7.5 pts), anchor_miss penalty (−1.76).",
        "",
        "Hypotheses:",
        "1. **Pro relabeling** (k01): gemini-2.5-pro anchor labels → fewer anchor_miss failures → penalty↓",
        "2. **target=4** (k02): fewer teacher candidates → only strongest selected → accept_rate↑ → precision↑",
        "3. **SAM3 all/min1** (k03): broader SAM3 refinement → better anchor grounding for isolated candidates",
        "4. **Permissive grounding** (k04): lower exclusion threshold + wider area window → more anchors survive",
        "",
        "## Prior Best",
        "",
        "| Run | Q3 | Accepted | Accept% | Images/200 | Rev local | Rev mixed | Anchor miss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    try:
        m = collect_metrics(H03_DEV200)
        lines.append(f"| `h03_aggressive_anchors` | {m['q3_score']} | {m['accepted_qas']} | {m['qa_accept_rate']:.3f} | {m['images_with_final_rows']}/200 | {m['reverse_local']} | {m['reverse_mixed']} | {m['anchor_missing']} |")
    except Exception:
        lines.append("| `h03_aggressive_anchors` | 80.24 | 392 | 0.807 | 106/200 | 7 | 2 | 57 |")

    lines.extend([
        "",
        "## K-Series Scoreboard",
        "",
        "| Run | Status | Accepted | Accept% | Images/200 | Q3 | Rev local | Rev mixed | Anchor miss | Ambig | Diversity |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for spec in specs:
        row = results.get(spec.name)
        if not row:
            lines.append(f"| `{spec.name}` | pending | - | - | - | - | - | - | - | - | - |")
            continue
        if row.get("status") not in {"ok", "cached"}:
            lines.append(f"| `{spec.name}` | {row.get('status')} | - | - | - | - | - | - | - | - | - |")
            continue
        m = row["metrics"]
        lines.append(
            f"| `{spec.name}` | {row.get('status')}"
            f" | {m['accepted_qas']}"
            f" | {m['qa_accept_rate']:.3f}"
            f" | {m['images_with_final_rows']}/200"
            f" | {m['q3_score']}"
            f" | {m['reverse_local']}"
            f" | {m['reverse_mixed']}"
            f" | {m['anchor_missing']}"
            f" | {m['ambiguity_high']}"
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
                f", rev-mixed=`{row['reverse_mixed']}`"
                f", anchor-miss=`{row['anchor_missing']}`"
            )

    lines.extend([
        "",
        "## Notes",
        "",
        "- All k-series runs reuse h03 OCR cache (`cache_level=ocr`).",
        "- Baseline fix: frontier teacher mixed-scope instruction added to dev40_complete.py.",
        "- h03 Q3 breakdown: precision=24.20 coverage=20.0 yield_eff=10.0 diversity=6.29 frontier=22.0 penalty=-2.27.",
        "- rev_mixed 2→5 would add +4.5 Q3 pts; accept_rate 0.807→0.85 adds +1.3 pts.",
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
        f"START {spec.name} cache={spec.cache_level} model={model_name}"
        f" relabel={spec.env.get('SGOCR_ANCHOR_RELABEL_MODE','?')}"
        f" target={target}"
        f" sam3_mode={spec.env.get('SGOCR_SAM3_APPLY_MODE','?')}"
        f" excl={spec.env.get('SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH','?')}"
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
        append_timeline(timeline_path, f"END {spec.name} ok accepted={metrics['accepted_qas']} images={metrics['images_with_final_rows']} q3={metrics['q3_score']} rev_mixed={metrics['reverse_mixed']} anchor_miss={metrics['anchor_missing']}")

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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "130_sgocr_dev200_k_sweep_2026-04-07.md"
    results: dict[str, dict[str, Any]] = {}
    append_timeline(timeline_path, f"BUNDLE {bundle_id} start specs={len(specs)} images=200 all_cache=ocr_from_h03")
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
