"""Dev200 L-series sweep: combine structural wins from k-series + deploy scorer/pipeline fixes.

Context:
  - k01 is the new Pareto frontier: Q3=81.84, 101/200 images, rev_mixed=5
    (first run to beat h03=80.24, gain was entirely from mixed-scope teacher fix)
  - k03 (SAM3 broad) achieved best gemini RG=0.712 and anchor_miss=63 vs h03=57
  - Pro relabeling WORSENED anchor quality (k01 anchor_miss=73 vs h03=57)
  - Q3 analysis: coverage/yield_eff still maxed; remaining levers are
    rev_mixed (7.5 pts max, k01 achieved 5 stochastically), precision, anchor_miss

Scorer/pipeline fixes deployed before this sweep (affect all future data):
  - NFKC unicode normalization in _strip_accents
  - Trailing OCR punct stripping in _partial_correct_direct_read
  - Substring fragment matching for short gold (<3 chars)
  - YES/NO teacher: _normalize_yesno_query_text() strips artifact OCR before embed
  - SGOCR_DR_AMBIGUITY_REJECT_SCORE: per-type ambiguity gate for DIRECT_READ

L-series hypotheses (all untested combos):
  l01  SAM3 broad + mixed-bias 0.65           novel: combines k03 and k01's winning features
  l02  SAM3 broad + mixed-bias 0.65 + DR=5   novel: adds DR ambiguity filter to l01
  l03  mixed-bias 0.65 only                  novel: isolate mixed-bias reproducibility vs k01
  l04  full stack + global ambi 6            novel: tightest quality gate on l02 base

All four reuse h03 OCR cache (all changes are downstream of OCR).
Base uses flash relabeling (confirmed superior to pro in k-series).
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
K01_DEV200 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev200_k_20260407_141551_k01_relabel_pro"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="L-series dev200 sweep: combine k-series structural wins + deploy scorer fixes.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev200_l_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=120)
    ap.add_argument("--limit", type=int, default=4)
    return ap.parse_args()


def build_specs(_bundle_id: str) -> list[AblationSpec]:
    # ----------------------------------------------------------------
    # Base: exact h03 settings with flash relabeling (confirmed best)
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
        # l01: SAM3 broad + mixed-scope bias 0.65
        #   k03 (SAM3 all/min1) reduced anchor_miss to 63 (best in k-series)
        #   and achieved highest gemini RG=0.712. k01 achieved rev_mixed=5 via
        #   the mixed-scope teacher fix, but this was stochastic (bias=0.50).
        #   Raising DIRECTIONAL_MIXED_BIAS to 0.65 makes mixed assignments
        #   ~30% more likely, increasing expected rev_mixed from ~2 to ~4-5.
        #   Flash relabeling (base) avoids k01's anchor_miss regression.
        #   This is the most theoretically grounded l-series config.
        # ============================================================
        AblationSpec(
            "l01_sam3_mixed65",
            "flash + SAM3 broad + mixed-bias 0.65 — combines k03 anchor quality with k01 rev_mixed gain",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14",
                 "--target-per-image": "5", "--max-detections": "128"},
            env={
                **base_env,
                "SGOCR_SAM3_APPLY_MODE": "all",
                "SGOCR_SAM3_TARGET_CLUSTER_MIN": "1",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.65",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.35",
            },
            cache_intermediate_dir=H03_INTERM,
        ),

        # ============================================================
        # l02: SAM3 broad + mixed-bias 0.65 + DR ambiguity reject=5
        #   Adds the new SGOCR_DR_AMBIGUITY_REJECT_SCORE=5 gate to l01.
        #   57% of DIRECT_READ is high-ambiguity (score ≥ 5); those rows
        #   score only 36% gemini-soft vs 60% for low-ambiguity DR.
        #   Rejecting them drops DR count ~57% but raises DR eval precision.
        #   Q3 impact: precision may rise (only clean DR accepted) but
        #   fewer accepted rows hurts precision term too — net effect unknown.
        #   Expected Q3 direction: +0 to +2 pts (ambiguity penalty↓ offsets
        #   any row-count drop since yield_eff is already capped).
        # ============================================================
        AblationSpec(
            "l02_sam3_mixed65_drfilter",
            "l01 + SGOCR_DR_AMBIGUITY_REJECT_SCORE=5 — adds high-ambiguity DR rejection",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14",
                 "--target-per-image": "5", "--max-detections": "128"},
            env={
                **base_env,
                "SGOCR_SAM3_APPLY_MODE": "all",
                "SGOCR_SAM3_TARGET_CLUSTER_MIN": "1",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.65",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.35",
                "SGOCR_DR_AMBIGUITY_REJECT_SCORE": "5",
            },
            cache_intermediate_dir=H03_INTERM,
        ),

        # ============================================================
        # l03: mixed-bias 0.65 only (no SAM3 change)
        #   k01 got rev_mixed=5 with DIRECTIONAL_MIXED_BIAS=0.50. k02/k03/k04
        #   all had the same bias and stayed at rev_mixed=2. Was k01 lucky
        #   or did pro relabeling seed a different assignment path?
        #   l03 tests the bias increase alone (flash relabeling, targeted SAM3)
        #   to determine if higher bias deterministically raises rev_mixed,
        #   or if some other factor was responsible in k01.
        # ============================================================
        AblationSpec(
            "l03_mixed65_only",
            "flash + mixed-bias 0.65 (no SAM3 change) — isolate mixed-scope bias effect",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14",
                 "--target-per-image": "5", "--max-detections": "128"},
            env={
                **base_env,
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.65",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.35",
            },
            cache_intermediate_dir=H03_INTERM,
        ),

        # ============================================================
        # l04: full stack + tighter global ambiguity gate 6
        #   l02 config plus SGOCR_AMBIGUITY_HARD_REJECT_SCORE=6 (down from 7).
        #   This tightens the global ambiguity rejection for ALL question types
        #   (current h03 threshold=7 means score ≥ 7 rejects; changing to 6
        #   adds one more ambiguity tier to the reject pool).
        #   The DR filter (REJECT_SCORE=5) is still more aggressive for DR.
        #   l04 tests whether a compound quality stack (SAM3 + mixed-bias +
        #   DR filter + global gate) outperforms l02's more targeted filtering.
        # ============================================================
        AblationSpec(
            "l04_full_quality_stack",
            "l02 + SGOCR_AMBIGUITY_HARD_REJECT_SCORE=6 — tightest quality gate across all types",
            "ocr",
            cli={"--grounding-threshold": "0.28", "--max-tags-per-image": "14",
                 "--target-per-image": "5", "--max-detections": "128"},
            env={
                **base_env,
                "SGOCR_SAM3_APPLY_MODE": "all",
                "SGOCR_SAM3_TARGET_CLUSTER_MIN": "1",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.65",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.35",
                "SGOCR_DR_AMBIGUITY_REJECT_SCORE": "5",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "6",
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
        "# SGOCR Dev200 L-Series Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Image universe: 200 images",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Design",
        "",
        "4-run sweep combining k-series structural wins + new scorer/pipeline fixes.",
        "All runs reuse h03 OCR cache — only downstream stages (grounding, SAM3, teacher, verifier) re-run.",
        "Flash relabeling used throughout (pro relabeling worsened anchor quality in k01).",
        "",
        "Baseline code changes (all l-series):",
        "  - NFKC unicode normalization + trailing-punct stripping + fragment substring in DR scorer",
        "  - YES/NO teacher: normalized OCR text before embedding (strips artifacts like `]`)",
        "  - SGOCR_DR_AMBIGUITY_REJECT_SCORE: per-type ambiguity gate for DIRECT_READ (new env var)",
        "",
        "K-series findings:",
        "  k01 Q3=81.84 (new best): gain entirely from rev_mixed 2→5 (mixed-scope bias, stochastic)",
        "  k03 Q3=78.27: SAM3 broad → anchor_miss=63 (vs h03=57), best gemini RG=0.712",
        "  Pro relabeling (k01): anchor_miss WORSENED 57→73 (abstract labels scored lower by grounder)",
        "",
        "Hypotheses:",
        "1. **SAM3 broad + mixed-bias 0.65** (l01): combine k03 anchor quality with higher mixed assignment rate",
        "2. **l01 + DR ambiguity filter=5** (l02): reject high-ambiguity DR — 57% of DR, only 36% soft accuracy",
        "3. **mixed-bias 0.65 only** (l03): isolate — does higher bias alone reproduce k01 rev_mixed=5?",
        "4. **Full quality stack** (l04): l02 + tighter global ambiguity gate (score≥6 rejects all types)",
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
    try:
        m = collect_metrics(K01_DEV200)
        lines.append(f"| `k01_relabel_pro` | {m['q3_score']} | {m['accepted_qas']} | {m['qa_accept_rate']:.3f} | {m['images_with_final_rows']}/200 | {m['reverse_local']} | {m['reverse_mixed']} | {m['anchor_missing']} |")
    except Exception:
        lines.append("| `k01_relabel_pro` | 81.84 | 351 | 0.736 | 101/200 | 6 | 5 | 73 |")

    lines.extend([
        "",
        "## L-Series Scoreboard",
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
        "- All l-series runs reuse h03 OCR cache (`cache_level=ocr`).",
        "- Flash relabeling throughout (pro worsened anchor_miss in k01).",
        "- k01 Q3 breakdown: precision=? coverage=20.0 yield_eff=10.0 frontier=26.5 (rev_mixed=5) penalty≈-3.5.",
        "- DR ambiguity reject (l02, l04): SGOCR_DR_AMBIGUITY_REJECT_SCORE=5 rejects score≥5 DR rows.",
        "- Mixed-scope bias 0.65 (l01/l02/l03/l04): directional RG mixed probability 50%→65%.",
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
        f" mixed_bias={spec.env.get('SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS','?')}"
        f" dr_ambi_reject={spec.env.get('SGOCR_DR_AMBIGUITY_REJECT_SCORE','off')}"
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "132_sgocr_dev200_l_sweep_2026-04-07.md"
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
