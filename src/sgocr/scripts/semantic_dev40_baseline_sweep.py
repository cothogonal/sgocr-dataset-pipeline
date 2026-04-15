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


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "ablate_gemini_flash_natural2q_40_min4clean"

# OCR cache source from sweep v2 (for reference runs)
CONTROL_B06 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b06_resolvability_loose"

# Prior sweep champions (for report baselines)
M14_CHAMPION = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_merge_sweep_20260406_094610_m14_b06_merge_local_extreme"
M11B_CHAMPION = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_merge_sweep_20260406_094610_m11b_det_box035_bigcrop_merge"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the 6-run baseline architecture sweep.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev40_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=60)
    ap.add_argument("--limit", type=int, default=6)
    return ap.parse_args()


def build_specs(bundle_id: str) -> list[AblationSpec]:
    # -----------------------------------------------------------------
    # Proposed baseline config: combines m14 local extreme + m11b
    # detection/crop settings + merge + improved spatial wording (code)
    # + softer ambiguity (7/6) + 14 tags.
    #
    # Flash teacher only.  Pro is dropped as teacher.
    # -----------------------------------------------------------------
    baseline_env = {
        # Merge stage (proven m01 config)
        "SGOCR_TEXT_MERGE_ENABLED": "1",
        "SGOCR_TEXT_MERGE_ANCHOR_OVERLAP_MIN": "0.08",
        "SGOCR_TEXT_MERGE_X_OVERLAP_MIN": "0.42",
        "SGOCR_TEXT_MERGE_Y_OVERLAP_MIN": "0.32",
        "SGOCR_TEXT_MERGE_GAP_RATIO_MAX": "2.0",
        "SGOCR_TEXT_MERGE_HEIGHT_RATIO_MAX": "2.2",
        "SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES": "3",
        # Detection: m11b sweet spot
        "SGOCR_DETECTOR_BOX_THRESH": "0.35",
        "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
        "SGOCR_OCR_CROP_PAD_RATIO": "0.10",
        "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.08",
        # Consensus: relaxed to match lower detection threshold
        "SGOCR_STRONG_CONSENSUS_FLOOR": "0.76",
        "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.82",
        "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
        "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.55",
        "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.62",
        # Anchor settings
        "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
        "SGOCR_ANCHOR_RELABEL_MODE": "flash",
        "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
        "SGOCR_GENERIC_ANCHOR_PENALTY": "0.10",
        "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.34",
        "SGOCR_FLORENCE_REGION_BONUS": "0.06",
        # Wording
        "SGOCR_LOCATION_WORDING_MODE": "lite",
        "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
        # Verifier thresholds
        "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
        "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
        "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
        "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "4",
        "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.35",
        # Ambiguity: softer (7/6) from m18 winning calibration
        "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
        "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
        # Local bias: m14 extreme settings
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.65",
        "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.35",
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.50",
        "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.30",
        # SAM3: targeted
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

    cli_baseline = {
        "--grounding-threshold": "0.28",
        "--max-tags-per-image": "14",
        "--target-per-image": "5",
        "--max-detections": "128",
    }

    # c01's intermediate dir will be used as OCR cache for c02-c06
    c01_intermediate = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / f"{bundle_id}_c01_baseline_candidate"

    return [
        # =============================================================
        # c01: THE PROPOSED BASELINE
        #   Combined m14 (local extreme) + m11b (detect 0.35, bigcrop)
        #   + merge + softer ambiguity (7/6) + 14 tags + new spatial
        #   wording code.  cache_level="none" — runs full pipeline.
        # =============================================================
        AblationSpec(
            "c01_baseline_candidate",
            "Proposed baseline: m14+m11b combined, detect 0.35, bigcrop 0.10, local extreme, merge, ambig 7/6, 14 tags, centroid spatial",
            "none",
            cli=dict(cli_baseline),
            env=dict(baseline_env),
        ),

        # =============================================================
        # c02: MODERATE LOCAL BIAS
        #   Same as c01 but dials back local bias to moderate levels.
        #   Tests: does extreme bias still help with improved spatial
        #   wording, or is moderate enough?
        # =============================================================
        AblationSpec(
            "c02_moderate_local",
            "c01 but with moderate local bias (0.40/0.20) — tests if extreme bias still needed with better spatial",
            "ocr",
            cli=dict(cli_baseline),
            env={
                **baseline_env,
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.40",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.20",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.25",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.12",
            },
            cache_intermediate_dir=c01_intermediate,
        ),

        # =============================================================
        # c03: TIGHTER AMBIGUITY
        #   Same as c01 but keeps strict ambiguity (6/5).
        #   Tests: does the softer 7/6 gate add value, or was it only
        #   useful with weaker spatial?
        # =============================================================
        AblationSpec(
            "c03_strict_ambiguity",
            "c01 but with strict ambiguity gate (6/5) — tests if softer gate adds value with better spatial",
            "ocr",
            cli=dict(cli_baseline),
            env={
                **baseline_env,
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "6",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "5",
            },
            cache_intermediate_dir=c01_intermediate,
        ),

        # =============================================================
        # c04: FEWER TAGS
        #   Same as c01 but with 10 tags (original) and standard
        #   grounding threshold.  Tests the marginal value of more tags.
        # =============================================================
        AblationSpec(
            "c04_fewer_tags",
            "c01 but with 10 tags and standard grounding threshold — tests marginal value of more tags",
            "ocr",
            cli={
                **cli_baseline,
                "--max-tags-per-image": "10",
                "--grounding-threshold": "0.34",
            },
            env=dict(baseline_env),
            cache_intermediate_dir=c01_intermediate,
        ),

        # =============================================================
        # c05: SMALLER CROPS
        #   Same as c01 but with 0.06 crop padding (between default
        #   0.04 and baseline 0.10).  Tests the marginal value of
        #   bigger crops at the detection level.
        # =============================================================
        AblationSpec(
            "c05_smaller_crops",
            "c01 but with crop pad 0.06 instead of 0.10 — tests marginal value of bigger crops",
            "none",
            cli=dict(cli_baseline),
            env={
                **baseline_env,
                "SGOCR_OCR_CROP_PAD_RATIO": "0.06",
                "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.05",
            },
        ),

        # =============================================================
        # c06: VARIED WORDING MODE
        #   Same as c01 but with location_wording_mode="varied".
        #   Tests if the teacher can exploit richer wording with the
        #   improved spatial semantics.
        # =============================================================
        AblationSpec(
            "c06_varied_wording",
            "c01 but with location_wording_mode=varied — tests richer teacher phrasing with improved spatial",
            "ocr",
            cli=dict(cli_baseline),
            env={
                **baseline_env,
                "SGOCR_LOCATION_WORDING_MODE": "varied",
            },
            cache_intermediate_dir=c01_intermediate,
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
        "# SGOCR Dev40 Baseline Architecture Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Source experiment: `{SOURCE_EXPERIMENT.name}`",
        f"- Prior champions: `{M14_CHAMPION.name}`, `{M11B_CHAMPION.name}`",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Design",
        "",
        "6-run sweep to cement a baseline architecture. c01 is the proposed baseline",
        "(m14+m11b combined + centroid-diff spatial wording + softer ambiguity + 14 tags).",
        "c02-c06 each remove or adjust one component to validate its contribution.",
        "",
        "## Planned Ablations",
        "",
    ]
    for spec in specs:
        lines.append(f"- `{spec.name}`: {spec.description}")
    lines.extend(
        [
            "",
            "## Scoreboard",
            "",
            "| Run | Status | Accepted | Accept% | Images | Q3 | Neg | Rev local | Rev mixed | Ambig high | Anchor miss | Diversity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    # Reference baselines from prior sweep
    for control_name, control_path in (
        ("m14_prior", M14_CHAMPION),
        ("m11b_prior", M11B_CHAMPION),
        ("recall_b06", CONTROL_B06),
    ):
        try:
            metrics = collect_metrics(control_path)
            lines.append(
                f"| `{control_name}` | reference"
                f" | {metrics['accepted_qas']}"
                f" | {metrics['qa_accept_rate']:.3f}"
                f" | {metrics['images_with_final_rows']}"
                f" | {metrics['q3_score']}"
                f" | {metrics['yesno_negative']}"
                f" | {metrics['reverse_local']}"
                f" | {metrics['reverse_mixed']}"
                f" | {metrics['ambiguity_high']}"
                f" | {metrics['anchor_missing']}"
                f" | {metrics['type_diversity']:.3f} |"
            )
        except Exception:
            lines.append(f"| `{control_name}` | error | - | - | - | - | - | - | - | - | - | - |")
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
            f" | {m['images_with_final_rows']}"
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
        ranked = sorted(complete, key=lambda row: (-row["q3_score"], -row["accepted_qas"], row["name"]))
        lines.extend(["", "## Rankings", ""])
        for idx, row in enumerate(ranked, 1):
            lines.append(
                f"{idx}. `{row['name']}`: Q3=`{row['q3_score']}`"
                f", accepted=`{row['accepted_qas']}`"
                f", rate=`{row['qa_accept_rate']:.3f}`"
                f", images=`{row['images_with_final_rows']}`"
                f", rev-local=`{row['reverse_local']}`"
                f", rev-mixed=`{row['reverse_mixed']}`"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Code changes: centroid-difference anchor-relative spatial wording, removed quadrant language.",
            "- c01 runs full pipeline (cache_level=none); c02-c04,c06 reuse c01 detection+OCR (cache_level=ocr).",
            "- c05 also runs full pipeline (different crop padding affects detection).",
            "- Flash teacher only. Pro is dropped from teacher configs.",
            f"- Live report: `{docs_path}`",
        ]
    )
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
        sys.executable,
        "-m",
        "sgocr.dev200_harness",
        "build-dev40-semantic",
        "--source-experiment-dir",
        str(SOURCE_EXPERIMENT),
        "--out-dir",
        str(out_dir),
        "--intermediate-dir",
        str(intermediate_dir),
        "--cache-intermediate-dir",
        str(cache_intermediate_dir),
        "--cache-level",
        spec.cache_level,
        "--model",
        model_name,
        "--workers",
        str(args.workers),
        "--max-side",
        str(args.max_side),
        "--target-per-image",
        spec.cli.get("--target-per-image", str(args.target_per_image)),
        "--max-detections",
        spec.cli.get("--max-detections", "72"),
        "--grounding-threshold",
        spec.cli.get("--grounding-threshold", "0.36"),
        "--max-tags-per-image",
        spec.cli.get("--max-tags-per-image", "8"),
    ]
    env = os.environ.copy()
    env.update(spec.env)
    append_timeline(timeline_path, f"START {spec.name} cache={spec.cache_level} model={model_name} cli={spec.cli} env_keys={sorted(spec.env.keys())}")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("CMD: " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        if spec.env:
            handle.write("ENV_OVERRIDES: " + json.dumps(spec.env, sort_keys=True) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            append_timeline(timeline_path, f"WATCH {spec.name} pid={proc.pid} gpu=[{gpu_snapshot()}] log={log_path.name}")
            write_text(report_path, render_report(bundle_id, specs, results, docs_path))
            write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
            time.sleep(max(5, int(args.watch_seconds)))
    if proc.returncode != 0:
        results[spec.name] = {"status": f"failed:{proc.returncode}", "experiment_name": experiment_name}
        append_timeline(timeline_path, f"END {spec.name} failed code={proc.returncode}")
    else:
        metrics = collect_metrics(out_dir)
        results[spec.name] = {"status": "ok", "metrics": metrics, "experiment_name": experiment_name}
        append_timeline(
            timeline_path,
            f"END {spec.name} ok accepted={metrics['accepted_qas']} images={metrics['images_with_final_rows']} q3={metrics['q3_score']}",
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "119_sgocr_dev40_baseline_sweep_2026-04-06.md"
    results: dict[str, dict[str, Any]] = {}
    append_timeline(
        timeline_path,
        f"BUNDLE {bundle_id} start specs={len(specs)} source={SOURCE_EXPERIMENT.name}",
    )
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
    append_timeline(timeline_path, f"BUNDLE {bundle_id} complete total_specs={len(specs)} completed={len([r for r in results.values() if r.get('status') in ('ok', 'cached')])}")


if __name__ == "__main__":
    main()
