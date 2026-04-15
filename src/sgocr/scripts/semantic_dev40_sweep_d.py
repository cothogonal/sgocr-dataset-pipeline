from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, LOGS_ROOT, REPO_ROOT
from ..run_quality import compute_run_quality
from .semantic_dev40_sweep_v2 import AblationSpec, append_timeline, gpu_snapshot, now_stamp, write_text


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "ablate_gemini_flash_natural2q_40_min4clean"
CONTROL_B07 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b07_text_recall_combo"
CONTROL_B07_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b07_text_recall_combo"
CONTROL_B06 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b06_resolvability_loose"
CONTROL_B06_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b06_resolvability_loose"
CONTROL_B05 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b05_consensus_relaxed"
CONTROL_B05_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b05_consensus_relaxed"
V3_BEST = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v3_20260405_102156_c02_b06_v8_control"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run semantic dev40 sweep d.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev40_sweep_d_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=60)
    ap.add_argument("--limit", type=int, default=6)
    return ap.parse_args()


def build_specs() -> list[AblationSpec]:
    clean_env = {
        "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
        "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
        "SGOCR_ANCHOR_RELABEL_MODE": "flash",
        "SGOCR_GENERIC_ANCHOR_PENALTY": "0.10",
        "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.34",
        "SGOCR_FLORENCE_REGION_BONUS": "0.06",
        "SGOCR_LOCATION_WORDING_MODE": "lite",
        "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
        "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "5",
        "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "4",
        "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "5",
        "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "5",
        "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.60",
        "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "5",
        "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "4",
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.22",
        "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.10",
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.22",
        "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.16",
    }
    recall_ensemble_env = {
        **clean_env,
        "SGOCR_DETECTOR_MODE": "ppocr_craft_ensemble",
        "SGOCR_DETECTOR_BOX_THRESH": "0.45",
        "SGOCR_DETECTOR_UNCLIP_RATIO": "1.90",
        "SGOCR_DETECTOR_MERGE_OVERLAP": "0.60",
        "SGOCR_CRAFT_TEXT_THRESHOLD": "0.36",
        "SGOCR_CRAFT_LINK_THRESHOLD": "0.32",
        "SGOCR_CRAFT_LOW_TEXT": "0.30",
        "SGOCR_OCR_CROP_PAD_RATIO": "0.08",
        "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.06",
        "SGOCR_STRONG_CONSENSUS_FLOOR": "0.76",
        "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.82",
        "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
        "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.55",
        "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.62",
    }
    return [
        AblationSpec(
            "d01_b07_clean_control",
            "b07 OCR cache + stronger anchor consolidation + lite location wording + ambiguity rejects",
            "ocr",
            cli={"--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env=dict(clean_env),
            cache_intermediate_dir=CONTROL_B07_INTERMEDIATE,
        ),
        AblationSpec(
            "d02_b06_clean_control",
            "b06 OCR cache + same clean grounding stack on looser resolvability base",
            "ocr",
            cli={"--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env=dict(clean_env),
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "d03_b05_clean_control",
            "b05 OCR cache + same clean grounding stack on relaxed-consensus base",
            "ocr",
            cli={"--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env=dict(clean_env),
            cache_intermediate_dir=CONTROL_B05_INTERMEDIATE,
        ),
        AblationSpec(
            "d04_recall_ensemble_flash",
            "Detector ensemble + text-recall geometry + Flash relabel + ambiguity cleanup",
            "none",
            cli={"--max-detections": "128", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env=dict(recall_ensemble_env),
        ),
        AblationSpec(
            "d05_recall_ensemble_cleaner",
            "Detector ensemble + cleaner negatives + slightly stricter anchor selection",
            "none",
            cli={"--max-detections": "128", "--grounding-threshold": "0.35", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                **recall_ensemble_env,
                "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.85",
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.12",
                "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.32",
                "SGOCR_ANCHOR_CONFLICT_OVERLAP": "0.74",
            },
        ),
        AblationSpec(
            "d06_recall_ensemble_pro_relabel",
            "Detector ensemble + Pro anchor relabel on the best high-recall geometry",
            "none",
            cli={"--max-detections": "128", "--grounding-threshold": "0.35", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                **recall_ensemble_env,
                "SGOCR_ANCHOR_RELABEL_MODE": "pro",
                "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.75",
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
        "# SGOCR Dev40 Sweep D",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Source experiment: `{SOURCE_EXPERIMENT.name}`",
        f"- Controls: `{CONTROL_B07.name}`, `{CONTROL_B06.name}`, `{CONTROL_B05.name}`",
        f"- Prior best reference: `{V3_BEST.name}`",
        f"- Updated: `{now_stamp()}`",
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
            "| Run | Status | Accepted | Accept% | Images | No | Rev local | Ambig high | Anchor miss | Ambig rej | Rev invalid | Quality |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for control_name, control_path in (
        ("control_b07", CONTROL_B07),
        ("control_b06", CONTROL_B06),
        ("control_b05", CONTROL_B05),
        ("best_v3", V3_BEST),
    ):
        metrics = collect_metrics(control_path)
        lines.append(
            f"| `{control_name}` | existing | {metrics['accepted_qas']} | {metrics['qa_accept_rate']:.3f} | {metrics['images_with_final_rows']} | {metrics['yesno_negative']} | {metrics['reverse_local']} | {metrics['ambiguity_high']} | {metrics['anchor_missing']} | {metrics['ambiguous_grounding']} | {metrics['reverse_invalid']} | {metrics['quality_score']:.2f} |"
        )
    for spec in specs:
        row = results.get(spec.name)
        if not row:
            lines.append(f"| `{spec.name}` | pending | - | - | - | - | - | - | - | - | - | - |")
            continue
        if row.get("status") not in {"ok", "cached"}:
            lines.append(f"| `{spec.name}` | {row.get('status')} | - | - | - | - | - | - | - | - | - | - |")
            continue
        metrics = row["metrics"]
        lines.append(
            f"| `{spec.name}` | {row.get('status')} | {metrics['accepted_qas']} | {metrics['qa_accept_rate']:.3f} | {metrics['images_with_final_rows']} | {metrics['yesno_negative']} | {metrics['reverse_local']} | {metrics['ambiguity_high']} | {metrics['anchor_missing']} | {metrics['ambiguous_grounding']} | {metrics['reverse_invalid']} | {metrics['quality_score']:.2f} |"
        )
    complete = [row["metrics"] | {"name": name} for name, row in results.items() if row.get("status") in {"ok", "cached"}]
    if complete:
        ranked = sorted(complete, key=lambda row: (-row["quality_score"], -row["accepted_qas"], row["name"]))
        lines.extend(["", "## Top Current Runs", ""])
        for row in ranked[:5]:
            lines.append(
                f"- `{row['name']}`: quality `{row['quality_score']:.2f}`, accepted `{row['accepted_qas']}`, images `{row['images_with_final_rows']}`, negatives `{row['yesno_negative']}`, reverse-local `{row['reverse_local']}`, ambiguity-high `{row['ambiguity_high']}`"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Sweep d targets anchor-label conflicts, local/global ambiguity, and missed text recall without weakening the verifier.",
            "- The frontier runs add a PP-OCR + CRAFT detector ensemble and stronger anchor conflict consolidation.",
            "- SAM 3 is not in this sweep because the current machine has no Hugging Face auth for checkpoint fetch.",
            f"- Live bundle report mirror: `{docs_path}`",
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
        append_timeline(timeline_path, f"END {spec.name} cached accepted={metrics['accepted_qas']} quality={metrics['quality_score']:.2f}")
        write_text(report_path, render_report(bundle_id, specs, results, docs_path))
        write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
        return

    cache_intermediate_dir = spec.cache_intermediate_dir or CONTROL_B07_INTERMEDIATE
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
    append_timeline(timeline_path, f"START {spec.name} cache={spec.cache_level} model={model_name} cli={spec.cli} env={spec.env}")
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
            append_timeline(timeline_path, f"check stuff! {spec.name} pid={proc.pid} gpu=[{gpu_snapshot()}] log={log_path.name}")
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
            f"END {spec.name} ok accepted={metrics['accepted_qas']} images={metrics['images_with_final_rows']} quality={metrics['quality_score']:.2f}",
        )
    write_text(report_path, render_report(bundle_id, specs, results, docs_path))
    write_text(docs_path, render_report(bundle_id, specs, results, docs_path))


def main() -> None:
    args = parse_args()
    specs = build_specs()[: args.limit]
    bundle_dir = LOGS_ROOT / args.bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = bundle_dir / "timeline.log"
    report_path = bundle_dir / "report.md"
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "113_sgocr_dev40_sweep_d_2026-04-05.md"
    results: dict[str, dict[str, Any]] = {}
    append_timeline(timeline_path, f"BUNDLE {args.bundle_id} start source={SOURCE_EXPERIMENT.name} control={CONTROL_B07.name}")
    write_text(report_path, render_report(args.bundle_id, specs, results, docs_path))
    write_text(docs_path, render_report(args.bundle_id, specs, results, docs_path))
    for spec in specs:
        run_ablation(
            bundle_dir=bundle_dir,
            bundle_id=args.bundle_id,
            spec=spec,
            args=args,
            docs_path=docs_path,
            report_path=report_path,
            results=results,
            specs=specs,
        )
    append_timeline(timeline_path, f"BUNDLE {args.bundle_id} complete")


if __name__ == "__main__":
    main()
