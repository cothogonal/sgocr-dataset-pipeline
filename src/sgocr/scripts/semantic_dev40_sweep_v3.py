from __future__ import annotations

import argparse
import sys
import time
import shlex
import os
import subprocess
from datetime import datetime
from pathlib import Path

from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, LOGS_ROOT, REPO_ROOT
from .semantic_dev40_sweep_v2 import (
    AblationSpec,
    append_timeline,
    collect_metrics,
    gpu_snapshot,
    now_stamp,
    write_text,
)


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "ablate_gemini_flash_natural2q_40_min4clean"
CONTROL_B07 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b07_text_recall_combo"
CONTROL_B07_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b07_text_recall_combo"
CONTROL_B06 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b06_resolvability_loose"
CONTROL_B06_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b06_resolvability_loose"
CONTROL_B05 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b05_consensus_relaxed"
CONTROL_B05_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b05_consensus_relaxed"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run semantic dev40 frontier ablation sweep.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev40_sweep_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=60)
    ap.add_argument("--limit", type=int, default=10)
    return ap.parse_args()


def build_specs() -> list[AblationSpec]:
    return [
        AblationSpec(
            "c01_b07_v8_control",
            "New v8 prompt/property stack on the clean b07 verified tuples",
            "verified",
            cli={"--target-per-image": "5"},
            cache_intermediate_dir=CONTROL_B07_INTERMEDIATE,
        ),
        AblationSpec(
            "c02_b06_v8_control",
            "New v8 stack on the loose-resolvability b06 verified tuples",
            "verified",
            cli={"--target-per-image": "5"},
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "c03_b05_v8_control",
            "New v8 stack on the relaxed-consensus b05 verified tuples",
            "verified",
            cli={"--target-per-image": "5"},
            cache_intermediate_dir=CONTROL_B05_INTERMEDIATE,
        ),
        AblationSpec(
            "c04_b07_supportive_consensus",
            "b07 OCR base + supportive anchor-prompt expansion + support-count scoring",
            "ocr",
            cli={"--max-detections": "112", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
            },
            cache_intermediate_dir=CONTROL_B07_INTERMEDIATE,
        ),
        AblationSpec(
            "c05_b07_supportive_flash",
            "b07 OCR base + supportive prompt expansion + Flash anchor relabel",
            "ocr",
            cli={"--max-detections": "112", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
                "SGOCR_ANCHOR_RELABEL_MODE": "flash",
            },
            cache_intermediate_dir=CONTROL_B07_INTERMEDIATE,
        ),
        AblationSpec(
            "c06_b07_supportive_pro",
            "b07 OCR base + supportive prompt expansion + Pro anchor relabel",
            "ocr",
            cli={"--max-detections": "112", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
                "SGOCR_ANCHOR_RELABEL_MODE": "pro",
            },
            cache_intermediate_dir=CONTROL_B07_INTERMEDIATE,
        ),
        AblationSpec(
            "c07_b07_supportive_flash_mixed",
            "b07 OCR base + Flash relabel + more mixed anchor-local/global reverse-ground answers",
            "ocr",
            cli={"--max-detections": "112", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
                "SGOCR_ANCHOR_RELABEL_MODE": "flash",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.24",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.12",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.24",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.18",
            },
            cache_intermediate_dir=CONTROL_B07_INTERMEDIATE,
        ),
        AblationSpec(
            "c08_b06_supportive_flash_clean",
            "b06 loose-resolvability OCR base + Flash relabel + stricter grounded negatives",
            "ocr",
            cli={"--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
                "SGOCR_ANCHOR_RELABEL_MODE": "flash",
                "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.50",
                "SGOCR_MAX_NEGATIVE_YESNO_PER_IMAGE": "1",
                "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "c09_b05_supportive_flash",
            "b05 relaxed-consensus OCR base + supportive prompt expansion + Flash relabel",
            "ocr",
            cli={"--max-detections": "96", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
                "SGOCR_ANCHOR_RELABEL_MODE": "flash",
            },
            cache_intermediate_dir=CONTROL_B05_INTERMEDIATE,
        ),
        AblationSpec(
            "c10_b07_aggressive_flash_clean",
            "b07 OCR base + aggressive prompt expansion + Flash relabel + cleaner anchor selection",
            "ocr",
            cli={"--max-detections": "112", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.08",
                "SGOCR_ANCHOR_RELABEL_MODE": "flash",
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.10",
                "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.36",
                "SGOCR_TEACHER_STRICTNESS": "very_strict",
                "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
            },
            cache_intermediate_dir=CONTROL_B07_INTERMEDIATE,
        ),
    ]


def render_report(bundle_id: str, specs: list[AblationSpec], results: dict[str, dict[str, object]], docs_path: Path) -> str:
    lines = [
        "# SGOCR Dev40 Frontier Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Source experiment: `{SOURCE_EXPERIMENT.name}`",
        f"- Primary control: `{CONTROL_B07.name}`",
        f"- Recall control: `{CONTROL_B06.name}`",
        f"- Consensus control: `{CONTROL_B05.name}`",
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
            "| Run | Status | Accepted | Accept% | Images | Yes | No | Rev global | Rev local | Anchor miss | Rev invalid | Quality |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for control_name, control_path in (
        ("control_b07", CONTROL_B07),
        ("control_b06", CONTROL_B06),
        ("control_b05", CONTROL_B05),
    ):
        metrics = collect_metrics(control_path)
        lines.append(
            f"| `{control_name}` | existing | {metrics['accepted_qas']} | {metrics['qa_accept_rate']:.3f} | {metrics['images_with_final_rows']} | {metrics['yesno_positive']} | {metrics['yesno_negative']} | {metrics['reverse_global']} | {metrics['reverse_local']} | {metrics['anchor_missing']} | {metrics['reverse_invalid']} | {metrics['quality_score']:.2f} |"
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
            f"| `{spec.name}` | {row.get('status')} | {metrics['accepted_qas']} | {metrics['qa_accept_rate']:.3f} | {metrics['images_with_final_rows']} | {metrics['yesno_positive']} | {metrics['yesno_negative']} | {metrics['reverse_global']} | {metrics['reverse_local']} | {metrics['anchor_missing']} | {metrics['reverse_invalid']} | {metrics['quality_score']:.2f} |"
        )
    complete = [row["metrics"] | {"name": name} for name, row in results.items() if row.get("status") in {"ok", "cached"}]
    if complete:
        ranked = sorted(complete, key=lambda row: (-row["quality_score"], -row["accepted_qas"], row["name"]))
        lines.extend(["", "## Top Current Runs", ""])
        for row in ranked[:5]:
            lines.append(
                f"- `{row['name']}`: quality `{row['quality_score']:.2f}`, accepted `{row['accepted_qas']}`, images `{row['images_with_final_rows']}`, negatives `{row['yesno_negative']}`, reverse-local `{row['reverse_local']}`"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This sweep targets stronger anchor labels, better box support, and cleaner anchor-local phrasing without giving up the b07 recall gains.",
            "- The frontier knobs are: supportive/aggressive prompt expansion, support-count scoring, Flash/Pro semantic anchor relabeling, and more mixed local/global reverse-ground answers.",
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
    results: dict[str, dict[str, object]],
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
        args.model,
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
    append_timeline(timeline_path, f"START {spec.name} cache={spec.cache_level} cli={spec.cli} env={spec.env}")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("CMD: " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        if spec.env:
            handle.write("ENV_OVERRIDES: " + str(spec.env) + "\n")
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "112_sgocr_dev40_frontier_sweep_v3_2026-04-05.md"
    results: dict[str, dict[str, object]] = {}
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
