from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, LOGS_ROOT, REPO_ROOT


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "ablate_gemini_flash_natural2q_40_min4clean"
CONTROL_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v1_20260405_000358_a12_recall_combo"
CONTROL_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v1_20260405_000358_a12_recall_combo"
V7_CONTROL = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "ablate_gemini_flash_natural2q_40_min4clean_semantic_dev40_v7"


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    cache_level: str
    cli: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    model_override: str | None = None
    cache_intermediate_dir: Path | None = None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run semantic dev40 follow-up ablation sweep.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev40_sweep_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
            "b01_det_recall_mild",
            "Lower detector box threshold, modest unclip, more detections",
            "none",
            cli={"--max-detections": "96", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_DETECTOR_BOX_THRESH": "0.45",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "1.80",
            },
        ),
        AblationSpec(
            "b02_det_recall_strong",
            "More aggressive detector recall and box expansion",
            "none",
            cli={"--max-detections": "112", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_DETECTOR_BOX_THRESH": "0.40",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
            },
        ),
        AblationSpec(
            "b03_crop_pad_008",
            "Larger OCR crops to reduce chopped letters",
            "none",
            cli={"--max-detections": "96", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_OCR_CROP_PAD_RATIO": "0.08",
                "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.06",
            },
        ),
        AblationSpec(
            "b04_crop_pad_012",
            "Even larger OCR crops for curved/chopped text stress test",
            "none",
            cli={"--max-detections": "96", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_OCR_CROP_PAD_RATIO": "0.12",
                "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.08",
            },
        ),
        AblationSpec(
            "b05_consensus_relaxed",
            "Lower OCR confidence floors to rescue more easy text",
            "none",
            cli={"--max-detections": "96", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_STRONG_CONSENSUS_FLOOR": "0.76",
                "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.82",
                "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
            },
        ),
        AblationSpec(
            "b06_resolvability_loose",
            "Slightly looser patch-level resolvability gate",
            "ocr",
            cli={"--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.55",
                "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.62",
            },
            cache_intermediate_dir=CONTROL_INTERMEDIATE,
        ),
        AblationSpec(
            "b07_text_recall_combo",
            "Detector recall + bigger crops + relaxed consensus + looser resolvability",
            "none",
            cli={"--max-detections": "112", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_DETECTOR_BOX_THRESH": "0.45",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "1.90",
                "SGOCR_OCR_CROP_PAD_RATIO": "0.08",
                "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.06",
                "SGOCR_STRONG_CONSENSUS_FLOOR": "0.76",
                "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.82",
                "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
                "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.55",
                "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.62",
            },
        ),
        AblationSpec(
            "b08_location_dedup",
            "Balanced wording with redundant location phrase cleanup",
            "verified",
            cli={"--target-per-image": "5"},
            env={
                "SGOCR_LOCATION_WORDING_MODE": "balanced",
                "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
                "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
                "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
                "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
            },
            cache_intermediate_dir=CONTROL_INTERMEDIATE,
        ),
        AblationSpec(
            "b09_text_recall_vstrict",
            "Text-recall combo plus stricter anti-editorial teacher",
            "none",
            cli={"--max-detections": "112", "--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={
                "SGOCR_DETECTOR_BOX_THRESH": "0.45",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "1.90",
                "SGOCR_OCR_CROP_PAD_RATIO": "0.08",
                "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.06",
                "SGOCR_STRONG_CONSENSUS_FLOOR": "0.76",
                "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.82",
                "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
                "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.55",
                "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.62",
                "SGOCR_LOCATION_WORDING_MODE": "balanced",
                "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
                "SGOCR_TEACHER_STRICTNESS": "very_strict",
            },
        ),
        AblationSpec(
            "b10_gemini_pro_recall",
            "Gemini Pro teacher on the best high-recall control geometry",
            "verified",
            cli={"--target-per-image": "5"},
            env={
                "SGOCR_LOCATION_WORDING_MODE": "balanced",
                "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
            },
            model_override="gemini-2.5-pro",
            cache_intermediate_dir=CONTROL_INTERMEDIATE,
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
    qtypes = Counter(row["tags"]["question_type"] for row in rows)
    yesno = [row for row in rows if row["tags"]["question_type"] == "YES_NO"]
    reverse = [row for row in rows if row["tags"]["question_type"] == "REVERSE_GROUND"]
    yesno_polarity = Counter(row["tags"].get("yesno_polarity") for row in yesno)
    reverse_scope = Counter((row.get("grounding") or {}).get("reverse_ground_scope_preference") for row in reverse)
    failure_counts = summary.get("failure_counts") or {}
    return {
        "accepted_qas": int(summary.get("accepted_qas") or 0),
        "generated_qas": int(summary.get("generated_qas") or 0),
        "qa_accept_rate": float(summary.get("qa_accept_rate") or 0.0),
        "images_with_final_rows": int(summary.get("images_with_final_rows") or 0),
        "yesno_positive": int(yesno_polarity.get("positive", 0)),
        "yesno_negative": int(yesno_polarity.get("negative", 0)),
        "reverse_local": int(reverse_scope.get("local", 0)),
        "reverse_global": int(reverse_scope.get("global", 0)),
        "reverse_mixed": int(reverse_scope.get("mixed", 0)),
        "anchor_missing": int(failure_counts.get("anchor_missing", 0)),
        "reverse_invalid": int(failure_counts.get("reverse_ground_answer_invalid", 0)),
        "validation_failed": int(failure_counts.get("validation_failed", 0)),
        "direct_read": int(qtypes.get("DIRECT_READ", 0)),
        "reverse_ground": int(qtypes.get("REVERSE_GROUND", 0)),
        "yes_no": int(qtypes.get("YES_NO", 0)),
        "anchor_property": int(qtypes.get("ANCHOR_PROPERTY", 0)),
        "mean_question_words": float(summary.get("mean_question_words") or 0.0),
        "quality_score": round(
            float(summary.get("accepted_qas") or 0)
            + 0.55 * float(summary.get("images_with_final_rows") or 0)
            - 1.25 * float(failure_counts.get("reverse_ground_answer_invalid", 0))
            - 1.0 * float(failure_counts.get("anchor_missing", 0))
            + 0.3 * float(yesno_polarity.get("negative", 0))
            + 0.3 * float(reverse_scope.get("local", 0)),
            2,
        ),
        "summary": summary,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def now_stamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def append_timeline(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now_stamp()}] {line}\n")


def gpu_snapshot() -> str:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or "nvidia-smi-unavailable"
    except Exception:
        return "nvidia-smi-unavailable"


def render_report(bundle_id: str, specs: list[AblationSpec], results: dict[str, dict[str, Any]], docs_path: Path) -> str:
    lines = [
        "# SGOCR Dev40 Follow-Up Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Source experiment: `{SOURCE_EXPERIMENT.name}`",
        f"- Control experiment: `{CONTROL_EXPERIMENT.name}`",
        f"- Legacy baseline: `{V7_CONTROL.name}`",
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
    control_metrics = collect_metrics(CONTROL_EXPERIMENT)
    v7_metrics = collect_metrics(V7_CONTROL)
    lines.append(
        f"| `control_a12` | existing | {control_metrics['accepted_qas']} | {control_metrics['qa_accept_rate']:.3f} | {control_metrics['images_with_final_rows']} | {control_metrics['yesno_positive']} | {control_metrics['yesno_negative']} | {control_metrics['reverse_global']} | {control_metrics['reverse_local']} | {control_metrics['anchor_missing']} | {control_metrics['reverse_invalid']} | {control_metrics['quality_score']:.2f} |"
    )
    lines.append(
        f"| `legacy_v7` | existing | {v7_metrics['accepted_qas']} | {v7_metrics['qa_accept_rate']:.3f} | {v7_metrics['images_with_final_rows']} | {v7_metrics['yesno_positive']} | {v7_metrics['yesno_negative']} | {v7_metrics['reverse_global']} | {v7_metrics['reverse_local']} | {v7_metrics['anchor_missing']} | {v7_metrics['reverse_invalid']} | {v7_metrics['quality_score']:.2f} |"
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
            "- This sweep focuses on easier-text recall, less chopped OCR crops, and lower-entropy location wording.",
            "- `quality` rewards accepted rows and coverage while penalizing anchor and reverse-ground failures.",
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

    cache_intermediate_dir = spec.cache_intermediate_dir or CONTROL_INTERMEDIATE
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
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "111_sgocr_dev40_overnight_sweep_v2_2026-04-05.md"
    results: dict[str, dict[str, Any]] = {}
    append_timeline(timeline_path, f"BUNDLE {args.bundle_id} start source={SOURCE_EXPERIMENT.name} control={CONTROL_EXPERIMENT.name}")
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
