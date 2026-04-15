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

from ..paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, LOGS_ROOT, REPO_ROOT


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "ablate_gemini_flash_natural2q_40_min4clean"
CONTROL_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "ablate_gemini_flash_natural2q_40_min4clean_semantic_dev40_v7"
CONTROL_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "ablate_gemini_flash_natural2q_40_min4clean_semantic_dev40_v7"


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    cache_level: str
    cli: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run overnight semantic dev40 ablation sweep.")
    ap.add_argument("--bundle-id", default=f"sgocr_dev40_sweep_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=4)
    ap.add_argument("--watch-seconds", type=int, default=60)
    ap.add_argument("--limit", type=int, default=12)
    return ap.parse_args()


def build_specs() -> list[AblationSpec]:
    return [
        AblationSpec("a01_ground_lo_034", "Lower grounding threshold for recall probe", "ocr", cli={"--grounding-threshold": "0.34"}),
        AblationSpec("a02_ground_hi_040", "Higher grounding threshold for precision probe", "ocr", cli={"--grounding-threshold": "0.40"}),
        AblationSpec("a03_tags_lo_6", "Fewer semantic tags per image", "ocr", cli={"--max-tags-per-image": "6"}),
        AblationSpec("a04_tags_hi_10", "More semantic tags per image", "ocr", cli={"--max-tags-per-image": "10"}),
        AblationSpec(
            "a05_anchor_strict",
            "Stricter anchor ordering against generic/oversized boxes",
            "ocr",
            env={
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.12",
                "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.30",
                "SGOCR_FLORENCE_REGION_BONUS": "0.06",
            },
        ),
        AblationSpec("a06_target3", "Smaller per-image QA budget", "verified", cli={"--target-per-image": "3"}),
        AblationSpec(
            "a07_neg_strict_250",
            "Stricter grounded-negative yes/no threshold",
            "verified",
            env={"SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.50"},
        ),
        AblationSpec(
            "a08_loc_lite",
            "Lower-entropy location wording and stricter disambiguation triggers",
            "verified",
            env={
                "SGOCR_LOCATION_WORDING_MODE": "lite",
                "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
                "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
                "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
                "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "5",
            },
        ),
        AblationSpec(
            "a09_teacher_vstrict",
            "Stricter anti-editorial teacher prompt",
            "verified",
            env={"SGOCR_TEACHER_STRICTNESS": "very_strict"},
        ),
        AblationSpec(
            "a10_local_bias_035",
            "More anchor-local reverse-ground answers for directional cases",
            "verified",
            env={
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.35",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.12",
            },
        ),
        AblationSpec(
            "a11_clean_combo",
            "Precision-oriented combo: stricter grounding, stricter negatives, lite wording",
            "ocr",
            cli={"--grounding-threshold": "0.38", "--max-tags-per-image": "8"},
            env={
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.12",
                "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.30",
                "SGOCR_FLORENCE_REGION_BONUS": "0.06",
                "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.50",
                "SGOCR_LOCATION_WORDING_MODE": "lite",
                "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
                "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
                "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
                "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "5",
                "SGOCR_TEACHER_STRICTNESS": "very_strict",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.24",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.10",
            },
        ),
        AblationSpec(
            "a12_recall_combo",
            "Recall-oriented combo: looser grounding, more tags, bigger budget",
            "ocr",
            cli={"--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"},
            env={"SGOCR_LOCATION_WORDING_MODE": "balanced"},
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
    metrics = {
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
            + 0.5 * float(summary.get("images_with_final_rows") or 0)
            - 1.5 * float(failure_counts.get("reverse_ground_answer_invalid", 0))
            - 1.0 * float(failure_counts.get("anchor_missing", 0))
            + 0.5 * float(reverse_scope.get("local", 0)),
            2,
        ),
        "summary": summary,
    }
    return metrics


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
        text = proc.stdout.strip()
        return text if text else "nvidia-smi-unavailable"
    except Exception:
        return "nvidia-smi-unavailable"


def render_report(bundle_id: str, specs: list[AblationSpec], results: dict[str, dict[str, Any]], docs_path: Path) -> str:
    lines = [
        f"# SGOCR Dev40 Overnight Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Source experiment: `{SOURCE_EXPERIMENT.name}`",
        f"- Control experiment: `{CONTROL_EXPERIMENT.name}`",
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
    lines.append(
        f"| `control_v7` | existing | {control_metrics['accepted_qas']} | {control_metrics['qa_accept_rate']:.3f} | {control_metrics['images_with_final_rows']} | {control_metrics['yesno_positive']} | {control_metrics['yesno_negative']} | {control_metrics['reverse_global']} | {control_metrics['reverse_local']} | {control_metrics['anchor_missing']} | {control_metrics['reverse_invalid']} | {control_metrics['quality_score']:.2f} |"
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
            f"| `{spec.name}` | ok | {metrics['accepted_qas']} | {metrics['qa_accept_rate']:.3f} | {metrics['images_with_final_rows']} | {metrics['yesno_positive']} | {metrics['yesno_negative']} | {metrics['reverse_global']} | {metrics['reverse_local']} | {metrics['anchor_missing']} | {metrics['reverse_invalid']} | {metrics['quality_score']:.2f} |"
        )
    complete = [row["metrics"] | {"name": name} for name, row in results.items() if row.get("status") == "ok"]
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
            "- `quality` is a simple heuristic: accepted rows and image coverage up, reverse-ground and anchor failures down.",
            "- All sweep runs keep the current strict verifier.",
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
        write_text(report_path, render_report(bundle_id, build_specs()[: args.limit], results, docs_path))
        write_text(docs_path, render_report(bundle_id, build_specs()[: args.limit], results, docs_path))
        return

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
        str(CONTROL_INTERMEDIATE),
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
            write_text(report_path, render_report(bundle_id, build_specs()[: args.limit], results, docs_path))
            write_text(docs_path, render_report(bundle_id, build_specs()[: args.limit], results, docs_path))
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
    write_text(report_path, render_report(bundle_id, build_specs()[: args.limit], results, docs_path))
    write_text(docs_path, render_report(bundle_id, build_specs()[: args.limit], results, docs_path))


def main() -> None:
    args = parse_args()
    specs = build_specs()[: args.limit]
    bundle_dir = LOGS_ROOT / args.bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = bundle_dir / "timeline.log"
    report_path = bundle_dir / "report.md"
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "110_sgocr_dev40_overnight_sweep_2026-04-04.md"
    results: dict[str, dict[str, Any]] = {}
    append_timeline(timeline_path, f"BUNDLE {args.bundle_id} start source={SOURCE_EXPERIMENT.name}")
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
        )
    append_timeline(timeline_path, f"BUNDLE {args.bundle_id} complete")


if __name__ == "__main__":
    main()
