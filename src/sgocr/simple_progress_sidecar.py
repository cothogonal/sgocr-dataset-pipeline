from __future__ import annotations

import argparse
import re
import time
from datetime import datetime
from pathlib import Path


STAGE_START_RE = re.compile(r"\[stage:(?P<stage>[^\]]+)\] start total=(?P<total>\d+)")
STAGE_PROGRESS_RE = re.compile(
    r"\[stage:(?P<stage>[^\]]+)\] progress completed=(?P<done>\d+)/(?P<total>\d+) pct=(?P<pct>\d+)% elapsed_s=(?P<elapsed>[0-9.]+) rate_per_s=(?P<rate>[0-9.]+)"
)
QWEN_FINISH_RE = re.compile(r"\[stage:qwen_detect_many\] finish completed=(?P<done>\d+)/(?P<total>\d+) pct=100% elapsed_s=(?P<elapsed>[0-9.]+) images=(?P<images>\d+)")
WRAPPER_STAGE_RE = re.compile(r"stage=(?P<stage>[a-zA-Z0-9_]+) counts=(?P<counts>\{.*\})")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Write a clean stage-level progress log for a live SGOCR run.")
    ap.add_argument("--build-log", required=True)
    ap.add_argument("--progress-log", required=True)
    ap.add_argument("--out-log", required=True)
    ap.add_argument("--poll-seconds", type=int, default=10)
    return ap.parse_args()


def _tail_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _latest_wrapper_stage(progress_text: str) -> str:
    stage = "unknown"
    for line in progress_text.splitlines():
        match = WRAPPER_STAGE_RE.search(line)
        if match:
            stage = str(match.group("stage"))
    return stage


def _stage_summary(build_text: str, wrapper_stage: str) -> tuple[str, str]:
    latest_stage_progress: dict[str, tuple[int, int, str, str]] = {}
    stage_starts: dict[str, int] = {}
    qwen_finish_counts = 0
    latest_qwen_progress: tuple[int, int, str, str] | None = None
    active_stage = None
    lines = build_text.splitlines()
    for line in lines:
        start_match = STAGE_START_RE.search(line)
        if start_match:
            active_stage = str(start_match.group("stage"))
            stage_starts[active_stage] = int(start_match.group("total"))
            if active_stage == "anchor_tag_discovery":
                qwen_finish_counts = 0
            continue
        progress_match = STAGE_PROGRESS_RE.search(line)
        if progress_match:
            stage = str(progress_match.group("stage"))
            latest_stage_progress[stage] = (
                int(progress_match.group("done")),
                int(progress_match.group("total")),
                str(progress_match.group("elapsed")),
                str(progress_match.group("rate")),
            )
            active_stage = stage
            if stage == "qwen_detect_many":
                latest_qwen_progress = latest_stage_progress[stage]
            continue
        if QWEN_FINISH_RE.search(line):
            qwen_finish_counts += 1

    if wrapper_stage == "anchor_tag_discovery":
        total = stage_starts.get("anchor_tag_discovery", 0)
        done = qwen_finish_counts
        pct = int((done * 100) / total) if total else 0
        qwen_suffix = ""
        if latest_qwen_progress is not None:
            q_done, q_total, q_elapsed, q_rate = latest_qwen_progress
            qwen_suffix = (
                f"; current_image={q_done}/{q_total} crops "
                f"({int((q_done * 100) / q_total) if q_total else 0}%) elapsed_s={q_elapsed} rate_per_s={q_rate}"
            )
        return (
            "anchor_tag_discovery",
            f"anchor_tag_discovery overall={done}/{total} images ({pct}%){qwen_suffix}",
        )
    if wrapper_stage == "anchor_grounding":
        done, total, elapsed, rate = latest_stage_progress.get("anchor_grounding", (0, 0, "0", "0"))
        return (
            "anchor_grounding",
            f"anchor_grounding overall={done}/{total} images ({int((done * 100) / total) if total else 0}%) elapsed_s={elapsed} rate_per_s={rate}",
        )
    if wrapper_stage == "verified_tuple_build":
        done, total, elapsed, rate = latest_stage_progress.get("verified_tuple_build", (0, 0, "0", "0"))
        return (
            "verified_tuple_build",
            f"verified_tuple_build overall={done}/{total} images ({int((done * 100) / total) if total else 0}%) elapsed_s={elapsed} rate_per_s={rate}",
        )
    return wrapper_stage, f"{wrapper_stage} waiting for simple stage counters"


def main() -> None:
    args = parse_args()
    build_log = Path(args.build_log)
    progress_log = Path(args.progress_log)
    out_log = Path(args.out_log)
    out_log.parent.mkdir(parents=True, exist_ok=True)
    last_line = None
    last_anchor_image_bucket = -1
    last_anchor_crop_bucket = -1
    while True:
        build_text = _tail_text(build_log)
        progress_text = _tail_text(progress_log)
        wrapper_stage = _latest_wrapper_stage(progress_text)
        stage_name, summary = _stage_summary(build_text, wrapper_stage)
        stamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{stamp}] stage={stage_name} {summary}"
        should_write = False
        if stage_name == "anchor_tag_discovery":
            image_match = re.search(r"overall=(\d+)/(\d+) images", summary)
            crop_match = re.search(r"current_image=(\d+)/(\d+) crops", summary)
            image_bucket = int(image_match.group(1)) // 10 if image_match else -1
            crop_bucket = int((int(crop_match.group(1)) * 10) / max(int(crop_match.group(2)), 1)) if crop_match else -1
            if image_bucket != last_anchor_image_bucket or crop_bucket != last_anchor_crop_bucket:
                last_anchor_image_bucket = image_bucket
                last_anchor_crop_bucket = crop_bucket
                should_write = True
        else:
            should_write = line != last_line
        if should_write:
            with out_log.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            last_line = line
        time.sleep(max(1, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
