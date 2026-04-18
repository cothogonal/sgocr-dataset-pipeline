from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io import write_json, write_jsonl


VALID_TEXT_RE = re.compile(r"[A-Za-z0-9]")
DEGENERATE_THIN_TEXT_RE = re.compile(r"^[1il|]+$", re.IGNORECASE)

REGION_PHRASES = {
    "ul": "upper-left area of the image",
    "uc": "top-center area of the image",
    "ur": "upper-right area of the image",
    "cl": "middle-left area of the image",
    "cc": "center of the image",
    "cr": "middle-right area of the image",
    "ll": "lower-left area of the image",
    "lc": "bottom-center area of the image",
    "lr": "lower-right area of the image",
}

REGION_SYNONYMS = {
    "ul": (
        "upper-left area of the image",
        "upper left area of the image",
        "top-left area of the image",
        "top left area of the image",
        "upper-left part of the image",
        "near the top-left of the image",
    ),
    "uc": (
        "top-center area of the image",
        "top center area of the image",
        "upper-center area of the image",
        "upper center area of the image",
        "near the top of the image",
        "upper middle of the image",
    ),
    "ur": (
        "upper-right area of the image",
        "upper right area of the image",
        "top-right area of the image",
        "top right area of the image",
        "upper-right part of the image",
        "near the top-right of the image",
    ),
    "cl": (
        "middle-left area of the image",
        "middle left area of the image",
        "left-middle area of the image",
        "left side of the image",
        "left half of the image",
        "middle-left part of the image",
    ),
    "cc": (
        "center of the image",
        "middle of the image",
        "central area of the image",
        "central part of the image",
        "near the middle of the image",
    ),
    "cr": (
        "middle-right area of the image",
        "middle right area of the image",
        "right-middle area of the image",
        "right side of the image",
        "right half of the image",
        "middle-right part of the image",
    ),
    "ll": (
        "lower-left area of the image",
        "lower left area of the image",
        "bottom-left area of the image",
        "bottom left area of the image",
        "lower-left part of the image",
        "near the bottom-left of the image",
    ),
    "lc": (
        "bottom-center area of the image",
        "bottom center area of the image",
        "lower-center area of the image",
        "lower center area of the image",
        "near the bottom of the image",
        "lower middle of the image",
    ),
    "lr": (
        "lower-right area of the image",
        "lower right area of the image",
        "bottom-right area of the image",
        "bottom right area of the image",
        "lower-right part of the image",
        "near the bottom-right of the image",
    ),
}


@dataclass(frozen=True)
class BootstrapTuple:
    image_id: str
    image_path: str
    image_width: int
    image_height: int
    source_split: str
    source_dataset: str
    ann_id: str
    answer: str
    answer_normalized: str
    text_polygon: list[float]
    text_bbox: list[float]
    region_key: str
    anchor_label: str
    anchor_synonyms: tuple[str, ...]
    relation: str
    unique: bool
    answer_level: str
    valid_text_count: int
    area_fraction: float
    text_length: int
    score: float
    density_bucket: str
    area_bucket: str

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["anchor_synonyms"] = list(self.anchor_synonyms)
        return row


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def is_valid_text(text: str) -> bool:
    text = (text or "").strip()
    if not text or text == ".":
        return False
    if len(text) < 2 or len(text) > 32:
        return False
    if len(text.split()) > 6:
        return False
    compact = re.sub(r"[\W_]+", "", text)
    if len(compact) >= 4:
        if DEGENERATE_THIN_TEXT_RE.fullmatch(compact):
            return False
        if len(set(compact.lower())) == 1 and compact[0].isalnum():
            return False
    return bool(VALID_TEXT_RE.search(text))


def region_key_for_bbox(bbox: list[float], width: int, height: int) -> str:
    x, y, w, h = [float(v) for v in bbox]
    cx = x + w / 2.0
    cy = y + h / 2.0
    x_bin = 0 if cx < width / 3.0 else 1 if cx < (2.0 * width / 3.0) else 2
    y_bin = 0 if cy < height / 3.0 else 1 if cy < (2.0 * height / 3.0) else 2
    x_keys = ("l", "c", "r")
    y_keys = ("u", "c", "l")
    return f"{y_keys[y_bin]}{x_keys[x_bin]}"


def density_bucket(count: int) -> str:
    if count <= 5:
        return "low"
    if count <= 14:
        return "medium"
    return "high"


def area_bucket(area_frac: float) -> str:
    if area_frac < 0.005:
        return "small"
    if area_frac < 0.02:
        return "medium"
    return "large"


def candidate_score(*, area_frac: float, text: str) -> float:
    length_score = min(len((text or "").strip()), 16) / 16.0
    return float(area_frac * 100.0 + length_score)


def load_textocr_bootstrap_candidates(raw_json_path: Path, images_root: Path) -> list[BootstrapTuple]:
    payload = json.loads(raw_json_path.read_text(encoding="utf-8"))
    imgs = payload.get("imgs", {})
    anns = payload.get("anns", {})
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in anns.values():
        image_id = str(ann.get("image_id") or "")
        if image_id:
            by_image[image_id].append(ann)

    out: list[BootstrapTuple] = []
    for image_id, img_meta in imgs.items():
        width = int(img_meta.get("width") or 0)
        height = int(img_meta.get("height") or 0)
        file_name = str(img_meta.get("file_name") or "")
        image_path = images_root / Path(file_name).name
        if width <= 0 or height <= 0 or not image_path.is_file():
            continue
        valid_anns = []
        for ann in by_image.get(str(image_id), []):
            text = str(ann.get("utf8_string") or "")
            if not is_valid_text(text):
                continue
            bbox = ann.get("bbox") or []
            points = ann.get("points") or []
            if len(bbox) != 4 or len(points) < 8:
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 1.0 or h <= 1.0:
                continue
            area_frac = float((w * h) / max(width * height, 1))
            region_key = region_key_for_bbox([x, y, w, h], width, height)
            valid_anns.append(
                {
                    "ann": ann,
                    "text": text.strip(),
                    "bbox": [x, y, w, h],
                    "points": [float(v) for v in points[:8]],
                    "area_frac": area_frac,
                    "region_key": region_key,
                    "score": candidate_score(area_frac=area_frac, text=text),
                }
            )
        if not valid_anns:
            continue
        by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in valid_anns:
            by_region[row["region_key"]].append(row)
        unique_region_rows = [rows[0] for rows in by_region.values() if len(rows) == 1]
        if not unique_region_rows:
            continue
        best = max(unique_region_rows, key=lambda row: (row["score"], row["area_frac"]))
        area_frac = float(best["area_frac"])
        out.append(
            BootstrapTuple(
                image_id=str(image_id),
                image_path=str(image_path),
                image_width=width,
                image_height=height,
                source_split="val",
                source_dataset="textocr_bootstrap",
                ann_id=str(best["ann"].get("id") or ""),
                answer=best["text"],
                answer_normalized=normalize_answer(best["text"]),
                text_polygon=list(best["points"]),
                text_bbox=list(best["bbox"]),
                region_key=str(best["region_key"]),
                anchor_label=REGION_PHRASES[str(best["region_key"])],
                anchor_synonyms=REGION_SYNONYMS[str(best["region_key"])],
                relation="in",
                unique=True,
                answer_level="word",
                valid_text_count=len(valid_anns),
                area_fraction=area_frac,
                text_length=len(best["text"]),
                score=float(best["score"]),
                density_bucket=density_bucket(len(valid_anns)),
                area_bucket=area_bucket(area_frac),
            )
        )
    return out


def select_dev_subset(candidates: list[BootstrapTuple], *, limit: int = 200, seed: int = 42) -> list[BootstrapTuple]:
    groups: dict[tuple[str, str, str], list[BootstrapTuple]] = defaultdict(list)
    for row in candidates:
        groups[(row.density_bucket, row.area_bucket, row.region_key)].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: (-row.score, row.image_id))
    rng = random.Random(seed)
    ordered_keys = sorted(groups.keys())
    selected: list[BootstrapTuple] = []
    seen: set[str] = set()
    while len(selected) < limit:
        progressed = False
        rng.shuffle(ordered_keys)
        for key in ordered_keys:
            rows = groups[key]
            while rows and rows[0].image_id in seen:
                rows.pop(0)
            if not rows:
                continue
            row = rows.pop(0)
            if row.image_id in seen:
                continue
            seen.add(row.image_id)
            selected.append(row)
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    selected.sort(key=lambda row: row.image_id)
    return selected


def build_and_write_dev_subset(
    *,
    raw_json_path: Path,
    images_root: Path,
    manifest_path: Path,
    tuples_path: Path,
    notes_path: Path,
    limit: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    candidates = load_textocr_bootstrap_candidates(raw_json_path, images_root)
    selected = select_dev_subset(candidates, limit=limit, seed=seed)
    manifest = {
        "name": "dev200_textocr_bootstrap_v1",
        "source": "TextOCR_0.1_val",
        "image_count": len(selected),
        "seed": int(seed),
        "selection_policy": "one unique region-grounded word tuple per image, stratified by density/area/region",
        "items": [row.to_dict() for row in selected],
    }
    notes_lines = [
        "# dev200_textocr_bootstrap_v1",
        "",
        "- bootstrap source: `TextOCR_0.1_val`",
        "- tuple type: `word-level`",
        "- grounding type: `image-region bootstrap`, not object-anchor grounding",
        f"- selected images: `{len(selected)}`",
        f"- seed: `{seed}`",
    ]
    write_json(manifest_path, manifest)
    write_jsonl(tuples_path, [row.to_dict() for row in selected])
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")
    return manifest
