from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .bootstrap import normalize_answer
from .bootstrap_kd import center_from_box, centroid_from_polygon, determine_relation, distance_point_to_box, overlap_fraction
from .semantic_dev40_tuning import load_semantic_dev40_tuning


STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "with",
    "up",
    "holding",
    "close",
    "view",
    "photo",
    "image",
    "there",
    "this",
    "that",
}

COLOR_TERMS = (
    "light blue",
    "dark blue",
    "navy blue",
    "sky blue",
    "bright red",
    "dark red",
    "light green",
    "dark green",
    "light gray",
    "dark gray",
    "light grey",
    "dark grey",
    "black",
    "white",
    "gray",
    "grey",
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "tan",
    "gold",
    "silver",
    "beige",
    "cream",
)

GENERIC_EXCLUDE = {
    "background",
    "room",
    "area",
    "image",
    "picture",
    "scene",
    "front",
    "back",
    "top",
    "bottom",
    "middle",
    "center",
}

NOISY_EXCLUDE = {
    "object",
    "objects",
    "detected",
    "detection",
    "pad",
    "padding",
    "unknown",
    "other",
    "artifact",
    "artifacts",
    "text",
    "words",
    "letters",
    "writing",
}

FALLBACK_TAGS = (
    "sign",
    "label",
    "poster",
    "box",
    "screen",
    "book",
    "phone",
    "bottle",
    "can",
    "bag",
    "shirt",
    "door",
    "window",
    "wall",
    "awning",
    "table",
    "counter",
    "shelf",
    "vehicle",
    "person",
)

SAFE_ANCHOR_TERMS = (
    "airplane tail",
    "plane tail",
    "vehicle side",
    "bus side",
    "car side",
    "car hood",
    "car bumper",
    "car windshield",
    "license plate",
    "baseball helmet",
    "helmet",
    "head",
    "face",
    "jersey chest",
    "shirt chest",
    "shirt sleeve",
    "jersey sleeve",
    "sleeve",
    "collar",
    "tail",
    "wing",
    "tube",
    "frame",
    "bottle label",
    "can label",
    "box side",
    "panel",
    "sign wall",
    "display wall",
    "poster wall",
    "sign panel",
    "display panel",
    "poster panel",
    "ad board",
    "store window",
    "car door",
    "document",
    "page",
    "sheet",
    "chart",
    "diagram",
    "flyer",
    "brochure",
    "booklet",
    "whiteboard",
    "chalkboard",
    "book cover",
    "storefront",
    "billboard",
    "nameplate",
    "banner",
    "poster",
    "screen",
    "monitor",
    "display",
    "menu",
    "sticker",
    "plaque",
    "awning",
    "window",
    "door",
    "wall",
    "roof",
    "board",
    "label",
    "sign",
    "tablet",
    "phone",
    "shirt",
    "jersey",
    "jacket",
    "sleeve",
    "uniform",
    "vest",
    "apron",
    "backpack",
    "package",
    "bottle",
    "can",
    "crate",
    "box",
    "bag",
    "shelf",
    "counter",
    "table",
    "desk",
    "rack",
    "building",
    "facade",
    "entrance",
    "vehicle",
    "train",
    "truck",
    "taxi",
    "bus",
    "car",
    "book",
)

SAFE_FALLBACK_TAGS = (
    "sign wall",
    "display panel",
    "document",
    "chart",
    "diagram",
    "box",
    "bottle",
    "can",
    "book",
    "phone",
    "tablet",
    "shirt",
    "door",
    "window",
    "wall",
    "awning",
    "table",
    "counter",
    "shelf",
    "board",
    "screen",
    "sign",
)

OPEN_COLOR_OBJECT_TERMS = (
    "helmet",
    "head",
    "face",
    "shirt",
    "jersey",
    "jacket",
    "sleeve",
    "collar",
    "chest",
    "bottle",
    "can",
    "box",
    "bag",
    "phone",
    "tablet",
    "screen",
    "monitor",
    "sign",
    "poster",
    "banner",
    "label",
    "document",
    "book",
    "page",
    "window",
    "door",
    "awning",
    "storefront",
    "table",
    "counter",
    "shelf",
    "vehicle",
    "car",
    "truck",
    "bus",
    "train",
    "airplane",
    "tail",
    "wing",
    "hood",
    "bumper",
    "windshield",
)

QWEN_LOCAL_DISCOVERY_CATEGORIES = (
    "label",
    "sign",
    "banner",
    "poster",
    "document",
    "chart",
    "diagram",
    "book",
    "book cover",
    "screen",
    "phone",
    "tablet",
    "bottle",
    "can",
    "box",
    "bag",
    "shirt",
    "jersey",
    "jacket",
    "window",
    "door",
    "awning",
    "storefront",
    "table",
    "counter",
    "shelf",
    "vehicle",
    "car",
    "bus",
    "truck",
    "train",
    "person",
)

QWEN_GLOBAL_INVENTORY_CATEGORIES = (
    "person",
    "face",
    "head",
    "helmet",
    "hat",
    "shirt",
    "jersey",
    "jacket",
    "bag",
    "backpack",
    "bottle",
    "can",
    "cup",
    "box",
    "package",
    "book",
    "page",
    "document",
    "label",
    "sticker",
    "sign",
    "poster",
    "banner",
    "menu",
    "screen",
    "monitor",
    "phone",
    "tablet",
    "chart",
    "diagram",
    "board",
    "whiteboard",
    "chalkboard",
    "window",
    "door",
    "wall",
    "building",
    "storefront",
    "table",
    "counter",
    "shelf",
    "vehicle",
    "car",
    "truck",
    "bus",
    "train",
    "airplane",
    "boat",
)

QWEN_INDEPENDENT_RAW_INVENTORY_CATEGORY_TEXT = (
    "all visible objects, object parts, surfaces, landmarks, and scene elements that could be useful as anchors for referring to text later, "
    "even if they do not themselves contain text; prefer specific visible object or object-part labels; include a visible color adjective when it is clear and stable; "
    "use 'with' to connect multiple descriptive attributes (e.g. 'man with dark hoodie' not 'man dark hoodie')"
)

GENERIC_TEXT_ANCHORS = {"sign", "poster", "label", "screen", "display", "board", "wall", "panel"}
GENERIC_PRIMARY_ANCHORS = {
    "sign",
    "poster",
    "label",
    "screen",
    "display",
    "board",
    "wall",
    "panel",
    "sign wall",
    "display wall",
    "poster wall",
    "sign panel",
    "display panel",
    "poster panel",
    "ad board",
}

ANCHOR_CATEGORIES = {
    "text_container": {"sign", "sign wall", "sign panel", "poster", "poster wall", "poster panel", "banner", "billboard", "label", "sticker", "menu", "screen", "monitor", "book", "book cover", "album", "cover", "board", "display wall", "display panel", "ad board", "plaque", "nameplate", "document", "page", "sheet", "chart", "diagram", "flyer", "brochure", "booklet"},
    "clothing": {"shirt", "jersey", "hat", "cap", "jacket", "sleeve", "uniform", "vest", "apron"},
    "vehicle": {"car", "car door", "bus", "truck", "van", "taxi", "train", "vehicle"},
    "building": {"building", "storefront", "wall", "door", "window", "store window", "awning", "facade", "entrance", "roof"},
    "container": {"box", "package", "bottle", "bottles", "bottle beer", "can", "bag", "backpack", "crate", "bin"},
    "furniture": {"shelf", "table", "counter", "desk", "rack"},
    "device": {"phone", "tablet", "monitor", "screen", "display"},
}


@dataclass(frozen=True)
class AnchorCandidate:
    label: str
    box: list[float]
    score: float
    source: str
    relevance: float
    caption: str | None = None


def expand_box(box: list[float], *, image_width: int, image_height: int, scale: float = 2.5) -> list[int]:
    x1, y1, x2, y2 = [float(value) for value in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half_w = max((x2 - x1) * scale / 2.0, 8.0)
    half_h = max((y2 - y1) * scale / 2.0, 8.0)
    left = max(0, int(math.floor(cx - half_w)))
    top = max(0, int(math.floor(cy - half_h)))
    right = min(int(image_width), int(math.ceil(cx + half_w)))
    bottom = min(int(image_height), int(math.ceil(cy + half_h)))
    return [left, top, right, bottom]


def crop_from_box(image: Image.Image, box: list[float], *, pad_ratio: float = 0.04) -> Image.Image:
    if pad_ratio == 0.04:
        pad_ratio = float(load_semantic_dev40_tuning().semantic_crop_pad_ratio)
    x1, y1, x2, y2 = [float(value) for value in box]
    pad = max((x2 - x1), (y2 - y1)) * pad_ratio
    left = max(0, int(math.floor(x1 - pad)))
    top = max(0, int(math.floor(y1 - pad)))
    right = min(image.width, int(math.ceil(x2 + pad)))
    bottom = min(image.height, int(math.ceil(y2 + pad)))
    return image.crop((left, top, right, bottom)).convert("RGB")


def extract_caption_tags(caption: str, *, max_tags: int = 6) -> list[str]:
    lower = re.sub(r"[^a-z0-9\s-]+", " ", str(caption or "").lower())
    lower = re.sub(r"\s+", " ", lower).strip()
    if not lower:
        return []

    matches: list[str] = []
    for phrase in sorted(FALLBACK_TAGS, key=lambda item: (-len(item), item)):
        if re.search(rf"\b{re.escape(phrase)}\b", lower):
            matches.append(phrase)

    words = [word for word in lower.split() if len(word) >= 3 and word not in STOPWORDS and not word.isdigit()]
    for idx, word in enumerate(words):
        matches.append(word)
        if word.endswith("s") and len(word) >= 5:
            matches.append(word[:-1])
        if idx + 1 < len(words):
            bigram = f"{word} {words[idx + 1]}"
            if all(part not in STOPWORDS for part in bigram.split()):
                matches.append(bigram)

    counts = Counter(matches)
    ordered = sorted(counts, key=lambda item: (-counts[item], len(item.split()), item))
    deduped: list[str] = []
    for item in ordered:
        if item in GENERIC_EXCLUDE:
            continue
        if item in deduped:
            continue
        if any(item in existing for existing in deduped if item != existing):
            continue
        deduped.append(item)
        if len(deduped) >= max_tags:
            break
    return deduped


def sanitize_anchor_label(raw_text: str) -> str | None:
    lowered = re.sub(r"[^a-z0-9\s-]+", " ", str(raw_text or "").lower())
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        return None
    color = extract_anchor_color(lowered)
    if lowered in GENERIC_EXCLUDE or lowered in NOISY_EXCLUDE:
        return None
    if "no object detected" in lowered or "object detected" in lowered:
        return None
    if "wall" in lowered:
        if "poster" in lowered:
            return "poster wall"
        if any(token in lowered for token in ("sign", "display", "ad", "billboard")):
            return "sign wall" if "sign" in lowered else "display wall"
        return "wall"
    if "panel" in lowered:
        if "poster" in lowered:
            return "poster panel"
        if any(token in lowered for token in ("sign", "display", "ad")):
            return "sign panel" if "sign" in lowered else "display panel"
        return "panel"
    if "board" in lowered and any(token in lowered for token in ("sign", "display", "ad")):
        return "ad board" if "ad" in lowered else "display panel"
    if "window" in lowered and "store" in lowered:
        return "store window"
    if "door" in lowered and "car" in lowered:
        return "car door"
    if any(token in lowered for token in ("document", "page", "sheet", "paper", "worksheet", "form", "handout")):
        return "document" if any(token in lowered for token in ("document", "sheet", "paper", "worksheet", "form", "handout")) else "page"
    if any(token in lowered for token in ("chart", "graph", "diagram")):
        return "chart" if "chart" in lowered or "graph" in lowered else "diagram"
    if any(token in lowered for token in ("flyer", "pamphlet", "leaflet")):
        return "flyer"
    if any(token in lowered for token in ("brochure", "booklet")):
        return "brochure" if "brochure" in lowered else "booklet"
    if any(token in lowered for token in ("can", "tin")):
        return f"{color} can" if color else "can"
    if any(token in lowered for token in ("bottle", "jar", "flask")):
        return f"{color} bottle" if color else "bottle"
    if color:
        tokens = lowered.split()
        filtered = [token for token in tokens if token not in {"the", "a", "an", "of", "on", "near", "part", "side"}]
        for width in (3, 2, 1):
            if len(filtered) >= width:
                phrase = " ".join(filtered[-width:])
                if phrase in SAFE_ANCHOR_TERMS:
                    return f"{color} {phrase}" if not phrase.startswith(color) else phrase
        for term in OPEN_COLOR_OBJECT_TERMS:
            if re.search(rf"\b{re.escape(term)}\b", lowered):
                return f"{color} {term}"
    if lowered.endswith("s") and lowered[:-1] in SAFE_ANCHOR_TERMS:
        lowered = lowered[:-1]
    for term in SAFE_ANCHOR_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            return f"{color} {term}" if color else term
    return None


def is_generic_anchor_label(raw_text: str) -> bool:
    clean = sanitize_anchor_label(raw_text)
    return bool(clean and clean in GENERIC_PRIMARY_ANCHORS)


def extract_anchor_color(raw_text: str) -> str | None:
    lowered = re.sub(r"[^a-z0-9\s-]+", " ", str(raw_text or "").lower())
    lowered = re.sub(r"\s+", " ", lowered).strip()
    if not lowered:
        return None
    for color in sorted(COLOR_TERMS, key=lambda item: (-len(item), item)):
        if re.search(rf"\b{re.escape(color)}\b", lowered):
            return color
    return None


def sanitize_anchor_tags(*raw_sources: str, max_tags: int = 6) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in raw_sources:
        clean = re.sub(r"[;,/|]+", " ", str(raw or ""))
        parts = [clean] + clean.split()
        for part in parts:
            label = sanitize_anchor_label(part)
            if not label or label in seen:
                continue
            seen.add(label)
            out.append(label)
            if len(out) >= max_tags:
                return out
    return out


def normalize_independent_anchor_label(raw_text: str) -> str | None:
    lowered = re.sub(r"[^a-z0-9\\s-]+", " ", str(raw_text or "").lower())
    lowered = re.sub(r"\\s+", " ", lowered).strip()
    if not lowered:
        return None
    if any(
        phrase in lowered
        for phrase in (
            "bbox",
            "coordinates",
            "json format",
            "category",
            "categories",
            "useful as anchors",
            "referring to text",
            "object parts surfaces",
            "report bbox",
            "locate every instance",
        )
    ):
        return None
    cleaned = sanitize_anchor_label(lowered)
    if cleaned:
        return cleaned
    tokens = [token for token in lowered.split() if token not in STOPWORDS]
    while tokens and tokens[0] in {"visible", "specific", "clear", "stable", "useful", "possible"}:
        tokens.pop(0)
    while tokens and tokens[-1] in {"visible", "specific", "clear", "stable", "useful", "possible"}:
        tokens.pop()
    if not tokens:
        return None
    phrase = " ".join(tokens[:5]).strip()
    if not phrase or phrase in GENERIC_EXCLUDE or phrase in NOISY_EXCLUDE:
        return None
    if phrase.endswith("s") and len(tokens) == 1 and len(phrase) >= 5 and phrase[:-1] not in GENERIC_EXCLUDE:
        phrase = phrase[:-1]
    if phrase in GENERIC_EXCLUDE or phrase in NOISY_EXCLUDE:
        return None
    return phrase


def _connected_text_component(
    *,
    primary_node_ids: list[str],
    all_text_nodes: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    nodes_by_id = {str(node["node_id"]): node for node in all_text_nodes}
    seed_ids = [node_id for node_id in primary_node_ids if node_id in nodes_by_id]
    if not seed_ids:
        return []
    component_ids = set(seed_ids)
    frontier = list(seed_ids)
    while frontier:
        current_id = frontier.pop()
        current_box = list(nodes_by_id[current_id]["bbox"])
        for other in all_text_nodes:
            other_id = str(other["node_id"])
            if other_id in component_ids:
                continue
            if text_boxes_connected(current_box, list(other["bbox"]), image_size):
                component_ids.add(other_id)
                frontier.append(other_id)
    return [nodes_by_id[node_id] for node_id in component_ids]


def categorize_anchor(label: str) -> str:
    lowered = str(label or "").strip().lower()
    for category, members in ANCHOR_CATEGORIES.items():
        if lowered in members or any(member in lowered or lowered in member for member in members):
            return category
    return "other"


def relation_between_text_and_anchor(text_box: list[float], anchor_box: list[float], image_size: tuple[int, int]) -> str | None:
    overlap = overlap_fraction(text_box, anchor_box)
    if overlap >= 0.35:
        return "on"

    tx1, ty1, tx2, ty2 = [float(value) for value in text_box]
    ax1, ay1, ax2, ay2 = [float(value) for value in anchor_box]
    text_area = max((tx2 - tx1) * (ty2 - ty1), 1e-6)
    anchor_area = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    area_ratio = max(text_area, anchor_area) / min(text_area, anchor_area)
    if area_ratio > 5.0:
        return None

    text_cx, text_cy = center_from_box(text_box)
    anchor_cx, anchor_cy = center_from_box(anchor_box)
    x_overlap = max(0.0, min(tx2, ax2) - max(tx1, ax1)) / max(min(tx2 - tx1, ax2 - ax1), 1e-6)
    y_overlap = max(0.0, min(ty2, ay2) - max(ty1, ay1)) / max(min(ty2 - ty1, ay2 - ay1), 1e-6)
    vertical_gap = max(0.0, max(ay1 - ty2, ty1 - ay2))
    horizontal_gap = max(0.0, max(ax1 - tx2, tx1 - ax2))
    width, height = image_size

    if text_cy < anchor_cy and x_overlap >= 0.20 and vertical_gap <= 0.20 * height:
        return "above"
    if text_cy > anchor_cy and x_overlap >= 0.20 and vertical_gap <= 0.20 * height:
        return "below"
    if text_cx < anchor_cx and y_overlap >= 0.20 and horizontal_gap <= 0.30 * width:
        return "left_of"
    if text_cx > anchor_cx and y_overlap >= 0.20 and horizontal_gap <= 0.30 * width:
        return "right_of"
    return None


def anchor_relevance(text_box: list[float], text_polygon: list[float], anchor_box: list[float], image_size: tuple[int, int]) -> float:
    overlap = overlap_fraction(text_box, anchor_box)
    distance = distance_point_to_box(centroid_from_polygon(text_polygon), anchor_box)
    scale = max(float(max(image_size)), 1.0)
    proximity = max(0.0, 1.0 - distance / scale)
    tx1, ty1, tx2, ty2 = [float(value) for value in text_box]
    ax1, ay1, ax2, ay2 = [float(value) for value in anchor_box]
    inter_x1 = max(tx1, ax1)
    inter_y1 = max(ty1, ay1)
    inter_x2 = min(tx2, ax2)
    inter_y2 = min(ty2, ay2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    text_area = max((tx2 - tx1) * (ty2 - ty1), 1.0)
    anchor_area = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    image_area = max(float(image_size[0]) * float(image_size[1]), 1.0)
    union_area = max(text_area + anchor_area - inter_area, 1.0)
    iou = inter_area / union_area
    anchor_area_frac = anchor_area / image_area
    size_ratio = anchor_area / text_area
    text_center = centroid_from_polygon(text_polygon)
    center_inside = 1.0 if ax1 <= text_center[0] <= ax2 and ay1 <= text_center[1] <= ay2 else 0.0
    fit_bonus = 0.22 if center_inside and 1.5 <= size_ratio <= 80.0 else 0.10 if center_inside and size_ratio <= 140.0 else 0.0
    penalty = 0.0
    if anchor_area_frac > 0.55:
        penalty += min(0.85, (anchor_area_frac - 0.55) * 1.7)
    if size_ratio > 140.0:
        penalty += min(0.75, math.log(size_ratio / 140.0 + 1.0) * 0.45)
    return float(max(0.0, overlap * 0.32 + iou * 0.28 + proximity * 0.18 + fit_bonus - penalty))


def _box_dims(box: list[float]) -> tuple[float, float]:
    return max(float(box[2]) - float(box[0]), 1.0), max(float(box[3]) - float(box[1]), 1.0)


def _box_gap(box_a: list[float], box_b: list[float]) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    gap_x = max(0.0, max(bx1 - ax2, ax1 - bx2))
    gap_y = max(0.0, max(by1 - ay2, ay1 - by2))
    return gap_x, gap_y


def _axis_overlap_fraction(box_a: list[float], box_b: list[float], *, axis: str) -> float:
    if axis == "x":
        a1, a2 = float(box_a[0]), float(box_a[2])
        b1, b2 = float(box_b[0]), float(box_b[2])
    else:
        a1, a2 = float(box_a[1]), float(box_a[3])
        b1, b2 = float(box_b[1]), float(box_b[3])
    overlap = max(0.0, min(a2, b2) - max(a1, b1))
    denom = max(min(a2 - a1, b2 - b1), 1e-6)
    return overlap / denom


def text_boxes_connected(box_a: list[float], box_b: list[float], image_size: tuple[int, int]) -> bool:
    if overlap_fraction(box_a, box_b) >= 0.05 or overlap_fraction(box_b, box_a) >= 0.05:
        return True
    width_a, height_a = _box_dims(box_a)
    width_b, height_b = _box_dims(box_b)
    gap_x, gap_y = _box_gap(box_a, box_b)
    x_overlap = _axis_overlap_fraction(box_a, box_b, axis="x")
    y_overlap = _axis_overlap_fraction(box_a, box_b, axis="y")
    max_height = max(height_a, height_b)
    max_span = max(width_a, height_a, width_b, height_b)
    if y_overlap >= 0.30 and gap_x <= 2.2 * max_height:
        return True
    if x_overlap >= 0.30 and gap_y <= 2.2 * max_height:
        return True
    center_dist = math.dist(center_from_box(box_a), center_from_box(box_b))
    return center_dist <= max(40.0, 0.11 * float(max(image_size)), 2.4 * max_span)


def _boxes_support_each_other(box_a: list[float], box_b: list[float]) -> bool:
    overlap_ab = overlap_fraction(box_a, box_b)
    overlap_ba = overlap_fraction(box_b, box_a)
    if overlap_ab >= 0.35 or overlap_ba >= 0.35:
        return True
    center_a = center_from_box(box_a)
    center_b = center_from_box(box_b)
    diag = max(
        math.hypot(float(box_a[2]) - float(box_a[0]), float(box_a[3]) - float(box_a[1])),
        math.hypot(float(box_b[2]) - float(box_b[0]), float(box_b[3]) - float(box_b[1])),
        1.0,
    )
    return math.dist(center_a, center_b) <= 0.22 * diag


def annotate_anchor_candidate_support(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = [dict(candidate) for candidate in candidates]
    for idx, row in enumerate(out):
        support_labels: set[str] = set()
        support_sources: set[str] = set()
        support_count = 0
        for other in out:
            if str(other.get("relation") or "") != str(row.get("relation") or ""):
                continue
            if not _boxes_support_each_other(list(row["box"]), list(other["box"])):
                continue
            support_count += 1
            support_sources.add(str(other.get("source") or "unknown"))
            support_labels.add(str(other.get("label") or ""))
        row["support_count"] = support_count
        row["source_support_count"] = len(support_sources)
        row["support_labels"] = sorted(label for label in support_labels if label)
        out[idx] = row
    return out


def consolidate_anchor_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    tuning = load_semantic_dev40_tuning()
    clusters: list[list[dict[str, Any]]] = []
    ordered = sorted(
        [dict(candidate) for candidate in candidates],
        key=lambda row: (-float(row.get("selection_score") or row.get("relevance") or 0.0), -float(row.get("score") or 0.0), row.get("label") or ""),
    )
    for row in ordered:
        placed = False
        for cluster in clusters:
            exemplar = cluster[0]
            if str(exemplar.get("relation") or "") != str(row.get("relation") or ""):
                continue
            overlap_ab = overlap_fraction(list(row["box"]), list(exemplar["box"]))
            overlap_ba = overlap_fraction(list(exemplar["box"]), list(row["box"]))
            iou = _anchor_box_iou(list(row["box"]), list(exemplar["box"]))
            if min(overlap_ab, overlap_ba) >= tuning.anchor_conflict_overlap or iou >= max(0.55, tuning.anchor_conflict_overlap - 0.10):
                cluster.append(row)
                placed = True
                break
        if not placed:
            clusters.append([row])

    consolidated: list[dict[str, Any]] = []
    for cluster in clusters:
        label_scores: Counter[str] = Counter()
        label_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in cluster:
            label = sanitize_anchor_label(str(row.get("label") or "")) or str(row.get("label") or "")
            row["label"] = label
            label_rows[label].append(row)
            score = float(row.get("selection_score") or row.get("relevance") or 0.0) + 0.10 * float(row.get("support_count") or 1.0)
            if label not in GENERIC_TEXT_ANCHORS:
                score += 0.08
            if len(label.split()) >= 2:
                score += 0.04
            if str(row.get("source") or "") == "florence_region":
                score += 0.03
            label_scores[label] += score
        best_label = max(
            label_scores.items(),
            key=lambda item: (item[1], item[0] not in GENERIC_TEXT_ANCHORS, len(item[0].split()), item[0]),
        )[0]
        representative = max(
            label_rows[best_label],
            key=lambda row: (float(row.get("selection_score") or row.get("relevance") or 0.0), float(row.get("score") or 0.0)),
        )
        merged = dict(representative)
        merged["label"] = best_label
        merged["alternate_labels"] = sorted(label_scores.keys())
        merged["label_conflict_count"] = max(0, len(label_scores) - 1)
        merged["support_labels"] = sorted({*(merged.get("support_labels") or []), *label_scores.keys()})
        merged["support_count"] = max(int(merged.get("support_count") or 1), len(cluster))
        merged["cluster_size"] = len(cluster)
        consolidated.append(merged)
    return consolidated


def expand_grounding_tags_for_node(
    *,
    base_tags: list[str],
    node: dict[str, Any],
    image_nodes: list[dict[str, Any]],
    image_size: tuple[int, int],
    mode: str,
) -> list[str]:
    if mode == "none":
        return []
    component_nodes = _connected_text_component(
        primary_node_ids=[str(node["node_id"])],
        all_text_nodes=image_nodes,
        image_size=image_size,
    )
    cluster_size = len(component_nodes) if component_nodes else 1
    lowered = {str(tag).strip().lower() for tag in base_tags if str(tag).strip()}
    extras: list[str] = []

    def add(tag: str) -> None:
        cleaned = sanitize_anchor_label(tag)
        if not cleaned:
            return
        if cleaned not in lowered and cleaned not in extras:
            extras.append(cleaned)

    if cluster_size >= 3:
        if {"wall", "poster", "sign", "board", "display"} & lowered:
            add("sign wall")
            add("display wall")
            add("sign panel")
            add("display panel")
            if mode == "aggressive":
                add("poster wall")
                add("poster panel")
        if {"board", "display", "sign"} & lowered and mode == "aggressive":
            add("ad board")

    if cluster_size >= 2 and mode == "aggressive":
        if "window" in lowered:
            add("store window")
        if "door" in lowered and "car" in lowered:
            add("car door")
    return extras


def local_text_location_metadata(
    *,
    primary_box: list[float],
    primary_node_ids: list[str],
    all_text_nodes: list[dict[str, Any]],
    image_size: tuple[int, int],
    coarse_phrase: str,
) -> dict[str, Any]:
    tuning = load_semantic_dev40_tuning()
    component_nodes = _connected_text_component(
        primary_node_ids=primary_node_ids,
        all_text_nodes=all_text_nodes,
        image_size=image_size,
    )
    if not component_nodes:
        return {
            "specific_location_phrase": coarse_phrase,
            "specific_location_synonyms": [coarse_phrase],
            "cluster_shape": "singleton",
            "cluster_size": 1,
            "cluster_resolvable_count": 0,
            "cluster_unresolvable_count": 0,
            "bucket_key": "singleton",
            "bucket_occupancy": 1,
            "cluster_bbox": primary_box,
            "layout_detail": "no nearby text neighbors",
        }
    component_boxes = [list(node["bbox"]) for node in component_nodes]
    cluster_bbox = [
        min(float(box[0]) for box in component_boxes),
        min(float(box[1]) for box in component_boxes),
        max(float(box[2]) for box in component_boxes),
        max(float(box[3]) for box in component_boxes),
    ]
    centers = [center_from_box(list(node["bbox"])) for node in component_nodes]
    x_values = [point[0] for point in centers]
    y_values = [point[1] for point in centers]
    x_range = max(x_values) - min(x_values) if x_values else 0.0
    y_range = max(y_values) - min(y_values) if y_values else 0.0
    bbox_w = max(cluster_bbox[2] - cluster_bbox[0], 1.0)
    bbox_h = max(cluster_bbox[3] - cluster_bbox[1], 1.0)
    bucket_names = []
    for center_x, center_y in centers:
        bucket_key, bucket_name = _bucket_name_from_fracs((center_x - cluster_bbox[0]) / bbox_w, (center_y - cluster_bbox[1]) / bbox_h)
        bucket_names.append((bucket_key, bucket_name))
    x_bucket_count = len({name.split("_")[1] for name, _ in bucket_names})
    y_bucket_count = len({name.split("_")[0] for name, _ in bucket_names})

    if len(component_nodes) <= 1:
        cluster_shape = "singleton"
    elif len(component_nodes) >= 4 and x_bucket_count >= 2 and y_bucket_count >= 2:
        cluster_shape = "grid"
    elif x_range > y_range * 1.6:
        cluster_shape = "row"
    elif y_range > x_range * 1.6:
        cluster_shape = "stack"
    else:
        cluster_shape = "cluster"

    primary_center = center_from_box(primary_box)
    primary_x, primary_y = primary_center
    x_sorted = sorted(centers, key=lambda point: point[0])
    y_sorted = sorted(centers, key=lambda point: point[1])
    x_rank = next((idx for idx, point in enumerate(x_sorted) if point == min(x_sorted, key=lambda item: abs(item[0] - primary_x) + abs(item[1] - primary_y))), 0)
    y_rank = next((idx for idx, point in enumerate(y_sorted) if point == min(y_sorted, key=lambda item: abs(item[0] - primary_x) + abs(item[1] - primary_y))), 0)
    n = len(component_nodes)

    descriptor = ""
    if n <= 1:
        descriptor = ""
    elif cluster_shape == "row":
        if n == 2:
            descriptor = "left text" if x_rank == 0 else "right text"
        elif x_rank == 0:
            descriptor = "leftmost text"
        elif x_rank == n - 1:
            descriptor = "rightmost text"
        else:
            descriptor = "middle text"
    elif cluster_shape == "stack":
        if n == 2:
            descriptor = "upper text" if y_rank == 0 else "lower text"
        elif y_rank == 0:
            descriptor = "uppermost text"
        elif y_rank == n - 1:
            descriptor = "lowest text"
        else:
            descriptor = "middle text"
    else:
        x_frac = (primary_x - cluster_bbox[0]) / bbox_w
        y_frac = (primary_y - cluster_bbox[1]) / bbox_h
        bucket_key, bucket_name = _bucket_name_from_fracs(x_frac, y_frac)
        horizontal = bucket_key.split("_")[1]
        vertical = bucket_key.split("_")[0]
        if horizontal != "center" and vertical != "middle":
            descriptor = f"{bucket_name} text"
        elif vertical != "middle":
            descriptor = f"{vertical} text"
        elif horizontal != "center":
            descriptor = f"{horizontal}-side text"
        else:
            descriptor = "central text"
    if cluster_shape in {"singleton", "row", "stack"}:
        if len(component_nodes) <= 1:
            bucket_key = "singleton"
        elif cluster_shape == "row":
            bucket_key = "left" if x_rank == 0 else "right" if x_rank == n - 1 else "middle"
        elif cluster_shape == "stack":
            bucket_key = "upper" if y_rank == 0 else "lower" if y_rank == n - 1 else "middle"
        else:
            bucket_key = "cluster"
    bucket_occupancy = sum(1 for key, _name in bucket_names if key == bucket_key)

    if descriptor:
        descriptor_norm = normalize_answer(descriptor)
        coarse_norm = normalize_answer(coarse_phrase)
        descriptor_tokens = set(descriptor_norm.replace("-", " ").split())
        coarse_tokens = set(coarse_norm.replace("-", " ").split())
        overlap = descriptor_tokens & coarse_tokens
        redundant = bool(tuning.avoid_redundant_location_phrases and len(overlap) >= 2)
        if redundant:
            primary_phrase = descriptor
            alternates = [
                descriptor,
                f"the {descriptor}",
                f"{descriptor} in that area",
            ]
        else:
            primary_phrase = f"{descriptor} in the {coarse_phrase}"
            alternates = [
                primary_phrase,
                f"{descriptor} near the {coarse_phrase}",
                f"{descriptor} within the {coarse_phrase}",
                f"{descriptor} around the {coarse_phrase}",
            ]
    else:
        primary_phrase = coarse_phrase
        alternates = [coarse_phrase]

    resolvable_count = sum(1 for node in component_nodes if bool(node.get("resolvable", False)))
    unresolvable_count = len(component_nodes) - resolvable_count
    detail = (
        f"{len(component_nodes)} nearby text boxes form a {cluster_shape}; "
        f"target is the {descriptor or 'only text'}; "
        f"{resolvable_count} resolvable and {unresolvable_count} unresolved"
    )
    return {
        "specific_location_phrase": primary_phrase,
        "specific_location_synonyms": list(dict.fromkeys(alternates)),
        "cluster_shape": cluster_shape,
        "cluster_size": len(component_nodes),
        "cluster_resolvable_count": resolvable_count,
        "cluster_unresolvable_count": unresolvable_count,
        "bucket_key": bucket_key,
        "bucket_occupancy": int(bucket_occupancy),
        "cluster_bbox": [round(value, 2) for value in cluster_bbox],
        "layout_detail": detail,
    }


def collect_semantic_kd_metadata(
    *,
    primary_box: list[float],
    primary_polygon: list[float],
    primary_node_ids: list[str],
    all_text_nodes: list[dict[str, Any]],
    all_anchors: list[dict[str, Any]],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    primary_center = centroid_from_polygon(primary_polygon)
    location_meta = local_text_location_metadata(
        primary_box=primary_box,
        primary_node_ids=primary_node_ids,
        all_text_nodes=all_text_nodes,
        image_size=image_size,
        coarse_phrase="area of the image",
    )
    neighbors = []
    primary_node_id_set = set(primary_node_ids)
    for other in all_text_nodes:
        if str(other["node_id"]) in primary_node_id_set:
            continue
        other_box = list(other["bbox"])
        dist = math.dist(primary_center, centroid_from_polygon(other["polygon"]))
        neighbors.append(
            {
                "node_id": str(other["node_id"]),
                "text": str(other["text"]),
                "confidence": round(float(other.get("confidence") or 0.0), 6),
                "consensus_tier": str(other.get("consensus_tier") or ""),
                "distance_px": round(float(dist), 1),
                "relation_to_primary": determine_relation(primary_box, other_box, image_size),
                "bbox": [round(float(value), 2) for value in other_box],
                "resolvable": bool(other.get("resolvable", True)),
            }
        )
    neighbors.sort(key=lambda row: (float(row["distance_px"]), row["node_id"]))

    nearby_anchors = []
    max_image_dim = float(max(image_size))
    for anchor in all_anchors:
        box = list(anchor["box"])
        overlap = overlap_fraction(primary_box, box)
        proximity = distance_point_to_box(primary_center, box)
        if overlap > 0.05 or proximity < 0.3 * max_image_dim:
            nearby_anchors.append(
                {
                    "label": str(anchor["label"]),
                    "box": [round(float(value), 2) for value in box],
                    "score": round(float(anchor.get("score") or 0.0), 6),
                    "overlap_with_text": round(float(overlap), 6),
                    "source": str(anchor.get("source") or "semantic"),
                }
            )
    nearby_anchors.sort(key=lambda row: (-float(row["score"]), row["label"]))

    return {
        "neighboring_text": neighbors[:10],
        "nearby_anchors": nearby_anchors[:10],
        "text_density": len(all_text_nodes),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "competing_tuples": 0,
        "teacher_answer_logprobs": None,
        "local_text_cluster_size": int(location_meta["cluster_size"]),
        "local_text_cluster_shape": str(location_meta["cluster_shape"]),
        "local_text_cluster_resolvable": int(location_meta["cluster_resolvable_count"]),
        "local_text_cluster_unresolvable": int(location_meta["cluster_unresolvable_count"]),
        "local_text_bucket_key": str(location_meta.get("bucket_key") or "singleton"),
        "local_text_bucket_occupancy": int(location_meta.get("bucket_occupancy") or 1),
        "local_text_cluster_bbox": [round(float(value), 2) for value in location_meta.get("cluster_bbox") or primary_box],
        "layout_detail": str(location_meta["layout_detail"]),
    }


def _dedupe_labels(labels: list[str], *, max_items: int = 12) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in labels:
        text = str(item or "").strip().lower()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _bucket_name_from_fracs(x_frac: float, y_frac: float) -> tuple[str, str]:
    horizontal = "left" if x_frac < 0.34 else "right" if x_frac > 0.66 else "center"
    vertical = "upper" if y_frac < 0.34 else "lower" if y_frac > 0.66 else "middle"
    return f"{vertical}_{horizontal}", f"{vertical}-{horizontal}"


def _anchor_box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1e-6)
    area_b = max((bx2 - bx1) * (by2 - by1), 1e-6)
    return inter_area / max(area_a + area_b - inter_area, 1e-6)


def remap_region_box_to_image(region_box: list[float], context_box: list[int]) -> list[float]:
    left, top, _, _ = [float(value) for value in context_box]
    x1, y1, x2, y2 = [float(value) for value in region_box[:4]]
    return [round(left + x1, 2), round(top + y1, 2), round(left + x2, 2), round(top + y2, 2)]


def anchor_candidate_viable(label: str, anchor_box: list[float], text_box: list[float], image_size: tuple[int, int]) -> bool:
    ax1, ay1, ax2, ay2 = [float(value) for value in anchor_box]
    tx1, ty1, tx2, ty2 = [float(value) for value in text_box]
    anchor_area = max((ax2 - ax1) * (ay2 - ay1), 1.0)
    text_area = max((tx2 - tx1) * (ty2 - ty1), 1.0)
    image_area = max(float(image_size[0]) * float(image_size[1]), 1.0)
    anchor_frac = anchor_area / image_area
    ratio = anchor_area / text_area
    generic = label in {"sign", "poster", "label", "screen", "display", "board"}
    if anchor_frac > 0.94:
        return False
    if generic and anchor_frac > 0.70:
        return False
    if generic and ratio > 220.0:
        return False
    return True


def anchor_local_location_metadata(text_box: list[float], anchor_box: list[float], relation: str, anchor_label: str) -> dict[str, Any]:
    label = sanitize_anchor_label(anchor_label) or str(anchor_label or "").strip().lower()
    if not label:
        return {"phrase": "", "synonyms": [], "clean": False, "mode": "none"}

    if relation == "above":
        return {
            "phrase": f"above the {label}",
            "synonyms": [f"above the {label}", f"over the {label}"],
            "clean": True,
            "mode": "directional",
        }
    if relation == "below":
        return {
            "phrase": f"below the {label}",
            "synonyms": [f"below the {label}", f"under the {label}"],
            "clean": True,
            "mode": "directional",
        }
    if relation == "left_of":
        return {
            "phrase": f"to the left of the {label}",
            "synonyms": [f"to the left of the {label}", f"left of the {label}"],
            "clean": True,
            "mode": "directional",
        }
    if relation == "right_of":
        return {
            "phrase": f"to the right of the {label}",
            "synonyms": [f"to the right of the {label}", f"right of the {label}"],
            "clean": True,
            "mode": "directional",
        }
    if relation != "on":
        return {"phrase": "", "synonyms": [], "clean": False, "mode": "none"}

    tx1, ty1, tx2, ty2 = [float(value) for value in text_box]
    ax1, ay1, ax2, ay2 = [float(value) for value in anchor_box]
    aw = max(ax2 - ax1, 1.0)
    ah = max(ay2 - ay1, 1.0)
    anchor_area = max(aw * ah, 1.0)
    text_area = max((tx2 - tx1) * (ty2 - ty1), 1.0)
    size_ratio = anchor_area / text_area
    if size_ratio < 1.45:
        return {"phrase": "", "synonyms": [], "clean": False, "mode": "none"}

    # Centroid difference: direction from anchor center to text center.
    # rx/ry are normalized by anchor half-dimensions so thresholds are
    # scale-invariant.  rx<0 = text is left of anchor center, ry<0 = above.
    text_cx = (tx1 + tx2) / 2.0
    text_cy = (ty1 + ty2) / 2.0
    anchor_cx = (ax1 + ax2) / 2.0
    anchor_cy = (ay1 + ay2) / 2.0
    rx = (text_cx - anchor_cx) / (aw / 2.0)
    ry = (text_cy - anchor_cy) / (ah / 2.0)
    abs_rx = abs(rx)
    abs_ry = abs(ry)

    # Need meaningful displacement from anchor center to produce useful
    # spatial language; if the text is roughly centered, bail out.
    if abs_rx < 0.30 and abs_ry < 0.30:
        return {"phrase": "", "synonyms": [], "clean": False, "mode": "none"}

    descriptor = ""
    mode = "none"
    is_left = rx < -0.30
    is_right = rx > 0.30
    is_top = ry < -0.30
    is_bottom = ry > 0.30

    if (is_left or is_right) and (is_top or is_bottom) and abs_rx >= 0.25 and abs_ry >= 0.25:
        vertical = "upper" if is_top else "lower"
        horizontal = "left" if is_left else "right"
        descriptor = f"{vertical} {horizontal}"
        mode = "corner"
    elif abs_rx > abs_ry + 0.10:
        if is_left:
            descriptor = "left"
            mode = "horizontal"
        elif is_right:
            descriptor = "right"
            mode = "horizontal"
    elif abs_ry > abs_rx + 0.10:
        if is_top:
            descriptor = "top"
            mode = "vertical"
        elif is_bottom:
            descriptor = "bottom"
            mode = "vertical"
    elif is_left or is_right:
        descriptor = "left" if is_left else "right"
        mode = "horizontal"
    elif is_top or is_bottom:
        descriptor = "top" if is_top else "bottom"
        mode = "vertical"

    if not descriptor:
        return {"phrase": "", "synonyms": [], "clean": False, "mode": "none"}

    if mode == "corner":
        natural_descriptor = descriptor.replace("upper", "top").replace("lower", "bottom")
        phrase = f"toward the {natural_descriptor} of the {label}"
        synonyms = [
            phrase,
            f"near the {natural_descriptor} of the {label}",
            f"on the {natural_descriptor.replace(' ', '-')} of the {label}",
            f"at the {natural_descriptor} of the {label}",
        ]
    elif mode == "horizontal":
        phrase = f"toward the {descriptor} of the {label}"
        synonyms = [
            phrase,
            f"on the {descriptor} side of the {label}",
            f"near the {descriptor} of the {label}",
        ]
    else:
        phrase = f"toward the {descriptor} of the {label}"
        synonyms = [
            phrase,
            f"near the {descriptor} of the {label}",
            f"at the {descriptor} of the {label}",
        ]
    return {
        "phrase": phrase,
        "synonyms": list(dict.fromkeys(synonyms)),
        "clean": True,
        "mode": mode,
    }
