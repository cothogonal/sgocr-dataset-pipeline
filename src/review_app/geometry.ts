import type { DisplayBox, Point } from "./types";

const REGION_GRID: Record<string, [number, number]> = {
  ul: [0, 0],
  uc: [1, 0],
  ur: [2, 0],
  cl: [0, 1],
  cc: [1, 1],
  cr: [2, 1],
  ll: [0, 2],
  lc: [1, 2],
  lr: [2, 2],
};

const REGION_LABEL_TO_KEY: Record<string, string> = {
  "upper-left area of the image": "ul",
  "upper left area of the image": "ul",
  "top-left area of the image": "ul",
  "top left area of the image": "ul",
  "top-center area of the image": "uc",
  "top center area of the image": "uc",
  "upper-center area of the image": "uc",
  "upper center area of the image": "uc",
  "upper-right area of the image": "ur",
  "upper right area of the image": "ur",
  "top-right area of the image": "ur",
  "top right area of the image": "ur",
  "middle-left area of the image": "cl",
  "middle left area of the image": "cl",
  "left-middle area of the image": "cl",
  "left side of the image": "cl",
  "center of the image": "cc",
  "middle of the image": "cc",
  "central area of the image": "cc",
  "middle-right area of the image": "cr",
  "middle right area of the image": "cr",
  "right-middle area of the image": "cr",
  "right side of the image": "cr",
  "lower-left area of the image": "ll",
  "lower left area of the image": "ll",
  "bottom-left area of the image": "ll",
  "bottom left area of the image": "ll",
  "bottom-center area of the image": "lc",
  "bottom center area of the image": "lc",
  "lower-center area of the image": "lc",
  "lower center area of the image": "lc",
  "lower-right area of the image": "lr",
  "lower right area of the image": "lr",
  "bottom-right area of the image": "lr",
  "bottom right area of the image": "lr",
};

export function asFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  return null;
}

export function normalizePolygon(raw: unknown): Point[] {
  if (!Array.isArray(raw)) {
    return [];
  }
  if (raw.length >= 6 && raw.every((item) => typeof item === "number")) {
    const points: Point[] = [];
    for (let index = 0; index + 1 < raw.length; index += 2) {
      const x = asFiniteNumber(raw[index]);
      const y = asFiniteNumber(raw[index + 1]);
      if (x == null || y == null) {
        continue;
      }
      points.push({ x, y });
    }
    return points;
  }
  const points: Point[] = [];
  for (const item of raw) {
    if (Array.isArray(item) && item.length >= 2) {
      const x = asFiniteNumber(item[0]);
      const y = asFiniteNumber(item[1]);
      if (x != null && y != null) {
        points.push({ x, y });
      }
    }
  }
  return points;
}

export function boxFromXYXY(raw: unknown): DisplayBox | null {
  if (!Array.isArray(raw) || raw.length < 4) {
    return null;
  }
  const x1 = asFiniteNumber(raw[0]);
  const y1 = asFiniteNumber(raw[1]);
  const x2 = asFiniteNumber(raw[2]);
  const y2 = asFiniteNumber(raw[3]);
  if (x1 == null || y1 == null || x2 == null || y2 == null) {
    return null;
  }
  return normalizeBox(x1, y1, x2, y2);
}

export function boxFromXYWH(raw: unknown): DisplayBox | null {
  if (!Array.isArray(raw) || raw.length < 4) {
    return null;
  }
  const x = asFiniteNumber(raw[0]);
  const y = asFiniteNumber(raw[1]);
  const width = asFiniteNumber(raw[2]);
  const height = asFiniteNumber(raw[3]);
  if (x == null || y == null || width == null || height == null) {
    return null;
  }
  return normalizeBox(x, y, x + width, y + height);
}

export function boxFromPolygon(points: Point[]): DisplayBox | null {
  if (points.length === 0) {
    return null;
  }
  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  return normalizeBox(Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys));
}

export function centroidFromPolygon(points: Point[]): Point | null {
  if (points.length === 0) {
    return null;
  }
  if (points.length < 3) {
    return {
      x: points.reduce((sum, point) => sum + point.x, 0) / points.length,
      y: points.reduce((sum, point) => sum + point.y, 0) / points.length,
    };
  }
  let area = 0;
  let cx = 0;
  let cy = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    const cross = current.x * next.y - next.x * current.y;
    area += cross;
    cx += (current.x + next.x) * cross;
    cy += (current.y + next.y) * cross;
  }
  if (Math.abs(area) < 1e-6) {
    const box = boxFromPolygon(points);
    return box ? { x: box.cx, y: box.cy } : null;
  }
  const divisor = area * 3;
  return { x: cx / divisor, y: cy / divisor };
}

export function centroidFromBox(box: DisplayBox | null): Point | null {
  if (!box) {
    return null;
  }
  return { x: box.cx, y: box.cy };
}

export function regionKeyFromLabel(anchorLabel: unknown): string | null {
  if (typeof anchorLabel !== "string") {
    return null;
  }
  return REGION_LABEL_TO_KEY[anchorLabel.toLowerCase()] || null;
}

export function boxFromRegionKey(regionKey: unknown, width: unknown, height: unknown): DisplayBox | null {
  if (typeof regionKey !== "string") {
    return null;
  }
  const bucket = REGION_GRID[regionKey];
  const imageWidth = asFiniteNumber(width);
  const imageHeight = asFiniteNumber(height);
  if (!bucket || imageWidth == null || imageHeight == null) {
    return null;
  }
  const [column, row] = bucket;
  const cellWidth = imageWidth / 3;
  const cellHeight = imageHeight / 3;
  return normalizeBox(column * cellWidth, row * cellHeight, (column + 1) * cellWidth, (row + 1) * cellHeight);
}

function normalizeBox(x1: number, y1: number, x2: number, y2: number): DisplayBox {
  const nx1 = Math.min(x1, x2);
  const ny1 = Math.min(y1, y2);
  const nx2 = Math.max(x1, x2);
  const ny2 = Math.max(y1, y2);
  const width = nx2 - nx1;
  const height = ny2 - ny1;
  return {
    x1: nx1,
    y1: ny1,
    x2: nx2,
    y2: ny2,
    width,
    height,
    cx: nx1 + width / 2,
    cy: ny1 + height / 2,
  };
}

