import { describe, expect, test } from "bun:test";

import { boxFromRegionKey, boxFromXYWH, centroidFromPolygon, normalizePolygon, regionKeyFromLabel } from "../../src/review_app/geometry";

describe("review geometry", () => {
  test("normalizes flat polygons and computes centroid", () => {
    const polygon = normalizePolygon([10, 10, 30, 10, 30, 30, 10, 30]);
    expect(polygon).toHaveLength(4);
    expect(centroidFromPolygon(polygon)).toEqual({ x: 20, y: 20 });
  });

  test("builds region box from bootstrap region key", () => {
    const box = boxFromRegionKey("ur", 900, 600);
    expect(box).toEqual({
      x1: 600,
      y1: 0,
      x2: 900,
      y2: 200,
      width: 300,
      height: 200,
      cx: 750,
      cy: 100,
    });
  });

  test("maps anchor labels back to region keys", () => {
    expect(regionKeyFromLabel("top right area of the image")).toBe("ur");
  });

  test("parses xywh boxes", () => {
    const box = boxFromXYWH([5, 7, 20, 30]);
    expect(box?.x2).toBe(25);
    expect(box?.y2).toBe(37);
  });
});
