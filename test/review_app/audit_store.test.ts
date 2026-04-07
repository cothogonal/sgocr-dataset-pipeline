import { afterEach, describe, expect, test } from "bun:test";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { loadAuditEntries, saveAuditEntry } from "../../src/review_app/audit_store";

const tempRoots: string[] = [];

afterEach(() => {
  while (tempRoots.length) {
    fs.rmSync(tempRoots.pop(), { recursive: true, force: true });
  }
});

describe("review audit store", () => {
  test("upserts audit state and writes jsonl", () => {
    const reviewRoot = fs.mkdtempSync(path.join(os.tmpdir(), "sgocr-audit-"));
    tempRoots.push(reviewRoot);

    saveAuditEntry(reviewRoot, "fixture_exp", {
      sampleId: "sample-1",
      sampleIndex: 0,
      imageId: "img-1",
      annId: "img-1_0",
      ocrCorrect: "yes",
      anchorGrounding: "marginal",
      spatialRelation: "yes",
      overallGo: "yes",
      questionFaithful: ["yes", "skip"],
      notes: "first pass",
      updatedAt: "2026-04-04T01:00:00Z",
    });

    saveAuditEntry(reviewRoot, "fixture_exp", {
      sampleId: "sample-1",
      sampleIndex: 0,
      imageId: "img-1",
      annId: "img-1_0",
      ocrCorrect: "yes",
      anchorGrounding: "yes",
      spatialRelation: "yes",
      overallGo: "yes",
      questionFaithful: ["yes", "yes"],
      notes: "updated",
      updatedAt: "2026-04-04T01:01:00Z",
    });

    const audit = loadAuditEntries(reviewRoot, "fixture_exp");
    expect(audit["sample-1"]?.anchorGrounding).toBe("yes");
    const jsonl = fs.readFileSync(path.join(reviewRoot, "fixture_exp", "audit_results.jsonl"), "utf8").trim().split("\n");
    expect(jsonl).toHaveLength(1);
    expect(jsonl[0]).toContain("\"updated\"");
  });
});
