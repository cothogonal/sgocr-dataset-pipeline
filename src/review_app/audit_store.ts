import fs from "node:fs";
import path from "node:path";

import type { AuditEntry } from "./types";

export function loadAuditEntries(reviewRoot: string, experimentName: string): Record<string, AuditEntry> {
  const statePath = auditStatePath(reviewRoot, experimentName);
  if (!fs.existsSync(statePath)) {
    return {};
  }
  return JSON.parse(fs.readFileSync(statePath, "utf8")) as Record<string, AuditEntry>;
}

export function saveAuditEntry(reviewRoot: string, experimentName: string, entry: AuditEntry): Record<string, AuditEntry> {
  const dir = path.join(reviewRoot, experimentName);
  fs.mkdirSync(dir, { recursive: true });
  const state = loadAuditEntries(reviewRoot, experimentName);
  state[entry.sampleId] = entry;
  const statePath = auditStatePath(reviewRoot, experimentName);
  const stateTmp = `${statePath}.tmp`;
  fs.writeFileSync(stateTmp, JSON.stringify(state, null, 2));
  fs.renameSync(stateTmp, statePath);

  const rows = Object.values(state).sort((left, right) => left.sampleIndex - right.sampleIndex);
  fs.writeFileSync(path.join(dir, "audit_results.jsonl"), rows.map((row) => JSON.stringify(row)).join("\n") + "\n");
  fs.appendFileSync(path.join(dir, "audit_events.jsonl"), `${JSON.stringify(entry)}\n`);
  return state;
}

function auditStatePath(reviewRoot: string, experimentName: string): string {
  return path.join(reviewRoot, experimentName, "audit_state.json");
}

