import fs from "node:fs";
import path from "node:path";

import { boxFromPolygon, boxFromRegionKey, boxFromXYWH, boxFromXYXY, centroidFromBox, centroidFromPolygon, normalizePolygon, regionKeyFromLabel } from "./geometry";
import type { ExperimentSummary, JsonValue, ReviewQuestion, ReviewSample, ReviewValidation } from "./types";

type LoaderOptions = {
  repoRoot: string;
  dataRoot: string;
  experimentsRoot: string;
};

export function listExperiments(options: LoaderOptions): ExperimentSummary[] {
  if (!fs.existsSync(options.experimentsRoot)) {
    return [];
  }
  return fs
    .readdirSync(options.experimentsRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => {
      const dirPath = path.join(options.experimentsRoot, entry.name);
      const summaryPath = path.join(dirPath, "summary.json");
      let summary: Record<string, JsonValue> = {};
      if (fs.existsSync(summaryPath)) {
        summary = JSON.parse(fs.readFileSync(summaryPath, "utf8"));
      }
      const experiment = asRecord(summary.experiment) || {};
      return {
        name: entry.name,
        path: dirPath,
        sampleCount: asNumber(asRecord(summary.stage_counts)?.final_qa_rows) ?? asNumber(summary.tuple_count) ?? asNumber(summary.accepted_qas) ?? asNumber(summary.input_tuple_count),
        qaAcceptRate: asNumber(summary.qa_accept_rate),
        tupleAcceptRate: asNumber(summary.tuple_full_accept_rate),
        model: asString(experiment.model),
        provider: asString(experiment.provider),
        promptVariant: asString(experiment.prompt_variant),
        inputTupleCount: asNumber(summary.input_tuple_count) ?? asNumber(asRecord(summary.stage_counts)?.verified_tuples),
        resolvableTupleCount: asNumber(summary.resolvable_tuple_count) ?? asNumber(asRecord(summary.stage_counts)?.verified_tuples),
        droppedTupleCount: asNumber(summary.dropped_tuple_count),
        generatedQas: asNumber(summary.generated_qas),
        acceptedQas: asNumber(summary.accepted_qas),
        sameImageUniverseCount: asNumber(summary.same_image_universe_count),
        resolvabilityStats: asRecord(summary.resolvability_stats),
        stageCounts: asRecord(summary.stage_counts),
        questionTypeCounts: asRecord(summary.question_type_counts),
        selectedQuestionTypeCounts: asRecord(summary.selected_question_type_counts),
        disabledQuestionTypes: Array.isArray(summary.disabled_question_types) ? summary.disabled_question_types : null,
        failureCounts: asRecord(summary.failure_counts),
      } satisfies ExperimentSummary;
    })
    .filter((entry) => primaryDatasetPath(entry.path) !== null)
    .sort((left, right) => left.name.localeCompare(right.name));
}

export function loadExperiment(name: string, options: LoaderOptions): { experiment: ExperimentSummary | null; samples: ReviewSample[] } {
  const experiments = listExperiments(options);
  const experiment = experiments.find((entry) => entry.name === name) || null;
  if (!experiment) {
    throw new Error(`Unknown experiment: ${name}`);
  }
  const datasetPath = primaryDatasetPath(experiment.path);
  if (!datasetPath) {
    throw new Error(`No dataset file found for experiment: ${name}`);
  }
  const rawIndex = buildRawIndex(path.join(experiment.path, "raw_results.jsonl"));
  const lines = fs
    .readFileSync(datasetPath, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const acceptedRows = lines.map((line) => JSON.parse(line) as Record<string, JsonValue>);
  const samples = acceptedRows.map((row, index) => normalizeReviewRow(row, index, options, rawIndex, "accepted"));
  const sampleIds = new Set(samples.map((sample) => sample.sampleId));
  const rawResultsPath = path.join(experiment.path, "raw_results.jsonl");
  if (fs.existsSync(rawResultsPath)) {
    const rejectedRows = fs
      .readFileSync(rawResultsPath, "utf8")
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => JSON.parse(line) as Record<string, JsonValue>)
      .filter((row) => {
        const sampleId = asString(row.sample_id);
        const acceptedCount = asNumber(asRecord(row.summary)?.accepted_count) ?? 0;
        const ok = asBoolean(row.ok) ?? false;
        return !!sampleId && !sampleIds.has(sampleId) && (!ok || acceptedCount !== 1);
      });
    const rejectedSamples = rejectedRows.map((row, index) =>
      normalizeReviewRow(row, samples.length + index, options, rawIndex, "rejected"),
    );
    samples.push(...rejectedSamples);
  }
  return { experiment, samples };
}

export function resolveImagePath(imagePath: string, options: Pick<LoaderOptions, "repoRoot" | "dataRoot">): string {
  const dataRoot = fs.realpathSync(options.dataRoot);
  const repoDataAlias = path.join(options.repoRoot, "data");
  let candidate = path.isAbsolute(imagePath) ? path.normalize(imagePath) : path.resolve(options.repoRoot, imagePath);
  if (candidate === repoDataAlias || candidate.startsWith(`${repoDataAlias}${path.sep}`)) {
    candidate = path.join(dataRoot, path.relative(repoDataAlias, candidate));
  }
  const resolved = fs.existsSync(candidate) ? fs.realpathSync(candidate) : candidate;
  const relative = path.relative(dataRoot, resolved);
  if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
    throw new Error(`Image path escapes data root: ${imagePath}`);
  }
  return resolved;
}

function normalizeReviewRow(
  row: Record<string, JsonValue>,
  index: number,
  options: LoaderOptions,
  rawIndex: Map<string, Record<string, JsonValue>>,
  sampleStatus: "accepted" | "rejected",
): ReviewSample {
  const normalized = normalizeSourceRow(row, rawIndex, index);
  const tuple = asRecord(normalized.tuple) || {};
  const grounding = asRecord(tuple.grounding) || {};
  const points = normalizePolygon(tuple.text_polygon);
  const textBox = boxFromPolygon(points) || boxFromXYWH(tuple.text_bbox);
  const regionKey = asString(tuple.region_key) || regionKeyFromLabel(tuple.anchor_label);
  const anchorBox = boxFromXYXY(tuple.anchor_box) || boxFromRegionKey(regionKey, tuple.image_width, tuple.image_height);
  const anchorBoxSource = boxFromXYXY(tuple.anchor_box) ? "tuple" : anchorBox ? "derived_region" : null;
  const queryAnchorBox = boxFromXYXY(grounding.query_anchor_box);
  const refBox = boxFromXYXY(tuple.ref_box);
  const imagePath = asString(tuple.image_path);
  const imageUrl = imagePath ? `/api/image?path=${encodeURIComponent(imagePath)}` : null;
  const questions = extractQuestions(normalized);
  const sampleId = asString(normalized.sample_id) || asString(tuple.sample_id) || `${asString(tuple.image_id) || "sample"}::${asString(tuple.ann_id) || index}`;
  if (imagePath) {
    try {
      resolveImagePath(imagePath, options);
    } catch {
      // Keep the sample loadable even if the file is missing; the UI surfaces the broken image path.
    }
  }
  return {
    sampleId,
    sampleIndex: index,
    sampleStatus,
    failureReason: asString(normalized.failure_reason) || asString(asRecord(normalized.filter_stage)?.reason),
    imageId: asString(tuple.image_id) || `sample-${index}`,
    annId: asString(tuple.ann_id),
    imagePath,
    imageUrl,
    imageWidth: asNumber(tuple.image_width),
    imageHeight: asNumber(tuple.image_height),
    tuple,
    geometry: {
      textPolygon: points,
      textCentroid: centroidFromPolygon(points),
      textBox,
      anchorBox,
      anchorCentroid: centroidFromBox(anchorBox),
      anchorBoxSource,
      queryAnchorBox,
      queryAnchorCentroid: centroidFromBox(queryAnchorBox),
      refBox,
      refCentroid: centroidFromBox(refBox),
    },
    questions,
    usage: asRecord(asRecord(normalized.result)?.usage) || null,
    summary: asRecord(normalized.summary) || null,
    filterStage: asRecord(normalized.filter_stage) || null,
    ok: asBoolean(normalized.ok) ?? true,
  };
}

function normalizeSourceRow(
  row: Record<string, JsonValue>,
  rawIndex: Map<string, Record<string, JsonValue>>,
  index: number,
): Record<string, JsonValue> {
  if (asRecord(row.tuple)) {
    return row;
  }
  const imageId = asString(row.image_id) || `sample-${index}`;
  const annId = asString(row.ann_id);
  const rowSampleId = asString(row.sample_id);
  const raw = (rowSampleId ? rawIndex.get(rowSampleId) : null) || rawIndex.get(sampleKey(imageId, annId)) || null;
  const rawTuple = asRecord(raw?.tuple) || {};
  const tuple = {
    ...rawTuple,
    ...row,
    image_id: imageId,
    ann_id: annId,
  };
  if (rawTuple.answer != null) {
    tuple.answer = rawTuple.answer;
  }
  if (rawTuple.answer_normalized != null) {
    tuple.answer_normalized = rawTuple.answer_normalized;
  }
  if (asString(row.answer)) {
    tuple.qa_answer = asString(row.answer);
  }
  if (asString(row.question)) {
    tuple.qa_question = asString(row.question);
  }
  return {
    ...raw,
    ...row,
    ok: raw?.ok ?? true,
    sample_id: rowSampleId || asString(raw?.sample_id),
    tuple,
    items: [{ question: asString(row.question) || "", answer: asString(row.answer) || "" }],
    validations: [{ accepted: true, answer_ok: true, anchor_ok: true, length_ok: true, duplicate_ok: true }],
    summary: raw?.summary || { accepted_count: 1 },
    filter_stage: raw?.filter_stage || null,
    result: raw?.result || null,
  };
}

function extractQuestions(row: Record<string, JsonValue>): ReviewQuestion[] {
  const directItems = Array.isArray(row.items) ? row.items : [];
  const parsedItems = Array.isArray(asRecord(row.result)?.parsed?.items) ? (asRecord(row.result)?.parsed?.items as JsonValue[]) : [];
  const items = directItems.length > 0 ? directItems : parsedItems;
  const validations = Array.isArray(row.validations) ? row.validations : [];
  return items.map((item, index) => {
    const record = asRecord(item) || {};
    const validation = (asRecord(validations[index]) || null) as ReviewValidation | null;
    return {
      index,
      question: asString(record.question) || "",
      answer: asString(record.answer) || "",
      validation,
    };
  });
}

function asRecord(value: JsonValue | undefined): Record<string, JsonValue> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, JsonValue>;
}

function asString(value: JsonValue | undefined): string | null {
  return typeof value === "string" ? value : null;
}

function asNumber(value: JsonValue | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asBoolean(value: JsonValue | undefined): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function primaryDatasetPath(experimentPath: string): string | null {
  const candidates = ["ocr_qa_dataset.jsonl", "accepted_dataset.jsonl", "raw_results.jsonl"];
  for (const candidate of candidates) {
    const fullPath = path.join(experimentPath, candidate);
    if (fs.existsSync(fullPath)) {
      return fullPath;
    }
  }
  return null;
}

function buildRawIndex(rawPath: string): Map<string, Record<string, JsonValue>> {
  const index = new Map<string, Record<string, JsonValue>>();
  if (!fs.existsSync(rawPath)) {
    return index;
  }
  const lines = fs
    .readFileSync(rawPath, "utf8")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  for (const line of lines) {
    const row = JSON.parse(line) as Record<string, JsonValue>;
    const sampleId = asString(row.sample_id);
    if (sampleId) {
      index.set(sampleId, row);
    }
    const tuple = asRecord(row.tuple) || {};
    const imageId = asString(tuple.image_id) || asString(row.image_id);
    if (!imageId) {
      continue;
    }
    index.set(sampleKey(imageId, asString(tuple.ann_id) || asString(row.ann_id)), row);
  }
  return index;
}

function sampleKey(imageId: string, annId: string | null): string {
  return `${imageId}::${annId || ""}`;
}
