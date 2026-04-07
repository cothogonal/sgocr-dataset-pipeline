const state = {
  experiments: [],
  experiment: null,
  samples: [],
  audit: {},
  filteredIndices: [],
  currentFilteredIndex: 0,
  saveTimer: null,
};

const els = {
  experimentSelect: document.getElementById("experimentSelect"),
  filterSelect: document.getElementById("filterSelect"),
  questionTypeFilter: document.getElementById("questionTypeFilter"),
  difficultyFilter: document.getElementById("difficultyFilter"),
  toggleStageFlow: document.getElementById("toggleStageFlow"),
  stageFlowBody: document.getElementById("stageFlowBody"),
  prevButton: document.getElementById("prevButton"),
  nextButton: document.getElementById("nextButton"),
  reviewCanvas: document.getElementById("reviewCanvas"),
  sourceImage: document.getElementById("sourceImage"),
  imageMeta: document.getElementById("imageMeta"),
  tagsSummary: document.getElementById("tagsSummary"),
  tupleSummary: document.getElementById("tupleSummary"),
  groundingSummary: document.getElementById("groundingSummary"),
  tupleJson: document.getElementById("tupleJson"),
  usageJson: document.getElementById("usageJson"),
  questionList: document.getElementById("questionList"),
  saveState: document.getElementById("saveState"),
  progressState: document.getElementById("progressState"),
  statusSummary: document.getElementById("statusSummary"),
  stageFlow: document.getElementById("stageFlow"),
  stageFailures: document.getElementById("stageFailures"),
  notesInput: document.getElementById("notesInput"),
  toggleTextPolygon: document.getElementById("toggleTextPolygon"),
  toggleTextBox: document.getElementById("toggleTextBox"),
  toggleAnchorBox: document.getElementById("toggleAnchorBox"),
  toggleRefBox: document.getElementById("toggleRefBox"),
  toggleNeighborBoxes: document.getElementById("toggleNeighborBoxes"),
  toggleNearbyAnchors: document.getElementById("toggleNearbyAnchors"),
  toggleCentroids: document.getElementById("toggleCentroids"),
  kdSummary: document.getElementById("kdSummary"),
  neighborTable: document.getElementById("neighborTable"),
  anchorTable: document.getElementById("anchorTable"),
};

const AUDIT_CHOICES = {
  ocrCorrect: ["yes", "no"],
  anchorGrounding: ["yes", "no", "marginal"],
  spatialRelation: ["yes", "no"],
  questionNatural: ["yes", "no"],
  answerCorrect: ["yes", "no"],
  overallGo: ["yes", "no", "marginal"],
};

bootstrap().catch((error) => {
  console.error(error);
  els.saveState.textContent = `Load failed: ${error.message}`;
});

async function bootstrap() {
  buildAuditControls();
  bindGlobalEvents();
  restoreStageFlowState();
  await loadSession(getInitialExperiment());
}

function getInitialExperiment() {
  return new URLSearchParams(window.location.search).get("experiment") || "";
}

async function loadSession(experimentName = "") {
  setSaveState("Loading");
  const url = new URL("/api/session", window.location.origin);
  if (experimentName) {
    url.searchParams.set("experiment", experimentName);
  }
  const payload = await fetchJson(url);
  state.experiments = payload.experiments || [];
  state.experiment = payload.experiment || null;
  state.samples = payload.samples || [];
  state.audit = payload.audit || {};

  renderExperimentSelect();
  renderFacetFilters();
  recomputeFilteredIndices();
  selectSampleByFilteredIndex(findFirstInterestingIndex());
  renderStatus();
  renderStageFlow();
  setSaveState("Ready");
}

function buildAuditControls() {
  document.querySelectorAll(".choice-row[data-field]").forEach((node) => {
    const field = node.getAttribute("data-field");
    for (const value of AUDIT_CHOICES[field]) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.field = field;
      button.dataset.value = value;
      button.textContent = value;
      button.addEventListener("click", () => {
        const entry = getCurrentAuditEntry();
        entry[field] = entry[field] === value ? "" : value;
        scheduleSave();
        renderCurrentSample();
      });
      node.appendChild(button);
    }
  });
}

function bindGlobalEvents() {
  els.experimentSelect.addEventListener("change", () => loadSession(els.experimentSelect.value));
  els.toggleStageFlow.addEventListener("click", toggleStageFlowVisibility);
  [els.filterSelect, els.questionTypeFilter, els.difficultyFilter].forEach((node) => node.addEventListener("change", () => {
    recomputeFilteredIndices();
    selectSampleByFilteredIndex(findFirstInterestingIndex());
  }));
  els.prevButton.addEventListener("click", () => selectSampleByFilteredIndex(state.currentFilteredIndex - 1));
  els.nextButton.addEventListener("click", () => selectSampleByFilteredIndex(state.currentFilteredIndex + 1));
  els.notesInput.addEventListener("input", () => {
    getCurrentAuditEntry().notes = els.notesInput.value;
    scheduleSave();
  });
  [els.toggleTextPolygon, els.toggleTextBox, els.toggleAnchorBox, els.toggleRefBox, els.toggleNeighborBoxes, els.toggleNearbyAnchors, els.toggleCentroids].forEach((node) => {
    node.addEventListener("change", renderCanvas);
  });
  window.addEventListener("keydown", (event) => {
    if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {
      return;
    }
    if (event.key === "ArrowRight") {
      selectSampleByFilteredIndex(state.currentFilteredIndex + 1);
    } else if (event.key === "ArrowLeft") {
      selectSampleByFilteredIndex(state.currentFilteredIndex - 1);
    }
  });
  els.sourceImage.addEventListener("load", () => {
    renderCanvas();
    const sample = currentSample();
    if (sample) {
      renderImageMeta(sample);
    }
  });
}

function restoreStageFlowState() {
  const collapsed = window.localStorage.getItem("sgocr.stageFlowCollapsed") === "1";
  setStageFlowCollapsed(collapsed);
}

function toggleStageFlowVisibility() {
  const collapsed = els.stageFlowBody.hidden;
  setStageFlowCollapsed(!collapsed);
}

function setStageFlowCollapsed(collapsed) {
  els.stageFlowBody.hidden = collapsed;
  els.toggleStageFlow.textContent = collapsed ? "Expand" : "Collapse";
  els.toggleStageFlow.setAttribute("aria-expanded", collapsed ? "false" : "true");
  window.localStorage.setItem("sgocr.stageFlowCollapsed", collapsed ? "1" : "0");
}

function renderExperimentSelect() {
  els.experimentSelect.innerHTML = "";
  for (const experiment of state.experiments) {
    const option = document.createElement("option");
    option.value = experiment.name;
    option.textContent = experiment.name;
    option.selected = state.experiment && experiment.name === state.experiment.name;
    els.experimentSelect.appendChild(option);
  }
}

function renderFacetFilters() {
  renderSelectOptions(els.questionTypeFilter, ["all", ...collectFacetValues("question_type")]);
  renderSelectOptions(els.difficultyFilter, ["all", ...collectFacetValues("difficulty")]);
}

function renderSelectOptions(select, values) {
  const current = select.value || "all";
  select.innerHTML = "";
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value === "all" ? "All" : value;
    option.selected = value === current || (!values.includes(current) && value === "all");
    select.appendChild(option);
  });
}

function collectFacetValues(key) {
  const values = new Set();
  state.samples.forEach((sample) => {
    const value = sample.tuple?.tags?.[key];
    if (typeof value === "string" && value) {
      values.add(value);
    }
  });
  return Array.from(values).sort();
}

function recomputeFilteredIndices() {
  const mode = els.filterSelect.value;
  state.filteredIndices = state.samples
    .map((sample, index) => ({ sample, index }))
    .filter(({ sample }) => matchFilter(sample, mode))
    .map(({ index }) => index);
  if (state.filteredIndices.length === 0) {
    state.filteredIndices = state.samples.map((_, index) => index);
  }
}

function matchFilter(sample, mode) {
  const entry = state.audit[sample.sampleId];
  const tags = sample.tuple.tags || {};
  if (mode === "needs_review") {
    if (entry && entry.overallGo) {
      return false;
    }
  }
  if (mode === "flagged" && (!entry || !(entry.overallGo === "no" || entry.anchorGrounding === "no" || entry.ocrCorrect === "no" || entry.spatialRelation === "no" || entry.questionFaithful.includes("no") || entry.questionNatural === "no" || entry.answerCorrect === "no"))) {
    return false;
  }
  if (mode === "final_dataset" && sample.tuple.resolvable === false) {
    return false;
  }
  if (mode === "final_dataset" && sample.sampleStatus !== "accepted") {
    return false;
  }
  if (mode === "rejected_only" && sample.sampleStatus !== "rejected") {
    return false;
  }
  if (mode === "dropped_resolvability" && sample.tuple.resolvable !== false) {
    return false;
  }
  if (els.questionTypeFilter.value !== "all" && tags.question_type !== els.questionTypeFilter.value) {
    return false;
  }
  if (els.difficultyFilter.value !== "all" && tags.difficulty !== els.difficultyFilter.value) {
    return false;
  }
  return true;
}

function findFirstInterestingIndex() {
  const firstUnreviewed = state.filteredIndices.findIndex((sampleIndex) => {
    const sample = state.samples[sampleIndex];
    const entry = state.audit[sample.sampleId];
    return !entry || !entry.overallGo;
  });
  return firstUnreviewed >= 0 ? firstUnreviewed : 0;
}

function selectSampleByFilteredIndex(filteredIndex) {
  if (state.filteredIndices.length === 0) {
    return;
  }
  state.currentFilteredIndex = Math.max(0, Math.min(filteredIndex, state.filteredIndices.length - 1));
  renderCurrentSample();
}

function renderCurrentSample() {
  const sample = currentSample();
  if (!sample) {
    return;
  }
  const audit = getCurrentAuditEntry();
  els.notesInput.value = audit.notes || "";
  updateChoiceRows(audit);
  renderStatus();
  renderImageMeta(sample);
  renderTagsSummary(sample);
  renderTupleSummary(sample);
  renderGroundingSummary(sample);
  renderQuestions(sample, audit);
  renderKDMetadata(sample);
  els.tupleJson.textContent = JSON.stringify(sample.tuple, null, 2);
  els.usageJson.textContent = JSON.stringify({ usage: sample.usage, summary: sample.summary, filterStage: sample.filterStage }, null, 2);

  if (sample.imageUrl && els.sourceImage.src !== sample.imageUrl) {
    els.sourceImage.src = sample.imageUrl;
  } else if (!sample.imageUrl) {
    const canvas = els.reviewCanvas;
    const ctx = canvas.getContext("2d");
    canvas.width = 800;
    canvas.height = 400;
    ctx.fillStyle = "#f7f1e4";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#6a655b";
    ctx.font = "18px Georgia";
    ctx.fillText("Missing image path", 24, 48);
  }
  renderCanvas();
}

function renderStatus() {
  const sample = currentSample();
  const reviewed = Object.values(state.audit).filter((entry) => entry.overallGo).length;
  els.progressState.textContent = `${state.currentFilteredIndex + 1} / ${state.filteredIndices.length} visible · ${reviewed} reviewed`;
  els.statusSummary.innerHTML = "";
  for (const pill of buildStatusPills(sample)) {
    els.statusSummary.appendChild(pill);
  }
}

function buildStatusPills(sample) {
  const pills = [];
  if (state.experiment) {
    pills.push(makePill(`experiment ${state.experiment.name}`));
    if (state.experiment.model) {
      pills.push(makePill(`model ${state.experiment.model}`));
    }
    if (state.experiment.promptVariant) {
      pills.push(makePill(`prompt ${state.experiment.promptVariant}`));
    }
    if (state.experiment.resolvableTupleCount != null && state.experiment.inputTupleCount != null) {
      pills.push(makePill(`resolvable ${state.experiment.resolvableTupleCount}/${state.experiment.inputTupleCount}`));
    }
  }
  if (sample) {
    pills.push(makePill(`image ${sample.imageId}`));
    if (sample.tuple.answer) {
      pills.push(makePill(`answer ${sample.tuple.answer}`));
    }
    if (sample.tuple.tags?.question_type) {
      pills.push(makePill(sample.tuple.tags.question_type));
    }
    if (sample.tuple.tags?.difficulty) {
      pills.push(makePill(sample.tuple.tags.difficulty));
    }
    pills.push(makePill(`questions ${sample.questions.length}`));
    pills.push(makePill(sample.tuple.resolvable === false ? "dropped by resolvability" : "kept in final dataset"));
  }
  return pills;
}

function renderStageFlow() {
  const experiment = state.experiment;
  els.stageFlow.innerHTML = "";
  els.stageFailures.innerHTML = "";
  if (!experiment) {
    return;
  }

  const stats = experiment.resolvabilityStats || {};
  const stageCounts = experiment.stageCounts || null;
  const stages = stageCounts ? buildGenericStages(stageCounts) : [
    {
      name: "Images",
      count: experiment.sameImageUniverseCount ?? experiment.inputTupleCount ?? state.samples.length,
      detail: "same validation image universe",
      tone: "info",
    },
    {
      name: "Primary Tuples",
      count: experiment.inputTupleCount ?? state.samples.length,
      detail: "bootstrap candidate tuples before 224px filter",
      tone: "info",
    },
    {
      name: "Text Nodes",
      count: stats.total_text_nodes ?? null,
      detail: "all valid text nodes found in those images",
      tone: "info",
    },
    {
      name: "Resolvable Nodes",
      count: stats.passing_text_nodes ?? null,
      detail: stats.dropped_text_nodes != null ? `${stats.dropped_text_nodes} nodes dropped` : "after 224px patch filter",
      tone: stats.dropped_text_nodes ? "fail" : "pass",
    },
    {
      name: "Final Tuples",
      count: experiment.resolvableTupleCount ?? experiment.inputTupleCount ?? state.samples.length,
      detail: experiment.droppedTupleCount != null ? `${experiment.droppedTupleCount} primary tuples dropped` : "tuples kept for QA",
      tone: experiment.droppedTupleCount ? "fail" : "pass",
    },
    {
      name: "QA Rows",
      count: experiment.generatedQas ?? experiment.sampleCount ?? state.samples.length,
      detail: "teacher-generated QA on kept tuples",
      tone: "info",
    },
    {
      name: "Accepted QA",
      count: experiment.acceptedQas ?? experiment.sampleCount ?? state.samples.length,
      detail: experiment.qaAcceptRate != null ? `${(experiment.qaAcceptRate * 100).toFixed(1)}% verifier pass` : "final review rows",
      tone: "pass",
    },
  ].filter((stage) => stage.count != null);

  stages.forEach((stage, index) => {
    els.stageFlow.appendChild(makeStageNode(stage));
    if (index < stages.length - 1) {
      const arrow = document.createElement("div");
      arrow.className = "stage-arrow";
      arrow.textContent = "→";
      els.stageFlow.appendChild(arrow);
    }
  });

  const failures = [];
  if (stats.dropped_text_nodes != null) {
    failures.push({
      title: "Tiny text removed",
      detail: `${stats.dropped_text_nodes} / ${stats.total_text_nodes} text nodes failed the 224px resolvability threshold`,
    });
  }
  if (experiment.droppedTupleCount != null && experiment.inputTupleCount != null) {
    failures.push({
      title: "Primary tuple loss",
      detail: `${experiment.droppedTupleCount} / ${experiment.inputTupleCount} primary tuples were excluded from the final dataset`,
    });
  }
  if (stageCounts && typeof stageCounts.resolvable_nodes === "number" && typeof stageCounts.grounded_nodes === "number" && stageCounts.resolvable_nodes > stageCounts.grounded_nodes) {
    failures.push({
      title: "Semantic anchor miss",
      detail: `${stageCounts.resolvable_nodes - stageCounts.grounded_nodes} resolvable nodes did not find a usable semantic anchor`,
    });
  }
  if (experiment.generatedQas != null && experiment.acceptedQas != null && experiment.generatedQas !== experiment.acceptedQas) {
    failures.push({
      title: "Verifier rejection",
      detail: `${experiment.generatedQas - experiment.acceptedQas} QA rows failed downstream verification`,
    });
  }

  if (!failures.length) {
    failures.push({
      title: "No downstream verifier loss",
      detail: "All generated QA rows in the current final dataset passed the verifier.",
    });
  }

  if (experiment.selectedQuestionTypeCounts) {
    failures.push({
      title: "Selected question mix",
      detail: Object.entries(experiment.selectedQuestionTypeCounts).map(([key, value]) => `${key}: ${value}`).join(" · "),
    });
  }
  if (Array.isArray(experiment.disabledQuestionTypes) && experiment.disabledQuestionTypes.length) {
    failures.push({
      title: "Disabled question types",
      detail: experiment.disabledQuestionTypes.join(", "),
    });
  }

  failures.forEach((failure) => {
    const chip = document.createElement("div");
    chip.className = "failure-chip";
    chip.innerHTML = `<strong>${failure.title}</strong>${failure.detail}`;
    els.stageFailures.appendChild(chip);
  });
}

function buildGenericStages(stageCounts) {
  const order = [
    ["images", "Images", "validation image universe"],
    ["detected_boxes", "Detected Boxes", "text detections before OCR consensus"],
    ["text_nodes", "Text Nodes", "all accepted OCR nodes before resolution filter"],
    ["resolvable_nodes", "Resolvable Nodes", "nodes still readable at 224px"],
    ["anchor_tag_rows", "Anchor Tags", "semantic tag proposals from context captions"],
    ["grounded_nodes", "Grounded Nodes", "resolvable nodes with at least one semantic anchor"],
    ["verified_tuples", "Verified Tuples", "all tuple objects after grouping"],
    ["candidate_tuples", "Candidates", "question-type candidates before selection"],
    ["selected_tuples", "Selected Tuples", "per-image tuples sent to the teacher"],
    ["raw_qa_rows", "Raw QA", "teacher outputs before verification"],
    ["final_qa_rows", "Accepted QA", "final rows shown in review"],
  ];
  return order
    .filter(([key]) => stageCounts[key] != null)
    .map(([key, name, detail], index, rows) => {
      const count = stageCounts[key];
      const prevCount = index > 0 ? stageCounts[rows[index - 1][0]] : null;
      const lost = typeof prevCount === "number" && typeof count === "number" && prevCount > count ? prevCount - count : 0;
      return {
        name,
        count,
        detail: lost > 0 ? `${detail} · ${lost} lost at this stage` : detail,
        tone: lost > 0 ? "fail" : key === "final_qa_rows" ? "pass" : "info",
      };
    });
}

function makeStageNode(stage) {
  const node = document.createElement("div");
  node.className = `stage-node ${stage.tone}`;
  node.innerHTML = `
    <div class="stage-name">${stage.name}</div>
    <div class="stage-count">${formatStageCount(stage.count)}</div>
    <div class="stage-detail">${stage.detail}</div>
  `;
  return node;
}

function makePill(text) {
  const span = document.createElement("span");
  span.className = "pill";
  span.textContent = text;
  return span;
}

function formatStageCount(value) {
  if (value == null) {
    return "n/a";
  }
  return Number(value).toLocaleString();
}

function renderImageMeta(sample) {
  const geometry = sample.geometry;
  const naturalWidth = els.sourceImage.naturalWidth || null;
  const naturalHeight = els.sourceImage.naturalHeight || null;
  const scaleX = sample.imageWidth && naturalWidth ? naturalWidth / sample.imageWidth : null;
  const scaleY = sample.imageHeight && naturalHeight ? naturalHeight / sample.imageHeight : null;
  const resolvability = sample.tuple.resolvability || {};
  const items = [
    ["tuple image", `${sample.imageWidth || "?"} × ${sample.imageHeight || "?"}`],
    ["served image", naturalWidth && naturalHeight ? `${naturalWidth} × ${naturalHeight}` : "loading"],
    ["coord scale", scaleX && scaleY ? `${scaleX.toFixed(4)} × ${scaleY.toFixed(4)}` : "n/a"],
    ["resolvable", sample.tuple.resolvable === false ? "no" : "yes"],
    ["text @224", resolvability.text_px_w != null && resolvability.text_px_h != null ? `${Number(resolvability.text_px_w).toFixed(2)} × ${Number(resolvability.text_px_h).toFixed(2)}` : "n/a"],
    ["text centroid", pointString(geometry.textCentroid)],
    ["text box", boxString(geometry.textBox)],
    ["anchor centroid", pointString(geometry.anchorCentroid)],
    ["anchor box", boxString(geometry.anchorBox)],
    ["query centroid", pointString(geometry.queryAnchorCentroid)],
    ["query box", boxString(geometry.queryAnchorBox)],
    ["ref centroid", pointString(geometry.refCentroid)],
    ["ref box", boxString(geometry.refBox)],
    ["anchor source", geometry.anchorBoxSource || "n/a"],
  ];
  els.imageMeta.innerHTML = "";
  for (const [label, value] of items) {
    els.imageMeta.appendChild(makeMetaItem(label, value));
  }
}

function renderTupleSummary(sample) {
  const tuple = sample.tuple;
  const items = [
    ["ann_id", tuple.ann_id || "n/a"],
    ["answer", tuple.answer || "n/a"],
    ["answer_normalized", tuple.answer_normalized || "n/a"],
    ["question_type", tuple.tags?.question_type || tuple.question_type || "n/a"],
    ["anchor_label", tuple.anchor_label || "n/a"],
    ["anchor_label_source", tuple.grounding?.anchor_label_source || tuple.anchor_label_source || "n/a"],
    ["anchor_category", tuple.tags?.anchor_category || "n/a"],
    ["region_key", tuple.region_key || "n/a"],
    ["relation", tuple.relation || "n/a"],
    ["unique", String(tuple.unique)],
    ["answer_level", tuple.answer_level || "n/a"],
    ["group_kind", tuple.group_kind || "n/a"],
    ["answer_source", tuple.tags?.answer_source || tuple.answer_source || "n/a"],
    ["difficulty", tuple.tags?.difficulty || "n/a"],
    ["yesno_polarity", tuple.tags?.yesno_polarity || "n/a"],
    ["anchor_property_type", tuple.tags?.anchor_property_type || "n/a"],
    ["valid_text_count", String(tuple.valid_text_count ?? "n/a")],
    ["area_fraction", tuple.area_fraction != null ? Number(tuple.area_fraction).toFixed(4) : "n/a"],
    ["density_bucket", tuple.density_bucket || "n/a"],
    ["area_bucket", tuple.area_bucket || "n/a"],
    ["score", tuple.score != null ? Number(tuple.score).toFixed(3) : "n/a"],
    ["competing_tuples", tuple.kd_metadata && tuple.kd_metadata.competing_tuples != null ? String(tuple.kd_metadata.competing_tuples) : "n/a"],
    ["image_path", sample.imagePath || "n/a"],
  ];
  els.tupleSummary.innerHTML = "";
  for (const [label, value] of items) {
    els.tupleSummary.appendChild(makeMetaItem(label, value));
  }
}

function renderTagsSummary(sample) {
  els.tagsSummary.innerHTML = "";
  const tags = sample.tuple.tags || {};
  ["question_type", "answer_type", "answer_source", "consensus_tier", "relation", "difficulty", "anchor_category"].forEach((key) => {
    if (tags[key]) {
      els.tagsSummary.appendChild(makePill(`${key} ${tags[key]}`));
    }
  });
  if (tags.unique === true) {
    els.tagsSummary.appendChild(makePill("unique true"));
  } else if (tags.unique === false) {
    els.tagsSummary.appendChild(makePill("unique false"));
  }
  if (tags.yesno_polarity) {
    els.tagsSummary.appendChild(makePill(`yesno ${tags.yesno_polarity}`));
  }
  if (tags.anchor_property_type) {
    els.tagsSummary.appendChild(makePill(`attr ${tags.anchor_property_type}`));
  }
  els.tagsSummary.appendChild(makePill(`status ${sample.sampleStatus}`));
  if (sample.failureReason) {
    els.tagsSummary.appendChild(makePill(`fail ${sample.failureReason}`));
  }
}

function renderGroundingSummary(sample) {
  const grounding = sample.tuple.grounding || {};
  const semantic = grounding.semantic_debug || {};
  const items = [
    ["text_node_ids", Array.isArray(grounding.text_node_ids) ? grounding.text_node_ids.join(", ") : sample.tuple.text_node_ids?.join?.(", ") || "n/a"],
    ["anchor_score", grounding.anchor_score != null ? String(grounding.anchor_score) : sample.tuple.anchor_score != null ? String(sample.tuple.anchor_score) : "n/a"],
    ["anchor_synonyms", Array.isArray(grounding.anchor_synonyms) ? grounding.anchor_synonyms.join(" | ") : Array.isArray(sample.tuple.anchor_synonyms) ? sample.tuple.anchor_synonyms.join(" | ") : "n/a"],
    ["ocr_confidence", grounding.ocr_confidence != null ? String(grounding.ocr_confidence) : sample.tuple.ocr_confidence != null ? String(sample.tuple.ocr_confidence) : "n/a"],
    ["text_bbox", Array.isArray(grounding.text_bbox) ? formatBBox(grounding.text_bbox) : sample.geometry.textBox ? boxString(sample.geometry.textBox) : "n/a"],
    ["anchor_box", Array.isArray(grounding.anchor_box) ? formatBBox(grounding.anchor_box) : sample.geometry.anchorBox ? boxString(sample.geometry.anchorBox) : "n/a"],
    ["ref_box", Array.isArray(grounding.ref_box) ? formatBBox(grounding.ref_box) : sample.geometry.refBox ? boxString(sample.geometry.refBox) : "n/a"],
    ["region_key", grounding.region_key || sample.tuple.region_key || "n/a"],
    ["ambiguity_level", sample.tuple.tags?.ambiguity_level || "n/a"],
    ["location_phrase", sample.tuple.location_phrase || "n/a"],
    ["anchor_region", grounding.anchor_region_phrase || "n/a"],
    ["anchor_local_phrase", grounding.anchor_local_phrase || sample.tuple.anchor_local_phrase || "n/a"],
    ["specific_location", grounding.specific_location_phrase || sample.tuple.specific_location_phrase || "n/a"],
    ["query_anchor_label", grounding.query_anchor_label || "n/a"],
    ["query_anchor_box", Array.isArray(grounding.query_anchor_box) ? formatBBox(grounding.query_anchor_box) : sample.geometry.queryAnchorBox ? boxString(sample.geometry.queryAnchorBox) : "n/a"],
    ["query_anchor_region", grounding.query_anchor_region_phrase || "n/a"],
    ["query_anchor_local", grounding.query_anchor_local_phrase || sample.tuple.query_anchor_local_phrase || "n/a"],
    ["query_location_phrase", grounding.query_location_phrase || "n/a"],
    ["query_specific_location", grounding.query_specific_location_phrase || sample.tuple.query_specific_location_phrase || "n/a"],
    ["reverse_scope_pref", grounding.reverse_ground_scope_preference || sample.tuple.reverse_ground_scope_preference || "n/a"],
    ["query_matches_text", grounding.query_grounding_matches_text === false ? "no" : grounding.query_grounding_matches_text === true ? "yes" : "n/a"],
    ["sample_status", sample.sampleStatus || "n/a"],
    ["failure_reason", sample.failureReason || "n/a"],
    ["anchor_source", grounding.anchor_source || sample.tuple.anchor_source || "n/a"],
    ["anchor_label_source", grounding.anchor_label_source || sample.tuple.anchor_label_source || "n/a"],
    ["raw_anchor_label", semantic.raw_anchor_label || "n/a"],
    ["semantic_caption", semantic.caption || "n/a"],
    ["discovered_tags", Array.isArray(semantic.discovered_tags) ? semantic.discovered_tags.join(" | ") : "n/a"],
    ["final_prompt_tags", Array.isArray(semantic.final_prompt_tags) ? semantic.final_prompt_tags.join(" | ") : "n/a"],
    ["anchor_support_count", semantic.anchor_support_count != null ? String(semantic.anchor_support_count) : "n/a"],
    ["source_support_count", semantic.anchor_source_support_count != null ? String(semantic.anchor_source_support_count) : "n/a"],
    ["support_labels", Array.isArray(semantic.anchor_support_labels) ? semantic.anchor_support_labels.join(" | ") : "n/a"],
    ["relabel_label", semantic.anchor_relabel?.label || "n/a"],
    ["relabel_model", semantic.anchor_relabel?.model || "n/a"],
    ["best_anchor_relevance", semantic.best_anchor_relevance != null ? String(semantic.best_anchor_relevance) : "n/a"],
    ["layout_detail", semantic.location_detail || sample.tuple.kd_metadata?.layout_detail || "n/a"],
    ["merge_orientation", semantic.merge_debug?.merge_orientation || "n/a"],
    ["merged_node_count", semantic.merge_debug?.merged_node_count != null ? String(semantic.merge_debug.merged_node_count) : "n/a"],
    ["merged_resolvable_count", semantic.merge_debug?.merged_resolvable_count != null ? String(semantic.merge_debug.merged_resolvable_count) : "n/a"],
    ["merged_unresolvable_count", semantic.merge_debug?.merged_unresolvable_count != null ? String(semantic.merge_debug.merged_unresolvable_count) : "n/a"],
    ["merge_cluster_labels", Array.isArray(semantic.merge_debug?.cluster_labels) ? semantic.merge_debug.cluster_labels.join(" | ") : "n/a"],
  ];
  els.groundingSummary.innerHTML = "";
  for (const [label, value] of items) {
    els.groundingSummary.appendChild(makeMetaItem(label, value));
  }
}

function makeMetaItem(label, value) {
  const item = document.createElement("div");
  item.className = "meta-item";
  const labelNode = document.createElement("div");
  labelNode.className = "label";
  labelNode.textContent = label;
  const valueNode = document.createElement("div");
  valueNode.className = "value";
  valueNode.textContent = value;
  item.append(labelNode, valueNode);
  return item;
}

function renderQuestions(sample, audit) {
  els.questionList.innerHTML = "";
  sample.questions.forEach((question, index) => {
    const card = document.createElement("div");
    card.className = "question-card";
    const q = document.createElement("div");
    q.className = "question-text";
    q.textContent = `${index + 1}. ${question.question}`;
    const a = document.createElement("div");
    a.className = "answer-text";
    a.textContent = `answer: ${question.answer}`;
    const badges = document.createElement("div");
    badges.className = "badges";
    const validation = question.validation || {};
    badges.append(
      makeBadge(`type ${sample.tuple.tags?.question_type || sample.tuple.question_type || "n/a"}`),
      makeBadge(`ans_src ${sample.tuple.tags?.answer_source || sample.tuple.answer_source || "n/a"}`),
      makeBadge(`accepted ${validation.accepted === true ? "yes" : validation.accepted === false ? "no" : "n/a"}`),
      makeBadge(`answer_ok ${validation.answer_ok === true ? "yes" : validation.answer_ok === false ? "no" : "n/a"}`),
      makeBadge(`anchor_ok ${validation.anchor_ok === true ? "yes" : validation.anchor_ok === false ? "no" : "n/a"}`),
      makeBadge(`len_ok ${validation.length_ok === true ? "yes" : validation.length_ok === false ? "no" : "n/a"}`),
    );
    const row = document.createElement("div");
    row.className = "choice-row";
    ["yes", "no", "skip"].forEach((value) => {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.value = value;
      button.textContent = `faithful ${value}`;
      if ((audit.questionFaithful[index] || "") === value) {
        button.classList.add("active");
      }
      button.addEventListener("click", () => {
        audit.questionFaithful[index] = audit.questionFaithful[index] === value ? "" : value;
        scheduleSave();
        renderCurrentSample();
      });
      row.appendChild(button);
    });
    card.append(q, a, badges, row);
    els.questionList.appendChild(card);
  });
}

function renderKDMetadata(sample) {
  const kd = sample.tuple.kd_metadata || {};
  const summaryItems = [
    ["text_density", kd.text_density != null ? String(kd.text_density) : "n/a"],
    ["competing_tuples", kd.competing_tuples != null ? String(kd.competing_tuples) : "n/a"],
    ["teacher_logprobs", kd.teacher_answer_logprobs ? "present" : "null"],
    ["image_size", Array.isArray(kd.image_size) ? kd.image_size.join(" × ") : "n/a"],
  ];
  els.kdSummary.innerHTML = "";
  for (const [label, value] of summaryItems) {
    els.kdSummary.appendChild(makeMetaItem(label, value));
  }
  renderTable(els.neighborTable, ["node_id", "text", "distance_px", "relation_to_primary", "resolvable", "bbox"], (kd.neighboring_text || []).map((row) => [
    row.node_id || "n/a",
    row.text || "",
    row.distance_px != null ? String(row.distance_px) : "n/a",
    row.relation_to_primary || "n/a",
    row.resolvable === false ? "no" : "yes",
    Array.isArray(row.bbox) ? formatBBox(row.bbox) : "n/a",
  ]));
  renderTable(els.anchorTable, ["label", "score", "overlap_with_text", "source", "region_key", "box"], (kd.nearby_anchors || []).map((row) => [
    row.label || "n/a",
    row.score != null ? String(row.score) : "n/a",
    row.overlap_with_text != null ? String(row.overlap_with_text) : row.overlap != null ? String(row.overlap) : "n/a",
    row.source || "n/a",
    row.region_key || "n/a",
    Array.isArray(row.box) ? formatBBox(row.box) : "n/a",
  ]));
}

function renderTable(table, headers, rows) {
  table.innerHTML = "";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headers.forEach((header) => {
    const cell = document.createElement("th");
    cell.textContent = header;
    headRow.appendChild(cell);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  if (!rows.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = headers.length;
    cell.textContent = "none";
    row.appendChild(cell);
    tbody.appendChild(row);
  } else {
    rows.forEach((values) => {
      const row = document.createElement("tr");
      values.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.appendChild(cell);
      });
      tbody.appendChild(row);
    });
  }
  table.appendChild(tbody);
}

function makeBadge(text) {
  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = text;
  return badge;
}

function updateChoiceRows(audit) {
  document.querySelectorAll(".choice-row[data-field]").forEach((row) => {
    const field = row.dataset.field;
    row.querySelectorAll("button").forEach((button) => {
      button.classList.toggle("active", audit[field] === button.dataset.value);
    });
  });
}

function currentSample() {
  if (state.filteredIndices.length === 0) {
    return null;
  }
  return state.samples[state.filteredIndices[state.currentFilteredIndex]] || null;
}

function getCurrentAuditEntry() {
  const sample = currentSample();
  if (!sample) {
    return null;
  }
  if (!state.audit[sample.sampleId]) {
    state.audit[sample.sampleId] = {
      sampleId: sample.sampleId,
      sampleIndex: sample.sampleIndex,
      imageId: sample.imageId,
      annId: sample.annId,
      ocrCorrect: "",
      anchorGrounding: "",
      spatialRelation: "",
      questionNatural: "",
      answerCorrect: "",
      overallGo: "",
      questionFaithful: Array.from({ length: sample.questions.length }, () => ""),
      tagsSnapshot: sample.tuple.tags || null,
      notes: "",
      updatedAt: new Date().toISOString(),
    };
  }
  const entry = state.audit[sample.sampleId];
  if (entry.questionFaithful.length < sample.questions.length) {
    entry.questionFaithful = sample.questions.map((_, index) => entry.questionFaithful[index] || "");
  }
  return entry;
}

function scheduleSave() {
  const audit = getCurrentAuditEntry();
  if (!audit) {
    return;
  }
  audit.updatedAt = new Date().toISOString();
  setSaveState("Saving");
  clearTimeout(state.saveTimer);
  state.saveTimer = setTimeout(() => saveCurrentAudit(), 250);
}

async function saveCurrentAudit() {
  const audit = getCurrentAuditEntry();
  if (!audit || !state.experiment) {
    return;
  }
  try {
    await fetchJson(`/api/audit?experiment=${encodeURIComponent(state.experiment.name)}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(audit),
    });
    setSaveState("Saved");
    renderStatus();
  } catch (error) {
    console.error(error);
    setSaveState(`Save failed: ${error.message}`);
  }
}

function setSaveState(text) {
  els.saveState.textContent = text;
}

async function fetchJson(url, init) {
  const response = await fetch(url, init);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

function renderCanvas() {
  const sample = currentSample();
  if (!sample) {
    return;
  }
  const image = els.sourceImage;
  if (!image.complete || !image.naturalWidth) {
    return;
  }
  const canvas = els.reviewCanvas;
  const ctx = canvas.getContext("2d");
  const targetWidth = sample.imageWidth || image.naturalWidth;
  const targetHeight = sample.imageHeight || image.naturalHeight;
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  if (els.toggleNearbyAnchors.checked) {
    (sample.tuple.kd_metadata?.nearby_anchors || []).forEach((anchor, index) => {
      if (Array.isArray(anchor.box) && anchor.box.length >= 4) {
        drawBox(ctx, toDisplayBox(anchor.box), "#6a4fb0", `nearby ${index + 1}`);
      }
    });
  }
  if (els.toggleNeighborBoxes.checked) {
    (sample.tuple.kd_metadata?.neighboring_text || []).forEach((neighbor, index) => {
      if (Array.isArray(neighbor.bbox) && neighbor.bbox.length >= 4) {
        drawBox(ctx, toDisplayBox(neighbor.bbox), "#d9841f", `nbr ${index + 1}`);
      }
    });
  }
  if (els.toggleAnchorBox.checked && sample.geometry.anchorBox) {
    drawBox(ctx, sample.geometry.anchorBox, "#0d5b55", sample.tuple.anchor_label || "anchor");
  }
  if (els.toggleAnchorBox.checked && sample.geometry.queryAnchorBox && sample.tuple.grounding?.query_grounding_matches_text === false) {
    const queryLabel = sample.tuple.grounding?.query_anchor_label || "query";
    drawBox(ctx, sample.geometry.queryAnchorBox, "#8e2f82", `query ${queryLabel}`);
  }
  if (els.toggleRefBox.checked && sample.geometry.refBox) {
    drawBox(ctx, sample.geometry.refBox, "#b1472f", sample.tuple.ref_label || "ref");
  }
  if (els.toggleTextPolygon.checked && sample.geometry.textPolygon.length) {
    drawPolygon(ctx, sample.geometry.textPolygon, "#2d6d39");
  }
  if (els.toggleTextBox.checked && sample.geometry.textBox) {
    drawBox(ctx, sample.geometry.textBox, "#2d6d39", "text");
  }
  if (els.toggleCentroids.checked) {
    drawCentroid(ctx, sample.geometry.textCentroid, "#2d6d39");
    drawCentroid(ctx, sample.geometry.anchorCentroid, "#0d5b55");
    if (sample.tuple.grounding?.query_grounding_matches_text === false) {
      drawCentroid(ctx, sample.geometry.queryAnchorCentroid, "#8e2f82");
    }
    drawCentroid(ctx, sample.geometry.refCentroid, "#b1472f");
  }
}

function drawPolygon(ctx, points, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = `${color}33`;
  ctx.lineWidth = 3;
  ctx.beginPath();
  points.forEach((point, index) => {
    if (index === 0) {
      ctx.moveTo(point.x, point.y);
    } else {
      ctx.lineTo(point.x, point.y);
    }
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawBox(ctx, box, color, label) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.setLineDash([10, 6]);
  ctx.strokeRect(box.x1, box.y1, box.width, box.height);
  ctx.setLineDash([]);
  ctx.fillStyle = color;
  ctx.font = "18px Georgia";
  ctx.fillText(label, box.x1 + 6, Math.max(20, box.y1 - 8));
  ctx.restore();
}

function drawCentroid(ctx, point, color) {
  if (!point) {
    return;
  }
  ctx.save();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function boxString(box) {
  if (!box) {
    return "n/a";
  }
  return `[${box.x1.toFixed(1)}, ${box.y1.toFixed(1)}] → [${box.x2.toFixed(1)}, ${box.y2.toFixed(1)}]`;
}

function formatBBox(box) {
  return `[${Number(box[0]).toFixed(1)}, ${Number(box[1]).toFixed(1)}] → [${Number(box[2]).toFixed(1)}, ${Number(box[3]).toFixed(1)}]`;
}

function pointString(point) {
  if (!point) {
    return "n/a";
  }
  return `(${point.x.toFixed(1)}, ${point.y.toFixed(1)})`;
}

function toDisplayBox(box) {
  const x1 = Number(box[0]);
  const y1 = Number(box[1]);
  const x2 = Number(box[2]);
  const y2 = Number(box[3]);
  return {
    x1,
    y1,
    x2,
    y2,
    width: x2 - x1,
    height: y2 - y1,
    cx: x1 + (x2 - x1) / 2,
    cy: y1 + (y2 - y1) / 2,
  };
}
