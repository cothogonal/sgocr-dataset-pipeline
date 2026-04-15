# SGOCR — Structured OCR Spatial QA Dataset Pipeline

SGOCR builds vision-necessary spatial QA datasets from OCR-rich image sources
(ChartQA, TextOCR, COCOText). It produces question-answer pairs that genuinely
require reading the image, verified via an image-dependence eval against a
frontier model.

The generated dataset is published on HuggingFace:
**[cothogonal/sgocr-spatial-qa](https://huggingface.co/datasets/cothogonal/sgocr-spatial-qa)** *(coming soon)*

---

## Layout

```text
sgocr/
  src/
    sgocr/           ← core library (importable as `sgocr`)
      __init__.py
      paths.py
      layout.py
      stages.py
      bootstrap.py
      secrets.py
      semantic_grounding.py
      semantic_dev40_tuning.py
      dev40_complete.py
      nemotron_frontend.py
      run_quality.py
      verify.py
      consensus.py
      ocr_runtime.py
      gemini_batch.py
      teacher/
      scripts/       ← sweep & eval entrypoints
        mixed_ita13_sweep.py
        dev200_eval.py
        dev5000_build.py
        ...
  test/
    test_paths.py
    test_layout.py
    test_stages.py
```

---

## Quickstart

```bash
# Set required API keys
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...

# Run the image-dependence eval on an existing experiment
PYTHONPATH=sgocr/src python -m sgocr.scripts.dev200_eval \
  run-frontier-benchmark \
  --experiment-dir /path/to/experiment \
  --model gemini:gemini-3-flash-preview \
  --workers 4

# Run a dataset sweep
PYTHONPATH=sgocr/src python -m sgocr.scripts.mixed_ita13_sweep \
  --bundle-id my_run_$(date +%Y%m%d_%H%M%S)
```

---

## Pipeline Overview

The following describes the `ita13_propbonus_offset20` (v0) build.

```
Stage A — OCR
  Nemotron OCR v2  (NVIDIA)
  → text bounding boxes + confidence scores → text_nodes.jsonl

Stage B — Anchor Candidate Generation + Groundback
  Qwen3-VL-8B-Instruct-FP8  (vLLM, independent raw visual inventory —
                              Qwen generates its own anchor vocabulary
                              without any Florence seed tags)
  Qwen3-VL-8B-Instruct-FP8  (groundback: re-localises each anchor label
                              with a single Qwen inference, rejects
                              candidates with IoU < 0.40)
  → grounded_anchors.jsonl

Stage C — Tuple Construction + Geometric Filtering
  centroid offset gate   (min spatial offset filters text trivially locatable by position alone)
  type-constrained selection  (target=5 QAs/image, RG hard cap=2, TP/AP selection bonus=0.5,
                               RG oversample boost=1.5)
  RG leakage check       (strips color/shape tokens from REVERSE_GROUND anchor labels, re-probes)
  → selected_tuples.jsonl

Stage D — Teacher QA Generation
  Gemini 2.5 Flash  (question + answer from (image, anchor, text_node) tuple)
  → raw_qa.jsonl

Stage E — Packaging
  → ocr_qa_dataset.jsonl
```

**Question types:**

| Type | Description |
|---|---|
| `DIRECT_READ` | Read a specific text element from the image |
| `REVERSE_GROUND` | Given text, locate or describe where it appears |
| `YES_NO` | Boolean question about text presence/property |
| `TEXT_PROPERTY` | Property of a text element (orientation, style, curvature…) |
| `ANCHOR_PROPERTY` | Property of the object the text is anchored to (color, material…) |

**Evaluation:**

Two passes run at the end of each sweep to measure dataset quality:

- **Inline frontier scoring** — Gemini 3 Flash Preview answers every accepted QA with the image in context. Soft-match against gold answer produces a per-run `sweep_score` for variant comparison.
- **Image-dependence eval** — Gemini 3 Flash Preview answers each QA twice: once with image+question, once with text-only. The gap between the two measures how much the image is actually required. *Vision-necessary rate* (image correct, text-only wrong) and *text-leaky rate* (text-only correct) are the primary output quality signals.

---

## Sample Rows

**DIRECT_READ** — *"What text is on the black laptop open lid in the left half of the image?"* → **ubuntu**

![DIRECT_READ sample](assets/sample_direct_read.jpg)

**REVERSE_GROUND** — *"Where is the text 'mandalina' located?"* → **upper text in the lower-left area of the image**

![REVERSE_GROUND sample](assets/sample_reverse_ground.jpg)

Both samples are from the `ita13_propbonus_offset20` v0 build (315 accepted QAs, target=5, cap=2, boost=1.5, offset=0.20, prop_bonus=0.5). Both rows share the same schema:

```json
{
  "sample_id":       "chartqa:shared:<hash>__DIRECT_READ",
  "image_id":        "chartqa:shared:<hash>",
  "image_path":      "data/vm_ssl/raw/chartqa/images/...",
  "image_width":     800,
  "image_height":    557,

  "question":        "What text is on the panel ...",
  "answer":          "2021",
  "question_type":   "DIRECT_READ",

  "anchor_label":    "panel",
  "anchor_box":      [x1, y1, x2, y2],
  "text_bbox":       [x, y, w, h],
  "ocr_confidence":  0.94,
  "dataset_source":  "chartqa_train",

  "tags": {
    "question_type":       "DIRECT_READ",
    "answer_type":         "text_string",
    "difficulty":          "hard",
    "quality_tier":        "tier_a | tier_b",
    "ambiguity_level":     "high | medium | low",
    "answer_source":       "mechanical | teacher_visual",
    "text_case":           "numeric | upper | lower | mixed",
    "anchor_category":     "text_container | other",
    "has_reference_object": false
  },

  "grounding": {
    "anchor_label":             "panel",
    "anchor_box":               [x1, y1, x2, y2],
    "anchor_score":             0.62,
    "relation":                 "on | near | above | ...",
    "anchor_region_phrase":     "lower-right area of the image",
    "specific_location_phrase": "lower-right text in the lower-right area of the image",
    "query_text_reference":     "2021"
  },

  "kd_metadata": {
    "text_density":                  59,
    "local_text_cluster_shape":      "grid | scattered | ...",
    "local_text_cluster_resolvable": 56,
    "layout_detail":                 "59 nearby text boxes form a grid ...",
    "neighboring_text":              [{ "text": "source", "distance_px": 25.3, "relation_to_primary": "below" }]
  },

  "inline_frontier": {
    "model":   "gemini-3-flash-preview",
    "correct": true,
    "score":   { "prediction_norm": "2021", "gold_norm": "2021", "soft_correct": true }
  }
}
```

Every row is grounded to a visible text element (`text_bbox`), an anchor object (`anchor_label` + `anchor_box`), and a natural-language spatial phrase — enough to reconstruct or audit the question without re-running the model.

---

## Key Tuning Parameters

| Env var | Default | Effect |
|---|---|---|
| `SGOCR_TARGET_PER_IMAGE` | `5` | Target QA pairs per image |
| `SGOCR_RG_PER_IMAGE_HARD_CAP` | `2` | Max REVERSE_GROUND per image |
| `SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS` | `0.5` | Score bonus for TP/AP to recover vision-nec% |
| `SGOCR_SPATIAL_MIN_CENTROID_OFFSET` | `0.15` | Spatial anchor quality gate |
| `SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST` | `1.5` | Oversampling boost for RG candidates |

---

## Secrets Policy

API keys must never be committed, logged, or passed as CLI flags.
Use environment variables only:

- `GEMINI_API_KEY`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`

---

## Tests

```bash
PYTHONPATH=sgocr/src .venv_local/bin/python -m unittest discover -s sgocr/test -v
```

## Review App

```bash
cd sgocr
bun run review -- --experiment ablate_gemini_flash_natural2q_40_min4clean
```
