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

Each run processes a dev set of OCR-rich images through these stages:

```
raw images
  → OCR extraction (Qwen2-VL)
  → anchor grounding (bbox → text label)
  → candidate generation (DR / RG / YN / TP / AP question types)
  → type-constrained selection (target N QAs per image)
  → inline frontier scoring (Gemini Flash)
  → image-dependence eval (image+Q vs text-only accuracy)
  → accepted dataset (ocr_qa_dataset.jsonl)
```

Question types:

| Type | Description |
|---|---|
| `DIRECT_READ` | Read a specific text element from the image |
| `REVERSE_GROUND` | Given text, locate or describe where it appears |
| `YES_NO` | Boolean question about text presence/property |
| `TEXT_PROPERTY` | Property of a text element (orientation, style…) |
| `ANCHOR_PROPERTY` | Property of the object the text is anchored to |

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
