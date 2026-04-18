# SGOCR Dataset Pipeline

SGOCR is a source-referenced OCR visual question answering dataset pipeline. It builds grounded QA rows from OCR-rich source images, with explicit text boxes, anchor boxes, source-image join keys, and provenance fields for auditing.

The v1 dataset is published here:

- **Dataset:** [dreeseaw/SGOCR](https://huggingface.co/datasets/dreeseaw/SGOCR)
- **Release:** `v1.0.0`
- **Rows:** 5,737 QA samples over 1,958 referenced images
- **Sources:** ChartQA, TextOCR, COCO / COCO-Text

This repository contains the minimal current pipeline code used for the v1 release path. Legacy sweep scripts, review-app code, cached artifacts, and old README sample assets have been removed from this public subtree.

## What The Dataset Contains

The published dataset is metadata-only: it does not redistribute the full upstream source-image corpora. Each QA row includes source join fields so users can reconstruct image paths from their local copies of ChartQA, TextOCR, and COCO train2014 / COCO-Text.

| Split | QA rows | Referenced images | Image policy |
|---|---:|---:|---|
| `train` | 5,737 | 1,958 | Metadata-only; join against upstream images |

| Question type | Rows in v1 |
|---|---:|
| `DIRECT_READ` | 2,778 |
| `YES_NO` | 1,635 |
| `TEXT_PROPERTY` | 915 |
| `REVERSE_GROUND` | 409 |

Load it directly from Hugging Face:

```python
from datasets import load_dataset

ds = load_dataset("dreeseaw/SGOCR", data_files="data/train.jsonl", split="train")
print(ds[0]["question"])
print(ds[0]["answer"])
print(ds[0]["source_dataset"], ds[0]["source_image_id"])
```

## Pipeline Shape

The v1 run was generated from the `sgocr_mixed3000_balanced_dam01_20260417_173000` production lane. The published rows correspond to the completed `primary_gemma` stage from that lane:

1. Build a balanced source universe: 1,000 ChartQA train images, 1,000 TextOCR train images, and 1,000 COCO-Text train images, excluding the prior 150-image champion manifest.
2. Run Nemotron OCR v2 to produce text detections and resolvable OCR nodes.
3. Discover local visual anchors with Florence tag discovery and a Gemma4 Ollama anchor backend.
4. Build verified `(image, text node, anchor)` tuples with geometric and ambiguity filters.
5. Generate candidate QA rows with Gemini 2.5 Flash.
6. Apply inline frontier checks and package accepted rows plus rejection/audit metadata.

The retained code also includes the current `balanced_dam01_r48` continuation path: Qwen3-VL DAM01 rescue over the top 48 weak-coverage images, merge of verified tuple caches, and a final Gemini teacher pass.

## Repository Layout

```text
src/sgocr/
  full_pipeline_dev40.py          # Main build orchestrator
  dev40_complete.py               # Compatibility re-export for pipeline modules
  bootstrap*.py                   # Source manifests, geometry, resolvability helpers
  semantic_dev40_tuning.py        # Env-var driven production tuning
  semantic_grounding.py           # Anchor discovery and grounding helpers
  nemotron_frontend.py            # Nemotron OCR v2 integration
  ollama_anchor.py                # Gemma/Ollama anchor backend
  qwen_anchor_vllm.py             # Qwen3-VL/vLLM anchor backend
  pipeline/                       # Anchor analysis, tuple building, teacher runtime, packaging
  scripts/
    mixed3000_balanced_dam01_run.py
    production_config.py
    dev200_harness.py
    dev200_eval.py
```

## Reproducing The Production Lane

Install dependencies in your environment, then set `PYTHONPATH` from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

Required local inputs:

| Input | Expected local location by default |
|---|---|
| ChartQA images | `data/vm_ssl/raw/chartqa/images/...` |
| TextOCR metadata | `data/vm_ssl/raw/textocr_full/TextOCR_0.1_train.json` |
| TextOCR images | `data/vm_ssl/raw/textocr_trainval/...` |
| COCO-Text train images | `data/vm_ssl/raw/coco_text_materialized/train/...` |
| Nemotron OCR v2 checkout or package | `SGOCR_NEMOTRON_SRC` or importable `nemotron_ocr` |
| Ollama Gemma model | `gemma4:e4b-it-q4_K_M` by default |

Required API keys:

```bash
export GEMINI_API_KEY=...
```

Run the same production lane:

```bash
PYTHONPATH=src python -m sgocr.scripts.mixed3000_balanced_dam01_run \
  --bundle-id sgocr_mixed3000_balanced_dam01_$(date +%Y%m%d_%H%M%S) \
  --source-name chartqa1000_textocr1000_cocotext1000_source_$(date +%Y%m%d_%H%M%S)
```

Outputs are written under `data/ocr_spatial_qa/final/mixed_dev3000/` and `data/ocr_spatial_qa/intermediate/mixed_dev3000/` by default. Override roots with `SGOCR_DATA_ROOT`, `SGOCR_LOGS_ROOT`, or `SGOCR_REPO_ROOT` if your source data lives elsewhere.

## Tests

The retained tests focus on core pipeline logic rather than historical sweep scripts:

```bash
PYTHONPATH=src python -m pytest test
```

## License And Data Terms

The SGOCR annotation files are provided for research and dataset-development use. The full upstream images are not redistributed here or in the Hugging Face dataset; users must obtain ChartQA, TextOCR, COCO, and COCO-Text from their official sources and comply with each upstream license and terms of use.
