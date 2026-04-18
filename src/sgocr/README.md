# sgocr/ — Spatial-Grounded OCR QA Dataset Pipeline

Generates training data by: detecting text → recognizing it → finding spatial anchors → building grounded QA pairs → scoring with a teacher model.

## pipeline/ — Core Pipeline Logic

The split pieces of the dataset generation engine. See `pipeline/README.md`.

## Entry Points

| File | Role |
|---|---|
| `dev40_complete.py` | Re-export shim → `pipeline/` (question building + teacher + assembly) |
| `full_pipeline_dev40.py` | Main orchestrator: `build_dev40_semantic_dataset()` ties all stages together |

## Core Modules

| File | Contents |
|---|---|
| `bootstrap.py` | TextOCR loading, region phrases, dev subset selection |
| `bootstrap_kd.py` | Knowledge distillation helpers, bbox geometry, resolvability |
| `consensus.py` | Multi-engine OCR consensus voting |
| `verify.py` | Automated QA pair verification |
| `layout.py` | Image layout analysis |
| `paths.py` | Canonical data paths (`OCR_SPATIAL_QA_FINAL_ROOT`, etc.) |
| `stages.py` | `PipelineStage` dataclass + `PIPELINE_STAGES` registry |
| `run_quality.py` | Dataset quality metrics (type coverage, answer distribution) |
| `semantic_grounding.py` | ML-backed grounding classes: `FlorenceTagger`, `GeminiAnchorRelabeler`, `GroundingDinoGrounder`, `Sam3Refiner` |
| `semantic_dev40_tuning.py` | Runtime tuning knobs (env-var driven) |

## External API Clients

| File | Backend |
|---|---|
| `gemini_batch.py` | Google Gemini batch API |
| `nemotron_frontend.py` | NVIDIA Nemotron OCR |
| `ollama_anchor.py` | Ollama (local Gemma) anchor discovery |
| `qwen_anchor_vllm.py` | Qwen3-VL via vLLM anchor grounding |
| `teacher/` | Teacher model HTTP clients + prompt templates |

## scripts/ — Sweep Scripts

| File | Role |
|---|---|
| `sweep_runner.py` | Config-driven sweep runner (new sweeps use JSON configs) |
| `configs/` | JSON sweep configs (e.g. `ita22.json`) |
| `mixed_ita*.py` | Legacy per-iteration sweep scripts (kept in place) |
