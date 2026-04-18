from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from ..dual_anchor import RescueSelectionConfig
from ..paths import REPO_ROOT, SRC_ROOT


TARGET_VARIANT = "balanced_dam01_r48"
TYPED_GATE_QUESTION_TYPES = "DIRECT_READ,YES_NO,TEXT_PROPERTY,ANCHOR_PROPERTY"

VARIANT_DESCRIPTIONS: dict[str, str] = {
    TARGET_VARIANT: (
        "Gemma primary anchor pass over the full source universe, Qwen3-VL DAM01 rescue on "
        "the top 48 weak-coverage images, then a final Gemini-2.5-Flash teacher pass from "
        "the merged verified tuple cache."
    ),
}

BASELINE_ENV: dict[str, str] = {
    "SGOCR_TEXT_MERGE_ENABLED": "1",
    "SGOCR_TEXT_MERGE_ANCHOR_OVERLAP_MIN": "0.08",
    "SGOCR_TEXT_MERGE_X_OVERLAP_MIN": "0.42",
    "SGOCR_TEXT_MERGE_Y_OVERLAP_MIN": "0.32",
    "SGOCR_TEXT_MERGE_GAP_RATIO_MAX": "2.0",
    "SGOCR_TEXT_MERGE_HEIGHT_RATIO_MAX": "2.2",
    "SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES": "3",
    "SGOCR_DETECTOR_BOX_THRESH": "0.35",
    "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
    "SGOCR_OCR_CROP_PAD_RATIO": "0.10",
    "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.08",
    "SGOCR_STRONG_CONSENSUS_FLOOR": "0.72",
    "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.79",
    "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
    "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.42",
    "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.56",
    "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
    "SGOCR_ANCHOR_RELABEL_MODE": "flash",
    "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
    "SGOCR_GENERIC_ANCHOR_PENALTY": "0.06",
    "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.34",
    "SGOCR_FLORENCE_REGION_BONUS": "0.06",
    "SGOCR_LOCATION_WORDING_MODE": "varied",
    "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
    "SGOCR_REVERSE_GROUND_ANSWER_STYLE": "frontier",
    "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
    "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
    "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
    "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "4",
    "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.35",
    "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
    "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
    "SGOCR_DR_AMBIGUITY_REJECT_SCORE": "5",
    "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.65",
    "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.35",
    "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.65",
    "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.35",
    "SGOCR_SAM3_REFINE_MODE": "top3",
    "SGOCR_SAM3_APPLY_MODE": "targeted",
    "SGOCR_SAM3_TOPK_PROMPTS": "3",
    "SGOCR_SAM3_CONFIDENCE_THRESHOLD": "0.35",
    "SGOCR_SAM3_BOX_THRESHOLD": "0.30",
    "SGOCR_SAM3_RELEVANCE_BONUS": "0.10",
    "SGOCR_SAM3_TARGET_SUPPORT_MAX": "1",
    "SGOCR_SAM3_TARGET_CLUSTER_MIN": "2",
    "SGOCR_SAM3_TARGET_AREA_START": "0.34",
    "SGOCR_TEACHER_STRICTNESS": "strict",
    "SGOCR_INLINE_FRONTIER_ENABLED": "1",
    "SGOCR_INLINE_FRONTIER_MODEL": "gemini-3-flash-preview",
    "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "1",
    "SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY": "0.12",
}

BASELINE_CLI: dict[str, str] = {
    "--grounding-threshold": "0.28",
    "--max-tags-per-image": "14",
    "--target-per-image": "5",
    "--max-detections": "128",
}


def build_variant_cmd(
    *,
    source_dir: Path,
    out_dir: Path,
    intermediate_dir: Path,
    cache_intermediate_dir: Path | None,
    cache_level: str,
    model: str,
    workers: int,
    max_side: int,
    device: str,
    env_overrides: dict[str, str],
    cli_overrides: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT)
    env.update(env_overrides)
    cmd = [
        sys.executable,
        "-m",
        "sgocr.scripts.dev200_harness",
        "build-dev40-semantic",
        "--source-experiment-dir",
        str(source_dir),
        "--out-dir",
        str(out_dir),
        "--intermediate-dir",
        str(intermediate_dir),
        "--model",
        model,
        "--device",
        device,
        "--workers",
        str(workers),
        "--max-side",
        str(max_side),
        "--cache-level",
        cache_level,
    ]
    if cache_intermediate_dir:
        cmd.extend(["--cache-intermediate-dir", str(cache_intermediate_dir)])
    for key, value in cli_overrides.items():
        cmd.extend([key, value])
    return cmd, env


def _common_env(args: Any) -> dict[str, str]:
    return {
        **BASELINE_ENV,
        "SGOCR_OCR_FRONTEND": "nemotron_v2",
        "SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND": "florence",
        "SGOCR_QWEN_ANCHOR_INVENTORY_MODE": "independent_raw",
        "SGOCR_ANCHOR_RELABEL_MODE": "none",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "SGOCR_SIBLING_DISAMBIGUATION_ENABLED": "1",
        "SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED": "1",
        "SGOCR_QWEN_OPEN_TAG_PROMPT_MODE": "color_specific",
        "SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT": "1",
        "SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED": "1",
        "SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES": str(int(args.group_min_instances)),
        "SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED": "0",
        "SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD": "0.92",
        "SGOCR_SAM3_REFINE_MODE": "none",
        "SGOCR_SPATIAL_MIN_CENTROID_OFFSET": str(float(args.spatial_min_centroid_offset)),
        "SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED": "1",
        "SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED": "1",
        "SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES": TYPED_GATE_QUESTION_TYPES,
        "SGOCR_RG_LEAKY_LABEL_HARD_REJECT_ENABLED": "1",
        "SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST": str(float(args.rg_candidate_oversample_boost)),
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED": "1",
        "SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD": "0.40",
        "SGOCR_ANCHOR_PROPERTY_SPECIFIC_THRESHOLD": "3",
        "SGOCR_ANCHOR_TEXT_REF_LABEL_FILTER_ENABLED": "1",
    }


def build_variants(args: Any) -> dict[str, dict[str, Any]]:
    common_cli = {
        **BASELINE_CLI,
        "--grounding-threshold": str(float(args.grounding_threshold)),
        "--max-tags-per-image": str(int(args.max_tags_per_image)),
    }
    common_env = _common_env(args)
    primary_env = {
        **common_env,
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "gemma4_ollama",
        "SGOCR_GEMMA_OLLAMA_MODEL": str(args.gemma_model),
        "SGOCR_GEMMA_OLLAMA_BASE_URL": str(args.gemma_base_url),
        "SGOCR_GEMMA_OLLAMA_NUM_CTX": str(int(args.gemma_num_ctx)),
        "SGOCR_GEMMA_ITA16_PROMPT_VARIANT": "antidoc",
        "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
    }
    rescue_qwen_env = {
        **common_env,
        "SGOCR_ANCHOR_CANDIDATE_BACKEND": "qwen3_vl_vllm",
        "SGOCR_QWEN_ANCHOR_MODEL": str(args.qwen_model),
        "SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION": str(float(args.qwen_gpu_memory_utilization)),
        "SGOCR_QWEN_ANCHOR_BATCH_SIZE": str(int(args.qwen_batch_size)),
        "SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN": str(int(args.qwen_max_model_len)),
        "SGOCR_RG_PER_IMAGE_HARD_CAP": "3",
        "SGOCR_QWEN_DAM01_PROMPT_ENABLED": "1",
    }
    balanced_selection_env = {
        **primary_env,
        "SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS": "0.30",
        "SGOCR_PER_IMAGE_ANCHOR_DIVERSITY_BONUS": "0.12",
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT": "1",
        "SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY": "0.18",
    }
    return {
        "primary_gemma": {
            "env": primary_env,
            "cli": {**common_cli, "--target-per-image": "8"},
        },
        TARGET_VARIANT: {
            "note": VARIANT_DESCRIPTIONS[TARGET_VARIANT],
            "rescue_config": RescueSelectionConfig(max_images=48),
            "qwen_prompt_mode": "dam01",
            "rescue_env": rescue_qwen_env,
            "rescue_cli": {**common_cli, "--target-per-image": "8"},
            "final_env": balanced_selection_env,
            "final_cli": {**common_cli, "--target-per-image": "8"},
        },
    }
