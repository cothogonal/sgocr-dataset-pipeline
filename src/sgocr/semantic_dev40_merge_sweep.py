from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, LOGS_ROOT, REPO_ROOT
from .run_quality import compute_run_quality
from .semantic_dev40_sweep_v2 import AblationSpec, append_timeline, gpu_snapshot, now_stamp, write_text


SOURCE_EXPERIMENT = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "ablate_gemini_flash_natural2q_40_min4clean"

# OCR cache sources from sweep v2
CONTROL_B06 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b06_resolvability_loose"
CONTROL_B06_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b06_resolvability_loose"
CONTROL_B07 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b07_text_recall_combo"
CONTROL_B07_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b07_text_recall_combo"
CONTROL_B05 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b05_consensus_relaxed"
CONTROL_B05_INTERMEDIATE = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "sgocr_dev40_sweep_v2_20260405_083742_b05_consensus_relaxed"

# Prior champion references
MERGE_V1 = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_merge_v1_20260405_225817_m01_targeted_sam_merge"
STRICT_CHAMPION = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / "sgocr_dev40_sweep_d_20260405_155213_d02_b06_clean_control"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the overnight merge + detection sweep (30 runs).")
    ap.add_argument("--bundle-id", default=f"sgocr_dev40_merge_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--target-per-image", type=int, default=5)
    ap.add_argument("--watch-seconds", type=int, default=60)
    ap.add_argument("--limit", type=int, default=30)
    return ap.parse_args()


def build_specs() -> list[AblationSpec]:
    # -----------------------------------------------------------------
    # Shared env blocks
    # -----------------------------------------------------------------
    # Base merge env: the proven m01 merge config
    merge_base = {
        "SGOCR_TEXT_MERGE_ENABLED": "1",
        "SGOCR_TEXT_MERGE_ANCHOR_OVERLAP_MIN": "0.08",
        "SGOCR_TEXT_MERGE_X_OVERLAP_MIN": "0.42",
        "SGOCR_TEXT_MERGE_Y_OVERLAP_MIN": "0.32",
        "SGOCR_TEXT_MERGE_GAP_RATIO_MAX": "2.0",
        "SGOCR_TEXT_MERGE_HEIGHT_RATIO_MAX": "2.2",
        "SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES": "3",
        "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "supportive",
        "SGOCR_ANCHOR_RELABEL_MODE": "flash",
        "SGOCR_ANCHOR_SUPPORT_BONUS": "0.06",
        "SGOCR_GENERIC_ANCHOR_PENALTY": "0.10",
        "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.34",
        "SGOCR_FLORENCE_REGION_BONUS": "0.06",
        "SGOCR_LOCATION_WORDING_MODE": "lite",
        "SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES": "1",
        "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "4",
        "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "3",
        "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "4",
        "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "4",
        "SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH": "2.35",
        "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "6",
        "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "5",
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.28",
        "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.12",
        "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.16",
        "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.08",
        "SGOCR_SAM3_REFINE_MODE": "top3",
        "SGOCR_SAM3_APPLY_MODE": "targeted",
        "SGOCR_SAM3_TOPK_PROMPTS": "3",
        "SGOCR_SAM3_CONFIDENCE_THRESHOLD": "0.35",
        "SGOCR_SAM3_BOX_THRESHOLD": "0.30",
        "SGOCR_SAM3_RELEVANCE_BONUS": "0.10",
        "SGOCR_SAM3_TARGET_SUPPORT_MAX": "1",
        "SGOCR_SAM3_TARGET_CLUSTER_MIN": "2",
        "SGOCR_SAM3_TARGET_AREA_START": "0.34",
    }

    # Detection-heavy base: ppocr+craft ensemble for text recall
    det_recall_base = {
        "SGOCR_DETECTOR_MODE": "ppocr_craft_ensemble",
        "SGOCR_DETECTOR_MERGE_OVERLAP": "0.60",
        "SGOCR_OCR_CROP_PAD_RATIO": "0.08",
        "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.06",
        "SGOCR_STRONG_CONSENSUS_FLOOR": "0.76",
        "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.82",
        "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
        "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.55",
        "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.62",
    }

    cli_standard = {"--grounding-threshold": "0.34", "--max-tags-per-image": "10", "--target-per-image": "5"}
    cli_det = {**cli_standard, "--max-detections": "128"}

    return [
        # =============================================================
        # TIER 1: Merge on high-recall OCR bases (3 runs)
        #   Expected champion candidates. Merge on the rich b06/b07/b05
        #   OCR caches that had 130/109/116 accepted pre-strict.
        # =============================================================
        AblationSpec(
            "m02_b06_merge_flash",
            "Merge stage on b06 loose-resolvability OCR cache + Flash teacher",
            "ocr",
            cli=dict(cli_standard),
            env=dict(merge_base),
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "m03_b07_merge_flash",
            "Merge stage on b07 text-recall-combo OCR cache + Flash teacher",
            "ocr",
            cli=dict(cli_standard),
            env=dict(merge_base),
            cache_intermediate_dir=CONTROL_B07_INTERMEDIATE,
        ),
        AblationSpec(
            "m04_b06_merge_pro",
            "Merge stage on b06 OCR cache + Pro teacher (teacher comparison vs m02)",
            "ocr",
            cli=dict(cli_standard),
            env={**merge_base, "SGOCR_ANCHOR_RELABEL_MODE": "pro"},
            model_override="gemini-2.5-pro",
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),

        # =============================================================
        # TIER 2a: Detection threshold reductions (7 runs)
        #   The user's favorite axis. Progressively lower detector
        #   thresholds with merge enabled to capture more text upstream.
        # =============================================================
        AblationSpec(
            "m05_det_box040_merge",
            "Detector box_thresh 0.40 + merge + PPOCR only",
            "none",
            cli=dict(cli_det),
            env={
                **merge_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.40",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "1.90",
            },
        ),
        AblationSpec(
            "m06_det_box035_merge",
            "Detector box_thresh 0.35 + merge + wider unclip",
            "none",
            cli=dict(cli_det),
            env={
                **merge_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.35",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
            },
        ),
        AblationSpec(
            "m07_det_box030_merge",
            "Detector box_thresh 0.30 aggressive recall probe + merge",
            "none",
            cli={**cli_standard, "--max-detections": "160"},
            env={
                **merge_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.30",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.20",
            },
        ),
        AblationSpec(
            "m08_ensemble_box045_merge",
            "PPOCR+CRAFT ensemble box_thresh 0.45 + merge (baseline ensemble)",
            "none",
            cli=dict(cli_det),
            env={
                **merge_base,
                **det_recall_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.45",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "1.90",
            },
        ),
        AblationSpec(
            "m09_ensemble_box040_merge",
            "PPOCR+CRAFT ensemble box_thresh 0.40 + merge (medium recall)",
            "none",
            cli=dict(cli_det),
            env={
                **merge_base,
                **det_recall_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.40",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.00",
            },
        ),
        AblationSpec(
            "m10_ensemble_box035_merge",
            "PPOCR+CRAFT ensemble box_thresh 0.35 + merge (aggressive recall)",
            "none",
            cli={**cli_standard, "--max-detections": "160"},
            env={
                **merge_base,
                **det_recall_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.35",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
                "SGOCR_CRAFT_TEXT_THRESHOLD": "0.32",
                "SGOCR_CRAFT_LINK_THRESHOLD": "0.28",
                "SGOCR_CRAFT_LOW_TEXT": "0.28",
            },
        ),
        AblationSpec(
            "m11_ensemble_box030_craft_low_merge",
            "PPOCR+CRAFT ensemble box_thresh 0.30 + low CRAFT thresholds + merge (maximum recall probe)",
            "none",
            cli={**cli_standard, "--max-detections": "192"},
            env={
                **merge_base,
                **det_recall_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.30",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.20",
                "SGOCR_CRAFT_TEXT_THRESHOLD": "0.28",
                "SGOCR_CRAFT_LINK_THRESHOLD": "0.25",
                "SGOCR_CRAFT_LOW_TEXT": "0.25",
                "SGOCR_CRAFT_LONG_SIZE": "1536",
            },
        ),

        AblationSpec(
            "m11b_det_box035_bigcrop_merge",
            "PPOCR box_thresh 0.35 + large crop pad 0.10 + merge (crop+detection recall combo)",
            "none",
            cli=dict(cli_det),
            env={
                **merge_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.35",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
                "SGOCR_OCR_CROP_PAD_RATIO": "0.10",
                "SGOCR_SEMANTIC_CROP_PAD_RATIO": "0.08",
                "SGOCR_STRONG_CONSENSUS_FLOOR": "0.76",
                "SGOCR_STANDARD_CONSENSUS_FLOOR": "0.82",
                "SGOCR_LOW_CONFIDENCE_FLOOR": "0.55",
                "SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES": "1.55",
                "SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES": "0.62",
            },
        ),

        # =============================================================
        # TIER 2b: Reverse-ground local/mixed recovery (3 runs)
        #   Every strict-verifier run has 0 local, 0 mixed reverse-
        #   ground answers. Push local bias hard and soften the reverse
        #   gate for rows with clean anchor-local evidence.
        # =============================================================
        AblationSpec(
            "m12_b06_merge_local_boost",
            "Merge on b06 + aggressive local reverse-ground bias + softer reverse ambiguity gate",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.48",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.25",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "7",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "m13_b06_merge_mixed_boost",
            "Merge on b06 + push both local and mixed reverse-ground answers",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.42",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.20",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.38",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.22",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "7",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "m14_b06_merge_local_extreme",
            "Merge on b06 + extreme local bias stress test",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.65",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.35",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.50",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.30",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "8",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),

        # =============================================================
        # TIER 2c: Anchor-missing reduction (3 runs)
        #   anchor_missing is the #1 waste bucket (21 in merge v1).
        #   Attack from multiple angles.
        # =============================================================
        AblationSpec(
            "m15_b06_merge_anchor_aggressive",
            "Merge on b06 + aggressive anchor expansion + lower generic penalty",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.04",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.09",
                "SGOCR_OVERSIZED_ANCHOR_AREA_START": "0.28",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "m16_b06_merge_anchor_pro_relabel",
            "Merge on b06 + Pro anchor relabel (richer label vocabulary)",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_ANCHOR_RELABEL_MODE": "pro",
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.05",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "m17_b06_merge_more_tags",
            "Merge on b06 + 14 tags per image + lower grounding threshold",
            "ocr",
            cli={"--grounding-threshold": "0.30", "--max-tags-per-image": "14", "--target-per-image": "5"},
            env={
                **merge_base,
                "SGOCR_FLORENCE_REGION_BONUS": "0.08",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),

        # =============================================================
        # TIER 2d: Ambiguity/verifier calibration (3 runs)
        # =============================================================
        AblationSpec(
            "m18_b06_merge_ambig_7_6",
            "Merge on b06 + ambiguity hard=7 reverse=6 (softer than m02)",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "m19_b06_merge_ambig_8_7",
            "Merge on b06 + ambiguity hard=8 reverse=7 (much softer, recall probe)",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "8",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "7",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "m20_b06_merge_ambig_5_4_strict",
            "Merge on b06 + ambiguity hard=5 reverse=4 (strictest, precision probe)",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "5",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "4",
                "SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD": "5",
                "SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD": "4",
                "SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD": "5",
                "SGOCR_PROPERTY_SPECIFIC_THRESHOLD": "5",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),

        # =============================================================
        # TIER 2e: Merge geometry ablations (3 runs)
        # =============================================================
        AblationSpec(
            "m21_b06_merge_wider_gap",
            "Merge on b06 + wider gap/height ratios + subsume>=2 (looser merge)",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_TEXT_MERGE_GAP_RATIO_MAX": "2.8",
                "SGOCR_TEXT_MERGE_HEIGHT_RATIO_MAX": "2.8",
                "SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES": "2",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "m22_b06_merge_tight_gap",
            "Merge on b06 + tight gap/height ratios (conservative merge)",
            "ocr",
            cli=dict(cli_standard),
            env={
                **merge_base,
                "SGOCR_TEXT_MERGE_GAP_RATIO_MAX": "1.4",
                "SGOCR_TEXT_MERGE_HEIGHT_RATIO_MAX": "1.6",
                "SGOCR_TEXT_MERGE_X_OVERLAP_MIN": "0.50",
                "SGOCR_TEXT_MERGE_Y_OVERLAP_MIN": "0.40",
            },
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),
        AblationSpec(
            "m23_b06_no_merge_control",
            "b06 with full strict verifier but NO merge (ablation control)",
            "ocr",
            cli=dict(cli_standard),
            env={k: v for k, v in merge_base.items() if not k.startswith("SGOCR_TEXT_MERGE")},
            cache_intermediate_dir=CONTROL_B06_INTERMEDIATE,
        ),

        # =============================================================
        # TIER 3a: Detection + merge combos (4 runs)
        #   Combine the best detection recall settings with merge.
        # =============================================================
        AblationSpec(
            "m24_ensemble_box040_merge_local",
            "PPOCR+CRAFT 0.40 + merge + local reverse-ground boost",
            "none",
            cli=dict(cli_det),
            env={
                **merge_base,
                **det_recall_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.40",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.00",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.45",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.22",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "7",
            },
        ),
        AblationSpec(
            "m25_ensemble_box040_merge_anchor_agg",
            "PPOCR+CRAFT 0.40 + merge + aggressive anchor expansion",
            "none",
            cli=dict(cli_det),
            env={
                **merge_base,
                **det_recall_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.40",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.00",
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.04",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.09",
            },
        ),
        AblationSpec(
            "m26_ensemble_box035_merge_ambig7",
            "PPOCR+CRAFT 0.35 + merge + softer ambiguity (7/6)",
            "none",
            cli={**cli_standard, "--max-detections": "160"},
            env={
                **merge_base,
                **det_recall_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.35",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
                "SGOCR_CRAFT_TEXT_THRESHOLD": "0.32",
                "SGOCR_CRAFT_LINK_THRESHOLD": "0.28",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
            },
        ),
        AblationSpec(
            "m27_det_box035_no_merge_control",
            "Detector box_thresh 0.35 PPOCR only NO merge (detection-only ablation control)",
            "none",
            cli=dict(cli_det),
            env={
                k: v for k, v in {
                    **merge_base,
                    "SGOCR_DETECTOR_BOX_THRESH": "0.35",
                    "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
                }.items() if not k.startswith("SGOCR_TEXT_MERGE")
            },
        ),

        # =============================================================
        # TIER 3b: Frontier combos (3 runs)
        #   Optimistic combos that stack multiple winning axes.
        # =============================================================
        AblationSpec(
            "m28_b05_merge_flash",
            "Merge on b05 relaxed-consensus OCR cache + Flash (third OCR base probe)",
            "ocr",
            cli=dict(cli_standard),
            env=dict(merge_base),
            cache_intermediate_dir=CONTROL_B05_INTERMEDIATE,
        ),
        AblationSpec(
            "m29_ensemble_box040_merge_kitchen_sink",
            "PPOCR+CRAFT 0.40 + merge + local boost + aggressive anchor + softer ambiguity (kitchen sink)",
            "none",
            cli=dict(cli_det),
            env={
                **merge_base,
                **det_recall_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.40",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.00",
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.04",
                "SGOCR_ANCHOR_SUPPORT_BONUS": "0.09",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.45",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.22",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.35",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.18",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
            },
        ),
        AblationSpec(
            "m30_ensemble_box035_merge_pro_kitchen_sink",
            "PPOCR+CRAFT 0.35 + merge + Pro relabel + local boost + softer ambiguity (max frontier)",
            "none",
            cli={**cli_standard, "--max-detections": "160"},
            env={
                **merge_base,
                **det_recall_base,
                "SGOCR_DETECTOR_BOX_THRESH": "0.35",
                "SGOCR_DETECTOR_UNCLIP_RATIO": "2.10",
                "SGOCR_CRAFT_TEXT_THRESHOLD": "0.32",
                "SGOCR_CRAFT_LINK_THRESHOLD": "0.28",
                "SGOCR_ANCHOR_RELABEL_MODE": "pro",
                "SGOCR_ANCHOR_PROMPT_EXPANSION_MODE": "aggressive",
                "SGOCR_GENERIC_ANCHOR_PENALTY": "0.04",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS": "0.48",
                "SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS": "0.25",
                "SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS": "0.38",
                "SGOCR_REVERSE_GROUND_ON_MIXED_BIAS": "0.22",
                "SGOCR_AMBIGUITY_HARD_REJECT_SCORE": "7",
                "SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE": "6",
            },
            model_override="gemini-2.5-pro",
        ),
    ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_metrics(experiment_dir: Path) -> dict[str, Any]:
    summary = read_json(experiment_dir / "summary.json")
    rows = load_rows(experiment_dir / "ocr_qa_dataset.jsonl")
    return compute_run_quality(summary, rows)


def render_report(bundle_id: str, specs: list[AblationSpec], results: dict[str, dict[str, Any]], docs_path: Path) -> str:
    lines = [
        "# SGOCR Dev40 Merge + Detection Overnight Sweep",
        "",
        f"- Bundle: `{bundle_id}`",
        f"- Source experiment: `{SOURCE_EXPERIMENT.name}`",
        f"- OCR caches: `{CONTROL_B06.name}`, `{CONTROL_B07.name}`, `{CONTROL_B05.name}`",
        f"- Prior merge reference: `{MERGE_V1.name}`",
        f"- Prior strict champion: `{STRICT_CHAMPION.name}`",
        f"- Updated: `{now_stamp()}`",
        "",
        "## Planned Ablations",
        "",
    ]
    for spec in specs:
        lines.append(f"- `{spec.name}`: {spec.description}")
    lines.extend(
        [
            "",
            "## Scoreboard",
            "",
            "| Run | Status | Accepted | Accept% | Images | No | Rev local | Rev mixed | Ambig high | Anchor miss | Ambig rej | Rev ambig | Rev invalid | Quality |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for control_name, control_path in (
        ("merge_v1", MERGE_V1),
        ("strict_champ_d02", STRICT_CHAMPION),
        ("recall_champ_b06", CONTROL_B06),
    ):
        try:
            metrics = collect_metrics(control_path)
            lines.append(
                f"| `{control_name}` | existing"
                f" | {metrics['accepted_qas']}"
                f" | {metrics['qa_accept_rate']:.3f}"
                f" | {metrics['images_with_final_rows']}"
                f" | {metrics['yesno_negative']}"
                f" | {metrics['reverse_local']}"
                f" | {metrics['reverse_mixed']}"
                f" | {metrics['ambiguity_high']}"
                f" | {metrics['anchor_missing']}"
                f" | {metrics['ambiguous_grounding']}"
                f" | {metrics['reverse_ambiguous']}"
                f" | {metrics['reverse_invalid']}"
                f" | {metrics['quality_score']:.2f} |"
            )
        except Exception:
            lines.append(f"| `{control_name}` | error | - | - | - | - | - | - | - | - | - | - | - | - |")
    for spec in specs:
        row = results.get(spec.name)
        if not row:
            lines.append(f"| `{spec.name}` | pending | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        if row.get("status") not in {"ok", "cached"}:
            lines.append(f"| `{spec.name}` | {row.get('status')} | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        m = row["metrics"]
        lines.append(
            f"| `{spec.name}` | {row.get('status')}"
            f" | {m['accepted_qas']}"
            f" | {m['qa_accept_rate']:.3f}"
            f" | {m['images_with_final_rows']}"
            f" | {m['yesno_negative']}"
            f" | {m['reverse_local']}"
            f" | {m['reverse_mixed']}"
            f" | {m['ambiguity_high']}"
            f" | {m['anchor_missing']}"
            f" | {m['ambiguous_grounding']}"
            f" | {m['reverse_ambiguous']}"
            f" | {m['reverse_invalid']}"
            f" | {m['quality_score']:.2f} |"
        )
    complete = [row["metrics"] | {"name": name} for name, row in results.items() if row.get("status") in {"ok", "cached"}]
    if complete:
        ranked = sorted(complete, key=lambda row: (-row["quality_score"], -row["accepted_qas"], row["name"]))
        lines.extend(["", "## Top Current Runs", ""])
        for row in ranked[:8]:
            lines.append(
                f"- `{row['name']}`: quality `{row['quality_score']:.2f}`"
                f", accepted `{row['accepted_qas']}`"
                f", images `{row['images_with_final_rows']}`"
                f", accept% `{row['qa_accept_rate']:.3f}`"
                f", neg `{row['yesno_negative']}`"
                f", rev-local `{row['reverse_local']}`"
                f", rev-mixed `{row['reverse_mixed']}`"
                f", ambig-high `{row['ambiguity_high']}`"
                f", anchor-miss `{row['anchor_missing']}`"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- 30-run overnight sweep combining merge stage with detection threshold reductions.",
            "- Tier 1 (m02-m04): merge on rich OCR bases, expected champion candidates.",
            "- Tier 2a (m05-m11): detection threshold reductions with merge, PPOCR-only and PPOCR+CRAFT ensemble.",
            "- Tier 2b (m12-m14): reverse-ground local/mixed recovery with progressively aggressive local bias.",
            "- Tier 2c (m15-m17): anchor-missing reduction via aggressive expansion, pro relabel, more tags.",
            "- Tier 2d (m18-m20): ambiguity gate calibration sweep.",
            "- Tier 2e (m21-m23): merge geometry ablations including no-merge control.",
            "- Tier 3a (m24-m27): detection + merge combos including detection-only control.",
            "- Tier 3b (m28-m30): frontier combos stacking multiple winning axes.",
            f"- Live bundle report mirror: `{docs_path}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_ablation(
    *,
    bundle_dir: Path,
    bundle_id: str,
    spec: AblationSpec,
    args: argparse.Namespace,
    docs_path: Path,
    report_path: Path,
    results: dict[str, dict[str, Any]],
    specs: list[AblationSpec],
) -> None:
    experiment_name = f"{bundle_id}_{spec.name}"
    out_dir = OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / experiment_name
    intermediate_dir = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / experiment_name
    log_path = bundle_dir / f"{spec.name}.log"
    timeline_path = bundle_dir / "timeline.log"
    if (out_dir / "summary.json").exists():
        metrics = collect_metrics(out_dir)
        results[spec.name] = {"status": "cached", "metrics": metrics, "experiment_name": experiment_name}
        append_timeline(timeline_path, f"END {spec.name} cached accepted={metrics['accepted_qas']} quality={metrics['quality_score']:.2f}")
        write_text(report_path, render_report(bundle_id, specs, results, docs_path))
        write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
        return

    cache_intermediate_dir = spec.cache_intermediate_dir or CONTROL_B06_INTERMEDIATE
    model_name = spec.model_override or args.model
    cmd = [
        sys.executable,
        "-m",
        "sgocr.dev200_harness",
        "build-dev40-semantic",
        "--source-experiment-dir",
        str(SOURCE_EXPERIMENT),
        "--out-dir",
        str(out_dir),
        "--intermediate-dir",
        str(intermediate_dir),
        "--cache-intermediate-dir",
        str(cache_intermediate_dir),
        "--cache-level",
        spec.cache_level,
        "--model",
        model_name,
        "--workers",
        str(args.workers),
        "--max-side",
        str(args.max_side),
        "--target-per-image",
        spec.cli.get("--target-per-image", str(args.target_per_image)),
        "--max-detections",
        spec.cli.get("--max-detections", "72"),
        "--grounding-threshold",
        spec.cli.get("--grounding-threshold", "0.36"),
        "--max-tags-per-image",
        spec.cli.get("--max-tags-per-image", "8"),
    ]
    env = os.environ.copy()
    env.update(spec.env)
    append_timeline(timeline_path, f"START {spec.name} cache={spec.cache_level} model={model_name} cli={spec.cli} env_keys={sorted(spec.env.keys())}")
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("CMD: " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        if spec.env:
            handle.write("ENV_OVERRIDES: " + json.dumps(spec.env, sort_keys=True) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            append_timeline(timeline_path, f"check stuff! {spec.name} pid={proc.pid} gpu=[{gpu_snapshot()}] log={log_path.name}")
            write_text(report_path, render_report(bundle_id, specs, results, docs_path))
            write_text(docs_path, render_report(bundle_id, specs, results, docs_path))
            time.sleep(max(5, int(args.watch_seconds)))
    if proc.returncode != 0:
        results[spec.name] = {"status": f"failed:{proc.returncode}", "experiment_name": experiment_name}
        append_timeline(timeline_path, f"END {spec.name} failed code={proc.returncode}")
    else:
        metrics = collect_metrics(out_dir)
        results[spec.name] = {"status": "ok", "metrics": metrics, "experiment_name": experiment_name}
        append_timeline(
            timeline_path,
            f"END {spec.name} ok accepted={metrics['accepted_qas']} images={metrics['images_with_final_rows']} quality={metrics['quality_score']:.2f}",
        )
    write_text(report_path, render_report(bundle_id, specs, results, docs_path))
    write_text(docs_path, render_report(bundle_id, specs, results, docs_path))


def main() -> None:
    args = parse_args()
    specs = build_specs()[: args.limit]
    bundle_dir = LOGS_ROOT / args.bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = bundle_dir / "timeline.log"
    report_path = bundle_dir / "report.md"
    docs_path = REPO_ROOT / "tasks" / "mm_bridge" / "docs" / "117_sgocr_dev40_merge_detection_sweep_2026-04-06.md"
    results: dict[str, dict[str, Any]] = {}
    append_timeline(
        timeline_path,
        f"BUNDLE {args.bundle_id} start specs={len(specs)} source={SOURCE_EXPERIMENT.name}",
    )
    write_text(report_path, render_report(args.bundle_id, specs, results, docs_path))
    write_text(docs_path, render_report(args.bundle_id, specs, results, docs_path))
    for spec in specs:
        run_ablation(
            bundle_dir=bundle_dir,
            bundle_id=args.bundle_id,
            spec=spec,
            args=args,
            docs_path=docs_path,
            report_path=report_path,
            results=results,
            specs=specs,
        )
    append_timeline(timeline_path, f"BUNDLE {args.bundle_id} complete total_specs={len(specs)} completed={len([r for r in results.values() if r.get('status') in ('ok', 'cached')])}")


if __name__ == "__main__":
    main()
