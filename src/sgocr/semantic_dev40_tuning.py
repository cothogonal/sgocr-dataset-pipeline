from __future__ import annotations

import os
from dataclasses import asdict, dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return str(raw).strip()


@dataclass(frozen=True)
class SemanticDev40Tuning:
    detector_mode: str = "ppocr"
    detector_box_thresh: float = 0.50
    detector_unclip_ratio: float = 1.50
    detector_merge_overlap: float = 0.62
    craft_text_threshold: float = 0.40
    craft_link_threshold: float = 0.35
    craft_low_text: float = 0.35
    craft_long_size: int = 1280
    ocr_crop_pad_ratio: float = 0.04
    semantic_crop_pad_ratio: float = 0.04
    resolvability_min_width_patches: float = 1.75
    resolvability_min_height_patches: float = 0.70
    text_merge_enabled: bool = False
    text_merge_anchor_overlap_min: float = 0.10
    text_merge_x_overlap_min: float = 0.45
    text_merge_y_overlap_min: float = 0.35
    text_merge_gap_ratio_max: float = 1.80
    text_merge_height_ratio_max: float = 1.90
    text_merge_subsume_min_nodes: int = 3
    strong_consensus_floor: float = 0.80
    standard_consensus_floor: float = 0.85
    low_confidence_floor: float = 0.60
    generic_anchor_penalty: float = 0.075
    oversized_anchor_area_start: float = 0.42
    florence_region_bonus: float = 0.045
    anchor_support_bonus: float = 0.040
    anchor_prompt_expansion_mode: str = "none"
    anchor_relabel_mode: str = "none"
    grounded_exclusion_min_strength: float = 2.25
    direct_read_specific_threshold: int = 3
    yesno_negative_specific_threshold: int = 2
    yesno_positive_specific_threshold: int = 3
    property_specific_threshold: int = 4
    max_negative_yesno_per_image: int = 1
    reverse_ground_directional_local_bias: float = 0.20
    reverse_ground_on_local_bias: float = 0.08
    reverse_ground_directional_mixed_bias: float = 0.10
    reverse_ground_on_mixed_bias: float = 0.06
    ambiguity_hard_reject_score: int = 5
    ambiguity_reverse_reject_score: int = 4
    anchor_conflict_overlap: float = 0.78
    sam3_refine_mode: str = "none"
    sam3_apply_mode: str = "all"
    sam3_topk_prompts: int = 2
    sam3_confidence_threshold: float = 0.45
    sam3_box_threshold: float = 0.35
    sam3_relevance_bonus: float = 0.05
    sam3_target_support_max: int = 1
    sam3_target_cluster_min: int = 2
    sam3_target_area_start: float = 0.42
    location_wording_mode: str = "balanced"
    avoid_redundant_location_phrases: bool = True
    teacher_strictness: str = "strict"

    def to_metadata(self) -> dict[str, float | int | str]:
        return asdict(self)


def load_semantic_dev40_tuning() -> SemanticDev40Tuning:
    tuning = SemanticDev40Tuning(
        detector_mode=_env_str("SGOCR_DETECTOR_MODE", "ppocr"),
        detector_box_thresh=_env_float("SGOCR_DETECTOR_BOX_THRESH", 0.50),
        detector_unclip_ratio=_env_float("SGOCR_DETECTOR_UNCLIP_RATIO", 1.50),
        detector_merge_overlap=_env_float("SGOCR_DETECTOR_MERGE_OVERLAP", 0.62),
        craft_text_threshold=_env_float("SGOCR_CRAFT_TEXT_THRESHOLD", 0.40),
        craft_link_threshold=_env_float("SGOCR_CRAFT_LINK_THRESHOLD", 0.35),
        craft_low_text=_env_float("SGOCR_CRAFT_LOW_TEXT", 0.35),
        craft_long_size=_env_int("SGOCR_CRAFT_LONG_SIZE", 1280),
        ocr_crop_pad_ratio=_env_float("SGOCR_OCR_CROP_PAD_RATIO", 0.04),
        semantic_crop_pad_ratio=_env_float("SGOCR_SEMANTIC_CROP_PAD_RATIO", 0.04),
        resolvability_min_width_patches=_env_float("SGOCR_RESOLVABILITY_MIN_WIDTH_PATCHES", 1.75),
        resolvability_min_height_patches=_env_float("SGOCR_RESOLVABILITY_MIN_HEIGHT_PATCHES", 0.70),
        text_merge_enabled=_env_int("SGOCR_TEXT_MERGE_ENABLED", 0) != 0,
        text_merge_anchor_overlap_min=_env_float("SGOCR_TEXT_MERGE_ANCHOR_OVERLAP_MIN", 0.10),
        text_merge_x_overlap_min=_env_float("SGOCR_TEXT_MERGE_X_OVERLAP_MIN", 0.45),
        text_merge_y_overlap_min=_env_float("SGOCR_TEXT_MERGE_Y_OVERLAP_MIN", 0.35),
        text_merge_gap_ratio_max=_env_float("SGOCR_TEXT_MERGE_GAP_RATIO_MAX", 1.80),
        text_merge_height_ratio_max=_env_float("SGOCR_TEXT_MERGE_HEIGHT_RATIO_MAX", 1.90),
        text_merge_subsume_min_nodes=_env_int("SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES", 3),
        strong_consensus_floor=_env_float("SGOCR_STRONG_CONSENSUS_FLOOR", 0.80),
        standard_consensus_floor=_env_float("SGOCR_STANDARD_CONSENSUS_FLOOR", 0.85),
        low_confidence_floor=_env_float("SGOCR_LOW_CONFIDENCE_FLOOR", 0.60),
        generic_anchor_penalty=_env_float("SGOCR_GENERIC_ANCHOR_PENALTY", 0.075),
        oversized_anchor_area_start=_env_float("SGOCR_OVERSIZED_ANCHOR_AREA_START", 0.42),
        florence_region_bonus=_env_float("SGOCR_FLORENCE_REGION_BONUS", 0.045),
        anchor_support_bonus=_env_float("SGOCR_ANCHOR_SUPPORT_BONUS", 0.040),
        anchor_prompt_expansion_mode=_env_str("SGOCR_ANCHOR_PROMPT_EXPANSION_MODE", "none"),
        anchor_relabel_mode=_env_str("SGOCR_ANCHOR_RELABEL_MODE", "none"),
        grounded_exclusion_min_strength=_env_float("SGOCR_GROUNDED_EXCLUSION_MIN_STRENGTH", 2.25),
        direct_read_specific_threshold=_env_int("SGOCR_DIRECT_READ_SPECIFIC_THRESHOLD", 3),
        yesno_negative_specific_threshold=_env_int("SGOCR_YESNO_NEGATIVE_SPECIFIC_THRESHOLD", 2),
        yesno_positive_specific_threshold=_env_int("SGOCR_YESNO_POSITIVE_SPECIFIC_THRESHOLD", 3),
        property_specific_threshold=_env_int("SGOCR_PROPERTY_SPECIFIC_THRESHOLD", 4),
        max_negative_yesno_per_image=_env_int("SGOCR_MAX_NEGATIVE_YESNO_PER_IMAGE", 1),
        reverse_ground_directional_local_bias=_env_float("SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS", 0.20),
        reverse_ground_on_local_bias=_env_float("SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS", 0.08),
        reverse_ground_directional_mixed_bias=_env_float("SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS", 0.10),
        reverse_ground_on_mixed_bias=_env_float("SGOCR_REVERSE_GROUND_ON_MIXED_BIAS", 0.06),
        ambiguity_hard_reject_score=_env_int("SGOCR_AMBIGUITY_HARD_REJECT_SCORE", 5),
        ambiguity_reverse_reject_score=_env_int("SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE", 4),
        anchor_conflict_overlap=_env_float("SGOCR_ANCHOR_CONFLICT_OVERLAP", 0.78),
        sam3_refine_mode=_env_str("SGOCR_SAM3_REFINE_MODE", "none"),
        sam3_apply_mode=_env_str("SGOCR_SAM3_APPLY_MODE", "all"),
        sam3_topk_prompts=_env_int("SGOCR_SAM3_TOPK_PROMPTS", 2),
        sam3_confidence_threshold=_env_float("SGOCR_SAM3_CONFIDENCE_THRESHOLD", 0.45),
        sam3_box_threshold=_env_float("SGOCR_SAM3_BOX_THRESHOLD", 0.35),
        sam3_relevance_bonus=_env_float("SGOCR_SAM3_RELEVANCE_BONUS", 0.05),
        sam3_target_support_max=_env_int("SGOCR_SAM3_TARGET_SUPPORT_MAX", 1),
        sam3_target_cluster_min=_env_int("SGOCR_SAM3_TARGET_CLUSTER_MIN", 2),
        sam3_target_area_start=_env_float("SGOCR_SAM3_TARGET_AREA_START", 0.42),
        location_wording_mode=_env_str("SGOCR_LOCATION_WORDING_MODE", "balanced"),
        avoid_redundant_location_phrases=_env_int("SGOCR_AVOID_REDUNDANT_LOCATION_PHRASES", 1) != 0,
        teacher_strictness=_env_str("SGOCR_TEACHER_STRICTNESS", "strict"),
    )
    if tuning.detector_mode not in {"ppocr", "ppocr_craft_ensemble"}:
        raise ValueError(f"Unsupported SGOCR_DETECTOR_MODE: {tuning.detector_mode}")
    if tuning.location_wording_mode not in {"lite", "balanced", "varied"}:
        raise ValueError(f"Unsupported SGOCR_LOCATION_WORDING_MODE: {tuning.location_wording_mode}")
    if tuning.teacher_strictness not in {"strict", "very_strict"}:
        raise ValueError(f"Unsupported SGOCR_TEACHER_STRICTNESS: {tuning.teacher_strictness}")
    if tuning.anchor_prompt_expansion_mode not in {"none", "supportive", "aggressive"}:
        raise ValueError(f"Unsupported SGOCR_ANCHOR_PROMPT_EXPANSION_MODE: {tuning.anchor_prompt_expansion_mode}")
    if tuning.anchor_relabel_mode not in {"none", "flash", "pro"}:
        raise ValueError(f"Unsupported SGOCR_ANCHOR_RELABEL_MODE: {tuning.anchor_relabel_mode}")
    if tuning.sam3_refine_mode not in {"none", "top1", "top2", "top3"}:
        raise ValueError(f"Unsupported SGOCR_SAM3_REFINE_MODE: {tuning.sam3_refine_mode}")
    if tuning.sam3_apply_mode not in {"all", "targeted"}:
        raise ValueError(f"Unsupported SGOCR_SAM3_APPLY_MODE: {tuning.sam3_apply_mode}")
    if tuning.detector_box_thresh <= 0 or tuning.detector_unclip_ratio <= 0 or tuning.detector_merge_overlap <= 0:
        raise ValueError("Detector thresholds must be > 0")
    if tuning.craft_text_threshold <= 0 or tuning.craft_link_threshold <= 0 or tuning.craft_low_text <= 0:
        raise ValueError("CRAFT thresholds must be > 0")
    if tuning.ocr_crop_pad_ratio < 0 or tuning.semantic_crop_pad_ratio < 0:
        raise ValueError("Crop pad ratios must be >= 0")
    if tuning.resolvability_min_width_patches <= 0 or tuning.resolvability_min_height_patches <= 0:
        raise ValueError("Resolvability thresholds must be > 0")
    if not 0.0 <= tuning.text_merge_anchor_overlap_min <= 1.0:
        raise ValueError("SGOCR_TEXT_MERGE_ANCHOR_OVERLAP_MIN must be in [0,1]")
    if not 0.0 <= tuning.text_merge_x_overlap_min <= 1.0:
        raise ValueError("SGOCR_TEXT_MERGE_X_OVERLAP_MIN must be in [0,1]")
    if not 0.0 <= tuning.text_merge_y_overlap_min <= 1.0:
        raise ValueError("SGOCR_TEXT_MERGE_Y_OVERLAP_MIN must be in [0,1]")
    if tuning.text_merge_gap_ratio_max <= 0.0:
        raise ValueError("SGOCR_TEXT_MERGE_GAP_RATIO_MAX must be > 0")
    if tuning.text_merge_height_ratio_max <= 0.0:
        raise ValueError("SGOCR_TEXT_MERGE_HEIGHT_RATIO_MAX must be > 0")
    if tuning.text_merge_subsume_min_nodes < 2:
        raise ValueError("SGOCR_TEXT_MERGE_SUBSUME_MIN_NODES must be >= 2")
    if tuning.strong_consensus_floor <= 0 or tuning.standard_consensus_floor <= 0 or tuning.low_confidence_floor <= 0:
        raise ValueError("Consensus thresholds must be > 0")
    if tuning.anchor_support_bonus < 0:
        raise ValueError("SGOCR_ANCHOR_SUPPORT_BONUS must be >= 0")
    if not 0.0 <= tuning.reverse_ground_directional_local_bias <= 1.0:
        raise ValueError("SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS must be in [0,1]")
    if not 0.0 <= tuning.reverse_ground_on_local_bias <= 1.0:
        raise ValueError("SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS must be in [0,1]")
    if not 0.0 <= tuning.reverse_ground_directional_mixed_bias <= 1.0:
        raise ValueError("SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS must be in [0,1]")
    if not 0.0 <= tuning.reverse_ground_on_mixed_bias <= 1.0:
        raise ValueError("SGOCR_REVERSE_GROUND_ON_MIXED_BIAS must be in [0,1]")
    if tuning.ambiguity_hard_reject_score < 0 or tuning.ambiguity_reverse_reject_score < 0:
        raise ValueError("Ambiguity reject scores must be >= 0")
    if not 0.0 <= tuning.anchor_conflict_overlap <= 1.0:
        raise ValueError("SGOCR_ANCHOR_CONFLICT_OVERLAP must be in [0,1]")
    if tuning.sam3_topk_prompts <= 0:
        raise ValueError("SGOCR_SAM3_TOPK_PROMPTS must be > 0")
    if not 0.0 <= tuning.sam3_confidence_threshold <= 1.0:
        raise ValueError("SGOCR_SAM3_CONFIDENCE_THRESHOLD must be in [0,1]")
    if not 0.0 <= tuning.sam3_box_threshold <= 1.0:
        raise ValueError("SGOCR_SAM3_BOX_THRESHOLD must be in [0,1]")
    if tuning.sam3_relevance_bonus < 0.0:
        raise ValueError("SGOCR_SAM3_RELEVANCE_BONUS must be >= 0")
    if tuning.sam3_target_support_max < 0:
        raise ValueError("SGOCR_SAM3_TARGET_SUPPORT_MAX must be >= 0")
    if tuning.sam3_target_cluster_min < 0:
        raise ValueError("SGOCR_SAM3_TARGET_CLUSTER_MIN must be >= 0")
    if tuning.sam3_target_area_start <= 0.0:
        raise ValueError("SGOCR_SAM3_TARGET_AREA_START must be > 0")
    if tuning.max_negative_yesno_per_image < 0:
        raise ValueError("SGOCR_MAX_NEGATIVE_YESNO_PER_IMAGE must be >= 0")
    return tuning
