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
    ocr_frontend: str = "classic"
    anchor_tag_discovery_backend: str = "florence"
    qwen_anchor_tag_discovery_vocab_mode: str = "constrained"
    anchor_candidate_backend: str = "florence_dino"
    qwen_anchor_inventory_mode: str = "selected_tags"
    qwen_anchor_inventory_pass_count: int = 1
    qwen_anchor_inventory_temperature: float = 0.0
    qwen_anchor_inventory_consensus_iou: float = 0.55
    qwen_anchor_inventory_min_support: int = 1
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
    anchor_type_soft_cap_count: int = 2
    anchor_type_soft_cap_penalty: float = 0.15
    reverse_ground_directional_local_bias: float = 0.20
    reverse_ground_on_local_bias: float = 0.08
    reverse_ground_directional_mixed_bias: float = 0.10
    reverse_ground_on_mixed_bias: float = 0.06
    ambiguity_hard_reject_score: int = 5
    ambiguity_reverse_reject_score: int = 4
    dr_ambiguity_reject_score: int = -1  # -1 = disabled; set to e.g. 5 to reject high-ambiguity DIRECT_READ
    inline_frontier_enabled: bool = True
    inline_frontier_model: str = "gemini-3-flash-preview"
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
    reverse_ground_answer_style: str = "standard"
    anchor_relabel_generic_mode: str = "standard"
    generic_anchor_retry_penalty: float = 0.10
    sibling_disambiguation_enabled: bool = False
    anchor_reference_color_enabled: bool = False
    suppress_anchor_local_without_competing_text: bool = False
    repeated_anchor_grouping_enabled: bool = False
    repeated_anchor_group_min_instances: int = 3
    cheap_ambiguity_proxy_enabled: bool = False
    cheap_ambiguity_proxy_reject_score: int = 6
    answer_leakage_filter_enabled: bool = False
    spatial_min_centroid_offset: float = 0.0
    per_image_anchor_diversity_bonus: float = 0.0
    anchor_color_mechanical_answer_enabled: bool = False
    qwen_structural_fallback_enabled: bool = False
    qwen_degenerate_label_threshold: float = 0.92
    qwen_min_anchor_detections_per_image: int = 0
    qwen_anti_ocr_prompt_enabled: bool = False
    qwen_degenerate_anchor_filter_enabled: bool = False
    rg_structural_anchor_filter_enabled: bool = False
    upstream_centroid_filter_enabled: bool = False
    inline_frontier_gate_enabled: bool = False
    inline_frontier_gate_word_f1_floor: float = -1.0
    vision_dependence_gate_enabled: bool = False
    inline_frontier_gate_question_types: str = "all"
    rg_vdep_check_enabled: bool = False
    rg_vdep_model: str = "gemini:gemini-3-flash-preview"
    rg_leakage_correction_enabled: bool = False
    rg_leaky_label_hard_reject_enabled: bool = False
    rg_candidate_oversample_boost: float = 0.0
    rg_per_image_hard_cap: int = 1
    property_candidate_selection_bonus: float = 0.0
    anchor_label_groundback_enabled: bool = False
    anchor_label_groundback_iou_threshold: float = 0.35
    qwen_open_tag_prompt_mode: str = "basic"
    teacher_answer_probe_count: int = 1
    teacher_answer_probe_temperature: float = 0.35
    qwen_anchor_model: str = "Qwen/Qwen3-VL-8B-Instruct-FP8"
    qwen_anchor_gpu_memory_utilization: float = 0.90
    qwen_anchor_batch_size: int = 2
    qwen_anchor_min_pixels: int = 64 * 32 * 32
    qwen_anchor_max_pixels: int = 9800 * 32 * 32
    qwen_anchor_max_model_len: int = 2048
    gemini_api_mode: str = "sync"
    gemini_batch_chunk_size: int = 48
    gemini_batch_poll_seconds: int = 15
    gemini_batch_timeout_seconds: int = 7200

    def to_metadata(self) -> dict[str, float | int | str]:
        return asdict(self)


def load_semantic_dev40_tuning() -> SemanticDev40Tuning:
    tuning = SemanticDev40Tuning(
        ocr_frontend=_env_str("SGOCR_OCR_FRONTEND", "classic"),
        anchor_tag_discovery_backend=_env_str("SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND", "florence"),
        qwen_anchor_tag_discovery_vocab_mode=_env_str("SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE", "constrained"),
        anchor_candidate_backend=_env_str("SGOCR_ANCHOR_CANDIDATE_BACKEND", "florence_dino"),
        qwen_anchor_inventory_mode=_env_str("SGOCR_QWEN_ANCHOR_INVENTORY_MODE", "selected_tags"),
        qwen_anchor_inventory_pass_count=_env_int("SGOCR_QWEN_ANCHOR_INVENTORY_PASS_COUNT", 1),
        qwen_anchor_inventory_temperature=_env_float("SGOCR_QWEN_ANCHOR_INVENTORY_TEMPERATURE", 0.0),
        qwen_anchor_inventory_consensus_iou=_env_float("SGOCR_QWEN_ANCHOR_INVENTORY_CONSENSUS_IOU", 0.55),
        qwen_anchor_inventory_min_support=_env_int("SGOCR_QWEN_ANCHOR_INVENTORY_MIN_SUPPORT", 1),
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
        anchor_type_soft_cap_count=_env_int("SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT", 2),
        anchor_type_soft_cap_penalty=_env_float("SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY", 0.15),
        reverse_ground_directional_local_bias=_env_float("SGOCR_REVERSE_GROUND_DIRECTIONAL_LOCAL_BIAS", 0.20),
        reverse_ground_on_local_bias=_env_float("SGOCR_REVERSE_GROUND_ON_LOCAL_BIAS", 0.08),
        reverse_ground_directional_mixed_bias=_env_float("SGOCR_REVERSE_GROUND_DIRECTIONAL_MIXED_BIAS", 0.10),
        reverse_ground_on_mixed_bias=_env_float("SGOCR_REVERSE_GROUND_ON_MIXED_BIAS", 0.06),
        ambiguity_hard_reject_score=_env_int("SGOCR_AMBIGUITY_HARD_REJECT_SCORE", 5),
        ambiguity_reverse_reject_score=_env_int("SGOCR_AMBIGUITY_REVERSE_REJECT_SCORE", 4),
        dr_ambiguity_reject_score=_env_int("SGOCR_DR_AMBIGUITY_REJECT_SCORE", -1),
        inline_frontier_enabled=_env_int("SGOCR_INLINE_FRONTIER_ENABLED", 1) != 0,
        inline_frontier_model=_env_str("SGOCR_INLINE_FRONTIER_MODEL", "gemini-3-flash-preview"),
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
        reverse_ground_answer_style=_env_str("SGOCR_REVERSE_GROUND_ANSWER_STYLE", "standard"),
        anchor_relabel_generic_mode=_env_str("SGOCR_ANCHOR_RELABEL_GENERIC_MODE", "standard"),
        generic_anchor_retry_penalty=_env_float("SGOCR_GENERIC_ANCHOR_RETRY_PENALTY", 0.10),
        sibling_disambiguation_enabled=_env_int("SGOCR_SIBLING_DISAMBIGUATION_ENABLED", 0) != 0,
        anchor_reference_color_enabled=_env_int("SGOCR_ANCHOR_REFERENCE_COLOR_ENABLED", 0) != 0,
        suppress_anchor_local_without_competing_text=_env_int("SGOCR_SUPPRESS_ANCHOR_LOCAL_WITHOUT_COMPETING_TEXT", 0) != 0,
        repeated_anchor_grouping_enabled=_env_int("SGOCR_REPEATED_ANCHOR_GROUPING_ENABLED", 0) != 0,
        repeated_anchor_group_min_instances=_env_int("SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES", 3),
        cheap_ambiguity_proxy_enabled=_env_int("SGOCR_CHEAP_AMBIGUITY_PROXY_ENABLED", 0) != 0,
        cheap_ambiguity_proxy_reject_score=_env_int("SGOCR_CHEAP_AMBIGUITY_PROXY_REJECT_SCORE", 6),
        answer_leakage_filter_enabled=_env_int("SGOCR_ANSWER_LEAKAGE_FILTER_ENABLED", 0) != 0,
        spatial_min_centroid_offset=_env_float("SGOCR_SPATIAL_MIN_CENTROID_OFFSET", 0.0),
        per_image_anchor_diversity_bonus=_env_float("SGOCR_PER_IMAGE_ANCHOR_DIVERSITY_BONUS", 0.0),
        anchor_color_mechanical_answer_enabled=_env_int("SGOCR_ANCHOR_COLOR_MECHANICAL_ANSWER_ENABLED", 0) != 0,
        qwen_structural_fallback_enabled=_env_int("SGOCR_QWEN_STRUCTURAL_FALLBACK_ENABLED", 0) != 0,
        qwen_degenerate_label_threshold=_env_float("SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD", 0.92),
        qwen_min_anchor_detections_per_image=_env_int("SGOCR_QWEN_MIN_ANCHOR_DETECTIONS_PER_IMAGE", 0),
        qwen_anti_ocr_prompt_enabled=_env_int("SGOCR_QWEN_ANTI_OCR_PROMPT_ENABLED", 0) != 0,
        qwen_degenerate_anchor_filter_enabled=_env_int("SGOCR_QWEN_DEGENERATE_ANCHOR_FILTER_ENABLED", 0) != 0,
        rg_structural_anchor_filter_enabled=_env_int("SGOCR_RG_STRUCTURAL_ANCHOR_FILTER_ENABLED", 0) != 0,
        upstream_centroid_filter_enabled=_env_int("SGOCR_UPSTREAM_CENTROID_FILTER_ENABLED", 0) != 0,
        inline_frontier_gate_enabled=_env_int("SGOCR_INLINE_FRONTIER_GATE_ENABLED", 0) != 0,
        inline_frontier_gate_word_f1_floor=_env_float("SGOCR_INLINE_FRONTIER_GATE_WORD_F1_FLOOR", -1.0),
        vision_dependence_gate_enabled=_env_int("SGOCR_VISION_DEPENDENCE_GATE_ENABLED", 0) != 0,
        inline_frontier_gate_question_types=_env_str("SGOCR_INLINE_FRONTIER_GATE_QUESTION_TYPES", "all"),
        rg_vdep_check_enabled=_env_int("SGOCR_RG_VDEP_CHECK_ENABLED", 0) != 0,
        rg_vdep_model=_env_str("SGOCR_RG_VDEP_MODEL", "gemini:gemini-3-flash-preview"),
        rg_leakage_correction_enabled=_env_int("SGOCR_RG_LEAKAGE_CORRECTION_ENABLED", 0) != 0,
        rg_leaky_label_hard_reject_enabled=_env_int("SGOCR_RG_LEAKY_LABEL_HARD_REJECT_ENABLED", 0) != 0,
        rg_candidate_oversample_boost=_env_float("SGOCR_RG_CANDIDATE_OVERSAMPLE_BOOST", 0.0),
        rg_per_image_hard_cap=_env_int("SGOCR_RG_PER_IMAGE_HARD_CAP", 1),
        property_candidate_selection_bonus=_env_float("SGOCR_PROPERTY_CANDIDATE_SELECTION_BONUS", 0.0),
        anchor_label_groundback_enabled=_env_int("SGOCR_ANCHOR_LABEL_GROUNDBACK_ENABLED", 0) != 0,
        anchor_label_groundback_iou_threshold=_env_float("SGOCR_ANCHOR_LABEL_GROUNDBACK_IOU_THRESHOLD", 0.35),
        qwen_open_tag_prompt_mode=_env_str("SGOCR_QWEN_OPEN_TAG_PROMPT_MODE", "basic"),
        teacher_answer_probe_count=_env_int("SGOCR_TEACHER_ANSWER_PROBE_COUNT", 1),
        teacher_answer_probe_temperature=_env_float("SGOCR_TEACHER_ANSWER_PROBE_TEMPERATURE", 0.35),
        qwen_anchor_model=_env_str("SGOCR_QWEN_ANCHOR_MODEL", "Qwen/Qwen3-VL-8B-Instruct-FP8"),
        qwen_anchor_gpu_memory_utilization=_env_float("SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION", 0.90),
        qwen_anchor_batch_size=_env_int("SGOCR_QWEN_ANCHOR_BATCH_SIZE", 2),
        qwen_anchor_min_pixels=_env_int("SGOCR_QWEN_ANCHOR_MIN_PIXELS", 64 * 32 * 32),
        qwen_anchor_max_pixels=_env_int("SGOCR_QWEN_ANCHOR_MAX_PIXELS", 9800 * 32 * 32),
        qwen_anchor_max_model_len=_env_int("SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN", 2048),
        gemini_api_mode=_env_str("SGOCR_GEMINI_API_MODE", "sync"),
        gemini_batch_chunk_size=_env_int("SGOCR_GEMINI_BATCH_CHUNK_SIZE", 48),
        gemini_batch_poll_seconds=_env_int("SGOCR_GEMINI_BATCH_POLL_SECONDS", 15),
        gemini_batch_timeout_seconds=_env_int("SGOCR_GEMINI_BATCH_TIMEOUT_SECONDS", 7200),
    )
    if tuning.ocr_frontend not in {"classic", "nemotron_v2"}:
        raise ValueError(f"Unsupported SGOCR_OCR_FRONTEND: {tuning.ocr_frontend}")
    if tuning.anchor_tag_discovery_backend not in {"florence", "qwen3_vl_vllm"}:
        raise ValueError(f"Unsupported SGOCR_ANCHOR_TAG_DISCOVERY_BACKEND: {tuning.anchor_tag_discovery_backend}")
    if tuning.qwen_anchor_tag_discovery_vocab_mode not in {"constrained", "open"}:
        raise ValueError(
            f"Unsupported SGOCR_QWEN_ANCHOR_TAG_DISCOVERY_VOCAB_MODE: {tuning.qwen_anchor_tag_discovery_vocab_mode}"
        )
    if tuning.anchor_candidate_backend not in {"florence_dino", "qwen3_vl_vllm"}:
        raise ValueError(f"Unsupported SGOCR_ANCHOR_CANDIDATE_BACKEND: {tuning.anchor_candidate_backend}")
    if tuning.qwen_anchor_inventory_mode not in {"selected_tags", "global_inventory", "independent_raw"}:
        raise ValueError(f"Unsupported SGOCR_QWEN_ANCHOR_INVENTORY_MODE: {tuning.qwen_anchor_inventory_mode}")
    if tuning.qwen_anchor_inventory_pass_count < 1:
        raise ValueError("SGOCR_QWEN_ANCHOR_INVENTORY_PASS_COUNT must be >= 1")
    if not 0.0 <= tuning.qwen_anchor_inventory_temperature <= 1.5:
        raise ValueError("SGOCR_QWEN_ANCHOR_INVENTORY_TEMPERATURE must be in [0, 1.5]")
    if not 0.0 < tuning.qwen_anchor_inventory_consensus_iou <= 1.0:
        raise ValueError("SGOCR_QWEN_ANCHOR_INVENTORY_CONSENSUS_IOU must be in (0, 1]")
    if tuning.qwen_anchor_inventory_min_support < 1:
        raise ValueError("SGOCR_QWEN_ANCHOR_INVENTORY_MIN_SUPPORT must be >= 1")
    if tuning.detector_mode not in {"ppocr", "ppocr_craft_ensemble"}:
        raise ValueError(f"Unsupported SGOCR_DETECTOR_MODE: {tuning.detector_mode}")
    if tuning.location_wording_mode not in {"lite", "balanced", "varied", "rich_local", "finalv0"}:
        raise ValueError(f"Unsupported SGOCR_LOCATION_WORDING_MODE: {tuning.location_wording_mode}")
    if tuning.reverse_ground_answer_style not in {"standard", "frontier", "frontier_rich"}:
        raise ValueError(f"Unsupported SGOCR_REVERSE_GROUND_ANSWER_STYLE: {tuning.reverse_ground_answer_style}")
    if tuning.anchor_relabel_generic_mode not in {"standard", "anti_generic"}:
        raise ValueError(f"Unsupported SGOCR_ANCHOR_RELABEL_GENERIC_MODE: {tuning.anchor_relabel_generic_mode}")
    if tuning.qwen_open_tag_prompt_mode not in {"basic", "color_specific"}:
        raise ValueError(f"Unsupported SGOCR_QWEN_OPEN_TAG_PROMPT_MODE: {tuning.qwen_open_tag_prompt_mode}")
    if tuning.repeated_anchor_group_min_instances < 2:
        raise ValueError("SGOCR_REPEATED_ANCHOR_GROUP_MIN_INSTANCES must be >= 2")
    if tuning.cheap_ambiguity_proxy_reject_score < 1:
        raise ValueError("SGOCR_CHEAP_AMBIGUITY_PROXY_REJECT_SCORE must be >= 1")
    if not 0.0 <= tuning.spatial_min_centroid_offset <= 0.5:
        raise ValueError("SGOCR_SPATIAL_MIN_CENTROID_OFFSET must be in [0, 0.5]")
    if tuning.inline_frontier_gate_word_f1_floor != -1.0 and not (
        0.0 <= tuning.inline_frontier_gate_word_f1_floor <= 1.0
    ):
        raise ValueError("SGOCR_INLINE_FRONTIER_GATE_WORD_F1_FLOOR must be -1.0 (disabled) or in [0, 1]")
    if tuning.per_image_anchor_diversity_bonus < 0.0:
        raise ValueError("SGOCR_PER_IMAGE_ANCHOR_DIVERSITY_BONUS must be >= 0")
    if not 0.0 <= tuning.qwen_degenerate_label_threshold <= 1.0:
        raise ValueError("SGOCR_QWEN_DEGENERATE_LABEL_THRESHOLD must be in [0, 1]")
    if tuning.qwen_min_anchor_detections_per_image < 0:
        raise ValueError("SGOCR_QWEN_MIN_ANCHOR_DETECTIONS_PER_IMAGE must be >= 0")
    if tuning.teacher_strictness not in {"strict", "very_strict"}:
        raise ValueError(f"Unsupported SGOCR_TEACHER_STRICTNESS: {tuning.teacher_strictness}")
    if tuning.gemini_api_mode not in {"sync", "batch"}:
        raise ValueError(f"Unsupported SGOCR_GEMINI_API_MODE: {tuning.gemini_api_mode}")
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
    if tuning.teacher_answer_probe_count < 1:
        raise ValueError("SGOCR_TEACHER_ANSWER_PROBE_COUNT must be >= 1")
    if not 0.0 <= tuning.teacher_answer_probe_temperature <= 1.5:
        raise ValueError("SGOCR_TEACHER_ANSWER_PROBE_TEMPERATURE must be in [0,1.5]")
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
    if tuning.anchor_type_soft_cap_count < 0:
        raise ValueError("SGOCR_ANCHOR_TYPE_SOFT_CAP_COUNT must be >= 0")
    if tuning.anchor_type_soft_cap_penalty < 0.0:
        raise ValueError("SGOCR_ANCHOR_TYPE_SOFT_CAP_PENALTY must be >= 0")
    if tuning.generic_anchor_retry_penalty < 0.0:
        raise ValueError("SGOCR_GENERIC_ANCHOR_RETRY_PENALTY must be >= 0")
    if not 0.0 < tuning.qwen_anchor_gpu_memory_utilization < 1.0:
        raise ValueError("SGOCR_QWEN_ANCHOR_GPU_MEMORY_UTILIZATION must be in (0,1)")
    if tuning.qwen_anchor_batch_size <= 0:
        raise ValueError("SGOCR_QWEN_ANCHOR_BATCH_SIZE must be > 0")
    if tuning.qwen_anchor_min_pixels <= 0 or tuning.qwen_anchor_max_pixels <= 0:
        raise ValueError("SGOCR_QWEN_ANCHOR_MIN_PIXELS and MAX_PIXELS must be > 0")
    if tuning.qwen_anchor_min_pixels > tuning.qwen_anchor_max_pixels:
        raise ValueError("SGOCR_QWEN_ANCHOR_MIN_PIXELS must be <= MAX_PIXELS")
    if tuning.qwen_anchor_max_model_len <= 0:
        raise ValueError("SGOCR_QWEN_ANCHOR_MAX_MODEL_LEN must be > 0")
    if tuning.gemini_batch_chunk_size <= 0:
        raise ValueError("SGOCR_GEMINI_BATCH_CHUNK_SIZE must be > 0")
    if tuning.gemini_batch_poll_seconds <= 0:
        raise ValueError("SGOCR_GEMINI_BATCH_POLL_SECONDS must be > 0")
    if tuning.gemini_batch_timeout_seconds <= 0:
        raise ValueError("SGOCR_GEMINI_BATCH_TIMEOUT_SECONDS must be > 0")
    return tuning
