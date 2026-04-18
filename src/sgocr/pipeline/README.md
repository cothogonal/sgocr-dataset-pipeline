# pipeline/ — Dataset Generation Engine

Split from `dev40_complete.py` and `full_pipeline_dev40.py` for LLM readability.
Each file is under 1,600 lines and handles one semantic concern.

| File | Lines | Contents |
|---|---|---|
| `anchor_analysis.py` | 1324 | Anchor extraction, sanitization, spatial relevance scoring, location metadata. Pure logic — no ML deps. |
| `question_builder.py` | 1591 | Question/candidate generation: `build_question_candidates`, `select_candidates`, `enforce_type_constraints`, quality scoring, ambiguity analysis |
| `teacher_runtime.py` | 1366 | Gemini teacher API: `run_teacher_batches`, inline frontier scoring, answer probes, prompt building, response parsing |
| `dataset_assembly.py` | 735 | Final sample assembly: `normalize_candidate_result`, `validate_candidate_output`, `row_to_final_sample`, `build_tags`, tag/bin helpers, geometry utilities |
| `tuple_builder.py` | 1031 | Tuple building/merging: `build_merged_sign_tuples_for_image`, `build_verified_tuples`, anchor clustering, component ordering, deduplication |
| `stages.py` | 1209 | Runtime pipeline stages: `run_detection_stage`, `run_recognition_stage`, `run_anchor_stage`, `refine_anchor_labels`, `build_consensus_nodes` |
