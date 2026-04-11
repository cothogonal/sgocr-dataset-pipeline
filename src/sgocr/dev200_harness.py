from __future__ import annotations

import argparse
from pathlib import Path

from .bootstrap import build_and_write_dev_subset
from .bootstrap_kd import materialize_bootstrap_kd_dataset
from .dev200_eval import _load_jsonl, _write_frontier_eval_index, compute_frontier_agreement, prepare_bundle_evals, run_frontier_benchmark
from .dev40_complete import build_dev40_complete_dataset
from .full_pipeline_dev40 import build_dev40_semantic_dataset
from .paths import OCR_SPATIAL_QA_FINAL_ROOT, OCR_SPATIAL_QA_INTERMEDIATE_ROOT, OCR_SPATIAL_QA_RAW_ROOT
from .teacher.bakeoff import ExperimentSpec, run_experiment


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build dev200 bootstrap data and run teacher bakeoff experiments.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_build = sub.add_parser("build-dev200")
    ap_build.add_argument("--raw-json", default="data/vm_ssl/raw/textocr_full/TextOCR_0.1_val.json")
    ap_build.add_argument("--images-root", default="data/vm_ssl/raw/textocr_trainval")
    ap_build.add_argument("--limit", type=int, default=200)
    ap_build.add_argument("--seed", type=int, default=42)

    ap_run = sub.add_parser("run-experiment")
    ap_run.add_argument("--name", required=True)
    ap_run.add_argument("--provider", choices=["gemini", "openai"], required=True)
    ap_run.add_argument("--model", required=True)
    ap_run.add_argument("--prompt-variant", required=True)
    ap_run.add_argument("--limit", type=int, default=16)
    ap_run.add_argument("--max-side", type=int, default=768)
    ap_run.add_argument("--workers", type=int, default=1)
    ap_run.add_argument("--tuples-path", default=str(OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / "bootstrap_tuples_dev200_v1.jsonl"))
    ap_run.add_argument("--out-dir", default="")

    ap_kd = sub.add_parser("materialize-bootstrap-kd")
    ap_kd.add_argument("--raw-json", default="data/vm_ssl/raw/textocr_full/TextOCR_0.1_val.json")
    ap_kd.add_argument("--source-experiment-dir", required=True)
    ap_kd.add_argument("--out-dir", default="")
    ap_kd.add_argument("--intermediate-dir", default="")

    ap_complete = sub.add_parser("build-dev40-complete")
    ap_complete.add_argument("--raw-json", default="data/vm_ssl/raw/textocr_full/TextOCR_0.1_val.json")
    ap_complete.add_argument("--source-experiment-dir", required=True)
    ap_complete.add_argument("--out-dir", default="")
    ap_complete.add_argument("--intermediate-dir", default="")
    ap_complete.add_argument("--model", default="gemini-2.5-flash")
    ap_complete.add_argument("--max-side", type=int, default=768)
    ap_complete.add_argument("--workers", type=int, default=4)
    ap_complete.add_argument("--target-per-image", type=int, default=4)

    ap_semantic = sub.add_parser("build-dev40-semantic")
    ap_semantic.add_argument("--source-experiment-dir", required=True)
    ap_semantic.add_argument("--out-dir", default="")
    ap_semantic.add_argument("--intermediate-dir", default="")
    ap_semantic.add_argument("--cache-intermediate-dir", default="")
    ap_semantic.add_argument("--model", default="gemini-2.5-flash")
    ap_semantic.add_argument("--device", default="auto")
    ap_semantic.add_argument("--workers", type=int, default=4)
    ap_semantic.add_argument("--max-side", type=int, default=768)
    ap_semantic.add_argument("--target-per-image", type=int, default=4)
    ap_semantic.add_argument("--max-detections", type=int, default=72)
    ap_semantic.add_argument("--grounding-threshold", type=float, default=0.30)
    ap_semantic.add_argument("--max-tags-per-image", type=int, default=8)
    ap_semantic.add_argument("--cache-level", choices=["none", "ocr", "verified"], default="verified")

    ap_prepare = sub.add_parser("prepare-bundle-evals")
    ap_prepare.add_argument("--bundle-id", default="sgocr_dev200_20260406_200346")

    ap_bench = sub.add_parser("run-frontier-benchmark")
    ap_bench.add_argument("--experiment-dir", required=True)
    ap_bench.add_argument("--model", action="append", dest="models", default=[])
    ap_bench.add_argument("--limit", type=int, default=0)
    ap_bench.add_argument("--question-type", action="append", dest="question_types", default=[])
    ap_bench.add_argument("--out-dir", default="")
    ap_bench.add_argument("--workers", type=int, default=1)

    ap_agree = sub.add_parser("compute-frontier-agreement")
    ap_agree.add_argument("--benchmark-dir", required=True)

    ap_backfill = sub.add_parser("backfill-frontier-evals-index", help="Rebuild frontier_evals_index.jsonl from existing evals/ dirs.")
    ap_backfill.add_argument("--experiment-dir", required=True, help="Path to a final experiment directory.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "build-dev200":
        raw_root = OCR_SPATIAL_QA_RAW_ROOT / "dev200"
        interm_root = OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200"
        build_and_write_dev_subset(
            raw_json_path=Path(args.raw_json),
            images_root=Path(args.images_root),
            manifest_path=raw_root / "dev200_manifest_v1.json",
            tuples_path=interm_root / "bootstrap_tuples_dev200_v1.jsonl",
            notes_path=raw_root / "dev200_notes.md",
            limit=int(args.limit),
            seed=int(args.seed),
        )
        return

    if args.cmd == "materialize-bootstrap-kd":
        out_dir = Path(args.out_dir) if args.out_dir else (OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / f"{Path(args.source_experiment_dir).name}_resolvable_kd_v1")
        intermediate_dir = (
            Path(args.intermediate_dir)
            if args.intermediate_dir
            else (OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / f"{Path(args.source_experiment_dir).name}_resolvable_kd_v1")
        )
        materialize_bootstrap_kd_dataset(
            source_experiment_dir=Path(args.source_experiment_dir),
            raw_json_path=Path(args.raw_json),
            out_dir=out_dir,
            intermediate_dir=intermediate_dir,
        )
        return

    if args.cmd == "build-dev40-complete":
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else (OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / f"{Path(args.source_experiment_dir).name}_complete_pipeline_dev40_v1")
        )
        intermediate_dir = (
            Path(args.intermediate_dir)
            if args.intermediate_dir
            else (OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / f"{Path(args.source_experiment_dir).name}_complete_pipeline_dev40_v1")
        )
        build_dev40_complete_dataset(
            source_experiment_dir=Path(args.source_experiment_dir),
            raw_json_path=Path(args.raw_json),
            out_dir=out_dir,
            intermediate_dir=intermediate_dir,
            model=str(args.model),
            max_side=int(args.max_side),
            workers=int(args.workers),
            target_per_image=int(args.target_per_image),
        )
        return

    if args.cmd == "build-dev40-semantic":
        out_dir = (
            Path(args.out_dir)
            if args.out_dir
            else (OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / f"{Path(args.source_experiment_dir).name}_semantic_dev40_v1")
        )
        intermediate_dir = (
            Path(args.intermediate_dir)
            if args.intermediate_dir
            else (OCR_SPATIAL_QA_INTERMEDIATE_ROOT / "dev200" / f"{Path(args.source_experiment_dir).name}_semantic_dev40_v1")
        )
        build_dev40_semantic_dataset(
            source_experiment_dir=Path(args.source_experiment_dir),
            out_dir=out_dir,
            intermediate_dir=intermediate_dir,
            cache_intermediate_dir=Path(args.cache_intermediate_dir) if args.cache_intermediate_dir else None,
            model=str(args.model),
            device=str(args.device),
            workers=int(args.workers),
            max_side=int(args.max_side),
            target_per_image=int(args.target_per_image),
            max_detections=int(args.max_detections),
            grounding_threshold=float(args.grounding_threshold),
            max_tags_per_image=int(args.max_tags_per_image),
            cache_level=str(args.cache_level),
        )
        return

    if args.cmd == "prepare-bundle-evals":
        prepare_bundle_evals(str(args.bundle_id))
        return

    if args.cmd == "run-frontier-benchmark":
        run_frontier_benchmark(
            experiment_dir=Path(args.experiment_dir),
            model_specs=list(args.models),
            limit=int(args.limit),
            only_question_types=list(args.question_types),
            out_dir=Path(args.out_dir) if args.out_dir else None,
            workers=int(args.workers),
        )
        return

    if args.cmd == "compute-frontier-agreement":
        compute_frontier_agreement(benchmark_dir=Path(args.benchmark_dir))
        return

    if args.cmd == "backfill-frontier-evals-index":
        experiment_dir = Path(args.experiment_dir)
        all_preds: list[dict] = []
        for pred_file in sorted((experiment_dir / "evals").glob("*/predictions.jsonl")):
            all_preds.extend(_load_jsonl(pred_file))
        _write_frontier_eval_index(experiment_dir, all_preds)
        print(f"Wrote frontier_evals_index.jsonl to {experiment_dir} ({len(all_preds)} prediction rows from {experiment_dir / 'evals'})")
        return

    out_dir = Path(args.out_dir) if args.out_dir else (OCR_SPATIAL_QA_FINAL_ROOT / "dev200" / args.name)
    run_experiment(
        ExperimentSpec(
            name=args.name,
            provider=args.provider,
            model=args.model,
            prompt_variant=args.prompt_variant,
            limit=int(args.limit),
            max_side=int(args.max_side),
            workers=int(args.workers),
        ),
        tuples_path=Path(args.tuples_path),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
