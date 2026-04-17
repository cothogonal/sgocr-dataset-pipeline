"""Expanded 9-image anchor smoke test (3 per dataset) for both Gemma and Qwen backends.

Usage:
    # Gemma (Ollama must be running):
    PYTHONPATH=sgocr/src .venv_local/bin/python -m sgocr.scripts.anchor_smoke_expanded --backend gemma

    # Qwen (Ollama must be stopped, GPU free):
    PYTHONPATH=sgocr/src .venv_local/bin/python -m sgocr.scripts.anchor_smoke_expanded --backend qwen
"""
from __future__ import annotations

import argparse
from typing import Any

from ..paths import REPO_ROOT

SMOKE_IMAGES = [
    # --- chartqa ---
    {
        "image_id": "chartqa:shared:06989fe3e21c9fed39f987c95534bd38d2e0e0d25cc9e8911bc04538ae5179e6",
        "image_path": "data/vm_ssl/raw/chartqa/images/06/06989fe3e21c9fed39f987c95534bd38d2e0e0d25cc9e8911bc04538ae5179e6.png",
        "dataset_source": "chartqa",
        "note": "bar chart (original smoke image)",
    },
    {
        "image_id": "chartqa:shared:0f35d793f45924aa03d6984acecf701cd5d4468773e148ef453f8c33beecb656",
        "image_path": "data/vm_ssl/raw/chartqa/images/0f/0f35d793f45924aa03d6984acecf701cd5d4468773e148ef453f8c33beecb656.png",
        "dataset_source": "chartqa",
        "note": "chartqa sample 2",
    },
    {
        "image_id": "chartqa:shared:248d8d9210bfcc00d49d4e3571c5e528b64284f5983becb2fe5bbb9fe375dbc3",
        "image_path": "data/vm_ssl/raw/chartqa/images/24/248d8d9210bfcc00d49d4e3571c5e528b64284f5983becb2fe5bbb9fe375dbc3.png",
        "dataset_source": "chartqa",
        "note": "chartqa sample 3",
    },
    # --- textocr ---
    {
        "image_id": "textocr:train:000811dda0037f67",
        "image_path": "data/vm_ssl/raw/textocr_trainval/000811dda0037f67.jpg",
        "dataset_source": "textocr",
        "note": "street scene (original smoke image)",
    },
    {
        "image_id": "textocr:train:00fdf42b979aacac",
        "image_path": "data/vm_ssl/raw/textocr_trainval/00fdf42b979aacac.jpg",
        "dataset_source": "textocr",
        "note": "textocr sample 2",
    },
    {
        "image_id": "textocr:train:01e1df88e5b1c438",
        "image_path": "data/vm_ssl/raw/textocr_trainval/01e1df88e5b1c438.jpg",
        "dataset_source": "textocr",
        "note": "textocr sample 3",
    },
    # --- coco_text ---
    {
        "image_id": "coco_text:train:108301",
        "image_path": "data/vm_ssl/raw/coco_text_materialized/train/COCO_train2014_000000108301.jpg",
        "dataset_source": "coco_text",
        "note": "office desk (original smoke image)",
    },
    {
        "image_id": "coco_text:train:114500",
        "image_path": "data/vm_ssl/raw/coco_text_materialized/train/COCO_train2014_000000114500.jpg",
        "dataset_source": "coco_text",
        "note": "coco_text sample 2",
    },
    {
        "image_id": "coco_text:train:185848",
        "image_path": "data/vm_ssl/raw/coco_text_materialized/train/COCO_train2014_000000185848.jpg",
        "dataset_source": "coco_text",
        "note": "ASHFIELD sign (merge ordering target)",
    },
]


def _run_gemma(images: list[dict]) -> dict[str, list[dict[str, Any]]]:
    from ..ollama_anchor import GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15, GemmaOllamaAnchorGrounder
    from ..qwen_anchor_vllm import QwenAnchorRequest

    grounder = GemmaOllamaAnchorGrounder()
    requests = [
        QwenAnchorRequest(
            image_id=img["image_id"],
            image_path=str(REPO_ROOT / img["image_path"]),
            categories=[],
            prompt_text=GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15,
            enforce_allowed_labels=False,
            normalize_open_labels=True,
        )
        for img in images
    ]
    return grounder.detect_many(requests)


def _run_qwen(images: list[dict]) -> dict[str, list[dict[str, Any]]]:
    from ..qwen_anchor_vllm import (
        INDEPENDENT_QWEN_INVENTORY_PROMPT_ITA15,
        QwenAnchorGrounderVLLM,
        QwenAnchorRequest,
    )

    grounder = QwenAnchorGrounderVLLM(
        model_name="Qwen/Qwen3-VL-8B-Instruct-FP8",
        gpu_memory_utilization=0.85,
        batch_size=3,
        max_model_len=2048,
    )
    requests = [
        QwenAnchorRequest(
            image_id=img["image_id"],
            image_path=str(REPO_ROOT / img["image_path"]),
            categories=[],
            prompt_text=INDEPENDENT_QWEN_INVENTORY_PROMPT_ITA15,
            enforce_allowed_labels=False,
            normalize_open_labels=True,
        )
        for img in images
    ]
    results = grounder.detect_many(requests)
    grounder.close()
    return results


def _print_results(
    backend: str,
    images: list[dict],
    results: dict[str, list[dict[str, Any]]],
) -> None:
    from ..qwen_anchor_vllm import is_degenerate_anchor_label, is_text_ref_anchor_label

    print(f"\n{'='*70}")
    print(f"  {backend.upper()} anchor smoke results — updated ITA15 prompt")
    print(f"{'='*70}\n")

    for img in images:
        iid = img["image_id"]
        rows = results.get(iid, [])
        flagged_degenerate = [r for r in rows if is_degenerate_anchor_label(str(r.get("label") or ""))]
        flagged_text_ref = [r for r in rows if is_text_ref_anchor_label(str(r.get("label") or ""))]
        clean = [r for r in rows if r not in flagged_degenerate and r not in flagged_text_ref]

        status = "PASS" if rows else "FAIL"
        short_id = iid.split(":")[-1][:20]
        print(f"[{status}] {img['dataset_source']:10s} {short_id:20s}  ({img['note']})")
        print(f"         total={len(rows)}  clean={len(clean)}  degenerate={len(flagged_degenerate)}  text_ref={len(flagged_text_ref)}")
        for r in rows:
            label = str(r.get("label") or "")
            box = [round(v, 0) for v in r["box"]]
            flags = []
            if is_degenerate_anchor_label(label):
                flags.append("DEG")
            if is_text_ref_anchor_label(label):
                flags.append("TXT_REF")
            flag_str = f" [{','.join(flags)}]" if flags else ""
            print(f"    {label!r:50s}  {box}{flag_str}")
        print()

    # Dataset-level summary table
    datasets = sorted({img["dataset_source"] for img in images})
    print(f"{'─'*70}")
    print(f"{'Dataset':12s}  {'Images':>6}  {'Total':>6}  {'Clean':>6}  {'Degenerate':>10}  {'Text-ref':>8}")
    print(f"{'─'*70}")
    for ds in datasets:
        ds_images = [img for img in images if img["dataset_source"] == ds]
        total = sum(len(results.get(img["image_id"], [])) for img in ds_images)
        deg = sum(
            sum(1 for r in results.get(img["image_id"], []) if is_degenerate_anchor_label(str(r.get("label") or "")))
            for img in ds_images
        )
        txt = sum(
            sum(1 for r in results.get(img["image_id"], []) if is_text_ref_anchor_label(str(r.get("label") or "")))
            for img in ds_images
        )
        clean = total - deg - txt
        print(f"{ds:12s}  {len(ds_images):>6}  {total:>6}  {clean:>6}  {deg:>10}  {txt:>8}")
    print(f"{'─'*70}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gemma", "qwen"], default="gemma")
    args = ap.parse_args()

    print(f"Running {args.backend.upper()} smoke test on {len(SMOKE_IMAGES)} images...\n", flush=True)

    if args.backend == "gemma":
        results = _run_gemma(SMOKE_IMAGES)
    else:
        results = _run_qwen(SMOKE_IMAGES)

    _print_results(args.backend, SMOKE_IMAGES, results)


if __name__ == "__main__":
    main()
