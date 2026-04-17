"""Dead-simple 3-image smoke test for GemmaOllamaAnchorGrounder.

Runs Gemma4 anchor inventory on one image from each dataset type
(chartqa, textocr, coco_text) and prints the detected anchors + a
crude "did anything come back?" pass/fail per image.
"""
from __future__ import annotations

from pathlib import Path

from ..ollama_anchor import GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA15, GemmaOllamaAnchorGrounder
from ..paths import REPO_ROOT
from ..qwen_anchor_vllm import QwenAnchorRequest

SMOKE_IMAGES = [
    {
        "image_id": "chartqa:shared:06989fe3e21c9fed39f987c95534bd38d2e0e0d25cc9e8911bc04538ae5179e6",
        "image_path": "data/vm_ssl/raw/chartqa/images/06/06989fe3e21c9fed39f987c95534bd38d2e0e0d25cc9e8911bc04538ae5179e6.png",
        "dataset_source": "chartqa_train",
    },
    {
        "image_id": "textocr:train:000811dda0037f67",
        "image_path": "data/vm_ssl/raw/textocr_trainval/000811dda0037f67.jpg",
        "dataset_source": "textocr_train",
    },
    {
        "image_id": "coco_text:train:108301",
        "image_path": "data/vm_ssl/raw/coco_text_materialized/train/COCO_train2014_000000108301.jpg",
        "dataset_source": "coco_text_train",
    },
]


def main() -> None:
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
        for img in SMOKE_IMAGES
    ]

    print("=== Gemma4 Ollama anchor smoke test ===\n", flush=True)
    results = grounder.detect_many(requests)

    for img in SMOKE_IMAGES:
        image_id = img["image_id"]
        rows = results.get(image_id, [])
        status = "PASS" if rows else "FAIL (no anchors)"
        print(f"[{status}] {image_id}  ({img['dataset_source']})")
        for r in rows:
            box = [round(v, 1) for v in r["box"]]
            print(f"  label={r['label']!r:45s}  box={box}")
        print()


if __name__ == "__main__":
    main()
