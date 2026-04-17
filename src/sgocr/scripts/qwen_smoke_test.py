"""Same 3-image smoke test as gemma_smoke_test.py but using QwenAnchorGrounderVLLM
with the updated ita15 prompts (no_text_ref + size-preference + chart-fluff suppression).
"""
from __future__ import annotations

from ..paths import REPO_ROOT
from ..qwen_anchor_vllm import (
    INDEPENDENT_QWEN_INVENTORY_PROMPT_ITA15,
    QwenAnchorGrounderVLLM,
    QwenAnchorRequest,
)

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

MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"


def main() -> None:
    grounder = QwenAnchorGrounderVLLM(
        model_name=MODEL,
        gpu_memory_utilization=0.85,
        batch_size=4,
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
        for img in SMOKE_IMAGES
    ]

    print("=== Qwen3-VL-8B ita15 prompt smoke test ===\n", flush=True)
    results = grounder.detect_many(requests)
    grounder.close()

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
