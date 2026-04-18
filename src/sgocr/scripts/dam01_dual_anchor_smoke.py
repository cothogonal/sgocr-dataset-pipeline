from __future__ import annotations

import argparse

from sgocr.dual_anchor import drain_ollama_model
from sgocr.ollama_anchor import (
    GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA16_ANTIDOC,
    GemmaOllamaAnchorGrounder,
)
from sgocr.qwen_anchor_vllm import (
    INDEPENDENT_QWEN_INVENTORY_PROMPT_DAM01,
    QwenAnchorGrounderVLLM,
    QwenAnchorRequest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test DAM dual-anchor handoff from Gemma to Qwen."
    )
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--gemma-model", default="gemma4:e4b-it-q4_K_M")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct-FP8")
    parser.add_argument("--min-free-mib", type=int, default=10000)
    parser.add_argument("--gpu-util", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_id = args.image_id
    image_path = args.image_path
    print(f"[smoke] image_id={image_id} image_path={image_path}", flush=True)

    gemma_request = QwenAnchorRequest(
        image_id=image_id,
        image_path=image_path,
        categories=[],
        prompt_text=GEMMA_INDEPENDENT_INVENTORY_PROMPT_ITA16_ANTIDOC,
        enforce_allowed_labels=False,
        normalize_open_labels=True,
    )
    gemma = GemmaOllamaAnchorGrounder(
        model=args.gemma_model,
        base_url="http://localhost:11434",
        num_ctx=4096,
        timeout_s=180,
    )
    gemma_result = gemma.detect_many([gemma_request])[image_id]
    print(f"[smoke] gemma_count={len(gemma_result)}", flush=True)
    print(
        f"[smoke] gemma_sample={[row.get('label') for row in gemma_result[:5]]}",
        flush=True,
    )

    print(f"[smoke] draining_ollama model={args.gemma_model}", flush=True)
    drain_ollama_model(
        model_name=args.gemma_model,
        min_free_mib=args.min_free_mib,
        timeout_s=45,
    )

    qwen_request = QwenAnchorRequest(
        image_id=image_id,
        image_path=image_path,
        categories=[],
        prompt_text=INDEPENDENT_QWEN_INVENTORY_PROMPT_DAM01,
        enforce_allowed_labels=False,
        normalize_open_labels=True,
    )
    qwen = QwenAnchorGrounderVLLM(
        model_name=args.qwen_model,
        gpu_memory_utilization=args.gpu_util,
        batch_size=1,
        min_pixels=64 * 32 * 32,
        max_pixels=9800 * 32 * 32,
        max_model_len=2048,
    )
    qwen_result = qwen.detect_many([qwen_request])[image_id]
    print(f"[smoke] qwen_count={len(qwen_result)}", flush=True)
    print(
        f"[smoke] qwen_sample={[row.get('label') for row in qwen_result[:5]]}",
        flush=True,
    )
    print(f"[smoke] second_model_ran={bool(qwen_result)}", flush=True)


if __name__ == "__main__":
    main()
