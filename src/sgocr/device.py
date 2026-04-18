from __future__ import annotations


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise SystemExit("CUDA requested but unavailable.")
        if requested == "mps":
            import torch

            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                raise SystemExit("MPS requested but unavailable.")
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
