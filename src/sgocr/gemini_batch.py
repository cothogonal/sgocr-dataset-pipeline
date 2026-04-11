from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any

from .secrets import GEMINI, get_secret


TERMINAL_BATCH_STATES = {
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
    "JOB_STATE_FAILED",
    "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_SUCCEEDED",
}


@dataclass(frozen=True)
class GeminiBatchRequest:
    key: str
    contents: list[Any]
    generation_config: dict[str, Any]
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class GeminiBatchResult:
    key: str
    raw_text: str
    usage: dict[str, Any]
    payload: dict[str, Any]
    error: str | None = None


def decode_inline_image_bytes(payload: Any) -> bytes:
    encoded = None
    if isinstance(payload, dict):
        encoded = payload.get("image_b64") or payload.get("data")
    else:
        encoded = getattr(payload, "image_b64", None) or getattr(payload, "data", None)
    if not encoded:
        raise ValueError("Image payload is missing inline base64 data")
    return base64.b64decode(str(encoded))


def image_part_from_payload(payload: Any) -> types.Part:
    _, types = _load_genai_sdk()
    mime_type = None
    if isinstance(payload, dict):
        mime_type = payload.get("mime_type")
    else:
        mime_type = getattr(payload, "mime_type", None)
    if not mime_type:
        raise ValueError("Image payload is missing mime_type")
    return types.Part.from_bytes(data=decode_inline_image_bytes(payload), mime_type=str(mime_type))


def batch_generate_json(
    *,
    model: str,
    requests: list[GeminiBatchRequest],
    display_name_prefix: str,
    chunk_size: int = 48,
    poll_interval_s: int = 15,
    timeout_s: int = 7200,
) -> dict[str, GeminiBatchResult]:
    genai, types = _load_genai_sdk()
    if not requests:
        return {}
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be > 0")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")

    client = genai.Client(api_key=get_secret(GEMINI))
    pending: list[dict[str, Any]] = []
    results: dict[str, GeminiBatchResult] = {}

    for chunk_index, start in enumerate(range(0, len(requests), chunk_size), start=1):
        chunk = requests[start : start + chunk_size]
        job = client.batches.create(
            model=model,
            src=[
                types.InlinedRequest(
                    contents=request.contents,
                    config=types.GenerateContentConfig(**request.generation_config),
                    metadata={"request_id": request.key, **(request.metadata or {})},
                )
                for request in chunk
            ],
            config=types.CreateBatchJobConfig(display_name=f"{display_name_prefix}-{chunk_index:03d}"),
        )
        pending.append({"job": job, "chunk": chunk})

    deadline = time.monotonic() + float(timeout_s)
    while pending:
        if time.monotonic() > deadline:
            unfinished = ", ".join(str(entry["job"].name) for entry in pending)
            raise TimeoutError(f"Timed out waiting for Gemini batch jobs: {unfinished}")

        next_pending: list[dict[str, Any]] = []
        for entry in pending:
            chunk = list(entry["chunk"])
            job = client.batches.get(name=str(entry["job"].name))
            state_name = _job_state_name(job.state)
            if state_name not in TERMINAL_BATCH_STATES:
                next_pending.append({"job": job, "chunk": chunk})
                continue
            _collect_job_results(job=job, chunk=chunk, model=model, results=results)
        pending = next_pending
        if pending:
            time.sleep(float(poll_interval_s))
    return results


def _collect_job_results(
    *,
    job: Any,
    chunk: list[GeminiBatchRequest],
    model: str,
    results: dict[str, GeminiBatchResult],
) -> None:
    _, types = _load_genai_sdk()
    state_name = _job_state_name(job.state)
    responses = list((job.dest or types.BatchJobDestination()).inlined_responses or [])
    by_key: dict[str, GeminiBatchResult] = {}

    for item in responses:
        metadata = dict(item.metadata or {})
        key = str(metadata.get("request_id") or "")
        if not key:
            continue
        if item.error is not None:
            by_key[key] = GeminiBatchResult(
                key=key,
                raw_text="",
                usage={},
                payload={},
                error=_job_error_text(item.error),
            )
            continue
        response = item.response
        payload = _response_payload_dict(response)
        raw_text = _response_text(response, payload)
        usage = payload.get("usageMetadata") or payload.get("usage_metadata") or {}
        by_key[key] = GeminiBatchResult(
            key=key,
            raw_text=raw_text,
            usage=dict(usage) if isinstance(usage, dict) else {},
            payload=payload,
            error=None,
        )

    job_error = _job_error_text(job.error)
    for request in chunk:
        result = by_key.get(request.key)
        if result is not None:
            results[request.key] = result
            continue
        fallback_error = job_error or f"Gemini batch finished in state {state_name} without a response"
        results[request.key] = GeminiBatchResult(
            key=request.key,
            raw_text="",
            usage={},
            payload={},
            error=fallback_error,
        )


def _job_state_name(state: Any) -> str:
    if state is None:
        return "JOB_STATE_UNSPECIFIED"
    value = getattr(state, "name", None)
    if value:
        return str(value)
    return str(state)


def _job_error_text(error: Any) -> str | None:
    if error is None:
        return None
    message = getattr(error, "message", None)
    code = getattr(error, "code", None)
    if message and code is not None:
        return f"{message} (code={code})"
    if message:
        return str(message)
    return str(error)


def _response_payload_dict(response: Any) -> dict[str, Any]:
    if response is None:
        return {}
    if hasattr(response, "model_dump"):
        dumped = response.model_dump(by_alias=True, exclude_none=True)
        if isinstance(dumped, dict):
            return dumped
    if isinstance(response, dict):
        return dict(response)
    return {}


def _response_text(response: Any, payload: dict[str, Any]) -> str:
    text = getattr(response, "text", None)
    if text:
        return str(text)
    for candidate in payload.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            part_text = part.get("text")
            if part_text:
                return str(part_text)
    raise RuntimeError("Gemini batch response did not contain text")


def _load_genai_sdk() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on env packaging
        raise ModuleNotFoundError(
            "Gemini batch mode requires the `google-genai` package. Install it in the active environment."
        ) from exc
    return genai, types
