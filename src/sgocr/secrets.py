from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SecretSpec:
    provider: str
    env_var: str


GEMINI = SecretSpec(provider="gemini", env_var="GEMINI_API_KEY")
OPENAI = SecretSpec(provider="openai", env_var="OPENAI_API_KEY")
ANTHROPIC = SecretSpec(provider="anthropic", env_var="ANTHROPIC_API_KEY")


class MissingSecretError(RuntimeError):
    """Raised when a required API key env var is missing."""


def get_secret(spec: SecretSpec) -> str:
    value = os.environ.get(spec.env_var)
    if value:
        return value
    raise MissingSecretError(
        f"Missing required API key environment variable: {spec.env_var}. "
        "Set it in your shell environment before running this stage."
    )


def redact_secret(value: str | None) -> str:
    if not value:
        return "<missing>"
    return "***REDACTED***"


def missing_secret_env_vars(specs: Iterable[SecretSpec]) -> list[str]:
    missing: list[str] = []
    for spec in specs:
        if not os.environ.get(spec.env_var):
            missing.append(spec.env_var)
    return missing
