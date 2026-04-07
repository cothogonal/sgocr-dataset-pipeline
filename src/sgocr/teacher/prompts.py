from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptVariant:
    name: str
    question_count: int
    allow_scene_context: bool
    strict_anchor: bool
    anti_hallucination: bool


PROMPT_VARIANTS: dict[str, PromptVariant] = {
    "strict_1q": PromptVariant(
        name="strict_1q",
        question_count=1,
        allow_scene_context=False,
        strict_anchor=True,
        anti_hallucination=True,
    ),
    "strict_2q": PromptVariant(
        name="strict_2q",
        question_count=2,
        allow_scene_context=False,
        strict_anchor=True,
        anti_hallucination=True,
    ),
    "natural_2q": PromptVariant(
        name="natural_2q",
        question_count=2,
        allow_scene_context=True,
        strict_anchor=False,
        anti_hallucination=True,
    ),
}


def response_schema(question_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": question_count,
                "maxItems": question_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "A natural question grounded to the specified text location."},
                        "answer": {"type": "string", "description": "Must exactly equal the provided answer string."},
                    },
                    "required": ["question", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def render_prompt(tuple_row: dict[str, Any], variant: PromptVariant) -> str:
    answer = str(tuple_row["answer"])
    anchor_label = str(tuple_row["anchor_label"])
    anchor_synonyms = tuple_row.get("anchor_synonyms") or [anchor_label]
    strict_anchor = (
        f'Use the exact phrase "{anchor_label}" in every question.'
        if variant.strict_anchor
        else f"Every question must clearly refer to this location. Allowed anchor phrasings include: {', '.join(f'\"{x}\"' for x in anchor_synonyms)}."
    )
    scene_context = (
        "You may add a small amount of visual context if it is plainly visible, but only if it helps the question feel natural."
        if variant.allow_scene_context
        else "Do not add extra scene details. Keep the question tightly tied to the verified location only."
    )
    anti_hallucination = (
        "Do not invent colors, materials, objects, or scene facts that are not clearly visible."
        if variant.anti_hallucination
        else ""
    )
    return (
        "You are generating OCR spatial QA training data.\n\n"
        "A verified fact about this image:\n"
        f'- Text reading: "{answer}"\n'
        f'- The text is located in the {anchor_label}\n'
        "- This location is already verified.\n\n"
        f"Write exactly {variant.question_count} natural questions.\n"
        f"{strict_anchor}\n"
        f"{scene_context}\n"
        f"{anti_hallucination}\n"
        f'The answer to every question must be exactly "{answer}".\n'
        "Keep each question between 5 and 35 words.\n"
        "Return JSON only."
    )
