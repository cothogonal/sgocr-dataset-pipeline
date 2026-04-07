from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .bootstrap import normalize_answer


def _norm_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def contains_anchor_phrase(question: str, anchor_synonyms: list[str] | tuple[str, ...]) -> bool:
    q = _norm_text(question)
    for phrase in anchor_synonyms:
        if _norm_text(str(phrase)) in q:
            return True
    return False


def looks_distinct(questions: list[str], threshold: float = 0.70) -> bool:
    if len(questions) < 2:
        return True
    token_sets = [set(_norm_text(q).split()) for q in questions]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            denom = max(len(a | b), 1)
            jac = len(a & b) / denom
            if jac > threshold:
                return False
    return True


@dataclass(frozen=True)
class QAValidation:
    answer_ok: bool
    anchor_ok: bool
    length_ok: bool
    duplicate_ok: bool
    accepted: bool


def validate_generated_items(tuple_row: dict[str, Any], items: list[dict[str, Any]]) -> tuple[list[QAValidation], dict[str, Any]]:
    answer_norm = normalize_answer(str(tuple_row.get("answer") or ""))
    anchor_synonyms = tuple_row.get("anchor_synonyms") or [tuple_row.get("anchor_label") or ""]
    questions = [str(item.get("question") or "") for item in items]
    dup_ok = looks_distinct(questions)
    validations: list[QAValidation] = []
    accepted = 0
    for item in items:
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "")
        answer_ok = normalize_answer(answer) == answer_norm
        anchor_ok = contains_anchor_phrase(question, anchor_synonyms)
        length_ok = 5 <= len(question.split()) <= 35
        row = QAValidation(
            answer_ok=answer_ok,
            anchor_ok=anchor_ok,
            length_ok=length_ok,
            duplicate_ok=dup_ok,
            accepted=bool(answer_ok and anchor_ok and length_ok and dup_ok),
        )
        validations.append(row)
        accepted += int(row.accepted)
    summary = {
        "generated_count": len(items),
        "accepted_count": accepted,
        "all_distinct": dup_ok,
        "all_answers_ok": all(v.answer_ok for v in validations) if validations else False,
        "all_anchor_ok": all(v.anchor_ok for v in validations) if validations else False,
    }
    return validations, summary
