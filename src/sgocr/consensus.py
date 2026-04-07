from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from .bootstrap import is_valid_text, normalize_answer
from .semantic_dev40_tuning import load_semantic_dev40_tuning


EDIT_RESCUE_MIN_LEN = 8
LOW_CONFIDENCE_FLOOR = 0.60
STRONG_CONSENSUS_FLOOR = 0.80
STANDARD_CONSENSUS_FLOOR = 0.85


@dataclass(frozen=True)
class OCRVote:
    model_name: str
    text: str
    confidence: float
    rotation: int = 0

    @property
    def normalized(self) -> str:
        return normalize_answer(self.text)


@dataclass(frozen=True)
class ConsensusDecision:
    accepted: bool
    text: str
    text_normalized: str
    confidence: float
    consensus_tier: str
    model_votes: tuple[dict[str, Any], ...]
    failure_reason: str | None = None


def votes_to_rows(votes: list[OCRVote]) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "model_name": vote.model_name,
            "text": vote.text,
            "text_normalized": vote.normalized,
            "confidence": round(float(vote.confidence), 6),
            "rotation": int(vote.rotation),
        }
        for vote in votes
    )


def geometric_mean(values: list[float]) -> float:
    clean = [max(float(value), 1e-6) for value in values if value is not None]
    if not clean:
        return 0.0
    return float(math.exp(sum(math.log(value) for value in clean) / len(clean)))


def edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, ch_left in enumerate(left, start=1):
        curr = [i]
        for j, ch_right in enumerate(right, start=1):
            insert_cost = curr[j - 1] + 1
            delete_cost = prev[j] + 1
            subst_cost = prev[j - 1] + (0 if ch_left == ch_right else 1)
            curr.append(min(insert_cost, delete_cost, subst_cost))
        prev = curr
    return prev[-1]


def is_numeric_only(text: str) -> bool:
    compact = re.sub(r"[\s\W_]+", "", str(text or ""))
    return bool(compact) and compact.isdigit()


def choose_consensus(votes: list[OCRVote]) -> ConsensusDecision:
    tuning = load_semantic_dev40_tuning()
    low_confidence_floor = float(tuning.low_confidence_floor)
    strong_consensus_floor = float(tuning.strong_consensus_floor)
    standard_consensus_floor = float(tuning.standard_consensus_floor)
    rows = votes_to_rows(votes)
    if not votes:
        return ConsensusDecision(
            accepted=False,
            text="",
            text_normalized="",
            confidence=0.0,
            consensus_tier="dropped",
            model_votes=rows,
            failure_reason="no_votes",
        )

    grouped: dict[str, list[OCRVote]] = {}
    non_empty_votes: list[OCRVote] = []
    for vote in votes:
        if not vote.normalized:
            continue
        non_empty_votes.append(vote)
        grouped.setdefault(vote.normalized, []).append(vote)

    if len(grouped) == 1:
        only_group = next(iter(grouped.values()))
        min_conf = min(vote.confidence for vote in only_group)
        best_vote = max(only_group, key=lambda vote: (vote.confidence, vote.model_name))
        accepted = bool(
            len(only_group) >= 3
            and len(non_empty_votes) >= 3
            and min_conf >= strong_consensus_floor
            and is_valid_text(best_vote.text)
        )
        return ConsensusDecision(
            accepted=accepted,
            text=best_vote.text if accepted else "",
            text_normalized=best_vote.normalized if accepted else "",
            confidence=float(min_conf if accepted else best_vote.confidence),
            consensus_tier="strong_consensus" if accepted else "dropped",
            model_votes=rows,
            failure_reason=None if accepted else "invalid_text_or_low_confidence",
        )

    agreeing_groups = sorted(grouped.values(), key=lambda group: (-len(group), -max(vote.confidence for vote in group)))
    if agreeing_groups and len(agreeing_groups[0]) >= 2:
        best_group = agreeing_groups[0]
        min_conf = min(vote.confidence for vote in best_group)
        best_vote = max(best_group, key=lambda vote: (vote.confidence, vote.model_name))
        if min_conf >= standard_consensus_floor and is_valid_text(best_vote.text):
            return ConsensusDecision(
                accepted=True,
                text=best_vote.text,
                text_normalized=best_vote.normalized,
                confidence=float(min_conf),
                consensus_tier="standard_consensus",
                model_votes=rows,
            )

    best_pair: tuple[OCRVote, OCRVote] | None = None
    for idx, left in enumerate(votes):
        for right in votes[idx + 1 :]:
            if not left.normalized or not right.normalized:
                continue
            if edit_distance(left.normalized, right.normalized) != 1:
                continue
            longer = left if len(left.normalized) >= len(right.normalized) else right
            if len(longer.normalized) < EDIT_RESCUE_MIN_LEN or is_numeric_only(longer.text):
                continue
            if is_valid_text(longer.text):
                best_pair = (left, right)
                break
        if best_pair is not None:
            break

    if best_pair is not None:
        best_vote = max(best_pair, key=lambda vote: (vote.confidence, vote.model_name))
        return ConsensusDecision(
            accepted=True,
            text=best_vote.text,
            text_normalized=best_vote.normalized,
            confidence=float(best_vote.confidence),
            consensus_tier="edit_rescue",
            model_votes=rows,
        )

    if all(vote.confidence < low_confidence_floor for vote in votes):
        return ConsensusDecision(
            accepted=False,
            text="",
            text_normalized="",
            confidence=max(vote.confidence for vote in votes),
            consensus_tier="dropped",
            model_votes=rows,
            failure_reason="low_confidence",
        )

    return ConsensusDecision(
        accepted=False,
        text="",
        text_normalized="",
        confidence=max(vote.confidence for vote in votes),
        consensus_tier="dropped",
        model_votes=rows,
        failure_reason="full_disagreement",
    )
