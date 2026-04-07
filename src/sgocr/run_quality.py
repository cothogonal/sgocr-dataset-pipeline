from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _question_type_diversity(qtypes: Counter[str | None]) -> float:
    """Shannon entropy of question-type distribution, normalized to [0, 1].

    A run that produces only DIRECT_READ scores 0.
    A run with perfectly balanced 5-type output scores 1.
    Practical runs land somewhere in between.
    """
    total = sum(qtypes.values())
    if total == 0:
        return 0.0
    max_entropy = math.log(5)  # 5 question types is the max
    if max_entropy == 0:
        return 0.0
    entropy = 0.0
    for count in qtypes.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)
    return min(entropy / max_entropy, 1.0)


def compute_q3_score(
    *,
    qa_accept_rate: float,
    images_with_final_rows: int,
    accepted_qas: int,
    generated_qas: int,
    anchor_missing: int,
    ambiguous_grounding: int,
    reverse_ambiguous: int,
    reverse_invalid: int,
    validation_failed: int,
    reverse_local: int,
    reverse_mixed: int,
    yesno_negative: int,
    type_diversity: float,
    num_types_with_coverage: int,
) -> float:
    """Q3 quality score — designed around three principles:

    1. Precision over yield: accept rate dominates, raw count does not.
       A sloppy run that passes 20 extra rows through a leaky gate should
       not outscore a clean run with fewer rows.

    2. Rate-based penalties: failure modes are penalized as fractions of
       generated QAs, not raw counts. This makes scores comparable across
       runs with different yield levels and across verifier generations.

    3. Heavy frontier reward: local/mixed reverse-ground answers are the
       hardest open problem and get meaningful credit. A run that cracks
       local grounding should visibly outscore one that doesn't.

    Rough scale: 0-100. A perfect run on dev40 would score ~85-95.
    A mediocre but functional run scores ~30-50.
    """

    # -- Precision: 0-30 pts --
    # Accept rate is the single most important signal.
    # Non-linear: reward the jump from 0.5 → 0.7 more than 0.8 → 0.9.
    precision = 30.0 * qa_accept_rate

    # -- Coverage: 0-20 pts --
    # What fraction of the dev40 image set produced usable rows?
    # This is the right normalization — raw image count is meaningless
    # without knowing the universe size.
    dev_images = 40.0
    coverage = 20.0 * min(images_with_final_rows / dev_images, 1.0)

    # -- Per-image yield efficiency: 0-10 pts --
    # Target is ~2.5 accepted per covered image. Diminishing returns
    # above that — we don't want to reward overproduction on easy images.
    per_image = accepted_qas / max(images_with_final_rows, 1)
    yield_eff = 10.0 * min(per_image / 3.5, 1.0)

    # -- Question-type diversity: 0-10 pts --
    # Split between Shannon entropy (smooth) and coverage count (discrete).
    # A dataset with all DIRECT_READ is less useful than balanced types.
    diversity = 5.0 * type_diversity + 2.0 * min(num_types_with_coverage / 4.0, 1.0)

    # -- Frontier grounding: 0-15 pts --
    # Local and mixed reverse-ground answers are the hardest open problem.
    # Each one that survives the strict verifier is genuinely valuable.
    # Negative yes/no also gets modest credit — useful when clean.
    frontier = (
        3.0 * min(reverse_local, 5)
        + 1.5 * min(reverse_mixed, 5)
        + 0.4 * min(yesno_negative, 10)
    )

    # -- Failure penalties: rate-based, scale-invariant --
    # These are fractions of generated QAs so runs with different yield
    # levels are comparable. Anchor-missing is the worst because it means
    # the teacher wasted an API call on something that could never pass.
    gen = max(generated_qas, 1)
    penalty = (
        15.0 * (anchor_missing / gen)
        + 10.0 * (ambiguous_grounding / gen)
        + 8.0 * (reverse_ambiguous / gen)
        + 5.0 * (reverse_invalid / gen)
        + 2.0 * (validation_failed / gen)
    )

    return round(precision + coverage + yield_eff + diversity + frontier - penalty, 2)


def compute_run_quality(summary: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    qtypes = Counter(row["tags"]["question_type"] for row in rows)
    yesno = [row for row in rows if row["tags"]["question_type"] == "YES_NO"]
    reverse = [row for row in rows if row["tags"]["question_type"] == "REVERSE_GROUND"]
    yesno_polarity = Counter(row["tags"].get("yesno_polarity") for row in yesno)
    reverse_scope = Counter((row.get("grounding") or {}).get("reverse_ground_scope_preference") for row in reverse)
    ambiguity = Counter(row["tags"].get("ambiguity_level") for row in rows)
    failure_counts = summary.get("failure_counts") or {}

    accepted_qas = int(summary.get("accepted_qas") or 0)
    generated_qas = int(summary.get("generated_qas") or 0)
    qa_accept_rate = float(summary.get("qa_accept_rate") or 0.0)
    images_with_final_rows = int(summary.get("images_with_final_rows") or 0)
    anchor_missing = int(failure_counts.get("anchor_missing", 0))
    ambiguous_grounding = int(failure_counts.get("ambiguous_grounding", 0))
    reverse_ambiguous = int(failure_counts.get("reverse_ground_ambiguous", 0))
    reverse_invalid = int(failure_counts.get("reverse_ground_answer_invalid", 0))
    validation_failed = int(failure_counts.get("validation_failed", 0))

    high_ambiguity = int(ambiguity.get("high", 0))
    high_ambiguity_fraction = float(high_ambiguity / accepted_qas) if accepted_qas else 1.0
    anchor_missing_rate = float(anchor_missing / generated_qas) if generated_qas else 0.0

    quality = round(
        15.0 * qa_accept_rate
        + 0.20 * accepted_qas
        + 0.55 * images_with_final_rows
        + 3.0 * (1.0 - high_ambiguity_fraction)
        - 0.45 * anchor_missing
        - 4.0 * anchor_missing_rate
        - 0.60 * ambiguous_grounding
        - 0.50 * reverse_ambiguous
        - 0.40 * reverse_invalid
        - 0.10 * validation_failed
        + 0.15 * float(yesno_polarity.get("negative", 0))
        + 0.15 * float(reverse_scope.get("local", 0))
        + 0.08 * float(reverse_scope.get("mixed", 0)),
        2,
    )

    yesno_neg = int(yesno_polarity.get("negative", 0))
    rev_local = int(reverse_scope.get("local", 0))
    rev_mixed = int(reverse_scope.get("mixed", 0))

    type_diversity = _question_type_diversity(qtypes)
    num_types_with_coverage = sum(1 for c in qtypes.values() if c >= 3)

    q3 = compute_q3_score(
        qa_accept_rate=qa_accept_rate,
        images_with_final_rows=images_with_final_rows,
        accepted_qas=accepted_qas,
        generated_qas=generated_qas,
        anchor_missing=anchor_missing,
        ambiguous_grounding=ambiguous_grounding,
        reverse_ambiguous=reverse_ambiguous,
        reverse_invalid=reverse_invalid,
        validation_failed=validation_failed,
        reverse_local=rev_local,
        reverse_mixed=rev_mixed,
        yesno_negative=yesno_neg,
        type_diversity=type_diversity,
        num_types_with_coverage=num_types_with_coverage,
    )

    return {
        "accepted_qas": accepted_qas,
        "generated_qas": generated_qas,
        "qa_accept_rate": qa_accept_rate,
        "images_with_final_rows": images_with_final_rows,
        "yesno_positive": int(yesno_polarity.get("positive", 0)),
        "yesno_negative": yesno_neg,
        "reverse_local": rev_local,
        "reverse_global": int(reverse_scope.get("global", 0)),
        "reverse_mixed": rev_mixed,
        "ambiguity_high": high_ambiguity,
        "high_ambiguity_fraction": high_ambiguity_fraction,
        "anchor_missing": anchor_missing,
        "anchor_missing_rate": anchor_missing_rate,
        "ambiguous_grounding": ambiguous_grounding,
        "reverse_ambiguous": reverse_ambiguous,
        "reverse_invalid": reverse_invalid,
        "validation_failed": validation_failed,
        "quality_score": quality,
        "q3_score": q3,
        "type_diversity": type_diversity,
        "num_types_with_coverage": num_types_with_coverage,
        "summary": summary,
        "question_types": dict(qtypes),
    }
