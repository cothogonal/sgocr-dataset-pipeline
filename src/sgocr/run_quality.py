from __future__ import annotations

import math
from collections import Counter
from typing import Any


def compute_answer_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute answer distribution quality signals from accepted QA rows.

    Signals:
    - yesno_yes_rate: fraction of YES_NO rows that expect "yes" (bias indicator)
    - top1_concentration: fraction of all rows where the most common answer appears
    - top10_concentration: fraction covered by the 10 most common answers
    - answer_entropy: Shannon entropy of answer distribution (higher = more diverse)
    """
    all_answers = [str(row.get("answer") or "").strip().lower() for row in rows if row.get("answer")]
    yesno_rows = [row for row in rows if str((row.get("tags") or {}).get("question_type") or "").upper() == "YES_NO"]
    yesno_yes = sum(1 for r in yesno_rows if str(r.get("answer") or "").strip().lower() == "yes")
    yesno_total = len(yesno_rows) or 1

    answer_counts = Counter(all_answers)
    total = len(all_answers) or 1
    top10 = answer_counts.most_common(10)
    top1_count = top10[0][1] if top10 else 0
    top10_count = sum(c for _, c in top10)

    entropy = 0.0
    for count in answer_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    return {
        "total_answers": total,
        "unique_answers": len(answer_counts),
        "yesno_yes_rate": round(yesno_yes / yesno_total, 4),
        "yesno_no_rate": round(1.0 - yesno_yes / yesno_total, 4),
        "top1_concentration": round(top1_count / total, 4),
        "top10_concentration": round(top10_count / total, 4),
        "answer_entropy": round(entropy, 4),
        "top10_answers": [(ans, cnt) for ans, cnt in top10],
    }


def compute_anchor_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate per-image anchor diversity from accepted QA rows.

    Since total Qwen detections aren't in the QA output, this measures the
    numerator only: distinct anchor labels referenced per image in the accepted set.
    A higher mean indicates the QAs spread across different visual regions.
    """
    by_image: dict[str, set[str]] = {}
    for row in rows:
        image_id = str(row.get("image_id") or "")
        anchor_label = str((row.get("tags") or {}).get("anchor_label") or "").strip().lower()
        if image_id:
            by_image.setdefault(image_id, set())
            if anchor_label:
                by_image[image_id].add(anchor_label)

    per_image_counts = [len(anchors) for anchors in by_image.values()]
    if not per_image_counts:
        return {"images": 0, "mean_unique_anchors": 0.0, "images_with_multiple_anchors": 0}

    n = len(per_image_counts)
    mean_unique = sum(per_image_counts) / n
    images_with_multiple = sum(1 for c in per_image_counts if c >= 2)
    return {
        "images": n,
        "mean_unique_anchors": round(mean_unique, 3),
        "median_unique_anchors": float(sorted(per_image_counts)[n // 2]),
        "images_with_multiple_anchors": images_with_multiple,
        "images_with_multiple_anchors_rate": round(images_with_multiple / n, 4),
    }


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

    inline_scored_rows = [row for row in rows if row.get("inline_frontier_correct") is not None]
    inline_frontier_mean = (
        sum(1.0 for row in inline_scored_rows if bool(row.get("inline_frontier_correct"))) / len(inline_scored_rows)
        if inline_scored_rows
        else float((summary.get("inline_frontier") or {}).get("mean_inline_frontier_correct") or 0.0)
    )
    inline_frontier_scored = (
        len(inline_scored_rows)
        if inline_scored_rows
        else int((summary.get("inline_frontier") or {}).get("scored_rows") or 0)
    )
    precision_first_score = round(inline_frontier_mean * math.sqrt(max(inline_frontier_scored, 0)), 4)
    sweep_score = precision_first_score if inline_frontier_scored > 0 else q3

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
        "q4_score": compute_q4_score_from_metrics({
            "accepted_qas": accepted_qas,
            "generated_qas": generated_qas,
            "qa_accept_rate": qa_accept_rate,
            "images_with_final_rows": images_with_final_rows,
            "yesno_positive": int(yesno_polarity.get("positive", 0)),
            "yesno_negative": yesno_neg,
            "reverse_local": rev_local,
            "reverse_global": int(reverse_scope.get("global", 0)),
            "reverse_mixed": rev_mixed,
            "high_ambiguity_fraction": high_ambiguity_fraction,
            "type_diversity": type_diversity,
            "num_types_with_coverage": num_types_with_coverage,
            "question_types": dict(qtypes),
        }),
        "inline_frontier_mean": round(inline_frontier_mean, 4),
        "inline_frontier_scored": inline_frontier_scored,
        "precision_first_score": precision_first_score,
        "sweep_score": sweep_score,
        "type_diversity": type_diversity,
        "num_types_with_coverage": num_types_with_coverage,
        "summary": summary,
        "question_types": dict(qtypes),
    }


# ---------------------------------------------------------------------------
# Q4 — Training Signal Score
# ---------------------------------------------------------------------------

def compute_q4_score_from_metrics(
    metrics: dict[str, Any],
    *,
    answer_distribution: dict[str, Any] | None = None,
    anchor_coverage: dict[str, Any] | None = None,
    image_dependence: dict[str, Any] | None = None,
) -> float:
    """Q4: Downstream Training Signal Score.

    Philosophy: maximize training utility for small VLMs, not perfection.
    - No frontier model calls (API-free).
    - Rewards yield, RG diversity, anchor quality, answer diversity.
    - Softly penalizes ambiguity and text-leakage.
    - Projects downstream VLM training signal.

    Rough scale: 0-100.  A perfect run scores ~80-95.
    A mediocre but functional run scores ~35-55.
    """
    accepted = int(metrics.get("accepted_qas") or 0)
    if accepted == 0:
        return 0.0

    images = int(metrics.get("images_with_final_rows") or 0)
    qtypes = metrics.get("question_types") or {}
    rg_count = int(qtypes.get("REVERSE_GROUND") or 0)
    dr_count = int(qtypes.get("DIRECT_READ") or 0)
    yn_count = int(qtypes.get("YES_NO") or 0)
    tp_count = int(qtypes.get("TEXT_PROPERTY") or 0)
    ap_count = int(qtypes.get("ANCHOR_PROPERTY") or 0)
    rev_local = int(metrics.get("reverse_local") or 0)
    rev_global = int(metrics.get("reverse_global") or 0)
    rev_mixed = int(metrics.get("reverse_mixed") or 0)
    yn_neg = int(metrics.get("yesno_negative") or 0)
    high_ambig_frac = float(metrics.get("high_ambiguity_fraction") or 0.0)
    type_diversity = float(metrics.get("type_diversity") or 0.0)
    num_types = int(metrics.get("num_types_with_coverage") or 0)

    # Optional external signals
    ans = answer_distribution or {}
    anc = anchor_coverage or {}
    idep = image_dependence or {}

    unique_ans = int(ans.get("unique_answers") or 0)
    ans_entropy = float(ans.get("answer_entropy") or 0.0)
    yesno_yes_rate = float(ans.get("yesno_yes_rate") or 0.5)
    mean_anchors = float(anc.get("mean_unique_anchors") or 0.0)
    multi_anchor_rate = float(anc.get("images_with_multiple_anchors_rate") or 0.0)
    text_leaky_rate = float(idep.get("text_leaky_rate") or -1.0)  # -1 means unavailable

    # =====================================================================
    # COMPONENT 1: Yield Base (0-20 pts)
    # =====================================================================
    # More data = more training signal, with diminishing returns.
    # Target: 300 rows saturates this component.
    yield_score = min(accepted / 300.0, 1.0) * 20.0

    # =====================================================================
    # COMPONENT 2: Image Coverage (0-10 pts)
    # =====================================================================
    # More images with rows = better generalization across visual styles.
    # Target: 120 images (80% of 150-image mixed150 pool) saturates.
    coverage_score = min(images / 120.0, 1.0) * 10.0

    # =====================================================================
    # COMPONENT 3: RG Force (0-25 pts)  ← main new component
    # =====================================================================
    # REVERSE_GROUND is the most spatially-reasoning-intensive type.
    # A "spatially-grounded" dataset must have meaningful RG presence.
    # Floor: 8%, target: 12%+, steep penalty below 5%.
    rg_pct = rg_count / accepted
    if rg_pct >= 0.12:
        rg_score = 25.0
    elif rg_pct >= 0.08:
        rg_score = 15.0 + (rg_pct - 0.08) / 0.04 * 10.0  # 15 → 25
    elif rg_pct >= 0.05:
        rg_score = 5.0 + (rg_pct - 0.05) / 0.03 * 10.0    # 5 → 15
    elif rg_pct >= 0.02:
        rg_score = (rg_pct - 0.02) / 0.03 * 5.0            # 0 → 5
    else:
        rg_score = 0.0

    # =====================================================================
    # COMPONENT 4: RG Mixed-Local-Global Diversity (0-10 pts)
    # =====================================================================
    # Copied from Q3's concept: reward diverse RG difficulty levels.
    # reverse_local (hardest), reverse_global (easiest), reverse_mixed (middle).
    # A run with all three types scores higher than one with only one type.
    rg_types_present = sum(1 for c in [rev_local, rev_global, rev_mixed] if c > 0)
    rg_total_scope = rev_local + rev_global + rev_mixed
    if rg_total_scope > 0:
        # Coverage: reward having at least 1 of each type (0-5 pts)
        rg_coverage = (rg_types_present / 3.0) * 5.0
        # Balance: Shannon entropy of scope distribution (0-5 pts)
        scope_counts = [c for c in [rev_local, rev_global, rev_mixed] if c > 0]
        if len(scope_counts) >= 2:
            scope_entropy = 0.0
            for c in scope_counts:
                p = c / rg_total_scope
                if p > 0:
                    scope_entropy -= p * math.log(p)
            max_scope_entropy = math.log(3)
            rg_balance = (scope_entropy / max_scope_entropy) * 5.0 if max_scope_entropy > 0 else 0.0
        else:
            rg_balance = 0.0
    else:
        rg_coverage = 0.0
        rg_balance = 0.0
    rg_mixed_score = rg_coverage + rg_balance

    # =====================================================================
    # COMPONENT 5: Answer Diversity (0-10 pts)
    # =====================================================================
    # High entropy + many unique answers = prevents mode collapse
    # during downstream fine-tuning.
    # Entropy target: 3.5+, Unique answers target: 90+
    entropy_score = min(ans_entropy / 4.0, 1.0) * 5.0
    unique_score = min(unique_ans / 100.0, 1.0) * 5.0
    answer_diversity_score = entropy_score + unique_score

    # =====================================================================
    # COMPONENT 6: Anchor Quality (0-10 pts)
    # =====================================================================
    # Multi-anchor images = richer spatial relationships per image.
    # Target: mean_anchors >= 1.5, multi_anchor_rate >= 25%
    anchor_mean_score = min(mean_anchors / 1.8, 1.0) * 5.0
    anchor_multi_score = min(multi_anchor_rate / 0.30, 1.0) * 5.0
    anchor_quality_score = anchor_mean_score + anchor_multi_score

    # =====================================================================
    # COMPONENT 7: Yes/No Balance (0-5 pts)
    # =====================================================================
    # Balanced yes/no = stable binary classification training.
    # Target: 0.35-0.65 YES rate.  Penalize extreme skew.
    yn_balance = 1.0 - abs(yesno_yes_rate - 0.5) * 2.0
    yn_balance = max(yn_balance, 0.0)
    yesno_score = yn_balance * 5.0

    # =====================================================================
    # COMPONENT 8: Text Quality Proxy (0-5 pts)
    # =====================================================================
    # Soft penalty on text-leaky rate.  Only applied if image_dependence
    # eval data is available (it costs tokens but is a one-time eval).
    if text_leaky_rate >= 0:
        if text_leaky_rate <= 0.35:
            text_score = 5.0
        elif text_leaky_rate <= 0.50:
            text_score = 5.0 - (text_leaky_rate - 0.35) / 0.15 * 3.0
        else:
            text_score = max(0.0, 2.0 - (text_leaky_rate - 0.50) / 0.30 * 2.0)
    else:
        # No eval data: give neutral score (neither reward nor penalty)
        text_score = 2.5

    # =====================================================================
    # COMPONENT 9: Disambiguity (0-5 pts)
    # =====================================================================
    # Soft penalty on ambiguity — some ambiguity is fine for generalization.
    # Target: < 60% high ambiguity.  Gentle slope above.
    if high_ambig_frac <= 0.60:
        disambig_score = 5.0
    elif high_ambig_frac <= 0.75:
        disambig_score = 5.0 - (high_ambig_frac - 0.60) / 0.15 * 3.0
    else:
        disambig_score = max(0.0, 2.0 - (high_ambig_frac - 0.75) / 0.25 * 2.0)

    # =====================================================================
    # TOTAL: 100 pts
    # =====================================================================
    total = (
        yield_score            #  0-20
        + coverage_score       #  0-10
        + rg_score             #  0-25
        + rg_mixed_score       #  0-10
        + answer_diversity_score  #  0-10
        + anchor_quality_score #  0-10
        + yesno_score          #  0- 5
        + text_score           #  0- 5
        + disambig_score       #  0- 5
    )

    return round(total, 2)


def compute_q4_score(
    *,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    answer_distribution: dict[str, Any] | None = None,
    anchor_coverage: dict[str, Any] | None = None,
    image_dependence: dict[str, Any] | None = None,
) -> float:
    """Compute Q4 score from raw QA rows (used during pipeline execution)."""
    qtypes = Counter(row["tags"]["question_type"] for row in rows)
    yesno = [row for row in rows if row["tags"]["question_type"] == "YES_NO"]
    reverse = [row for row in rows if row["tags"]["question_type"] == "REVERSE_GROUND"]
    yesno_polarity = Counter(row["tags"].get("yesno_polarity") for row in yesno)
    reverse_scope = Counter((row.get("grounding") or {}).get("reverse_ground_scope_preference") for row in reverse)
    ambiguity = Counter(row["tags"].get("ambiguity_level") for row in rows)

    accepted_qas = int(summary.get("accepted_qas") or 0)
    images_with_final_rows = int(summary.get("images_with_final_rows") or 0)
    high_ambiguity = int(ambiguity.get("high", 0))
    high_ambiguity_fraction = float(high_ambiguity / accepted_qas) if accepted_qas else 1.0
    yesno_neg = int(yesno_polarity.get("negative", 0))
    rev_local = int(reverse_scope.get("local", 0))
    rev_global = int(reverse_scope.get("global", 0))
    rev_mixed = int(reverse_scope.get("mixed", 0))
    type_diversity = _question_type_diversity(qtypes)
    num_types_with_coverage = sum(1 for c in qtypes.values() if c >= 3)

    metrics = {
        "accepted_qas": accepted_qas,
        "images_with_final_rows": images_with_final_rows,
        "question_types": dict(qtypes),
        "reverse_local": rev_local,
        "reverse_global": rev_global,
        "reverse_mixed": rev_mixed,
        "yesno_negative": yesno_neg,
        "high_ambiguity_fraction": high_ambiguity_fraction,
        "type_diversity": type_diversity,
        "num_types_with_coverage": num_types_with_coverage,
    }

    return compute_q4_score_from_metrics(
        metrics,
        answer_distribution=answer_distribution,
        anchor_coverage=anchor_coverage,
        image_dependence=image_dependence,
    )
