"""Re-score all mixed_dev150 sweep reports with Q4 (Training Signal Score).

Usage:
    python -m sgocr.rescore_q4

Or run directly:
    python sgocr/src/sgocr/rescore_q4.py

Reads all report JSONs from the mixed_dev150 final output directory,
computes Q4 scores, and prints a ranked table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure sgocr package is importable when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sgocr.run_quality import compute_q4_score_from_metrics

REPORT_DIR = Path(
    "/home/wdree/percy/vqafromscratch/data/ocr_spatial_qa/final/mixed_dev150"
)


def rescore_report(report_path: Path) -> list[dict]:
    """Load a report JSON and compute Q4 for each variant."""
    with open(report_path) as f:
        data = json.load(f)

    results = []
    bundle = data.get("bundle_id", report_path.stem)

    for variant_name, vdata in data.get("results", {}).items():
        status = vdata.get("status", "unknown")
        if status not in ("ok", "cached"):
            continue

        m = vdata.get("metrics") or {}
        adist = vdata.get("answer_distribution") or {}
        acov = vdata.get("anchor_coverage") or {}
        idep = vdata.get("image_dependence") or {}

        # Build the metrics dict expected by compute_q4_score_from_metrics
        qtypes = m.get("question_types") or {}
        metrics = {
            "accepted_qas": m.get("accepted_qas", 0),
            "images_with_final_rows": m.get("images_with_final_rows", 0),
            "question_types": qtypes,
            "reverse_local": m.get("reverse_local", 0),
            "reverse_global": m.get("reverse_global", 0),
            "reverse_mixed": m.get("reverse_mixed", 0),
            "yesno_negative": m.get("yesno_negative", 0),
            "high_ambiguity_fraction": m.get("high_ambiguity_fraction", 0.0),
            "type_diversity": m.get("type_diversity", 0.0),
            "num_types_with_coverage": m.get("num_types_with_coverage", 0),
        }

        q4 = compute_q4_score_from_metrics(
            metrics,
            answer_distribution=adist,
            anchor_coverage=acov,
            image_dependence=idep,
        )

        rg_count = qtypes.get("REVERSE_GROUND", 0)
        accepted = m.get("accepted_qas", 0)
        rg_pct = rg_count / accepted if accepted else 0

        results.append({
            "bundle": bundle,
            "variant": variant_name,
            "status": status,
            "q4_score": q4,
            "sweep_score": m.get("sweep_score", 0),
            "q3_score": m.get("q3_score", 0),
            "accepted": accepted,
            "images": m.get("images_with_final_rows", 0),
            "rg_count": rg_count,
            "rg_pct": round(rg_pct * 100, 1),
            "vision_nec": round((idep.get("vision_necessary_rate") or 0) * 100, 1),
            "text_leaky": round((idep.get("text_leaky_rate") or 0) * 100, 1),
            "answer_entropy": adist.get("answer_entropy", 0),
            "unique_ans": adist.get("unique_answers", 0),
            "mean_anchors": acov.get("mean_unique_anchors", 0),
            "multi_anchor_pct": round((acov.get("images_with_multiple_anchors_rate") or 0) * 100, 1),
            "high_ambig_pct": round(m.get("high_ambiguity_fraction", 0) * 100, 1),
            "yesno_yes_rate": round(adist.get("yesno_yes_rate", 0) * 100, 1),
        })

    return results


def main():
    # Collect all report JSONs, ordered by sweep
    report_files = sorted(REPORT_DIR.glob("sgocr_mixed_ita*_report.json"))
    # Also include the earlier finalv0 reports
    report_files += sorted(REPORT_DIR.glob("sgocr_mixed_finalv0_*_report.json"))
    report_files = sorted(set(report_files))

    if not report_files:
        print(f"No report JSONs found in {REPORT_DIR}")
        sys.exit(1)

    all_results = []
    for rp in report_files:
        all_results.extend(rescore_report(rp))

    if not all_results:
        print("No valid variants found in reports.")
        sys.exit(1)

    # Rank by Q4
    ranked = sorted(all_results, key=lambda r: (-r["q4_score"], -r["accepted"]))

    # Print ranked table
    print()
    print("=" * 140)
    print(f"{'RANK':>4}  {'Q4':>6}  {'Sweep':>7}  {'Q3':>6}  {'Acc':>5}  {'Img':>4}  "
          f"{'RG%':>5}  {'RG#':>3}  {'VisNec%':>7}  {'Leaky%':>6}  {'Ent':>4}  "
          f"{'UA':>3}  {'Anc':>4}  {'MA%':>4}  {'Amb%':>4}  {'YN%':>4}  VARIANT")
    print("-" * 140)

    for i, r in enumerate(ranked, 1):
        print(
            f"{i:>4}  {r['q4_score']:>6.2f}  {r['sweep_score']:>7.4f}  "
            f"{r['q3_score']:>6.2f}  {r['accepted']:>5}  {r['images']:>4}  "
            f"{r['rg_pct']:>5.1f}  {r['rg_count']:>3}  {r['vision_nec']:>7.1f}  "
            f"{r['text_leaky']:>6.1f}  {r['answer_entropy']:>4.2f}  "
            f"{r['unique_ans']:>3}  {r['mean_anchors']:>4.2f}  "
            f"{r['multi_anchor_pct']:>4.1f}  {r['high_ambig_pct']:>4.1f}  "
            f"{r['yesno_yes_rate']:>4.1f}  {r['variant']}"
        )

    print("=" * 140)

    # Component breakdown for top 10
    print()
    print("Q4 COMPONENT BREAKDOWN — Top 10 Variants")
    print("=" * 140)

    for i, r in enumerate(ranked[:10], 1):
        qtypes = {"REVERSE_GROUND": r["rg_count"]}
        metrics = {
            "accepted_qas": r["accepted"],
            "images_with_final_rows": r["images"],
            "question_types": qtypes,
            "reverse_local": 0,  # Not available from report summary alone
            "reverse_global": 0,
            "reverse_mixed": 0,
            "yesno_negative": 0,
            "high_ambiguity_fraction": r["high_ambig_pct"] / 100,
            "type_diversity": 0,
            "num_types_with_coverage": 0,
        }

        # Re-compute with component breakdown
        accepted = r["accepted"]
        images = r["images"]
        rg_pct = r["rg_pct"] / 100
        ans_entropy = r["answer_entropy"]
        unique_ans = r["unique_ans"]
        mean_anchors = r["mean_anchors"]
        multi_anchor_rate = r["multi_anchor_pct"] / 100
        yesno_yes_rate = r["yesno_yes_rate"] / 100
        high_ambig_frac = r["high_ambig_pct"] / 100
        text_leaky_rate = r["text_leaky"] / 100 if r["text_leaky"] > 0 else -1

        # Component scores
        yield_s = min(accepted / 300.0, 1.0) * 20.0
        cover_s = min(images / 120.0, 1.0) * 10.0

        if rg_pct >= 0.12:
            rg_s = 25.0
        elif rg_pct >= 0.08:
            rg_s = 15.0 + (rg_pct - 0.08) / 0.04 * 10.0
        elif rg_pct >= 0.05:
            rg_s = 5.0 + (rg_pct - 0.05) / 0.03 * 10.0
        elif rg_pct >= 0.02:
            rg_s = (rg_pct - 0.02) / 0.03 * 5.0
        else:
            rg_s = 0.0

        rg_mix_s = 0.0  # Can't compute without reverse_local/global/mixed
        ans_div_s = min(ans_entropy / 4.0, 1.0) * 5.0 + min(unique_ans / 100.0, 1.0) * 5.0
        anc_s = min(mean_anchors / 1.8, 1.0) * 5.0 + min(multi_anchor_rate / 0.30, 1.0) * 5.0
        yn_s = max(0.0, 1.0 - abs(yesno_yes_rate - 0.5) * 2.0) * 5.0

        if text_leaky_rate >= 0:
            if text_leaky_rate <= 0.35:
                txt_s = 5.0
            elif text_leaky_rate <= 0.50:
                txt_s = 5.0 - (text_leaky_rate - 0.35) / 0.15 * 3.0
            else:
                txt_s = max(0.0, 2.0 - (text_leaky_rate - 0.50) / 0.30 * 2.0)
        else:
            txt_s = 2.5

        if high_ambig_frac <= 0.60:
            amb_s = 5.0
        elif high_ambig_frac <= 0.75:
            amb_s = 5.0 - (high_ambig_frac - 0.60) / 0.15 * 3.0
        else:
            amb_s = max(0.0, 2.0 - (high_ambig_frac - 0.75) / 0.25 * 2.0)

        total = yield_s + cover_s + rg_s + rg_mix_s + ans_div_s + anc_s + yn_s + txt_s + amb_s

        print(f"\n  #{i}: {r['variant']}  (Q4={r['q4_score']:.2f})")
        print(f"      Yield: {yield_s:.1f}/20  Cover: {cover_s:.1f}/10  "
              f"RG: {rg_s:.1f}/25  RG-div: {rg_mix_s:.1f}/10  "
              f"Ans-div: {ans_div_s:.1f}/10  Anchor: {anc_s:.1f}/10  "
              f"Y/N: {yn_s:.1f}/5  Text: {txt_s:.1f}/5  Disambig: {amb_s:.1f}/5")

    print()


if __name__ == "__main__":
    main()
