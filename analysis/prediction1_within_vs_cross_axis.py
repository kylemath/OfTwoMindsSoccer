#!/usr/bin/env python3
"""
Prediction 1 — Within-axis interference exceeds cross-axis interference.
=========================================================================
Uses the Kaggle dataset (Kicker_Side × Goalie_Side × Outcome) to test
whether goalkeeper save rate differs between within-axis mismatches
(both on horizontal axis) and cross-axis mismatches (horizontal vs centre).

Framework prediction: within-axis mismatches → higher conversion rate
(harder for the keeper to correct within the same motor subspace).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact, chi2_contingency

from utils import (
    load_kaggle, setup_plotting, save_fig, add_stat,
    KEEPER_BLUE, STRIKER_RED, FIELD_GREEN, WARN_AMBER, MUTED_GREY, PALETTE,
)


def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, centre - margin), min(1, centre + margin)


def classify_mismatch(row):
    """Classify each penalty into match type based on kicker/goalie sides."""
    ks, gs = row["Kicker_Side"], row["Goalie_Side"]
    if ks == gs:
        return "Match"
    # Within-axis: both on horizontal (L↔R)
    if {ks, gs} == {"L", "R"}:
        return "Within-axis\nmismatch"
    # Cross-axis: one is C, other is L or R
    return "Cross-axis\nmismatch"


def run():
    print("\n" + "=" * 60)
    print("PREDICTION 1: Within-axis vs. cross-axis interference")
    print("=" * 60)

    setup_plotting()
    df = load_kaggle()

    # ------------------------------------------------------------------
    # 1. Classify each penalty
    # ------------------------------------------------------------------
    df["mismatch"] = df.apply(classify_mismatch, axis=1)
    counts = df.groupby("mismatch")["Outcome"].agg(["sum", "count"])
    counts.columns = ["goals", "total"]
    counts["saves"] = counts["total"] - counts["goals"]
    counts["conversion"] = counts["goals"] / counts["total"]
    print("\nConversion rates by mismatch type:")
    print(counts.to_string())

    # ------------------------------------------------------------------
    # 2. Statistical test: within-axis vs cross-axis mismatch
    # ------------------------------------------------------------------
    within = df[df["mismatch"] == "Within-axis\nmismatch"]
    cross = df[df["mismatch"] == "Cross-axis\nmismatch"]
    match = df[df["mismatch"] == "Match"]

    # 2×2 table: rows = {within, cross}, cols = {goal, save/miss}
    table = np.array([
        [within["Outcome"].sum(), len(within) - within["Outcome"].sum()],
        [cross["Outcome"].sum(), len(cross) - cross["Outcome"].sum()],
    ])
    odds_ratio, p_fisher = fisher_exact(table, alternative="greater")
    print(f"\nFisher exact (within > cross conversion): OR={odds_ratio:.3f}, p={p_fisher:.4f}")

    # Also chi-squared for the full 3-group comparison
    table_3 = np.array([
        [match["Outcome"].sum(), len(match) - match["Outcome"].sum()],
        [within["Outcome"].sum(), len(within) - within["Outcome"].sum()],
        [cross["Outcome"].sum(), len(cross) - cross["Outcome"].sum()],
    ])
    chi2, p_chi2, dof, _ = chi2_contingency(table_3)
    print(f"Chi-squared (3 groups): χ²={chi2:.2f}, df={dof}, p={p_chi2:.4f}")

    # ------------------------------------------------------------------
    # 3. Register LaTeX stats
    # ------------------------------------------------------------------
    n_total = len(df)
    add_stat("PredOneNTotal", str(n_total))
    add_stat("PredOneNWithin", str(len(within)))
    add_stat("PredOneNCross", str(len(cross)))
    add_stat("PredOneNMatch", str(len(match)))

    within_label = "Within-axis\nmismatch"
    cross_label = "Cross-axis\nmismatch"
    add_stat("PredOneConvWithin", f"{counts.loc[within_label, 'conversion']:.1%}")
    add_stat("PredOneConvCross", f"{counts.loc[cross_label, 'conversion']:.1%}")
    add_stat("PredOneConvMatch", f"{counts.loc['Match', 'conversion']:.1%}")

    add_stat("PredOneFisherOR", f"{odds_ratio:.2f}")
    add_stat("PredOneFisherP", f"{p_fisher:.4f}")
    add_stat("PredOneChiSq", f"{chi2:.2f}")
    add_stat("PredOneChiP", f"{p_chi2:.4f}")

    # ------------------------------------------------------------------
    # 4a. Figure — bar chart of conversion rates by mismatch type
    # ------------------------------------------------------------------
    order = ["Match", "Cross-axis\nmismatch", "Within-axis\nmismatch"]
    colors = [MUTED_GREY, KEEPER_BLUE, STRIKER_RED]
    conv = [counts.loc[o, "conversion"] for o in order]
    totals = [counts.loc[o, "total"] for o in order]

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    goals = [counts.loc[o, "goals"] for o in order]
    cis = [_wilson_ci(int(g), int(t)) for g, t in zip(goals, totals)]
    yerr_lo = [c - ci[1] for c, ci in zip(conv, cis)]
    yerr_hi = [ci[2] - c for c, ci in zip(conv, cis)]
    bars = ax.bar(order, conv, yerr=[yerr_lo, yerr_hi], capsize=4,
                  color=colors, edgecolor="white", linewidth=1.2, error_kw={"lw": 1.2})
    for bar, c, t in zip(bars, conv, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{c:.1%}\n(n={t})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Conversion rate (goal scored)")
    ax.set_ylim(0, 1.08)
    ax.set_title("Prediction 1: Conversion by mismatch type", fontsize=11, fontweight="bold")
    ax.axhline(y=df["Outcome"].mean(), ls="--", color=MUTED_GREY, alpha=0.5, label="Overall mean")
    ax.legend(fontsize=8)
    save_fig(fig, "empirical_pred1_conversion_by_mismatch")

    # ------------------------------------------------------------------
    # 4b. Figure — full 3×3 heatmap
    # ------------------------------------------------------------------
    ct = pd.crosstab(df["Kicker_Side"], df["Goalie_Side"], values=df["Outcome"],
                     aggfunc="mean").reindex(index=["L", "C", "R"], columns=["L", "C", "R"])
    ct_n = pd.crosstab(df["Kicker_Side"], df["Goalie_Side"]).reindex(
        index=["L", "C", "R"], columns=["L", "C", "R"])

    annot = ct.copy()
    for r in annot.index:
        for c in annot.columns:
            annot.loc[r, c] = f"{ct.loc[r, c]:.0%}\n(n={ct_n.loc[r, c]})"

    fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(ct, annot=annot, fmt="", cmap="RdYlGn", vmin=0.5, vmax=1.0,
                linewidths=1.5, linecolor="white", cbar_kws={"label": "Conversion rate"},
                ax=ax2)
    ax2.set_xlabel("Goalkeeper dive direction")
    ax2.set_ylabel("Kicker shot direction")
    ax2.set_title("Prediction 1: Decision matrix", fontsize=11, fontweight="bold")
    save_fig(fig2, "empirical_pred1_heatmap")

    print("  ✓ Prediction 1 complete.\n")


if __name__ == "__main__":
    from utils import init_stats, write_stats
    init_stats()
    run()
    write_stats()
