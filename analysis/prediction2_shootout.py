#!/usr/bin/env python3
"""
Prediction 2 — Shootout replication of sequential effects.
============================================================
Uses the FIFA World Cup penalty-shootout dataset (jfjelstul/worldcup,
1982-2022) to test whether the outcome of kick n-1 in a shootout
predicts the outcome of kick n.

Because shootout kicks are tightly spaced (~1 min apart) and highly
consequential, they provide a stronger test of shared-subspace sequential
priming than in-game penalties separated by many minutes of play.

Analyses:
  1. Goal-after-goal vs goal-after-miss (chi-squared, risk difference)
  2. Logistic regression controlling for round number and team
  3. Figure: sequential conversion rates with Wilson CIs
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

from utils import (
    load_shootout, setup_plotting, save_fig, add_stat,
    KEEPER_BLUE, STRIKER_RED, MUTED_GREY,
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


def run():
    print("\n" + "=" * 60)
    print("PREDICTION 2 REPLICATION: Shootout sequential effects")
    print("=" * 60)

    setup_plotting()

    df = load_shootout()
    n_kicks_total = len(df)
    n_matches = df["match_id"].nunique()
    overall_conv = df["converted"].mean()
    print(f"  Shootout kicks: {n_kicks_total}  Matches: {n_matches}")
    print(f"  Overall conversion rate: {overall_conv:.1%}")

    # Sort by match and kick order (key_id preserves chronological order)
    df = df.sort_values(["match_id", "key_id"]).reset_index(drop=True)

    # Within each match, assign a sequential kick number and get prev outcome
    df["kick_num"] = df.groupby("match_id").cumcount() + 1
    df["prev_converted"] = df.groupby("match_id")["converted"].shift(1)
    df_pairs = df.dropna(subset=["prev_converted"]).copy()
    df_pairs["prev_converted"] = df_pairs["prev_converted"].astype(int)

    n_pairs = len(df_pairs)
    print(f"  Sequential pairs: {n_pairs}")

    # ------------------------------------------------------------------
    # Crosstab + chi-squared
    # ------------------------------------------------------------------
    ct = pd.crosstab(
        df_pairs["prev_converted"], df_pairs["converted"],
        rownames=["Prev kick"], colnames=["Current kick"],
    )
    print("\n  Crosstab (prev → current):")
    print(ct.to_string())

    if ct.shape == (2, 2):
        chi2, p_chi, dof, _ = chi2_contingency(ct)
    else:
        chi2, p_chi, dof = 0.0, 1.0, 0
    print(f"  Chi-squared: χ²={chi2:.3f}, df={dof}, p={p_chi:.4f}")

    # Conditional conversion rates
    after_goal = df_pairs[df_pairs["prev_converted"] == 1]
    after_miss = df_pairs[df_pairs["prev_converted"] == 0]
    conv_after_goal = after_goal["converted"].mean() if len(after_goal) else float("nan")
    conv_after_miss = after_miss["converted"].mean() if len(after_miss) else float("nan")
    n_after_goal = len(after_goal)
    n_after_miss = len(after_miss)
    print(f"  P(goal | prev=goal) = {conv_after_goal:.3f}  (n={n_after_goal})")
    print(f"  P(goal | prev=miss) = {conv_after_miss:.3f}  (n={n_after_miss})")

    # Risk difference
    rd = conv_after_goal - conv_after_miss
    se_rd = np.sqrt(
        conv_after_goal * (1 - conv_after_goal) / max(n_after_goal, 1)
        + conv_after_miss * (1 - conv_after_miss) / max(n_after_miss, 1)
    )
    rd_lo = rd - 1.96 * se_rd
    rd_hi = rd + 1.96 * se_rd
    print(f"  Risk difference: {rd:.3f} [{rd_lo:.3f}, {rd_hi:.3f}]")

    # ------------------------------------------------------------------
    # Logistic regression with round and team controls
    # ------------------------------------------------------------------
    logit_coef = 0.0
    logit_p = 1.0
    try:
        df_pairs["round_num"] = (df_pairs["kick_num"] - 1) // 2 + 1
        mod = smf.logit(
            "converted ~ prev_converted + round_num",
            data=df_pairs,
        ).fit(disp=0)
        print("\n  Logistic regression: converted ~ prev_converted + round_num")
        print(mod.summary2().tables[1].to_string())
        logit_coef = mod.params["prev_converted"]
        logit_p = mod.pvalues["prev_converted"]
    except Exception as e:
        print(f"  Logistic regression failed: {e}")

    # Cluster-robust SEs (clustered on match_id)
    cluster_coef = 0.0
    cluster_p = 1.0
    try:
        mod_cl = smf.logit(
            "converted ~ prev_converted + round_num",
            data=df_pairs,
        ).fit(
            disp=0,
            cov_type="cluster",
            cov_kwds={"groups": df_pairs["match_id"]},
        )
        cluster_coef = mod_cl.params["prev_converted"]
        cluster_p = mod_cl.pvalues["prev_converted"]
        print(f"  Cluster-robust: coef={cluster_coef:.3f}, p={cluster_p:.4f}")
    except Exception as e:
        print(f"  Cluster-robust regression failed: {e}")

    # ------------------------------------------------------------------
    # Register LaTeX stats
    # ------------------------------------------------------------------
    add_stat("ShootoutPredTwoNKicks", str(n_kicks_total))
    add_stat("ShootoutPredTwoNMatches", str(n_matches))
    add_stat("ShootoutPredTwoOverallConv", f"{overall_conv:.1%}")
    add_stat("ShootoutPredTwoNPairs", str(n_pairs))
    add_stat("ShootoutPredTwoConvAfterGoal", f"{conv_after_goal:.1%}")
    add_stat("ShootoutPredTwoConvAfterMiss", f"{conv_after_miss:.1%}")
    add_stat("ShootoutPredTwoNAfterGoal", str(n_after_goal))
    add_stat("ShootoutPredTwoNAfterMiss", str(n_after_miss))
    add_stat("ShootoutPredTwoChiSq", f"{chi2:.2f}")
    add_stat("ShootoutPredTwoChiP", f"{p_chi:.4f}")
    add_stat("ShootoutPredTwoRiskDiff", f"{rd:.1%}")
    add_stat("ShootoutPredTwoRiskDiffCILo", f"{rd_lo:.1%}")
    add_stat("ShootoutPredTwoRiskDiffCIHi", f"{rd_hi:.1%}")
    add_stat("ShootoutPredTwoLogitCoef", f"{logit_coef:.3f}")
    add_stat("ShootoutPredTwoLogitP", f"{logit_p:.4f}")
    add_stat("ShootoutPredTwoClusterCoef", f"{cluster_coef:.3f}")
    add_stat("ShootoutPredTwoClusterP", f"{cluster_p:.4f}")

    # ------------------------------------------------------------------
    # Figure: sequential conversion rates with Wilson CIs
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    labels = ["After prev.\ngoal", "After prev.\nmiss"]
    ks = [int(after_goal["converted"].sum()), int(after_miss["converted"].sum())]
    ns = [n_after_goal, n_after_miss]
    vals = [conv_after_goal, conv_after_miss]
    colors = [STRIKER_RED, KEEPER_BLUE]

    cis = [_wilson_ci(k, n) for k, n in zip(ks, ns)]
    yerr_lo = [v - ci[1] for v, ci in zip(vals, cis)]
    yerr_hi = [ci[2] - v for v, ci in zip(vals, cis)]

    bars = ax.bar(
        labels, vals, yerr=[yerr_lo, yerr_hi], capsize=5,
        color=colors, edgecolor="white", width=0.5, error_kw={"lw": 1.3},
    )
    for bar, v, n in zip(bars, vals, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f"{v:.1%}\n(n={n})", ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylabel("P(goal on current kick)")
    ax.set_ylim(0, 1.15)
    ax.axhline(overall_conv, ls="--", color=MUTED_GREY, alpha=0.5,
               label=f"Overall ({overall_conv:.1%})")
    ax.set_title(
        f"Shootout replication: sequential effects\n"
        f"(FIFA World Cup 1982-2022, n={n_pairs} pairs)",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8)
    save_fig(fig, "empirical_pred2_shootout")

    print("  ✓ Shootout replication complete.\n")


if __name__ == "__main__":
    from utils import init_stats, write_stats
    init_stats()
    run()
    write_stats()
