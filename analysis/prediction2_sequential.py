#!/usr/bin/env python3
"""
Prediction 2 — Task-history ghosts across successive kicks (exploratory).
==========================================================================
Uses the Bundesliga dataset to test whether, in matches with multiple
penalties, the outcome of penalty n-1 predicts penalty n.

Limitations: these are in-game penalties (not shootouts), separated by
many minutes of play. Results are therefore exploratory and likely
conservative — if sequential effects appear even across long inter-penalty
intervals, this would be notable.

Also analyses the multi-league dataset for goalkeeper side repetition patterns
within consecutive same-goalkeeper sequences.

Revision: now disaggregates saves vs misses, adds covariate-controlled
logistic regression, reports risk difference with 95% CI, and adds
Wilson CIs to bar charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, binomtest
import statsmodels.formula.api as smf

from utils import (
    load_bundesliga, load_kaggle, setup_plotting, save_fig, add_stat,
    KEEPER_BLUE, STRIKER_RED, FIELD_GREEN, WARN_AMBER, MUTED_GREY,
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
    print("PREDICTION 2: Sequential effects (exploratory)")
    print("=" * 60)

    setup_plotting()

    # ==================================================================
    # Part A: Bundesliga — same-match sequential penalty outcomes
    # ==================================================================
    df = load_bundesliga()

    # Create match identifiers using date + sorted club pair
    df["club_pair"] = df.apply(
        lambda r: "_".join(sorted([str(r["gkclub"]), str(r["ptclub"])])), axis=1
    )
    df["match_id"] = df["date"].astype(str) + "_" + df["club_pair"]

    # Sort by match and minute
    df = df.sort_values(["match_id", "minute"]).reset_index(drop=True)

    # Keep only multi-penalty matches
    match_counts = df.groupby("match_id").size()
    multi_matches = match_counts[match_counts >= 2].index
    df_multi = df[df["match_id"].isin(multi_matches)].copy()

    n_multi_matches = len(multi_matches)
    n_multi_penalties = len(df_multi)
    print(f"\nMulti-penalty matches: {n_multi_matches}")
    print(f"Penalties in multi-penalty matches: {n_multi_penalties}")

    # For each penalty, get the previous penalty's outcome in the same match
    df_multi["prev_goal"] = df_multi.groupby("match_id")["goal"].shift(1)
    df_pairs = df_multi.dropna(subset=["prev_goal"]).copy()
    df_pairs["prev_goal"] = df_pairs["prev_goal"].astype(int)

    # Test: does prev penalty outcome predict current outcome?
    ct = pd.crosstab(df_pairs["prev_goal"], df_pairs["goal"],
                     rownames=["Prev penalty"], colnames=["Current penalty"])
    print("\nCrosstab (prev penalty → current penalty):")
    print(ct)

    if ct.shape == (2, 2):
        chi2, p_chi, dof, _ = chi2_contingency(ct)
        print(f"Chi-squared: χ²={chi2:.3f}, df={dof}, p={p_chi:.4f}")
    else:
        chi2, p_chi = 0.0, 1.0
        print("Insufficient categories for chi-squared test.")

    # Conditional probabilities
    goal_after_goal = df_pairs[df_pairs["prev_goal"] == 1]["goal"].mean()
    goal_after_nongoal = df_pairs[df_pairs["prev_goal"] == 0]["goal"].mean()
    n_after_goal = len(df_pairs[df_pairs["prev_goal"] == 1])
    n_after_nongoal = len(df_pairs[df_pairs["prev_goal"] == 0])

    print(f"\nP(goal | prev=goal) = {goal_after_goal:.3f}  (n={n_after_goal})")
    print(f"P(goal | prev=non-goal) = {goal_after_nongoal:.3f}  (n={n_after_nongoal})")

    # ------------------------------------------------------------------
    # Risk difference + 95% CI
    # ------------------------------------------------------------------
    risk_diff = goal_after_goal - goal_after_nongoal
    se_rd = np.sqrt(
        goal_after_goal * (1 - goal_after_goal) / n_after_goal +
        goal_after_nongoal * (1 - goal_after_nongoal) / n_after_nongoal
    )
    rd_ci_lo = risk_diff - 1.96 * se_rd
    rd_ci_hi = risk_diff + 1.96 * se_rd
    print(f"Risk difference: {risk_diff:.3f} [{rd_ci_lo:.3f}, {rd_ci_hi:.3f}]")

    # ------------------------------------------------------------------
    # Disaggregate: save vs miss (prev non-goal breakdown)
    # ------------------------------------------------------------------
    df_multi["saved"] = (df_multi["result"] == "gehalten").astype(int)
    df_multi["prev_saved"] = df_multi.groupby("match_id")["saved"].shift(1)
    df_pairs_detail = df_multi.dropna(subset=["prev_goal"]).copy()
    df_pairs_detail["prev_goal"] = df_pairs_detail["prev_goal"].astype(int)
    df_pairs_detail["prev_saved"] = df_pairs_detail["prev_saved"].astype(int)

    # Previous was a save (goalkeeper stopped it)
    prev_save_only = df_pairs_detail[(df_pairs_detail["prev_goal"] == 0) & (df_pairs_detail["prev_saved"] == 1)]
    # Previous was a miss (kicker missed)
    prev_miss_only = df_pairs_detail[(df_pairs_detail["prev_goal"] == 0) & (df_pairs_detail["prev_saved"] == 0)]

    goal_after_save_only = prev_save_only["goal"].mean() if len(prev_save_only) > 0 else float("nan")
    goal_after_miss = prev_miss_only["goal"].mean() if len(prev_miss_only) > 0 else float("nan")
    n_after_save_only = len(prev_save_only)
    n_after_miss = len(prev_miss_only)

    print(f"\nDisaggregated non-goals:")
    print(f"  P(goal | prev=save) = {goal_after_save_only:.3f}  (n={n_after_save_only})")
    print(f"  P(goal | prev=miss) = {goal_after_miss:.3f}  (n={n_after_miss})")

    # ------------------------------------------------------------------
    # Logistic regression with covariates
    # ------------------------------------------------------------------
    df_pairs_lr = df_pairs.copy()
    # Ensure numeric types
    for col in ["goaldiff", "minute", "homegame", "ptexp", "gkexp"]:
        if col in df_pairs_lr.columns:
            df_pairs_lr[col] = pd.to_numeric(df_pairs_lr[col], errors="coerce")

    df_pairs_lr = df_pairs_lr.dropna(subset=["goaldiff", "minute", "homegame"])

    try:
        mod = smf.logit("goal ~ prev_goal + goaldiff + minute + homegame", data=df_pairs_lr).fit(disp=0)
        print("\nLogistic regression: goal ~ prev_goal + goaldiff + minute + homegame")
        print(mod.summary2().tables[1].to_string())
        logit_coef = mod.params["prev_goal"]
        logit_p = mod.pvalues["prev_goal"]
    except Exception as e:
        print(f"Logistic regression failed: {e}")
        logit_coef = 0.0
        logit_p = 1.0

    # ------------------------------------------------------------------
    # Register LaTeX stats
    # ------------------------------------------------------------------
    add_stat("PredTwoNMultiMatches", str(n_multi_matches))
    add_stat("PredTwoNPairs", str(len(df_pairs)))
    add_stat("PredTwoGoalAfterGoal", f"{goal_after_goal:.1%}")
    add_stat("PredTwoGoalAfterSave", f"{goal_after_nongoal:.1%}")
    add_stat("PredTwoNAfterGoal", str(n_after_goal))
    add_stat("PredTwoNAfterSave", str(n_after_nongoal))
    add_stat("PredTwoChiSq", f"{chi2:.2f}")
    add_stat("PredTwoChiP", f"{p_chi:.4f}")
    add_stat("PredTwoRiskDiff", f"{risk_diff:.1%}")
    add_stat("PredTwoRiskDiffCILo", f"{rd_ci_lo:.1%}")
    add_stat("PredTwoRiskDiffCIHi", f"{rd_ci_hi:.1%}")
    add_stat("PredTwoLogitCoef", f"{logit_coef:.3f}")
    add_stat("PredTwoLogitP", f"{logit_p:.4f}")
    add_stat("PredTwoGoalAfterSaveOnly", f"{goal_after_save_only:.1%}")
    add_stat("PredTwoGoalAfterMiss", f"{goal_after_miss:.1%}")

    # ==================================================================
    # Part B: Multi-league — goalkeeper side repetition
    # ==================================================================
    dfk = load_kaggle()

    # Check for consecutive same-goalkeeper penalties (proxy for same match)
    dfk["prev_gk"] = dfk["goalkeeper_name"].shift(1)
    dfk["prev_gk_side"] = dfk["Goalie_Side"].shift(1)
    dfk["prev_country"] = dfk["Country"].shift(1)

    same_match = dfk[
        (dfk["goalkeeper_name"] == dfk["prev_gk"]) &
        (dfk["Country"] == dfk["prev_country"])
    ].copy()

    if len(same_match) > 0:
        same_match["gk_repeated"] = (same_match["Goalie_Side"] == same_match["prev_gk_side"]).astype(int)
        repeat_rate = same_match["gk_repeated"].mean()
        n_same = len(same_match)
        n_repeat = same_match["gk_repeated"].sum()
        # Test against chance = 1/3 (three options: L, C, R)
        btest = binomtest(n_repeat, n_same, 1/3, alternative="greater")
        p_binom = btest.pvalue

        print(f"\nMulti-league same-goalkeeper side repetition:")
        print(f"  Repeat rate: {repeat_rate:.1%} ({n_repeat}/{n_same})")
        print(f"  Binomial test vs. 1/3 chance: p={p_binom:.4f}")

        add_stat("PredTwoKaggleNSameMatch", str(n_same))
        add_stat("PredTwoKaggleRepeatRate", f"{repeat_rate:.1%}")
        add_stat("PredTwoKaggleBinomP", f"{p_binom:.4f}")
    else:
        print("\nNo consecutive same-goalkeeper penalties found in multi-league data.")
        add_stat("PredTwoKaggleNSameMatch", "0")
        add_stat("PredTwoKaggleRepeatRate", "N/A")
        add_stat("PredTwoKaggleBinomP", "N/A")

    # ==================================================================
    # Figure — conditional probabilities with 95% Wilson CIs
    # ==================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel A: Bundesliga sequential
    labels_a = ["After prev.\ngoal", "After prev.\nnon-goal"]
    k_a = [df_pairs[df_pairs["prev_goal"] == 1]["goal"].sum(),
           df_pairs[df_pairs["prev_goal"] == 0]["goal"].sum()]
    ns_a = [n_after_goal, n_after_nongoal]
    vals_a = [goal_after_goal, goal_after_nongoal]
    colors_a = [STRIKER_RED, KEEPER_BLUE]

    # Wilson CIs
    cis_a = [_wilson_ci(int(k), n) for k, n in zip(k_a, ns_a)]
    yerr_lo_a = [v - ci[1] for v, ci in zip(vals_a, cis_a)]
    yerr_hi_a = [ci[2] - v for v, ci in zip(vals_a, cis_a)]

    bars_a = ax1.bar(labels_a, vals_a, yerr=[yerr_lo_a, yerr_hi_a], capsize=4,
                     color=colors_a, edgecolor="white", width=0.55, error_kw={"lw": 1.2})
    for bar, v, n in zip(bars_a, vals_a, ns_a):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                 f"{v:.1%}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("P(goal on current penalty)")
    ax1.set_ylim(0, 1.1)
    baseline = df["goal"].mean()
    ax1.axhline(baseline, ls="--", color=MUTED_GREY, alpha=0.5, label=f"Overall ({baseline:.1%})")
    ax1.set_title("A. Bundesliga: sequential\npenalty outcomes", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)

    # Panel B: Multi-league goalkeeper repetition
    if len(same_match) > 0:
        labels_b = ["Repeat\nside", "Switch\nside"]
        vals_b = [repeat_rate, 1 - repeat_rate]
        colors_b = [WARN_AMBER, FIELD_GREEN]
        # CIs for repeat/switch
        ci_rep = _wilson_ci(n_repeat, n_same)
        ci_sw = _wilson_ci(n_same - n_repeat, n_same)
        yerr_lo_b = [repeat_rate - ci_rep[1], (1 - repeat_rate) - ci_sw[1]]
        yerr_hi_b = [ci_rep[2] - repeat_rate, ci_sw[2] - (1 - repeat_rate)]

        bars_b = ax2.bar(labels_b, vals_b, yerr=[yerr_lo_b, yerr_hi_b], capsize=4,
                         color=colors_b, edgecolor="white", width=0.55, error_kw={"lw": 1.2})
        for bar, v in zip(bars_b, vals_b):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                     f"{v:.1%}", ha="center", va="bottom", fontsize=9)
        ax2.axhline(1/3, ls="--", color=MUTED_GREY, alpha=0.5, label="Chance (1/3)")
        ax2.set_ylabel("Proportion of same-GK pairs")
        ax2.set_ylim(0, 1.1)
        ax2.set_title(f"B. Multi-league: GK side repetition\n(n={n_same} pairs)",
                      fontsize=10, fontweight="bold")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("B. Multi-league: GK side repetition", fontsize=10, fontweight="bold")

    fig.suptitle("Prediction 2: Sequential effects (exploratory)",
                 fontsize=11, fontweight="bold", y=1.04)
    save_fig(fig, "empirical_pred2_sequential")

    print("  ✓ Prediction 2 complete.\n")


if __name__ == "__main__":
    from utils import init_stats, write_stats
    init_stats()
    run()
    write_stats()
