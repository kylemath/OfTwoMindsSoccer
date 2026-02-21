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

Robustness additions:
- Time-decay analysis (delta_minute interaction)
- Kick-n-minus-2 ghost test
- Cluster-robust standard errors
- Three-level bar figure (After Goal / After Save / After Miss)
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

    # ==================================================================
    # Robustness 1: Time-decay analysis (delta_minute interaction)
    # ==================================================================
    print("\n--- Robustness: Time-decay analysis ---")
    df_multi["prev_minute"] = df_multi.groupby("match_id")["minute"].shift(1)
    df_td = df_multi.dropna(subset=["prev_goal", "prev_minute"]).copy()
    df_td["prev_goal"] = df_td["prev_goal"].astype(int)
    df_td["delta_minute"] = pd.to_numeric(df_td["minute"], errors="coerce") - pd.to_numeric(df_td["prev_minute"], errors="coerce")
    df_td = df_td.dropna(subset=["delta_minute"])
    df_td["delta_minute"] = df_td["delta_minute"].clip(lower=0)

    td_mean = df_td["delta_minute"].mean()
    td_median = df_td["delta_minute"].median()
    print(f"  Delta-minute: mean={td_mean:.1f}, median={td_median:.1f}")

    td_interaction_coef = 0.0
    td_interaction_p = 1.0
    try:
        df_td_lr = df_td.copy()
        for col in ["goaldiff", "minute", "homegame"]:
            if col in df_td_lr.columns:
                df_td_lr[col] = pd.to_numeric(df_td_lr[col], errors="coerce")
        df_td_lr = df_td_lr.dropna(subset=["goaldiff", "minute", "homegame"])
        df_td_lr["prev_goal_x_delta"] = df_td_lr["prev_goal"] * df_td_lr["delta_minute"]
        mod_td = smf.logit(
            "goal ~ prev_goal + delta_minute + prev_goal_x_delta + goaldiff + minute + homegame",
            data=df_td_lr,
        ).fit(disp=0)
        print("  Logistic regression with time-decay interaction:")
        print(mod_td.summary2().tables[1].to_string())
        td_interaction_coef = mod_td.params.get("prev_goal_x_delta", 0.0)
        td_interaction_p = mod_td.pvalues.get("prev_goal_x_delta", 1.0)
    except Exception as e:
        print(f"  Time-decay regression failed: {e}")

    # Time-decay figure: 3×2 — bars + sliding window for each subset
    from matplotlib.patches import Patch
    fig_td, axes = plt.subplots(3, 2, figsize=(13, 11))

    # Tag shooting-team perspective
    df_multi["prev_ptclub"] = df_multi.groupby("match_id")["ptclub"].shift(1)
    df_td_fig = df_multi.dropna(subset=["prev_goal", "prev_minute"]).copy()
    df_td_fig["prev_goal"] = df_td_fig["prev_goal"].astype(int)
    df_td_fig["delta_minute"] = (
        pd.to_numeric(df_td_fig["minute"], errors="coerce")
        - pd.to_numeric(df_td_fig["prev_minute"], errors="coerce")
    ).clip(lower=0)
    df_td_fig = df_td_fig.dropna(subset=["delta_minute"])
    df_td_fig["same_team"] = (df_td_fig["ptclub"] == df_td_fig["prev_ptclub"]).astype(int)

    legend_handles = [
        Patch(facecolor=STRIKER_RED, label="Prev = Goal"),
        Patch(facecolor=KEEPER_BLUE, label="Prev = Non-goal"),
    ]

    def _plot_tertile_bars(ax, data, title):
        data = data.copy()
        data["delta_bin"] = pd.qcut(data["delta_minute"], q=3, duplicates="drop")
        grouped = (
            data.groupby(["delta_bin", "prev_goal"], observed=True)["goal"]
            .agg(["mean", "count"]).reset_index()
        )
        bins_u = sorted(data["delta_bin"].unique(), key=lambda x: x.left)
        bin_labels = [f"{b.left:.0f}-{b.right:.0f} min" for b in bins_u]
        x_pos = np.arange(len(bins_u))
        width = 0.35
        for i, bg in enumerate(bins_u):
            sub = grouped[grouped["delta_bin"] == bg]
            for _, row in sub.iterrows():
                offset = -width / 2 if int(row["prev_goal"]) == 0 else width / 2
                colour = KEEPER_BLUE if int(row["prev_goal"]) == 0 else STRIKER_RED
                n_val = int(row["count"])
                p_val, ci_lo, ci_hi = _wilson_ci(int(round(row["mean"] * n_val)), n_val)
                ax.bar(
                    x_pos[i] + offset, p_val, width=width, color=colour,
                    yerr=[[p_val - ci_lo], [ci_hi - p_val]], capsize=3,
                    error_kw={"lw": 1},
                )
                ax.text(
                    x_pos[i] + offset, ci_hi + 0.02,
                    f"n={n_val}", ha="center", va="bottom", fontsize=6.5,
                )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_ylabel("P(goal on current penalty)")
        ax.set_xlabel("Inter-penalty interval")
        ax.set_ylim(0, 1.15)
        ax.legend(handles=legend_handles, fontsize=7, loc="upper right")
        ax.set_title(title, fontsize=10, fontweight="bold")

    def _plot_sliding_window(ax, data, title):
        half_w = 2.5
        delta_max = data["delta_minute"].quantile(0.95)
        centers = np.arange(half_w, delta_max + 0.5, 1.0)
        min_n_window = 5

        for prev_val in [0, 1]:
            sub = data[data["prev_goal"] == prev_val]
            xs, ys, ci_los, ci_his = [], [], [], []
            for c in centers:
                mask = (sub["delta_minute"] >= c - half_w) & (sub["delta_minute"] <= c + half_w)
                window = sub.loc[mask, "goal"]
                if len(window) < min_n_window:
                    continue
                k = int(window.sum())
                n = len(window)
                p, lo, hi = _wilson_ci(k, n)
                xs.append(c)
                ys.append(p)
                ci_los.append(lo)
                ci_his.append(hi)
            xs, ys = np.array(xs), np.array(ys)
            ci_los, ci_his = np.array(ci_los), np.array(ci_his)
            colour = STRIKER_RED if prev_val == 1 else KEEPER_BLUE
            label = "Prev = Goal" if prev_val == 1 else "Prev = Non-goal"
            ax.plot(xs, ys, color=colour, lw=2, label=label, zorder=3)
            ax.fill_between(xs, ci_los, ci_his, color=colour, alpha=0.15, zorder=2)

        baseline = data["goal"].mean()
        ax.axhline(baseline, ls="--", color=MUTED_GREY, alpha=0.6,
                   label=f"Overall ({baseline:.1%})", zorder=1)
        ax.set_ylabel("P(goal on current penalty)")
        ax.set_xlabel("Inter-penalty interval (min, 5-min window)")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(centers[0], centers[-1])
        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")

    n_all = len(df_td_fig)
    n_same = int(df_td_fig["same_team"].sum())
    n_opp = int((df_td_fig["same_team"] == 0).sum())

    subsets = [
        (df_td_fig, "All consecutive penalties", n_all),
        (df_td_fig[df_td_fig["same_team"] == 1], "Same team shoots again", n_same),
        (df_td_fig[df_td_fig["same_team"] == 0], "Opposing team shoots next", n_opp),
    ]
    panel_letters = ["A", "B", "C", "D", "E", "F"]

    for row_idx, (data, label, n) in enumerate(subsets):
        ax_bar = axes[row_idx, 0]
        ax_line = axes[row_idx, 1]
        ltr_bar = panel_letters[row_idx * 2]
        ltr_line = panel_letters[row_idx * 2 + 1]
        _plot_tertile_bars(ax_bar, data, f"{ltr_bar}. {label} — tertiles (n={n})")
        _plot_sliding_window(ax_line, data, f"{ltr_line}. {label} — sliding window (n={n})")

    fig_td.suptitle("Sequential effect by inter-penalty interval",
                    fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig_td, "empirical_pred2_time_decay")

    # ==================================================================
    # Cross-dataset robustness: sequential effect in all three datasets
    # ==================================================================
    print("\n--- Cross-dataset robustness ---")
    from utils import load_shootout

    # --- Bundesliga (already computed above) ---
    bund_pairs = df_pairs.copy()
    bund_pairs["prev_goal"] = bund_pairs["prev_goal"].astype(int)

    # --- Kaggle multi-league: same-GK proxy for same-match pairs ---
    dfk = load_kaggle()
    dfk = dfk.sort_values(["goalkeeper_name", "Country", "Minute"]).reset_index(drop=True)
    dfk["prev_gk"] = dfk["goalkeeper_name"].shift(1)
    dfk["prev_country"] = dfk["Country"].shift(1)
    dfk["prev_minute"] = dfk["Minute"].shift(1)
    dfk["prev_outcome"] = dfk["Outcome"].shift(1)
    kag_pairs = dfk[
        (dfk["goalkeeper_name"] == dfk["prev_gk"])
        & (dfk["Country"] == dfk["prev_country"])
    ].copy()
    kag_pairs["delta_minute"] = kag_pairs["Minute"] - kag_pairs["prev_minute"]
    kag_pairs = kag_pairs[kag_pairs["delta_minute"] > 0].copy()
    kag_pairs["prev_outcome"] = kag_pairs["prev_outcome"].astype(int)

    # --- World Cup shootout: reconstruct alternating order ---
    dfs = load_shootout()
    dfs = dfs.sort_values(["match_id", "key_id"]).reset_index(drop=True)

    def _interleave_shootout(match_df):
        teams = match_df["team_name"].unique()
        if len(teams) != 2:
            return match_df
        t1 = match_df[match_df["team_name"] == teams[0]].reset_index(drop=True)
        t2 = match_df[match_df["team_name"] == teams[1]].reset_index(drop=True)
        t1["kick_num"] = range(1, len(t1) + 1)
        t2["kick_num"] = range(1, len(t2) + 1)
        home_first = int(t1["home_team"].iloc[0]) == 1
        t1["shoot_order"] = t1["kick_num"] * 2 - (1 if home_first else 0)
        t2["shoot_order"] = t2["kick_num"] * 2 - (1 if not home_first else 0)
        return pd.concat([t1, t2]).sort_values("shoot_order")

    dfs_ordered = pd.concat(
        [_interleave_shootout(g) for _, g in dfs.groupby("match_id")]
    ).reset_index(drop=True)
    dfs_ordered["prev_converted"] = dfs_ordered.groupby("match_id")["converted"].shift(1)
    shoot_pairs = dfs_ordered.dropna(subset=["prev_converted"]).copy()
    shoot_pairs["prev_converted"] = shoot_pairs["prev_converted"].astype(int)

    # Compute stats per dataset
    datasets_info = []
    for label, data, prev_col, goal_col in [
        ("Bundesliga\n(in-game, 1963–2017)", bund_pairs, "prev_goal", "goal"),
        ("Multi-league\n(in-game, 2020–2025)", kag_pairs, "prev_outcome", "Outcome"),
        ("World Cup shootouts\n(1982–2022)", shoot_pairs, "prev_converted", "converted"),
    ]:
        after_g = data[data[prev_col] == 1]
        after_ng = data[data[prev_col] == 0]
        p_g = after_g[goal_col].mean() if len(after_g) > 0 else 0.0
        p_ng = after_ng[goal_col].mean() if len(after_ng) > 0 else 0.0
        n_g = len(after_g)
        n_ng = len(after_ng)
        k_g = int(after_g[goal_col].sum()) if len(after_g) > 0 else 0
        k_ng = int(after_ng[goal_col].sum()) if len(after_ng) > 0 else 0
        datasets_info.append((label, p_g, n_g, k_g, p_ng, n_ng, k_ng))
        print(f"  {label.replace(chr(10), ' ')}: "
              f"P(goal|prev=goal)={p_g:.3f} (n={n_g}), "
              f"P(goal|prev=nongoal)={p_ng:.3f} (n={n_ng})")

    # --- Figure: cross-dataset comparison (3×2: bars + sliding window per dataset) ---
    fig_xd, axes_xd = plt.subplots(3, 2, figsize=(13, 11))

    legend_handles_xd = [
        Patch(facecolor=STRIKER_RED, label="Prev = Goal"),
        Patch(facecolor=KEEPER_BLUE, label="Prev = Non-goal"),
    ]

    def _plot_tertile_bars_xd(ax, data, delta_col, prev_col, goal_col, title):
        data = data.copy()
        try:
            data["delta_bin"] = pd.qcut(data[delta_col], q=3, duplicates="drop")
        except ValueError:
            ax.text(0.5, 0.5, "Insufficient\nvariation\nfor tertiles",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=10, fontweight="bold")
            return
        grouped = (
            data.groupby(["delta_bin", prev_col], observed=True)[goal_col]
            .agg(["mean", "count"]).reset_index()
        )
        bins_u = sorted(data["delta_bin"].unique(), key=lambda x: x.left)
        bin_labels = [f"{b.left:.0f}-{b.right:.0f} min" for b in bins_u]
        x_pos = np.arange(len(bins_u))
        width = 0.35
        for i, bg in enumerate(bins_u):
            sub = grouped[grouped["delta_bin"] == bg]
            for _, row in sub.iterrows():
                offset = -width / 2 if int(row[prev_col]) == 0 else width / 2
                colour = KEEPER_BLUE if int(row[prev_col]) == 0 else STRIKER_RED
                n_val = int(row["count"])
                p_val, ci_lo, ci_hi = _wilson_ci(int(round(row["mean"] * n_val)), n_val)
                ax.bar(
                    x_pos[i] + offset, p_val, width=width, color=colour,
                    yerr=[[p_val - ci_lo], [ci_hi - p_val]], capsize=3,
                    error_kw={"lw": 1},
                )
                ax.text(
                    x_pos[i] + offset, ci_hi + 0.02,
                    f"n={n_val}", ha="center", va="bottom", fontsize=6.5,
                )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_ylabel("P(goal on current penalty)")
        ax.set_xlabel("Inter-penalty interval")
        ax.set_ylim(0, 1.15)
        ax.legend(handles=legend_handles_xd, fontsize=7, loc="upper right")
        ax.set_title(title, fontsize=10, fontweight="bold")

    def _plot_sliding_window_xd(ax, data, delta_col, prev_col, goal_col, title):
        half_w = 2.5
        try:
            delta_max = data[delta_col].quantile(0.95)
        except Exception:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=10, fontweight="bold")
            return
        centers = np.arange(half_w, delta_max + 0.5, 1.0)
        if len(centers) < 3:
            ax.text(0.5, 0.5, "Insufficient time\nvariation", ha="center",
                    va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(title, fontsize=10, fontweight="bold")
            return
        min_n_window = 3
        for prev_val in [0, 1]:
            sub = data[data[prev_col] == prev_val]
            all_ys = np.full(len(centers), np.nan)
            all_ci_los = np.full(len(centers), np.nan)
            all_ci_his = np.full(len(centers), np.nan)
            for idx, c in enumerate(centers):
                mask = (sub[delta_col] >= c - half_w) & (sub[delta_col] <= c + half_w)
                window = sub.loc[mask, goal_col]
                if len(window) < min_n_window:
                    continue
                k = int(window.sum())
                n = len(window)
                p, lo, hi = _wilson_ci(k, n)
                all_ys[idx] = p
                all_ci_los[idx] = lo
                all_ci_his[idx] = hi
            # Interpolate small gaps (up to 3 consecutive NaNs)
            s = pd.Series(all_ys)
            s = s.interpolate(method="linear", limit=3)
            all_ys = s.values
            s_lo = pd.Series(all_ci_los).interpolate(method="linear", limit=3)
            s_hi = pd.Series(all_ci_his).interpolate(method="linear", limit=3)
            all_ci_los, all_ci_his = s_lo.values, s_hi.values
            valid = ~np.isnan(all_ys)
            colour = STRIKER_RED if prev_val == 1 else KEEPER_BLUE
            label = "Prev = Goal" if prev_val == 1 else "Prev = Non-goal"
            if valid.any():
                ax.plot(centers[valid], all_ys[valid], color=colour, lw=2,
                        label=label, zorder=3)
                ax.fill_between(centers[valid], all_ci_los[valid],
                                all_ci_his[valid], color=colour, alpha=0.15, zorder=2)
        baseline = data[goal_col].mean()
        ax.axhline(baseline, ls="--", color=MUTED_GREY, alpha=0.6,
                   label=f"Overall ({baseline:.1%})", zorder=1)
        ax.set_ylabel("P(goal on current penalty)")
        ax.set_xlabel("Inter-penalty interval (min, 5-min window)")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(centers[0], centers[-1])
        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")

    # Prepare Bundesliga time-decay data
    bund_td = df_td_fig.copy()

    # Prepare Kaggle time-decay data
    kag_td = kag_pairs.copy()

    # Shootout has no meaningful time intervals — show by kick round instead
    shoot_pairs["prev_team"] = dfs_ordered.groupby("match_id")["team_name"].shift(1)
    shoot_pairs = shoot_pairs.dropna(subset=["prev_team"])
    shoot_pairs["same_team_shoot"] = (
        shoot_pairs["team_name"] == shoot_pairs["prev_team"]
    ).astype(int)

    xd_datasets = [
        (bund_td, "delta_minute", "prev_goal", "goal",
         "Bundesliga (in-game, 1963–2017)"),
        (kag_td, "delta_minute", "prev_outcome", "Outcome",
         "Multi-league (in-game, 2020–2025)"),
    ]

    for row_idx, (data, delta_col, prev_col, goal_col, label) in enumerate(xd_datasets):
        n = len(data)
        ltr_bar = chr(ord("A") + row_idx * 2)
        ltr_line = chr(ord("A") + row_idx * 2 + 1)
        _plot_tertile_bars_xd(
            axes_xd[row_idx, 0], data, delta_col, prev_col, goal_col,
            f"{ltr_bar}. {label} — tertiles (n={n})",
        )
        _plot_sliding_window_xd(
            axes_xd[row_idx, 1], data, delta_col, prev_col, goal_col,
            f"{ltr_line}. {label} — sliding window (n={n})",
        )

    # Row 3: Shootout — bar chart by kick round (early/mid/late) + overall effect
    shoot_round = dfs_ordered.copy()
    shoot_round["round_num"] = shoot_round.groupby("match_id").cumcount() + 1
    shoot_round["prev_conv"] = shoot_round.groupby("match_id")["converted"].shift(1)
    shoot_round = shoot_round.dropna(subset=["prev_conv"]).copy()
    shoot_round["prev_conv"] = shoot_round["prev_conv"].astype(int)
    shoot_round["round_bin"] = pd.cut(
        shoot_round["round_num"],
        bins=[1, 4, 7, 13],
        labels=["Kicks 2-4", "Kicks 5-7", "Kicks 8+"],
    )
    shoot_round = shoot_round.dropna(subset=["round_bin"])

    ax_sb = axes_xd[2, 0]
    grouped_s = (
        shoot_round.groupby(["round_bin", "prev_conv"], observed=True)["converted"]
        .agg(["mean", "count"]).reset_index()
    )
    bins_s = ["Kicks 2-4", "Kicks 5-7", "Kicks 8+"]
    x_pos_s = np.arange(len(bins_s))
    width = 0.35
    for i, bl in enumerate(bins_s):
        sub = grouped_s[grouped_s["round_bin"] == bl]
        for _, row in sub.iterrows():
            offset = -width / 2 if int(row["prev_conv"]) == 0 else width / 2
            colour = KEEPER_BLUE if int(row["prev_conv"]) == 0 else STRIKER_RED
            n_val = int(row["count"])
            p_val, ci_lo, ci_hi = _wilson_ci(int(round(row["mean"] * n_val)), n_val)
            ax_sb.bar(
                x_pos_s[i] + offset, p_val, width=width, color=colour,
                yerr=[[p_val - ci_lo], [ci_hi - p_val]], capsize=3,
                error_kw={"lw": 1},
            )
            ax_sb.text(
                x_pos_s[i] + offset, ci_hi + 0.02,
                f"n={n_val}", ha="center", va="bottom", fontsize=6.5,
            )
    ax_sb.set_xticks(x_pos_s)
    ax_sb.set_xticklabels(bins_s, fontsize=8)
    ax_sb.set_ylabel("P(goal on current penalty)")
    ax_sb.set_xlabel("Kick position in shootout")
    ax_sb.set_ylim(0, 1.15)
    ax_sb.legend(handles=legend_handles_xd, fontsize=7, loc="upper right")
    ax_sb.set_title(f"E. World Cup shootouts (1982–2022) — by round (n={len(shoot_round)})",
                    fontsize=10, fontweight="bold")

    # Panel F: shootout by kick number (line plot)
    ax_sk = axes_xd[2, 1]
    for prev_val in [0, 1]:
        sub = shoot_round[shoot_round["prev_conv"] == prev_val]
        by_kick = sub.groupby("round_num")["converted"].agg(["mean", "sum", "count"])
        by_kick = by_kick[by_kick["count"] >= 5]
        xs = by_kick.index.values
        ys = by_kick["mean"].values
        ci_data = [_wilson_ci(int(r["sum"]), int(r["count"])) for _, r in by_kick.iterrows()]
        ci_los = np.array([c[1] for c in ci_data])
        ci_his = np.array([c[2] for c in ci_data])
        colour = STRIKER_RED if prev_val == 1 else KEEPER_BLUE
        label = "Prev = Goal" if prev_val == 1 else "Prev = Non-goal"
        ax_sk.plot(xs, ys, "o-", color=colour, lw=2, markersize=5, label=label, zorder=3)
        ax_sk.fill_between(xs, ci_los, ci_his, color=colour, alpha=0.15, zorder=2)
    baseline_s = shoot_round["converted"].mean()
    ax_sk.axhline(baseline_s, ls="--", color=MUTED_GREY, alpha=0.6,
                  label=f"Overall ({baseline_s:.1%})", zorder=1)
    ax_sk.set_ylabel("P(goal on current penalty)")
    ax_sk.set_xlabel("Kick number in shootout")
    ax_sk.set_ylim(0, 1.05)
    ax_sk.legend(fontsize=7)
    ax_sk.set_title(f"F. World Cup shootouts — by kick number (n={len(shoot_round)})",
                    fontsize=10, fontweight="bold")

    fig_xd.suptitle("Sequential effect across datasets",
                    fontsize=12, fontweight="bold", y=1.01)
    save_fig(fig_xd, "empirical_pred2_cross_dataset")

    # ==================================================================
    # Robustness 2: Kick-n-minus-2 ghost test
    # ==================================================================
    print("\n--- Robustness: Kick-n-minus-2 ghost test ---")
    df_multi["prev_goal_n2"] = df_multi.groupby("match_id")["goal"].shift(2)
    df_triples = df_multi.dropna(subset=["prev_goal", "prev_goal_n2"]).copy()
    df_triples["prev_goal"] = df_triples["prev_goal"].astype(int)
    df_triples["prev_goal_n2"] = df_triples["prev_goal_n2"].astype(int)
    n_triples = len(df_triples)
    print(f"  Triples available (kick n with n-1 and n-2): {n_triples}")

    ghost_coef = 0.0
    ghost_p = 1.0
    if n_triples >= 10:
        try:
            for col in ["goaldiff", "minute", "homegame"]:
                if col in df_triples.columns:
                    df_triples[col] = pd.to_numeric(df_triples[col], errors="coerce")
            df_triples_lr = df_triples.dropna(subset=["goaldiff", "minute", "homegame"])
            mod_ghost = smf.logit(
                "goal ~ prev_goal + prev_goal_n2 + goaldiff + minute + homegame",
                data=df_triples_lr,
            ).fit(disp=0)
            print("  Ghost logistic regression:")
            print(mod_ghost.summary2().tables[1].to_string())
            ghost_coef = mod_ghost.params.get("prev_goal_n2", 0.0)
            ghost_p = mod_ghost.pvalues.get("prev_goal_n2", 1.0)
        except Exception as e:
            print(f"  Ghost regression failed: {e}")
    else:
        print("  Too few triples for ghost test.")

    # ==================================================================
    # Robustness 3: Cluster-robust standard errors
    # ==================================================================
    print("\n--- Robustness: Cluster-robust SEs ---")
    cluster_coef = 0.0
    cluster_p = 1.0
    try:
        mod_cluster = smf.logit(
            "goal ~ prev_goal + goaldiff + minute + homegame", data=df_pairs_lr
        ).fit(
            disp=0,
            cov_type="cluster",
            cov_kwds={"groups": df_pairs_lr["match_id"]},
        )
        print("  Cluster-robust logistic regression:")
        print(mod_cluster.summary2().tables[1].to_string())
        cluster_coef = mod_cluster.params["prev_goal"]
        cluster_p = mod_cluster.pvalues["prev_goal"]
    except Exception as e:
        print(f"  Cluster-robust regression failed: {e}")

    # ------------------------------------------------------------------
    # Register LaTeX stats (original + robustness)
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

    # Robustness stats
    add_stat("PredTwoTimeDeltaMean", f"{td_mean:.1f}")
    add_stat("PredTwoTimeDeltaMedian", f"{td_median:.1f}")
    add_stat("PredTwoInteractionCoef", f"{td_interaction_coef:.3f}")
    add_stat("PredTwoInteractionP", f"{td_interaction_p:.4f}")
    add_stat("PredTwoNTriples", str(n_triples))
    add_stat("PredTwoGhostCoef", f"{ghost_coef:.3f}")
    add_stat("PredTwoGhostP", f"{ghost_p:.4f}")
    add_stat("PredTwoClusterLogitCoef", f"{cluster_coef:.3f}")
    add_stat("PredTwoClusterLogitP", f"{cluster_p:.4f}")

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Panel A: Bundesliga sequential — THREE bars: After Goal / After Save / After Miss
    labels_a = ["After prev.\ngoal", "After prev.\nsave", "After prev.\nmiss"]
    k_goal = int(df_pairs[df_pairs["prev_goal"] == 1]["goal"].sum())
    k_save = int(prev_save_only["goal"].sum()) if n_after_save_only > 0 else 0
    k_miss = int(prev_miss_only["goal"].sum()) if n_after_miss > 0 else 0
    k_a = [k_goal, k_save, k_miss]
    ns_a = [n_after_goal, n_after_save_only, n_after_miss]
    vals_a = [goal_after_goal,
              goal_after_save_only if n_after_save_only > 0 else 0.0,
              goal_after_miss if n_after_miss > 0 else 0.0]
    colors_a = [STRIKER_RED, KEEPER_BLUE, WARN_AMBER]

    cis_a = [_wilson_ci(k, n) for k, n in zip(k_a, ns_a)]
    yerr_lo_a = [v - ci[1] for v, ci in zip(vals_a, cis_a)]
    yerr_hi_a = [ci[2] - v for v, ci in zip(vals_a, cis_a)]

    bars_a = ax1.bar(labels_a, vals_a, yerr=[yerr_lo_a, yerr_hi_a], capsize=4,
                     color=colors_a, edgecolor="white", width=0.5, error_kw={"lw": 1.2})
    for bar, v, n in zip(bars_a, vals_a, ns_a):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                 f"{v:.1%}\n(n={n})", ha="center", va="bottom", fontsize=8)
    ax1.set_ylabel("P(goal on current penalty)")
    ax1.set_ylim(0, 1.15)
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
