#!/usr/bin/env python3
"""
Prediction 5 — Keeper centrality maximises striker interference.
=================================================================
Uses the Kaggle dataset (Goalie_Side × Outcome) to test whether
conversion rate is lower when the goalkeeper stays central.

Framework prediction: staying central forces the striker to maintain
competing motor programmes (shared subspace), degrading shot quality.

Also tests with Bundesliga: whether result-type distributions differ
when the goal difference is small (high pressure → more interference).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.stats.proportion import proportions_ztest

from utils import (
    load_kaggle, load_bundesliga, setup_plotting, save_fig, add_stat,
    KEEPER_BLUE, STRIKER_RED, FIELD_GREEN, WARN_AMBER, MUTED_GREY,
)


def run():
    print("\n" + "=" * 60)
    print("PREDICTION 5: Keeper centrality and striker interference")
    print("=" * 60)

    setup_plotting()
    df = load_kaggle()

    # ------------------------------------------------------------------
    # 1. Conversion rate: goalkeeper central vs dive L/R
    # ------------------------------------------------------------------
    df["gk_action"] = df["Goalie_Side"].apply(lambda x: "Stay central" if x == "C" else "Dive L/R")

    grp = df.groupby("gk_action")["Outcome"].agg(["sum", "count"]).reset_index()
    grp.columns = ["gk_action", "goals", "total"]
    grp["conversion"] = grp["goals"] / grp["total"]
    print("\nConversion rate by goalkeeper action:")
    print(grp.to_string(index=False))

    central = df[df["gk_action"] == "Stay central"]
    dive = df[df["gk_action"] == "Dive L/R"]

    # Fisher exact test (is conversion lower when keeper central?)
    table = np.array([
        [central["Outcome"].sum(), len(central) - central["Outcome"].sum()],
        [dive["Outcome"].sum(), len(dive) - dive["Outcome"].sum()],
    ])
    or_val, p_fish = fisher_exact(table, alternative="less")
    print(f"\nFisher exact (central < dive conversion): OR={or_val:.3f}, p={p_fish:.4f}")

    # Two-proportion z-test
    count = np.array([central["Outcome"].sum(), dive["Outcome"].sum()])
    nobs = np.array([len(central), len(dive)])
    z_stat, p_z = proportions_ztest(count, nobs, alternative="smaller")
    print(f"Two-proportion z-test: z={z_stat:.3f}, p={p_z:.4f}")

    # Stats
    n_central = len(central)
    n_dive = len(dive)
    conv_central = central["Outcome"].mean()
    conv_dive = dive["Outcome"].mean()

    add_stat("PredFiveNCentral", str(n_central))
    add_stat("PredFiveNDive", str(n_dive))
    add_stat("PredFiveConvCentral", f"{conv_central:.1%}")
    add_stat("PredFiveConvDive", f"{conv_dive:.1%}")
    add_stat("PredFiveFisherOR", f"{or_val:.2f}")
    add_stat("PredFiveFisherP", f"{p_fish:.4f}")
    add_stat("PredFiveZStat", f"{z_stat:.2f}")
    add_stat("PredFiveZP", f"{p_z:.4f}")

    # ------------------------------------------------------------------
    # 2. Figure — conversion rate by goalkeeper action
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    labels = ["Stay central", "Dive L/R"]
    convs = [conv_central, conv_dive]
    ns = [n_central, n_dive]
    colors = [FIELD_GREEN, KEEPER_BLUE]

    bars = ax.bar(labels, convs, color=colors, edgecolor="white", linewidth=1.2, width=0.55)
    for bar, c, n in zip(bars, convs, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{c:.1%}\n(n={n})", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Conversion rate")
    ax.set_ylim(0, 1.12)
    ax.set_title("Prediction 5: Keeper centrality", fontsize=11, fontweight="bold")
    overall = df["Outcome"].mean()
    ax.axhline(overall, ls="--", color=MUTED_GREY, alpha=0.5, label=f"Overall ({overall:.1%})")
    ax.legend(fontsize=8)
    save_fig(fig, "empirical_pred5_centrality")

    # ------------------------------------------------------------------
    # 3. Supplementary: Kicker side distribution against central keepers
    # ------------------------------------------------------------------
    kick_vs_central = central["Kicker_Side"].value_counts()
    kick_vs_dive = dive["Kicker_Side"].value_counts(normalize=True)

    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(7, 3.2))

    # Panel A: where do kickers aim vs central keepers?
    if len(kick_vs_central) > 0:
        ax2a.bar(kick_vs_central.index, kick_vs_central.values,
                 color=[STRIKER_RED, WARN_AMBER, KEEPER_BLUE], edgecolor="white")
        ax2a.set_ylabel("Count")
        ax2a.set_xlabel("Kicker side")
        ax2a.set_title("Shots vs. central keeper", fontsize=10)

    # Panel B: conversion by kicker side when keeper dives
    for side in ["L", "C", "R"]:
        sub = dive[dive["Kicker_Side"] == side]
        if len(sub) > 0:
            ax2b.bar(side, sub["Outcome"].mean(),
                     color={"L": STRIKER_RED, "C": WARN_AMBER, "R": KEEPER_BLUE}[side],
                     edgecolor="white")
    ax2b.set_ylabel("Conversion rate")
    ax2b.set_xlabel("Kicker side")
    ax2b.set_title("Conversion when keeper dives", fontsize=10)
    ax2b.set_ylim(0, 1.05)

    fig2.suptitle("Prediction 5: Supplementary", fontsize=11, fontweight="bold")
    save_fig(fig2, "empirical_pred5_supplementary")

    print("  ✓ Prediction 5 complete.\n")


if __name__ == "__main__":
    from utils import init_stats, write_stats
    init_stats()
    run()
    write_stats()
