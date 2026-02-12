#!/usr/bin/env python3
"""
Prediction 6 — Pre-commitment reduces interference (motor purity).
===================================================================
Uses the Kaggle dataset to test whether "natural-side" shots (kicker foot
congruent with shot direction, suggesting pre-committed motor programme)
have higher conversion rates than "unnatural-side" shots.

Right-footed kickers shooting left = natural (cross-body power shot).
Left-footed kickers shooting right = natural (cross-body power shot).
Opposite = unnatural, likely more reactive / conflicted.

Also tests: does kicker-goalie side matching interact with foot congruence?

Framework prediction: natural-side kicks reflect a single, clean motor
programme (reduced shared-subspace interference) → higher conversion.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
import statsmodels.formula.api as smf

from utils import (
    load_kaggle, setup_plotting, save_fig, add_stat,
    KEEPER_BLUE, STRIKER_RED, FIELD_GREEN, WARN_AMBER, MUTED_GREY,
)


def classify_congruence(row):
    """
    Classify shot as natural (cross-body) or unnatural.
    Right foot → natural target is Left (cross-body power).
    Left foot  → natural target is Right (cross-body power).
    Centre shots are ambiguous (classified separately).
    """
    foot, side = row["Kicker_Foot"], row["Kicker_Side"]
    if side == "C":
        return "Centre"
    if (foot == "R" and side == "L") or (foot == "L" and side == "R"):
        return "Natural\n(cross-body)"
    return "Unnatural\n(same-side)"


def run():
    print("\n" + "=" * 60)
    print("PREDICTION 6: Pre-commitment / motor purity")
    print("=" * 60)

    setup_plotting()
    df = load_kaggle()

    # ------------------------------------------------------------------
    # 1. Classify each kick
    # ------------------------------------------------------------------
    df["congruence"] = df.apply(classify_congruence, axis=1)

    grp = df.groupby("congruence")["Outcome"].agg(["sum", "count"]).reset_index()
    grp.columns = ["congruence", "goals", "total"]
    grp["conversion"] = grp["goals"] / grp["total"]
    print("\nConversion rate by foot-side congruence:")
    print(grp.to_string(index=False))

    natural = df[df["congruence"] == "Natural\n(cross-body)"]
    unnatural = df[df["congruence"] == "Unnatural\n(same-side)"]
    centre = df[df["congruence"] == "Centre"]

    # ------------------------------------------------------------------
    # 2. Fisher exact: natural vs unnatural
    # ------------------------------------------------------------------
    table = np.array([
        [natural["Outcome"].sum(), len(natural) - natural["Outcome"].sum()],
        [unnatural["Outcome"].sum(), len(unnatural) - unnatural["Outcome"].sum()],
    ])
    or_val, p_fish = fisher_exact(table, alternative="greater")
    print(f"\nFisher exact (natural > unnatural): OR={or_val:.3f}, p={p_fish:.4f}")

    # ------------------------------------------------------------------
    # 3. Logistic regression controlling for Goalie_Side match
    # ------------------------------------------------------------------
    df_lr = df[df["congruence"] != "Centre"].copy()
    df_lr["natural"] = (df_lr["congruence"] == "Natural\n(cross-body)").astype(int)
    df_lr["gk_match"] = (df_lr["Kicker_Side"] == df_lr["Goalie_Side"]).astype(int)

    mod = smf.logit("Outcome ~ natural + gk_match", data=df_lr).fit(disp=0)
    print("\nLogistic regression: Outcome ~ natural + gk_match")
    print(mod.summary2().tables[1].to_string())

    # ------------------------------------------------------------------
    # 4. Register stats
    # ------------------------------------------------------------------
    conv_nat = natural["Outcome"].mean()
    conv_unnat = unnatural["Outcome"].mean()
    conv_centre = centre["Outcome"].mean() if len(centre) > 0 else float("nan")

    add_stat("PredSixNNatural", str(len(natural)))
    add_stat("PredSixNUnnatural", str(len(unnatural)))
    add_stat("PredSixNCentre", str(len(centre)))
    add_stat("PredSixConvNatural", f"{conv_nat:.1%}")
    add_stat("PredSixConvUnnatural", f"{conv_unnat:.1%}")
    add_stat("PredSixConvCentre", f"{conv_centre:.1%}")
    add_stat("PredSixFisherOR", f"{or_val:.2f}")
    add_stat("PredSixFisherP", f"{p_fish:.4f}")
    add_stat("PredSixLogitNatCoef", f"{mod.params['natural']:.3f}")
    add_stat("PredSixLogitNatP", f"{mod.pvalues['natural']:.4f}")
    add_stat("PredSixLogitMatchCoef", f"{mod.params['gk_match']:.3f}")
    add_stat("PredSixLogitMatchP", f"{mod.pvalues['gk_match']:.4f}")

    # ------------------------------------------------------------------
    # 5. Figure — conversion by congruence
    # ------------------------------------------------------------------
    order = ["Natural\n(cross-body)", "Centre", "Unnatural\n(same-side)"]
    colors = [FIELD_GREEN, WARN_AMBER, STRIKER_RED]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    for i, (label, col) in enumerate(zip(order, colors)):
        sub = grp[grp["congruence"] == label]
        if len(sub) > 0:
            c = sub["conversion"].values[0]
            n = sub["total"].values[0]
            bar = ax.bar(i, c, color=col, edgecolor="white", linewidth=1.2, width=0.6)
            ax.text(i, c + 0.015, f"{c:.1%}\n(n={n})", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("Conversion rate")
    ax.set_ylim(0, 1.12)
    ax.set_title("Prediction 6: Motor purity (foot–side congruence)",
                 fontsize=11, fontweight="bold")
    overall = df["Outcome"].mean()
    ax.axhline(overall, ls="--", color=MUTED_GREY, alpha=0.5, label=f"Overall ({overall:.1%})")
    ax.legend(fontsize=8)
    save_fig(fig, "empirical_pred6_congruence")

    # ------------------------------------------------------------------
    # 6. Figure — interaction: congruence × gk_match
    # ------------------------------------------------------------------
    df_lr["gk_label"] = df_lr["gk_match"].map({0: "Keeper wrong", 1: "Keeper correct"})
    df_lr["cong_label"] = df_lr["natural"].map({1: "Natural", 0: "Unnatural"})

    interact = df_lr.groupby(["cong_label", "gk_label"])["Outcome"].agg(["mean", "count"]).reset_index()

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    x = np.arange(2)
    width = 0.32
    for i, gk in enumerate(["Keeper wrong", "Keeper correct"]):
        sub = interact[interact["gk_label"] == gk]
        vals = [sub[sub["cong_label"] == c]["mean"].values[0] if len(sub[sub["cong_label"] == c]) > 0 else 0
                for c in ["Natural", "Unnatural"]]
        ns = [sub[sub["cong_label"] == c]["count"].values[0] if len(sub[sub["cong_label"] == c]) > 0 else 0
              for c in ["Natural", "Unnatural"]]
        col = KEEPER_BLUE if gk == "Keeper correct" else STRIKER_RED
        bars = ax2.bar(x + i * width, vals, width, label=gk, color=col,
                       edgecolor="white", linewidth=1)
        for j, (v, n) in enumerate(zip(vals, ns)):
            ax2.text(x[j] + i * width, v + 0.015, f"{v:.0%}\n({n})",
                     ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x + width / 2)
    ax2.set_xticklabels(["Natural\n(cross-body)", "Unnatural\n(same-side)"], fontsize=9)
    ax2.set_ylabel("Conversion rate")
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Prediction 6: Congruence × Keeper match",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)
    save_fig(fig2, "empirical_pred6_interaction")

    print("  ✓ Prediction 6 complete.\n")


if __name__ == "__main__":
    from utils import init_stats, write_stats
    init_stats()
    run()
    write_stats()
