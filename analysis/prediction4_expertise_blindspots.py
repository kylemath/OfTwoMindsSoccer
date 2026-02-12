#!/usr/bin/env python3
"""
Prediction 4 — Gain modulation creates paradoxical expertise blind spots.
=========================================================================
Uses the Bundesliga dataset (goalkeeper experience → save rate) to test
whether the relationship between goalkeeper experience and save rate is
non-monotonic, consistent with increased gain modulation creating rigidity.

Framework prediction: very experienced goalkeepers may show a plateau or
decline in save rate, especially against "unreadable" shots (post/bar hits
suggest well-struck shots to corners).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import statsmodels.api as sm
import statsmodels.formula.api as smf

from utils import (
    load_bundesliga, setup_plotting, save_fig, add_stat,
    KEEPER_BLUE, STRIKER_RED, FIELD_GREEN, WARN_AMBER, MUTED_GREY,
)


def run():
    print("\n" + "=" * 60)
    print("PREDICTION 4: Expertise blind spots (gain modulation)")
    print("=" * 60)

    setup_plotting()
    df = load_bundesliga()

    # ------------------------------------------------------------------
    # 1. Compute save rate per goalkeeper-experience level
    # ------------------------------------------------------------------
    # Binary: 1 = save (gehalten), 0 = goal or miss
    df["saved"] = (df["result"] == "gehalten").astype(int)

    # Group by experience bins
    max_exp = int(df["gkexp"].max())
    exp_group = df.groupby("gkexp").agg(
        n_penalties=("saved", "count"),
        n_saves=("saved", "sum"),
    ).reset_index()
    exp_group["save_rate"] = exp_group["n_saves"] / exp_group["n_penalties"]

    print("\nSave rate by goalkeeper experience (seasons):")
    print(exp_group.to_string(index=False))

    n_total = len(df)
    overall_save = df["saved"].mean()
    add_stat("PredFourNTotal", str(n_total))
    add_stat("PredFourOverallSave", f"{overall_save:.1%}")

    # ------------------------------------------------------------------
    # 2. Logistic regression: save ~ gkexp + gkexp^2 (test quadratic)
    # ------------------------------------------------------------------
    df["gkexp2"] = df["gkexp"] ** 2

    # Linear model
    mod_lin = smf.logit("saved ~ gkexp", data=df).fit(disp=0)
    # Quadratic model
    mod_quad = smf.logit("saved ~ gkexp + gkexp2", data=df).fit(disp=0)

    print("\n--- Linear logistic regression ---")
    print(mod_lin.summary2().tables[1].to_string())
    print(f"AIC = {mod_lin.aic:.1f}")

    print("\n--- Quadratic logistic regression ---")
    print(mod_quad.summary2().tables[1].to_string())
    print(f"AIC = {mod_quad.aic:.1f}")

    # Likelihood ratio test for quadratic term
    lr_stat = -2 * (mod_lin.llf - mod_quad.llf)
    from scipy.stats import chi2
    lr_p = chi2.sf(lr_stat, df=1)
    print(f"\nLR test for quadratic term: χ²={lr_stat:.3f}, p={lr_p:.4f}")

    # Spearman correlation: experience vs save rate (at group level)
    rho, p_spearman = spearmanr(exp_group["gkexp"], exp_group["save_rate"])
    print(f"Spearman ρ (exp vs save rate): {rho:.3f}, p={p_spearman:.4f}")

    # Peak of quadratic
    b0, b1, b2 = mod_quad.params["Intercept"], mod_quad.params["gkexp"], mod_quad.params["gkexp2"]
    if b2 != 0:
        peak_exp = -b1 / (2 * b2)
        print(f"Predicted peak save rate at experience = {peak_exp:.1f} seasons")
    else:
        peak_exp = float("nan")

    add_stat("PredFourLinCoef", f"{mod_lin.params['gkexp']:.4f}")
    add_stat("PredFourLinP", f"{mod_lin.pvalues['gkexp']:.4f}")
    add_stat("PredFourQuadCoef", f"{mod_quad.params['gkexp2']:.5f}")
    add_stat("PredFourQuadP", f"{mod_quad.pvalues['gkexp2']:.4f}")
    add_stat("PredFourLRChi", f"{lr_stat:.2f}")
    add_stat("PredFourLRP", f"{lr_p:.4f}")
    add_stat("PredFourPeakExp", f"{peak_exp:.1f}")
    add_stat("PredFourAICLin", f"{mod_lin.aic:.1f}")
    add_stat("PredFourAICQuad", f"{mod_quad.aic:.1f}")
    add_stat("PredFourSpearmanRho", f"{rho:.3f}")
    add_stat("PredFourSpearmanP", f"{p_spearman:.4f}")

    # ------------------------------------------------------------------
    # 3. Figure — save rate vs experience with logistic fit
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Scatter: group-level save rates (size ∝ n)
    ax.scatter(exp_group["gkexp"], exp_group["save_rate"],
               s=exp_group["n_penalties"] * 0.5, alpha=0.6,
               color=KEEPER_BLUE, edgecolor="white", linewidth=0.5,
               label="Observed (size prop. to n)")

    # Fitted curves
    x_pred = np.linspace(0, max_exp, 200)
    # Linear fit
    logit_lin = mod_lin.params["Intercept"] + mod_lin.params["gkexp"] * x_pred
    y_lin = 1 / (1 + np.exp(-logit_lin))
    ax.plot(x_pred, y_lin, color=MUTED_GREY, ls="--", lw=1.5, label="Linear fit")
    # Quadratic fit
    logit_quad = b0 + b1 * x_pred + b2 * x_pred ** 2
    y_quad = 1 / (1 + np.exp(-logit_quad))
    ax.plot(x_pred, y_quad, color=STRIKER_RED, lw=2, label="Quadratic fit")

    if not np.isnan(peak_exp) and 0 < peak_exp < max_exp:
        ax.axvline(peak_exp, ls=":", color=WARN_AMBER, lw=1.2, alpha=0.7)
        ax.text(peak_exp + 0.3, ax.get_ylim()[1] * 0.95,
                f"Peak ≈ {peak_exp:.0f} seasons", fontsize=8, color=WARN_AMBER)

    ax.set_xlabel("Goalkeeper experience (seasons)")
    ax.set_ylabel("Save rate")
    ax.set_title("Prediction 4: Expertise and save rate", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    save_fig(fig, "empirical_pred4_expertise")

    # ------------------------------------------------------------------
    # 4. Figure — save rate by shot type and experience
    # ------------------------------------------------------------------
    # Split into "readable" (goals = Tor, saves = gehalten) and
    # "unreadable" (post/bar/high misses — suggest well-placed shots)
    df["shot_type"] = df["result_en"].apply(
        lambda x: "Readable" if x in ("Goal", "Save") else "Unreadable\n(post/bar/high)"
    )
    # For "readable" shots, save rate = saves / (goals + saves)
    readable = df[df["shot_type"] == "Readable"].copy()
    readable["exp_bin"] = pd.cut(readable["gkexp"], bins=[0, 3, 6, 9, 99],
                                  labels=["0-3", "4-6", "7-9", "10+"], right=True)
    save_by_bin = readable.groupby("exp_bin", observed=True)["saved"].agg(["mean", "count"]).reset_index()
    save_by_bin.columns = ["Experience bin", "Save rate", "n"]

    fig2, ax2 = plt.subplots(figsize=(4.5, 3.5))
    colors_bin = [KEEPER_BLUE, FIELD_GREEN, WARN_AMBER, STRIKER_RED]
    bars = ax2.bar(save_by_bin["Experience bin"], save_by_bin["Save rate"],
                   color=colors_bin, edgecolor="white", linewidth=1.2)
    for bar, row in zip(bars, save_by_bin.itertuples()):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{row[2]:.1%}\n(n={row[3]})", ha="center", va="bottom", fontsize=8)
    ax2.set_ylabel("Save rate (readable shots)")
    ax2.set_xlabel("Goalkeeper experience (seasons)")
    ax2.set_title("Prediction 4: Save rate by experience bin", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, max(save_by_bin["Save rate"]) + 0.06)
    save_fig(fig2, "empirical_pred4_expertise_bins")

    print("  ✓ Prediction 4 complete.\n")


if __name__ == "__main__":
    from utils import init_stats, write_stats
    init_stats()
    run()
    write_stats()
