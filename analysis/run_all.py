#!/usr/bin/env python3
"""
Master analysis script — runs all prediction tests and writes out
LaTeX stat macros + figures consumed by the manuscript build.

Usage:
    python analysis/run_all.py          (from project root)
    make analysis                       (via Makefile)
"""

import sys
import os

# Ensure the analysis package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import init_stats, write_stats, setup_plotting, ensure_dirs

# Import each prediction module
import prediction1_within_vs_cross_axis as pred1
import prediction2_sequential as pred2
import prediction4_expertise_blindspots as pred4
import prediction5_keeper_centrality as pred5
import prediction6_precommitment as pred6


def main():
    print("=" * 60)
    print("  PENALTY KICK SUBSPACE ANALYSIS — FULL PIPELINE")
    print("=" * 60)

    # Initialise shared state
    init_stats()
    setup_plotting()
    ensure_dirs()

    # Run each prediction analysis (order matches manuscript)
    pred1.run()
    pred2.run()
    pred4.run()
    pred5.run()
    pred6.run()

    # Flush all LaTeX stat macros to disk
    write_stats()

    print("=" * 60)
    print("  ALL ANALYSES COMPLETE")
    print(f"  Stats written to manuscript/generated_stats.tex")
    print(f"  Figures written to manuscript/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
