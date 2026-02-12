"""
Shared utilities for penalty kick analysis scripts.
Handles data loading, LaTeX stat macro generation, and common plotting setup.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths (all relative to the project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "manuscript", "figures")
STATS_FILE = os.path.join(PROJECT_ROOT, "manuscript", "generated_stats.tex")

KAGGLE_CSV = os.path.join(DATA_DIR, "kaggle_penalty_kicks_2020_2025.csv")
BUNDESLIGA_CSV = os.path.join(DATA_DIR, "bundesliga_penalties_1963_2017.csv")


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_kaggle():
    """Load and return the Kaggle penalty-kick dataset (2020-2025)."""
    df = pd.read_csv(KAGGLE_CSV)
    df["Country"] = df["Country"].str.strip()
    return df


def load_bundesliga():
    """Load and return the Bundesliga penalty dataset (1963-2017)."""
    df = pd.read_csv(BUNDESLIGA_CSV)
    # Recode result to English and to binary (goal=1, save/miss=0)
    result_map = {
        "Tor": "Goal",
        "gehalten": "Save",
        "vorbei": "Miss",
        "drüber": "Miss (high)",
        "Latte": "Miss (crossbar)",
        "Pfosten": "Miss (post)",
    }
    df["result_en"] = df["result"].map(result_map)
    df["goal"] = (df["result"] == "Tor").astype(int)
    return df


# ---------------------------------------------------------------------------
# LaTeX stats-macro helpers
# ---------------------------------------------------------------------------
_stat_lines: list[str] = []


def init_stats():
    """Clear the accumulated stat lines (called once at the start of run_all)."""
    global _stat_lines
    _stat_lines = []


def add_stat(name: str, value: str):
    """
    Register a LaTeX command \\name that expands to *value*.
    name should be a valid LaTeX command name (letters only, no backslash).
    """
    # Escape special LaTeX characters in value
    safe = (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("#", "\\#")
        .replace("$", "\\$")
    )
    # \% produces a literal percent sign in LaTeX
    safe = safe.replace("%", "\\%")
    _stat_lines.append(f"\\newcommand{{\\{name}}}{{{safe}}}")


def write_stats():
    """Flush all accumulated stat macros to the generated_stats.tex file."""
    header = (
        "% ============================================================\n"
        "% AUTO-GENERATED — do not edit by hand.\n"
        "% Regenerated every time `make` or `run_all.py` is executed.\n"
        "% ============================================================\n"
    )
    with open(STATS_FILE, "w") as f:
        f.write(header)
        f.write("\n".join(_stat_lines))
        f.write("\n")


# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------
# Consistent, publication-quality style
KEEPER_BLUE = "#3B82F6"
STRIKER_RED = "#EF4444"
FIELD_GREEN = "#10B981"
WARN_AMBER = "#F59E0B"
MUTED_GREY = "#6B7280"

PALETTE = [KEEPER_BLUE, STRIKER_RED, FIELD_GREEN, WARN_AMBER, MUTED_GREY]


def setup_plotting():
    """Apply global matplotlib / seaborn style."""
    matplotlib.use("Agg")  # non-interactive backend
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.1,
        rc={
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "axes.edgecolor": MUTED_GREY,
            "axes.labelcolor": "#1F2937",
            "text.color": "#1F2937",
        },
    )


def save_fig(fig, name, tight=True):
    """Save a figure as both PNG and PDF in the manuscript figures directory."""
    ensure_dirs()
    if tight:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Saved {name}.png / .pdf")
