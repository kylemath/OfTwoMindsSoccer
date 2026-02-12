# ============================================================
# TaskNaturePaper — Reproducible Build
# ============================================================
# `make`           → run analysis + compile manuscript
# `make analysis`  → run Python analysis scripts only
# `make manuscript` → compile LaTeX only (assumes stats exist)
# `make clean`     → remove build artefacts
# ============================================================

VENV       := .venv
PYTHON     := $(VENV)/bin/python3
PIP        := $(VENV)/bin/pip
ANALYSIS   := analysis/run_all.py
TEX_DIR    := manuscript
TEX_MAIN   := $(TEX_DIR)/main.tex
STATS_FILE := $(TEX_DIR)/generated_stats.tex

# Data files that analysis depends on
DATA_DEPS  := data/kaggle_penalty_kicks_2020_2025.csv \
              data/bundesliga_penalties_1963_2017.csv

# All analysis scripts (rebuild if any change)
ANALYSIS_SCRIPTS := $(wildcard analysis/*.py)

# Generated figures (representative sample — the Python scripts produce these)
GEN_FIGS := $(TEX_DIR)/figures/empirical_pred1_conversion_by_mismatch.pdf \
            $(TEX_DIR)/figures/empirical_pred1_heatmap.pdf \
            $(TEX_DIR)/figures/empirical_pred2_sequential.pdf \
            $(TEX_DIR)/figures/empirical_pred4_expertise.pdf \
            $(TEX_DIR)/figures/empirical_pred4_expertise_bins.pdf \
            $(TEX_DIR)/figures/empirical_pred5_centrality.pdf \
            $(TEX_DIR)/figures/empirical_pred6_congruence.pdf \
            $(TEX_DIR)/figures/empirical_pred6_interaction.pdf

# ============================================================
# Default target: full rebuild
# ============================================================
.PHONY: all analysis manuscript clean venv

all: analysis manuscript

# ============================================================
# Virtual environment
# ============================================================
$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r analysis/requirements.txt

venv: $(VENV)/bin/activate

# ============================================================
# Analysis: run Python scripts → produce figures + stats
# ============================================================
$(STATS_FILE) $(GEN_FIGS) &: $(ANALYSIS_SCRIPTS) $(DATA_DEPS) | $(VENV)/bin/activate
	@echo "──────────────────────────────────────────"
	@echo "  Running analysis pipeline..."
	@echo "──────────────────────────────────────────"
	$(PYTHON) $(ANALYSIS)

analysis: $(STATS_FILE)

# ============================================================
# Manuscript: compile LaTeX (depends on stats + figures)
# ============================================================
$(TEX_DIR)/main.pdf: $(TEX_MAIN) $(STATS_FILE) $(GEN_FIGS) $(wildcard $(TEX_DIR)/figures/*.png) $(wildcard $(TEX_DIR)/*.bib)
	@echo "──────────────────────────────────────────"
	@echo "  Compiling manuscript..."
	@echo "──────────────────────────────────────────"
	cd $(TEX_DIR) && pdflatex -interaction=nonstopmode main.tex
	cd $(TEX_DIR) && bibtex main || true
	cd $(TEX_DIR) && pdflatex -interaction=nonstopmode main.tex
	cd $(TEX_DIR) && pdflatex -interaction=nonstopmode main.tex

manuscript: $(TEX_DIR)/main.pdf

# ============================================================
# Clean build artefacts (keeps data + venv)
# ============================================================
clean:
	cd $(TEX_DIR) && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz main.pdf
	rm -f $(STATS_FILE)
	rm -f $(TEX_DIR)/figures/empirical_*.png $(TEX_DIR)/figures/empirical_*.pdf
