#!/bin/bash
# ============================================================
# Compile script for: The Subspace Penalty Kick
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================"
echo "  The Subspace Penalty Kick - Build Script"
echo "============================================"
echo ""

# --- Step 1: Generate Figures ---
echo "[1/3] Generating figures..."
if command -v python3 &> /dev/null; then
    python3 generate_figures.py
elif command -v python &> /dev/null; then
    python generate_figures.py
else
    echo "WARNING: Python not found. Skipping figure generation."
    echo "         Install Python 3 and run: pip install matplotlib numpy"
fi
echo ""

# --- Step 2: Compile LaTeX ---
echo "[2/3] Compiling LaTeX..."
if command -v pdflatex &> /dev/null; then
    # First pass
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

    # Bibliography
    if command -v bibtex &> /dev/null; then
        bibtex main > /dev/null 2>&1 || true
    else
        echo "WARNING: bibtex not found. References may not render."
    fi

    # Second and third pass for cross-references
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 || true

    echo "PDF generated: main.pdf"
else
    echo "ERROR: pdflatex not found."
    echo "Install a TeX distribution:"
    echo "  macOS:  brew install --cask mactex"
    echo "  Ubuntu: sudo apt install texlive-full"
    echo "  or use: https://www.overleaf.com"
    exit 1
fi
echo ""

# --- Step 3: Clean up ---
echo "[3/3] Cleaning auxiliary files..."
rm -f main.aux main.bbl main.blg main.log main.out main.toc main.nav main.snm 2>/dev/null || true
echo ""

# --- Done ---
echo "============================================"
echo "  Build complete!"
echo "  Output: ${SCRIPT_DIR}/main.pdf"
echo "============================================"

# Open PDF if on macOS
if [[ "$OSTYPE" == "darwin"* ]] && [ -f main.pdf ]; then
    echo "Opening PDF..."
    open main.pdf
fi
