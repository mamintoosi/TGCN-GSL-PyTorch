#!/bin/bash
# ============================================================
# Regenerate All Publication Figures with Updated Model Names
# ============================================================
# This script regenerates all figures with the new naming:
#   NoGraph         -> T-GCN-NoSpatial
#   MultiGraph_fixed -> T-GCN-MultiGSL
#   GatedMulti      -> T-GCN-MultiGSL-Mix
#
# No experiments need to be rerun — only display labels changed.
# ============================================================

set -e

REPO="/data/git/mamintoosi/TGCN-GSL-PyTorch"
PYTHON="/data/python-envs/pytorch/bin/python"

cd "$REPO"

echo "========================================"
echo "Regenerating Publication Figures"
echo "========================================"

echo ""
echo "--- Phase 1: Main figures (fig1-fig7) ---"
$PYTHON paper/generate_figures.py
echo ""

echo "--- Phase 2: Extra figures (fig8-fig9) ---"
$PYTHON paper/generate_figures_extra.py
echo ""

echo "--- Phase 3: Compile LaTeX ---"
cd paper
export PATH="/data/texlive/2026/bin/x86_64-linux:$PATH"
pdflatex -interaction=nonstopmode sn-article.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode sn-article.tex > /dev/null 2>&1
echo "PDF compiled: paper/sn-article.pdf ($(stat -c%s sn-article.pdf) bytes)"

echo ""
echo "========================================"
echo "ALL DONE"
echo "Output: paper/figures/*.pdf"
echo "PDF:    paper/sn-article.pdf"
echo "========================================"
