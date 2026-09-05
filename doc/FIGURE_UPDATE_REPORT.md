# Figure Update Report — Sep 5, 2026

## What Changed

### Figure 9: Predicted vs Actual Time Series

**Before:** Side-by-side 3×2 layout (each method in separate column)
**After:** Combined 3×1 layout (both methods overlaid per node)

**Changes made:**
1. `paper/generate_figures_extra.py` — rewrote `fig9_predicted_vs_actual()`:
   - 3 rows × 1 column (was 3×2)
   - Both T-GCN-NoSpatial and T-GCN-MultiGSL-Mix drawn on same axis per node
   - Added `compute_lag()` function for cross-correlation lag analysis
   - Legend shows both RMSE and lag (e.g., "T-GCN-MultiGSL-Mix (RMSE=0.1234, lag=+2)")
2. Regenerated `paper/figures/fig9_predicted_vs_actual.pdf` and `.png`
3. Added `paper/sections/results.tex` paragraph + Figure~9 reference
4. Updated `doc/RESPONSE_TO_REVIEWERS.md`: Reviewer 1 Q2 marked as addressed

### Figure 8: Convergence Curves

**Added** to `paper/sections/discussion.tex` as new subsection "Training Dynamics" with Figure~8 reference.

## Temporal Lag Analysis

Cross-correlation analysis on the three most variable Los-loop nodes (PH=1, seed=42):

| Node | Lag (steps) | Correlation |
|------|------------|-------------|
| 149  | +2         | 0.9967      |
| 163  | +1         | 0.9914      |
| 12   | +1         | 0.9952      |

**Interpretation:** The 1–2 step lag is inherent to MSE-trained next-step predictors. The model learns a smoothed version of the signal; when the actual value changes rapidly, the prediction takes 1–2 steps to catch up. This is NOT a bug — it is expected behavior and is noted in the figure caption.

## Paper Structure (modular)

```
paper/
  sn-article.tex              ← Main file (compile this)
  commands.tex                ← Macros
  sections/
    abstract.tex
    introduction.tex
    background.tex
    method.tex
    experiments.tex
    results.tex               ← fig9 added here
    discussion.tex            ← fig8 added here
    limitations.tex
    conclusion.tex
  appendix/
    original_gsl_results.tex
    convergence.tex
    bibliometric.tex
    additional_diagnostics.tex
  figures/
    fig1_graph_comparison.pdf/png
    fig2_rmse_comparison.pdf/png
    fig3_multiseed_boxplot.pdf/png
    fig4_param_control.pdf/png
    fig5_lag_ablation.pdf/png
    fig6_threshold_sensitivity.pdf/png
    fig7_lag_edge_stats.pdf/png
    fig8_convergence.pdf/png       ← NEW in this session
    fig9_predicted_vs_actual.pdf/png  ← REGENERATED in this session
  sn-article_original.tex     ← Backup of original 1166-line manuscript
```

## Compilation

```bash
cd paper
export PATH="/data/texlive/2026/bin/x86_64-linux:$PATH"
pdflatex -interaction=nonstopmode sn-article.tex
bibtex sn-article
pdflatex -interaction=nonstopmode sn-article.tex
pdflatex -interaction=nonstopmode sn-article.tex
```

## Git Commits in This Session

1. `21409ed` — Add 7 publication figures, update results with figure refs
2. `cf2cfb0` — Add checkpoint training script, extra figure generator
3. `5f0c8b6` — Add fig8/fig9 to paper with combined predicted-vs-actual layout
4. `fc78d8e` — Remove flattened manuscript; use modular sn-article.tex
