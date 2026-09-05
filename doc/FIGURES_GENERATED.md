# Figures Generated for Revised Paper

**Date:** 2026-09-05
**Script:** `paper/generate_figures.py`
**Output:** `paper/figures/`

## Generated Figures (no retraining required)

All figures use EXISTING saved results from Stages 24–26. No new experiments were run.

| File | Description | Data Source |
|------|-------------|-------------|
| `fig1_graph_comparison.pdf` | Physical vs DAGMA graph heatmaps + degree distribution | `los_adj.csv`, `los_ph1_seed42_L3_lag_*.npy` |
| `fig2_rmse_comparison.pdf` | RMSE bar chart for all methods | `stage26_results_los_ph1_seed42.json` |
| `fig3_multiseed_boxplot.pdf` | 5-seed box plot + mean±std bars | `stage26_validation_A_losloop_ph1.csv` |
| `fig4_param_control.pdf` | Parameter-matched control comparison | `stage26_validation_B_losloop_ph1.csv` |
| `fig5_lag_ablation.pdf` | Horizontal bar chart of lag contributions | `stage26_validation_C_losloop_ph1.csv` |
| `fig6_threshold_sensitivity.pdf` | RMSE vs threshold/edge count | `stage26_results_los_ph1_seed42.json` |
| `fig7_lag_edge_stats.pdf` | Lag edge counts, Jaccard overlap, weight distributions | `los_ph1_seed42_L3_lag_*.npy`, `sz_ph1_seed42_L3_lag_*.npy` |

## Figures That Require Retraining

Two reviewer-requested figure types cannot be generated from saved results:

### 1. Predicted vs Actual Time Series (Reviewer 1, Q2)
- **Why:** No model checkpoints (`.pt`/`.pth`) were saved during training
- **What's needed:** Save model state_dict after training, then load for inference
- **Effort:** ~10 lines of code change in training loop + ~20 lines for plotting script
- **Recommendation:** Moderate priority — useful but not essential for the revision

### 2. Training Convergence Curves (Reviewer 2, Comment 5)
- **Why:** Training loss per epoch was not logged to files
- **What's needed:** Record train_loss and val_loss per epoch, save to CSV/JSON
- **Effort:** ~5 lines of code in training loop + ~15 lines for plotting
- **Recommendation:** Low priority — the paper no longer emphasizes convergence. Current convergence figures were moved to appendix.

## How to Regenerate

```bash
cd /data/git/mamintoosi/TGCN-GSL-PyTorch
/data/python-envs/pytorch/bin/python paper/generate_figures.py
```

All figures are saved as both PDF (publication) and PNG (preview) in `paper/figures/`.
