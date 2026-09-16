# Demo: predicted vs actual (Figures 9–10)

Small committed assets so you can regenerate the manuscript-style overlays
without the full `results/` tree (~150 MB, gitignored).

| File | Role |
|------|------|
| `assets/pred_vs_actual_los_ph1_seed42.npz` | Los-loop, 3 nodes × 100 steps |
| `assets/pred_vs_actual_sz_ph1_seed42.npz` | SZ-Taxi, 3 nodes × 100 steps |
| `plot_pred_vs_actual.py` | Plot Actual vs Physical vs Mix |
| `build_pred_vs_actual_assets.py` | Rebuild `.npz` from local checkpoints |

```bash
python demo/plot_pred_vs_actual.py
```

Outputs land in `demo/figures/`.
