#!/usr/bin/env python3
"""Two-panel PH1-6 mean-RMSE trends (Los-loop | SZ-Taxi), T-GCN family.

PH1-4: main Tables 2 (canonical five-seed means).
PH5-6: results/stage59_ph56_horizon (Los, incl. Mix) and
       results/supplementary_ph56 (SZ).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "revised_version" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PH = np.arange(1, 7)

# Mean RMSE only (main tables + supplementary).
LOS = {
    "T-GCN (Physical)": [7.88, 8.13, 8.37, 8.66, 8.68, 8.97],
    "T-GCN-GSL": [5.86, 6.30, 6.66, 7.04, 7.22, 7.62],
    "T-GCN-cGSL": [5.82, 6.35, 6.67, 7.09, 7.21, 7.60],
    "T-GCN-MultiGSL": [4.84, 5.45, 5.94, 6.26, 6.61, 6.88],
    "T-GCN-MultiGSL-Mix": [4.49, 5.08, 5.55, 5.86, 6.23, 6.51],
    "Graph-free (NoSpatial)": [5.25, 5.76, 6.11, 6.58, 6.77, 7.16],
}

SZ = {
    "T-GCN (Physical)": [5.45, 5.55, 5.60, 5.63, 5.63, 5.69],
    "T-GCN-GSL": [4.28, 4.31, 4.33, 4.37, 4.39, 4.43],
    "T-GCN-cGSL": [4.30, 4.34, 4.38, 4.41, 4.43, 4.46],
    "T-GCN-MultiGSL": [4.13, 4.16, 4.20, 4.22, 4.26, 4.28],
    "Graph-free (NoSpatial)": [4.12, 4.16, 4.19, 4.23, 4.24, 4.27],
}

STYLE = {
    "T-GCN (Physical)": dict(color="#F44336", marker="s", ls="-", lw=1.8),
    "T-GCN-GSL": dict(color="#4CAF50", marker="o", ls="-", lw=1.5),
    "T-GCN-cGSL": dict(color="#8BC34A", marker="^", ls="--", lw=1.5),
    "T-GCN-MultiGSL": dict(color="#009688", marker="D", ls="-", lw=1.5),
    "T-GCN-MultiGSL-Mix": dict(color="#7B1FA2", marker="*", ls="-", lw=2.0, markersize=10),
    "Graph-free (NoSpatial)": dict(color="#455A64", marker="x", ls=":", lw=1.4),
}


def plot_panel(ax, series, title):
    for name, ys in series.items():
        ax.plot(PH, ys, label=name, **STYLE[name])
    ax.set_xlabel("Prediction horizon (PH)")
    ax.set_ylabel("Test RMSE (mean over 5 seeds)")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(PH)
    ax.grid(True, ls=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=False)
plot_panel(axes[0], LOS, "(a) Los-loop (5 min)")
plot_panel(axes[1], SZ, "(b) SZ-Taxi (15 min)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    fontsize=8.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)
fig.suptitle("T-GCN family: mean test RMSE vs prediction horizon (PH1–6)", y=1.03, fontsize=12)
fig.tight_layout(rect=(0, 0.14, 1, 0.98))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"ph16_tgcn_trends.{ext}", bbox_inches="tight", dpi=200)
plt.close(fig)
print("Wrote", OUT / "ph16_tgcn_trends.pdf")
