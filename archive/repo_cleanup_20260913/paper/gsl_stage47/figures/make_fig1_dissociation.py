"""Generate Figure 1 (consumption dissociation) for Stage 47 Results.

Data: gsl_stage41/stage41_summary.csv, Los-loop PH1, seeds 42-46.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

labels = [
    "T-GCN\n(Physical)",
    "GCN\n(Physical)",
    "GCN-MultiGSL\n(static union)",
    "T-GCN-MultiGSL\n(per-timestep)",
    "T-GCN-MultiGSL-Mix\n(gated)",
]
means = [7.877, 8.145, 9.780, 4.841, 4.491]
stds = [0.284, 0.112, 0.142, 0.115, 0.140]
nospatial = 5.251

colors = ["#9aa0a6", "#9aa0a6", "#c45c48", "#2a9d8f", "#1f7a6f"]

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
x = np.arange(len(labels))
ax.bar(
    x,
    means,
    yerr=stds,
    capsize=4,
    width=0.68,
    color=colors,
    edgecolor="#222222",
    linewidth=0.6,
    error_kw=dict(elinewidth=1.0, ecolor="#333333", capthick=1.0),
    zorder=3,
)

ax.axhline(nospatial, color="#1d3557", linestyle="--", linewidth=1.2, zorder=2)
ax.text(
    len(labels) - 0.35,
    nospatial + 0.12,
    "T-GCN-NoSpatial  5.25",
    color="#1d3557",
    fontsize=8,
    ha="right",
    va="bottom",
)

for xi, m, s in zip(x, means, stds):
    ax.text(xi, m + s + 0.12, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

ax.annotate(
    "",
    xy=(3, 5.05),
    xytext=(2, 9.55),
    arrowprops=dict(
        arrowstyle="<->",
        color="#333333",
        lw=1.0,
        connectionstyle="arc3,rad=-0.25",
    ),
)
ax.text(
    2.5,
    7.35,
    "identical multi-lag\nartifacts (union holds\n28 of 30 lag slots)",
    ha="center",
    va="center",
    fontsize=8,
    color="#222222",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.95),
)

ax.set_ylabel("Test RMSE (km/h)", fontsize=10)
ax.set_title("Los-loop, PH1: consumption of the same multi-lag graphs", fontsize=11, pad=10)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.set_ylim(0, 11.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", linestyle=":", alpha=0.45, zorder=0)
ax.tick_params(axis="y", labelsize=9)

fig.text(
    0.5,
    0.01,
    "Bars: 5-seed mean RMSE; whiskers: sample SD (seeds 42–46). "
    "Static statistical-dependency graphs; no causal interpretation.",
    ha="center",
    fontsize=7.5,
    color="#444444",
)

fig.tight_layout(rect=[0, 0.04, 1, 1])
out_dir = Path(__file__).resolve().parent
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "fig1_dissociation.pdf", bbox_inches="tight")
fig.savefig(out_dir / "fig1_dissociation.png", bbox_inches="tight")
print("Wrote", out_dir / "fig1_dissociation.pdf")
print("Wrote", out_dir / "fig1_dissociation.png")
