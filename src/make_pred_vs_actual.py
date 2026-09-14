#!/usr/bin/env python3
"""Predicted vs actual overlays (Los-loop and SZ-Taxi), legend lower-right."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper" / "revised_version" / "figures"
CKPT = ROOT / "results" / "stage26_checkpoint"

COL_NS = "#2196F3"  # used only if Physical preds missing
# Lighter than Mix so the two series are easy to separate
COL_PHYS = "#FF8A80"
COL_MIX = "#7B1FA2"  # purple: distinct from Physical red / dashed light red


def squeeze_ph1(a):
    a = np.load(a) if not isinstance(a, np.ndarray) else a
    if a.ndim == 3 and a.shape[1] == 1:
        a = a[:, 0, :]
    return a


def high_var_nodes(y, k=3):
    v = y.var(axis=0)
    return list(np.argsort(v)[-k:][::-1])


def plot_pair(nograph_dir, mix_dir, nodes, title, stem, compare_label="Physical"):
    yt = squeeze_ph1(nograph_dir / "y_true.npy")
    yp_n = squeeze_ph1(nograph_dir / "y_pred.npy")
    yp_m = squeeze_ph1(mix_dir / "y_pred.npy")
    # Dashed comparison series; solid for Actual and Mix
    col_cmp = COL_PHYS if "Physical" in compare_label else COL_NS
    T = min(100, yt.shape[0])
    t = np.arange(T)
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True)
    for ax, node in zip(axes, nodes):
        ax.plot(t, yt[:T, node], color="black", lw=1.4, label="Actual", zorder=3)
        ax.plot(
            t,
            yp_n[:T, node],
            color=col_cmp,
            lw=1.2,
            ls="--",
            alpha=0.95,
            label=compare_label,
        )
        ax.plot(
            t,
            yp_m[:T, node],
            color=COL_MIX,
            lw=1.2,
            alpha=0.95,
            label="T-GCN-MultiGSL-Mix",
        )
        ax.set_ylabel(f"Node {node}\n(norm.)", fontsize=9)
        ax.legend(loc="lower right", fontsize=7.5, ncol=3, framealpha=0.92)
        ax.grid(True, ls=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Test time step")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print("Wrote", stem, "nodes", nodes, "compare", compare_label)


def main():
    # Prefer Physical T-GCN predictions if trained; else graph-free (legacy)
    los_ref = CKPT / "los_ph1_seed42_physical"
    sz_ref = CKPT / "sz_ph1_seed42_physical"
    los_label = "T-GCN (Physical)" if (los_ref / "y_pred.npy").exists() else "Graph-free baseline"
    sz_label = "T-GCN (Physical)" if (sz_ref / "y_pred.npy").exists() else "Graph-free baseline"
    plot_pair(
        los_ref if (los_ref / "y_pred.npy").exists() else CKPT / "los_ph1_seed42_nograph",
        CKPT / "los_ph1_seed42_gated_multi",
        [149, 163, 12],
        "Los-loop PH1, seed 42: predicted vs actual (normalized)\n"
        "three high-variance nodes; 100 consecutive test steps",
        "pred_vs_actual_los_ph1",
        compare_label=los_label,
    )
    sz_ref_use = sz_ref if (sz_ref / "y_pred.npy").exists() else CKPT / "sz_ph1_seed42_nograph"
    sz_y = squeeze_ph1(sz_ref_use / "y_true.npy")
    sz_nodes = high_var_nodes(sz_y, 3)
    plot_pair(
        sz_ref_use,
        CKPT / "sz_ph1_seed42_gated_multi",
        sz_nodes,
        "SZ-Taxi PH1, seed 42: predicted vs actual (normalized)\n"
        "three high-variance nodes; 100 consecutive test steps",
        "pred_vs_actual_sz_ph1",
        compare_label=sz_label,
    )


if __name__ == "__main__":
    main()
