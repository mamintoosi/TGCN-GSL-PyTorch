#!/usr/bin/env python3
"""
generate_figures_extra.py — Convergence curves + predicted vs actual plots.

Generates figures from SAVED CHECKPOINTS (requires stage26_train_with_logging.py first).

Usage:
    cd /data/git/mamintoosi/TGCN-GSL-PyTorch
    /data/python-envs/pytorch/bin/python paper/generate_figures_extra.py

Output:
    paper/figures/fig8_convergence.pdf
    paper/figures/fig9_predicted_vs_actual.pdf
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper" / "figures"
CKPT_DIR = ROOT / "results" / "stage26_checkpoint"

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'NoGraph': '#2196F3',
    'MultiGraph': '#4CAF50',
    'GatedMulti': '#E91E63',
}


def load_loss_history(method, dataset_prefix, ph, seed):
    path = CKPT_DIR / f"{dataset_prefix}_ph{ph}_seed{seed}_{method}" / "train_loss_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_predictions(method, dataset_prefix, ph, seed):
    base = CKPT_DIR / f"{dataset_prefix}_ph{ph}_seed{seed}_{method}"
    pred_path = base / "y_pred.npy"
    true_path = base / "y_true.npy"
    if not pred_path.exists() or not true_path.exists():
        return None, None
    return np.load(pred_path), np.load(true_path)


# ============================================================
# Figure 8: Training Convergence Curves
# ============================================================
def fig8_convergence():
    """Plot training loss curves for NoGraph, MultiGraph, GatedMulti."""
    print("Generating Figure 8: Convergence Curves...")

    methods = ['nograph', 'multi_graph_fixed', 'gated_multi']
    display_names = ['T-GCN-NoSpatial', 'T-GCN-MultiGSL', 'T-GCN-MultiGSL-Mix']
    colors = [COLORS['NoGraph'], COLORS['MultiGraph'], COLORS['GatedMulti']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: Los-loop convergence
    found_any = False
    for method, name, color in zip(methods, display_names, colors):
        history = load_loss_history(method, 'los', 1, 42)
        if history:
            losses = history['train_losses']
            ax1.plot(range(1, len(losses)+1), losses, color=color, linewidth=2, label=name)
            found_any = True
    if found_any:
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Training Loss (MSE)')
        ax1.set_title('(a) Los-loop PH=1 (seed=42)')
        ax1.legend()
        ax1.set_yscale('log')
    else:
        ax1.text(0.5, 0.5, 'No checkpoints found.\nRun stage26_train_with_logging.py first.',
                ha='center', va='center', transform=ax1.transAxes, fontsize=10, color='red')
        ax1.set_title('(a) Los-loop PH=1')

    # Panel B: SZ-Taxi convergence (if available)
    found_any = False
    for method, name, color in zip(methods, display_names, colors):
        history = load_loss_history(method, 'sz', 1, 42)
        if history:
            losses = history['train_losses']
            ax2.plot(range(1, len(losses)+1), losses, color=color, linewidth=2, label=name)
            found_any = True
    if found_any:
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Training Loss (MSE)')
        ax2.set_title('(b) SZ-Taxi PH=1 (seed=42)')
        ax2.legend()
        ax2.set_yscale('log')
    else:
        ax2.text(0.5, 0.5, 'No checkpoints found.\nRun stage26_train_with_logging.py first.',
                ha='center', va='center', transform=ax2.transAxes, fontsize=10, color='red')
        ax2.set_title('(b) SZ-Taxi PH=1')

    plt.suptitle('Training Convergence', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_convergence.pdf")
    plt.savefig(FIG_DIR / "fig8_convergence.png")
    plt.close()
    print(f"  Saved: fig8_convergence.pdf")


# ============================================================
# Figure 9: Predicted vs Actual Time Series
# ============================================================
def fig9_predicted_vs_actual():
    """Plot predicted vs actual traffic speed for selected nodes and time windows."""
    print("Generating Figure 9: Predicted vs Actual...")

    methods = ['nograph', 'gated_multi']
    display_names = ['T-GCN-NoSpatial', 'T-GCN-MultiGSL-Mix']
    colors = [COLORS['NoGraph'], COLORS['GatedMulti']]

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Select 3 representative nodes (by variance in ground truth)
    preds_all = {}
    gts_all = {}
    for method in methods:
        pred, gt = load_predictions(method, 'los', 1, 42)
        if pred is not None:
            preds_all[method] = pred
            gts_all[method] = gt

    if not preds_all:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No predictions found.\nRun stage26_train_with_logging.py first.',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10, color='red')
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig9_predicted_vs_actual.pdf")
        plt.savefig(FIG_DIR / "fig9_predicted_vs_actual.png")
        plt.close()
        print(f"  Saved: fig9_predicted_vs_actual.pdf (placeholder)")
        return

    # Use ground truth from first available method
    gt_key = list(gts_all.keys())[0]
    gt_all = gts_all[gt_key]  # (n_test, pre_len, N)

    # Flatten to (n_test * pre_len, N) for easier slicing
    n_test, pre_len, N = gt_all.shape
    gt_flat = gt_all.reshape(-1, N)

    # Select nodes by variance
    node_vars = np.var(gt_flat, axis=0)
    top_nodes = np.argsort(node_vars)[-3:][::-1]  # top 3 most variable

    # Select time window (100 consecutive steps)
    n_steps = min(100, gt_flat.shape[0])
    start_step = (gt_flat.shape[0] - n_steps) // 2

    for row, node_idx in enumerate(top_nodes):
        gt_series = gt_flat[start_step:start_step+n_steps, node_idx]

        for col, (method, name, color) in enumerate(zip(methods, display_names, colors)):
            ax = axes[row, col]

            if method in preds_all:
                pred_flat = preds_all[method].reshape(-1, N)
                pred_series = pred_flat[start_step:start_step+n_steps, node_idx]
                rmse = np.sqrt(np.mean((pred_series - gt_series) ** 2))

                ax.plot(gt_series, color='black', linewidth=1.5, label='Actual', alpha=0.8)
                ax.plot(pred_series, color=color, linewidth=1.5, label=name, alpha=0.8)
                ax.set_title(f'Node {node_idx} — {name} (RMSE={rmse:.4f})', fontsize=10)
                ax.legend(fontsize=8, loc='upper right')
            else:
                ax.text(0.5, 0.5, f'{name}\nnot available',
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)

            ax.set_xlabel('Time Step' if row == 2 else '')
            ax.set_ylabel('Speed (normalized)' if col == 0 else '')

    plt.suptitle('Predicted vs Actual Traffic Speed — Los-loop PH=1 (seed=42)\n'
                 'Three most variable nodes, 100 consecutive test steps',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_predicted_vs_actual.pdf")
    plt.savefig(FIG_DIR / "fig9_predicted_vs_actual.png")
    plt.close()
    print(f"  Saved: fig9_predicted_vs_actual.pdf")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING EXTRA FIGURES (convergence + predicted vs actual)")
    print("=" * 70)

    fig8_convergence()
    fig9_predicted_vs_actual()

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Output: {FIG_DIR}")
    print("=" * 70)
