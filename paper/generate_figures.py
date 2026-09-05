#!/usr/bin/env python3
"""
generate_figures.py — Publication-quality figures for the revised paper.

Generates all figures from EXISTING saved results (no retraining required).
Figures that require retraining are noted but not generated.

Usage:
    cd /data/git/mamintoosi/TGCN-GSL-PyTorch
    /data/python-envs/pytorch/bin/python paper/generate_figures.py

Output:
    paper/figures/fig1_graph_comparison.pdf
    paper/figures/fig2_rmse_comparison.pdf
    paper/figures/fig3_multiseed_boxplot.pdf
    paper/figures/fig4_param_control.pdf
    paper/figures/fig5_lag_ablation.pdf
    paper/figures/fig6_threshold_sensitivity.pdf
    paper/figures/fig7_lag_edge_stats.pdf
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
ROOT = Path(__file__).resolve().parent.parent  # repo root
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

# Publication style
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
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette (consistent across figures)
COLORS = {
    'NoGraph': '#2196F3',
    'Physical': '#F44336',
    'Corr': '#FF9800',
    'SingleDAG': '#9C27B0',
    'UnionGraph': '#795548',
    'MultiGraph': '#4CAF50',
    'WeightedMulti': '#00BCD4',
    'T-GCN-MultiGSL-Mix': '#E91E63',
    'ParamMatch': '#607D8B',
}


# ============================================================
# Figure 1: Physical vs Learned Graph Visualization
# ============================================================
def fig1_graph_comparison():
    """Side-by-side adjacency heatmaps: Physical vs DAGMA lag-specific graphs."""
    print("Generating Figure 1: Physical vs DAGMA Graph Comparison...")
    
    # Load physical adjacency
    phys_adj = np.loadtxt(DATA_DIR / "los_adj.csv", delimiter=',')
    
    # Load DAGMA lag-specific graphs (thresholded at 0.1)
    lag_1 = np.load(RESULTS_DIR / "stage26_validation" / "los_ph1_seed42_L3_lag_1.npy")
    lag_2 = np.load(RESULTS_DIR / "stage26_validation" / "los_ph1_seed42_L3_lag_2.npy")
    lag_3 = np.load(RESULTS_DIR / "stage26_validation" / "los_ph1_seed42_L3_lag_3.npy")
    
    # Apply threshold
    thr = 0.1
    lag_1_thr = (np.abs(lag_1) > thr).astype(float)
    lag_2_thr = (np.abs(lag_2) > thr).astype(float)
    lag_3_thr = (np.abs(lag_3) > thr).astype(float)
    
    # Combine for visualization
    lag_union = np.clip(lag_1_thr + lag_2_thr + lag_3_thr, 0, 3)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: Physical
    im0 = axes[0].imshow(phys_adj, cmap='Blues', aspect='equal', interpolation='nearest')
    axes[0].set_title(f'(a) Physical Graph\n({int(phys_adj.sum())} edges, {phys_adj.shape[0]} nodes)')
    axes[0].set_xlabel('Node')
    axes[0].set_ylabel('Node')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Panel B: Multi-lag DAGMA union
    im1 = axes[1].imshow(lag_union, cmap='YlOrRd', aspect='equal', interpolation='nearest',
                         vmin=0, vmax=3)
    n_edges = int((lag_union > 0).sum())
    axes[1].set_title(f'(b) Multi-Lag DAGMA (union)\n({n_edges} edges, threshold={thr})')
    axes[1].set_xlabel('Node')
    axes[1].set_ylabel('Node')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3],
                 label='# lag graphs')
    
    # Panel C: Degree comparison
    phys_deg = phys_adj.sum(axis=1)
    dag_deg = (lag_union > 0).astype(float).sum(axis=1)
    
    axes[2].hist(phys_deg, bins=30, alpha=0.6, color=COLORS['Physical'], 
                 label=f'Physical (mean={phys_deg.mean():.1f})', density=True)
    axes[2].hist(dag_deg, bins=15, alpha=0.6, color=COLORS['T-GCN-MultiGSL-Mix'],
                 label=f'DAGMA (mean={dag_deg.mean():.1f})', density=True)
    axes[2].set_xlabel('Node Degree')
    axes[2].set_ylabel('Density')
    axes[2].set_title('(c) Node Degree Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_graph_comparison.pdf")
    plt.savefig(FIG_DIR / "fig1_graph_comparison.png")
    plt.close()
    print(f"  Saved: fig1_graph_comparison.pdf")


# ============================================================
# Figure 2: Main RMSE Comparison (Los-loop, all methods)
# ============================================================
def fig2_rmse_comparison():
    """Bar chart of all methods on Los-loop PH=1 from Stage 26 results."""
    print("Generating Figure 2: RMSE Comparison...")
    
    with open(RESULTS_DIR / "stage26_validation" / "stage26_results_los_ph1_seed42.json") as f:
        data = json.load(f)
    
    # Select key methods for clean visualization
    key_methods = [
        ('Physical', 'Physical'),
        ('Corr-K8', 'Corr-K8'),
        ('Corr-K16', 'Corr-K16'),
        ('SingleDAG_thr0.1', 'SingleDAG\n(thr=0.1)'),
        ('SingleDAG_thr0.3', 'SingleDAG\n(thr=0.3)'),
        ('T-GCN-NoSpatial', 'T-GCN\\nNoSpatial'),
        ('UnionGraph_thr0.1', 'Union\nGraph'),
        ('MultiGraphTGCN_thr0.1', 'T-GCN-\\nMultiGSL'),
        ('WeightedMulti_thr0.1', 'Weighted\nMulti'),
        ('GatedMulti_thr0.1', 'T-GCN-MultiGSL-\\nMix'),
    ]
    
    results_map = {r['method']: r for r in data['results']}
    
    labels = []
    rmse_vals = []
    n_edges = []
    colors = []
    color_map = {
        'Physical': COLORS['Physical'],
        'Corr-K8': COLORS['Corr'], 'Corr-K16': COLORS['Corr'],
        'SingleDAG_thr0.1': COLORS['SingleDAG'], 'SingleDAG_thr0.3': COLORS['SingleDAG'],
        'NoGraph': COLORS['NoGraph'],
        'UnionGraph_thr0.1': COLORS['UnionGraph'],
        'MultiGraphTGCN_thr0.1': COLORS['MultiGraph'],
        'WeightedMulti_thr0.1': COLORS['WeightedMulti'],
        'GatedMulti_thr0.1': COLORS['T-GCN-MultiGSL-Mix'],
    }
    
    for method_key, label in key_methods:
        if method_key in results_map:
            r = results_map[method_key]
            labels.append(label)
            rmse_vals.append(r['rmse'])
            n_edges.append(r['n_edges'])
            colors.append(color_map.get(method_key, '#999999'))
    
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(len(labels)), rmse_vals, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add edge count annotations
    for i, (bar, ne) in enumerate(zip(bars, n_edges)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{ne}e', ha='center', va='bottom', fontsize=8, fontstyle='italic')
    
    # Add NoGraph baseline line
    nograph_rmse = results_map['NoGraph']['rmse']
    ax.axhline(y=nograph_rmse, color=COLORS['NoGraph'], linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(labels)-0.5, nograph_rmse + 0.05, 'T-GCN-NoSpatial baseline', 
            fontsize=8, color=COLORS['NoGraph'], ha='right')
    
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('RMSE')
    ax.set_title('Los-loop PH=1: RMSE by Method (seed=42, threshold=0.1)')
    ax.set_ylim(0, max(rmse_vals) * 1.15)
    
    # Highlight GatedMulti
    bars[-1].set_edgecolor(COLORS['T-GCN-MultiGSL-Mix'])
    bars[-1].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_rmse_comparison.pdf")
    plt.savefig(FIG_DIR / "fig2_rmse_comparison.png")
    plt.close()
    print(f"  Saved: fig2_rmse_comparison.pdf")


# ============================================================
# Figure 3: Multi-Seed Box Plot (Experiment A)
# ============================================================
def fig3_multiseed_boxplot():
    """Box plot of T-GCN-NoSpatial vs T-GCN-MultiGSL vs T-GCN-MultiGSL-Mix across 5 seeds."""
    print("Generating Figure 3: Multi-Seed Box Plot...")
    
    # Read CSV
    csv_path = RESULTS_DIR / "stage26_validation" / "stage26_validation_A_losloop_ph1.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    methods = ['NoGraph', 'MultiGraphTGCN_fixed', 'GatedMultiGraphTGCN']
    display_names = ['T-GCN-\nNoSpatial', 'T-GCN-\nMultiGSL', 'T-GCN-MultiGSL-\nMix']
    
    data_by_method = {m: [] for m in methods}
    for row in rows:
        data_by_method[row['method']].append(float(row['rmse']))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [2, 1]})
    
    # Panel A: Box plot
    box_data = [data_by_method[m] for m in methods]
    bp = ax1.boxplot(box_data, tick_labels=display_names, patch_artist=True,
                     widths=0.5, showmeans=True, meanprops=dict(marker='D', markerfacecolor='white'))
    
    box_colors = [COLORS['NoGraph'], COLORS['MultiGraph'], COLORS['T-GCN-MultiGSL-Mix']]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, m in enumerate(methods):
        seeds = data_by_method[m]
        jitter = np.random.uniform(-0.1, 0.1, len(seeds))
        ax1.scatter([i+1+j for j in jitter], seeds, color='black', s=30, zorder=5, alpha=0.7)
    
    ax1.set_ylabel('RMSE')
    ax1.set_title('(a) RMSE Distribution Across 5 Seeds')
    ax1.axhline(y=np.mean(data_by_method['NoGraph']), color=COLORS['NoGraph'], 
                linestyle=':', alpha=0.5)
    
    # Panel B: Bar chart of means with error bars
    means = [np.mean(data_by_method[m]) for m in methods]
    stds = [np.std(data_by_method[m]) for m in methods]
    
    bars = ax2.bar(range(len(methods)), means, yerr=stds, color=box_colors,
                   edgecolor='black', linewidth=0.5, capsize=5, error_kw={'linewidth': 1})
    
    for i, (m, mean, std) in enumerate(zip(methods, means, stds)):
        ax2.text(i, mean + std + 0.03, f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontsize=8)
    
    # Improvement annotation
    nograph_mean = means[0]
    gated_mean = means[2]
    improvement = (nograph_mean - gated_mean) / nograph_mean * 100
    ax2.annotate(f'−{improvement:.1f}%', xy=(2, gated_mean), xytext=(2.3, gated_mean + 0.3),
                fontsize=10, fontweight='bold', color=COLORS['T-GCN-MultiGSL-Mix'],
                arrowprops=dict(arrowstyle='->', color=COLORS['T-GCN-MultiGSL-Mix']))
    
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(display_names, fontsize=9)
    ax2.set_ylabel('RMSE (mean ± std)')
    ax2.set_title('(b) Mean RMSE ± Std')
    ax2.set_ylim(0, max(m+s for m, s in zip(means, stds)) * 1.2)
    
    plt.suptitle('Los-loop PH=1: 5-Seed Validation', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_multiseed_boxplot.pdf")
    plt.savefig(FIG_DIR / "fig3_multiseed_boxplot.png")
    plt.close()
    print(f"  Saved: fig3_multiseed_boxplot.pdf")


# ============================================================
# Figure 4: Parameter-Matched Control (Experiment B)
# ============================================================
def fig4_param_control():
    """Bar chart showing parameter count vs RMSE."""
    print("Generating Figure 4: Parameter-Matched Control...")
    
    csv_path = RESULTS_DIR / "stage26_validation" / "stage26_validation_B_losloop_ph1.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    labels = ['T-GCN-NoSpatial\n(h=64)', 'T-GCN-NoSpatial\n(h=74)', 'T-GCN-MultiGSL-Mix\n(h=64)']
    params = [int(r['n_params']) for r in rows]
    rmses = [float(r['rmse']) for r in rows]
    colors = [COLORS['NoGraph'], COLORS['ParamMatch'], COLORS['T-GCN-MultiGSL-Mix']]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Panel A: Parameter count
    bars1 = ax1.bar(range(3), params, color=colors, edgecolor='black', linewidth=0.5)
    for bar, p in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                f'{p:,}', ha='center', va='bottom', fontsize=10)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Number of Parameters')
    ax1.set_title('(a) Parameter Count')
    ax1.set_ylim(0, max(params) * 1.2)
    
    # Panel B: RMSE
    bars2 = ax2.bar(range(3), rmses, color=colors, edgecolor='black', linewidth=0.5)
    for bar, r in zip(bars2, rmses):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{r:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Annotation: h=64 to h=74 improvement
    delta_param = params[1] - params[0]
    delta_rmse = rmses[0] - rmses[1]
    ax2.annotate(f'+{delta_param:,} params\n→ −{delta_rmse:.4f} RMSE\n({delta_rmse/rmses[0]*100:.1f}%)',
                xy=(1, rmses[1]), xytext=(1.5, rmses[1] + 0.3),
                fontsize=8, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Annotation: T-GCN-MultiGSL-Mix improvement
    gated_improve = rmses[0] - rmses[2]
    ax2.annotate(f'Gating:\n−{gated_improve:.4f} RMSE\n({gated_improve/rmses[0]*100:.1f}%)',
                xy=(2, rmses[2]), xytext=(2.4, rmses[2] + 0.4),
                fontsize=8, ha='center', color=COLORS['T-GCN-MultiGSL-Mix'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['T-GCN-MultiGSL-Mix']))
    
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('RMSE')
    ax2.set_title('(b) Forecasting Performance')
    ax2.set_ylim(0, max(rmses) * 1.25)
    
    plt.suptitle('Parameter-Matched Control (Los-loop PH=1, seed=42)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_param_control.pdf")
    plt.savefig(FIG_DIR / "fig4_param_control.png")
    plt.close()
    print(f"  Saved: fig4_param_control.pdf")


# ============================================================
# Figure 5: Lag Ablation (Experiment C)
# ============================================================
def fig5_lag_ablation():
    """Bar chart showing which lags contribute."""
    print("Generating Figure 5: Lag Ablation...")
    
    csv_path = RESULTS_DIR / "stage26_validation" / "stage26_validation_C_losloop_ph1.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Sort by RMSE (best first, excluding NoGraph)
    dag_rows = [r for r in rows if r['method'] != 'NoGraph']
    dag_rows.sort(key=lambda r: float(r['rmse']))
    
    # Add NoGraph at the end
    nograph_row = [r for r in rows if r['method'] == 'NoGraph'][0]
    
    labels = []
    rmses = []
    edges = []
    lags_used = []
    
    for r in dag_rows:
        lags = r.get('lags_used', '').replace('lag_', '')
        labels.append(lags)
        rmses.append(float(r['rmse']))
        edges.append(int(r['n_edges']))
        lags_used.append(r.get('lags_used', ''))
    
    labels.append('T-GCN-NoSpatial')
    rmses.append(float(nograph_row['rmse']))
    edges.append(int(nograph_row['n_edges']))
    lags_used.append('none')
    
    # Color by number of lags
    n_lags = []
    for lu in lags_used:
        if lu == 'none':
            n_lags.append(0)
        else:
            n_lags.append(lu.count('+') + 1)
    
    cmap = plt.cm.YlOrRd
    norm = plt.Normalize(vmin=0, vmax=3)
    colors = [cmap(norm(nl)) if nl > 0 else COLORS['NoGraph'] for nl in n_lags]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    bars = ax.barh(range(len(labels)-1, -1, -1), rmses, color=colors, 
                   edgecolor='black', linewidth=0.5, height=0.6)
    
    for i, (bar, rmse, ne, nl) in enumerate(zip(bars, rmses, edges, n_lags)):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{rmse:.4f}  ({ne} edges)', ha='left', va='center', fontsize=9)
    
    ax.set_yticks(range(len(labels)-1, -1, -1))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('RMSE')
    ax.set_title('Lag Ablation — Los-loop PH=1 (seed=42)\nHigher bars = worse performance')
    ax.set_xlim(0, max(rmses) * 1.25)
    
    # Add NoGraph baseline line
    ax.axvline(x=float(nograph_row['rmse']), color=COLORS['NoGraph'], 
               linestyle='--', alpha=0.5, linewidth=1)
    
    # Legend for lag count
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=cmap(norm(1)), edgecolor='black', label='1 lag'),
        Patch(facecolor=cmap(norm(2)), edgecolor='black', label='2 lags'),
        Patch(facecolor=cmap(norm(3)), edgecolor='black', label='3 lags (all)'),
        Patch(facecolor=COLORS['NoGraph'], edgecolor='black', label='T-GCN-NoSpatial'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_lag_ablation.pdf")
    plt.savefig(FIG_DIR / "fig5_lag_ablation.png")
    plt.close()
    print(f"  Saved: fig5_lag_ablation.pdf")


# ============================================================
# Figure 6: Threshold Sensitivity (from Stage 26 JSON)
# ============================================================
def fig6_threshold_sensitivity():
    """RMSE vs edge count for different DAGMA thresholds."""
    print("Generating Figure 6: Threshold Sensitivity...")
    
    with open(RESULTS_DIR / "stage26_validation" / "stage26_results_los_ph1_seed42.json") as f:
        data = json.load(f)
    
    # Extract SingleDAG results at different thresholds
    single_dag = [r for r in data['results'] if r['family'] == 'C_single_dagma']
    single_dag.sort(key=lambda r: r.get('threshold', 0))
    
    thresholds = [r.get('threshold', 0) for r in single_dag]
    rmses = [r['rmse'] for r in single_dag]
    n_edges = [r['n_edges'] for r in single_dag]
    
    # Also add T-GCN-NoSpatial and T-GCN-MultiGSL-Mix
    nograph_rmse = next(r['rmse'] for r in data['results'] if r['method'] == 'NoGraph')
    gated_rmse = next(r['rmse'] for r in data['results'] if r['method'] == 'GatedMulti_thr0.1')
    gated_edges = next(r['n_edges'] for r in data['results'] if r['method'] == 'GatedMulti_thr0.1')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Panel A: RMSE vs threshold
    ax1.plot(thresholds, rmses, 'o-', color=COLORS['SingleDAG'], linewidth=2, markersize=8,
             label='Single-lag DAGMA')
    ax1.axhline(y=nograph_rmse, color=COLORS['NoGraph'], linestyle='--', linewidth=1.5,
                label=f'T-GCN-NoSpatial ({nograph_rmse:.3f})')
    ax1.axhline(y=gated_rmse, color=COLORS['T-GCN-MultiGSL-Mix'], linestyle='--', linewidth=1.5,
                label=f'T-GCN-MultiGSL-Mix ({gated_rmse:.3f})')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('RMSE')
    ax1.set_title('(a) RMSE vs Threshold')
    ax1.set_xscale('log')
    ax1.legend(fontsize=8)
    
    # Panel B: RMSE vs edge count
    ax2.plot(n_edges, rmses, 's-', color=COLORS['SingleDAG'], linewidth=2, markersize=8,
             label='Single-lag DAGMA')
    ax2.scatter([gated_edges], [gated_rmse], color=COLORS['T-GCN-MultiGSL-Mix'], marker='*', 
                s=200, zorder=5, label=f'T-GCN-MultiGSL-Mix ({gated_edges} edges)')
    ax2.axhline(y=nograph_rmse, color=COLORS['NoGraph'], linestyle='--', linewidth=1.5,
                label=f'T-GCN-NoSpatial')
    ax2.set_xlabel('Number of Edges')
    ax2.set_ylabel('RMSE')
    ax2.set_title('(b) RMSE vs Edge Count')
    ax2.legend(fontsize=8)
    
    # Annotate thresholds
    for thr, rmse, ne in zip(thresholds, rmses, n_edges):
        ax2.annotate(f'thr={thr}', (ne, rmse), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, fontstyle='italic')
    
    plt.suptitle('DAGMA Threshold Sensitivity — Los-loop PH=1', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_threshold_sensitivity.pdf")
    plt.savefig(FIG_DIR / "fig6_threshold_sensitivity.png")
    plt.close()
    print(f"  Saved: fig6_threshold_sensitivity.pdf")


# ============================================================
# Figure 7: Lag-Specific Graph Edge Statistics
# ============================================================
def fig7_lag_edge_stats():
    """Analyze edge overlap and statistics across lags."""
    print("Generating Figure 7: Lag-Specific Graph Edge Statistics...")
    
    thr = 0.1
    
    # Los-loop
    lag1 = np.load(RESULTS_DIR / "stage26_validation" / "los_ph1_seed42_L3_lag_1.npy")
    lag2 = np.load(RESULTS_DIR / "stage26_validation" / "los_ph1_seed42_L3_lag_2.npy")
    lag3 = np.load(RESULTS_DIR / "stage26_validation" / "los_ph1_seed42_L3_lag_3.npy")
    
    lag1_bin = (np.abs(lag1) > thr).astype(int)
    lag2_bin = (np.abs(lag2) > thr).astype(int)
    lag3_bin = (np.abs(lag3) > thr).astype(int)
    
    # Compute edge sets
    def get_edges(A):
        return set(zip(*np.where(A > 0)))
    
    e1, e2, e3 = get_edges(lag1_bin), get_edges(lag2_bin), get_edges(lag3_bin)
    
    # Jaccard overlap
    def jaccard(s1, s2):
        if len(s1 | s2) == 0:
            return 0.0
        return len(s1 & s2) / len(s1 | s2)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: Edge count per lag (bar chart for both datasets)
    # Also load SZ-Taxi
    sz_lag1 = (np.abs(np.load(RESULTS_DIR / "stage26_validation" / "sz_ph1_seed42_L3_lag_1.npy")) > thr).astype(int)
    sz_lag2 = (np.abs(np.load(RESULTS_DIR / "stage26_validation" / "sz_ph1_seed42_L3_lag_2.npy")) > thr).astype(int)
    sz_lag3 = (np.abs(np.load(RESULTS_DIR / "stage26_validation" / "sz_ph1_seed42_L3_lag_3.npy")) > thr).astype(int)
    
    los_edges = [len(e1), len(e2), len(e3)]
    sz_edges = [int(sz_lag1.sum()), int(sz_lag2.sum()), int(sz_lag3.sum())]
    
    x = np.arange(3)
    w = 0.35
    axes[0].bar(x - w/2, los_edges, w, color=COLORS['T-GCN-MultiGSL-Mix'], label='Los-loop', edgecolor='black', linewidth=0.5)
    axes[0].bar(x + w/2, sz_edges, w, color=COLORS['NoGraph'], label='SZ-Taxi', edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Lag 1', 'Lag 2', 'Lag 3'])
    axes[0].set_ylabel('Number of Edges')
    axes[0].set_title('(a) Edges per Lag')
    axes[0].legend()
    
    # Panel B: Jaccard overlap matrix (Los-loop)
    lags = [e1, e2, e3]
    jaccard_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            jaccard_matrix[i, j] = jaccard(lags[i], lags[j])
    
    im = axes[1].imshow(jaccard_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(['Lag 1', 'Lag 2', 'Lag 3'])
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(['Lag 1', 'Lag 2', 'Lag 3'])
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{jaccard_matrix[i,j]:.2f}', ha='center', va='center', fontsize=11)
    axes[1].set_title('(b) Jaccard Overlap (Los-loop)')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Panel C: Edge weight distributions per lag
    lag1_weights = lag1[lag1_bin > 0]
    lag2_weights = lag2[lag2_bin > 0]
    lag3_weights = lag3[lag3_bin > 0]
    
    if len(lag1_weights) > 0:
        axes[2].hist(np.abs(lag1_weights), bins=15, alpha=0.6, color=COLORS['NoGraph'],
                     label=f'Lag 1 (n={len(lag1_weights)})', density=True)
    if len(lag2_weights) > 0:
        axes[2].hist(np.abs(lag2_weights), bins=10, alpha=0.6, color=COLORS['MultiGraph'],
                     label=f'Lag 2 (n={len(lag2_weights)})', density=True)
    if len(lag3_weights) > 0:
        axes[2].hist(np.abs(lag3_weights), bins=15, alpha=0.6, color=COLORS['T-GCN-MultiGSL-Mix'],
                     label=f'Lag 3 (n={len(lag3_weights)})', density=True)
    axes[2].set_xlabel('|DAGMA Weight|')
    axes[2].set_ylabel('Density')
    axes[2].set_title('(c) Edge Weight Distribution')
    axes[2].legend(fontsize=8)
    
    plt.suptitle('Lag-Specific Graph Analysis — Los-loop (threshold=0.1)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_lag_edge_stats.pdf")
    plt.savefig(FIG_DIR / "fig7_lag_edge_stats.png")
    plt.close()
    print(f"  Saved: fig7_lag_edge_stats.pdf")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 70)
    
    fig1_graph_comparison()
    fig2_rmse_comparison()
    fig3_multiseed_boxplot()
    fig4_param_control()
    fig5_lag_ablation()
    fig6_threshold_sensitivity()
    fig7_lag_edge_stats()
    
    print("\n" + "=" * 70)
    print("ALL FIGURES GENERATED SUCCESSFULLY")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 70)
