#!/usr/bin/env python3
"""
Clean-room controlled experiment: Physical vs GSL vs cGSL for Traffic Prediction.

This script runs the COMPLETE experiment hierarchy:
1. Physical graph baseline (GCN, TGCN)
2. GSL graph (DAGMA, directed acyclic)
3. cGSL graph (DAGMA, symmetrized cyclic)
4. Random sparse physical graph (controlled ablation)

For each configuration:
- Dataset: SZ-Taxi, Los-loop
- Model: GCN, TGCN
- Prediction Horizon: 1, 2, 3, 4
- Seed: 42 (default)

All conditions are IDENTICAL except the graph structure.

Output: results/clean_reimplementation/experiment_results.json

Usage:
    cd TGCN-GSL-PyTorch
    python gsl_clean/run_experiment.py
    python gsl_clean/run_experiment.py --dataset shenzhen --model GCN
    python gsl_clean/run_experiment.py --seeds 42 43 44
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gsl_clean.config import ExperimentConfig, DATASET_DAGMA_CONFIGS, DATA_PATHS
from gsl_clean.data_pipeline import (
    load_data,
    generate_sequences,
    prepare_dagma_input,
)
from gsl_clean.graph_utils import (
    build_gsl_adjacency,
    build_cgsl_adjacency,
    graph_statistics,
    print_graph_stats,
)
from models.gcn import GCN
from models.tgcn import TGCN
from tasks.supervised import SupervisedForecastTask


# ============================================================
# Seed management
# ============================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Graph loading
# ============================================================

def load_graph(
    dataset_name: str,
    pre_len: int,
    graph_type: int,
    feat: np.ndarray,
    seq_len: int = 12,
    split_ratio: float = 0.8,
) -> Tuple[np.ndarray, Dict]:
    """
    Load or construct adjacency matrix based on graph_type.

    Args:
        dataset_name: "shenzhen" or "losloop"
        pre_len: prediction horizon (1-4)
        graph_type: 0=physical, 1=GSL, 2=cGSL, 3=random-sparse
        feat: raw feature data (T, N)
        seq_len: historical window
        split_ratio: train/test split

    Returns:
        adj: Adjacency matrix (N, N) as float32
        stats: Graph statistics dictionary
    """
    # Load physical adjacency
    _, adj_physical = load_data(dataset_name)

    if graph_type == 0:
        # Physical graph
        adj = adj_physical.copy()
        label = f"Physical ({dataset_name})"

    elif graph_type in (1, 2):
        # GSL or cGSL from existing DAGMA results
        w_est_path = f"data/W_est_{dataset_name}_pre_len{pre_len}.npy"
        if not os.path.exists(w_est_path):
            raise FileNotFoundError(
                f"W_est file not found: {w_est_path}. "
                "Run DAGMA first or use graph_type=0."
            )
        W_est = np.load(w_est_path)

        if graph_type == 1:
            adj = build_gsl_adjacency(W_est)
            label = f"GSL ({dataset_name}, PH={pre_len})"
        else:
            adj = build_cgsl_adjacency(W_est)
            label = f"cGSL ({dataset_name}, PH={pre_len})"

    elif graph_type == 3:
        # Random sparse physical graph
        sparse_path = f"data/sparse_random_{dataset_name}_pre_len{pre_len}.npy"
        if os.path.exists(sparse_path):
            adj = np.load(sparse_path).astype(np.float32)
            label = f"Random-Sparse ({dataset_name}, PH={pre_len})"
        else:
            raise FileNotFoundError(
                f"Sparse random graph not found: {sparse_path}. "
                "Run generate_sparse_random_graphs.py first."
            )
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    stats = graph_statistics(adj, label)
    return adj, stats


# ============================================================
# Training
# ============================================================

def train_and_evaluate(
    adj: np.ndarray,
    model_name: str,
    dataset_name: str,
    train_X: np.ndarray,
    train_Y: np.ndarray,
    test_X: np.ndarray,
    test_Y: np.ndarray,
    config: ExperimentConfig,
    device: str = "cuda",
) -> Dict:
    """
    Train model and evaluate on test set.

    Returns:
        Dictionary of metrics and timing information.
    """
    set_seed(config.seed)

    seq_len = config.seq_len
    hidden_dim = config.hidden_dim
    pre_len = config.pre_len

    # Create model
    if model_name == "GCN":
        model = GCN(adj=adj, seq_len=seq_len, hidden_dim=hidden_dim)
        loss_name = "mse"
    elif model_name == "TGCN":
        model = TGCN(adj=adj, hidden_dim=hidden_dim)
        loss_name = "mse_with_regularizer"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Compute feat_max_val for proper denormalization
    feat_raw, _ = load_data(dataset_name)
    feat_max_val = float(np.max(feat_raw))

    model_task = SupervisedForecastTask(
        model=model,
        loss=loss_name,
        pre_len=pre_len,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        feat_max_val=feat_max_val,
    )

    # Move to device
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    actual_device = "cuda" if use_cuda else "cpu"
    model = model.to(actual_device)
    if model_task.regressor is not None:
        model_task.regressor = model_task.regressor.to(actual_device)

    optimizer = model_task.configure_optimizer()

    # Convert to tensors
    train_X_t = torch.FloatTensor(train_X)
    train_Y_t = torch.FloatTensor(train_Y)
    test_X_t = torch.FloatTensor(test_X)
    test_Y_t = torch.FloatTensor(test_Y)

    train_dataset = torch.utils.data.TensorDataset(train_X_t, train_Y_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )

    # Train
    start_train = time.time()
    for epoch in range(config.max_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(actual_device)
            y_batch = y_batch.to(actual_device)
            optimizer.zero_grad()
            loss = model_task.training_step((x_batch, y_batch))
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_train

    # Evaluate
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_X_t, test_Y_t),
        batch_size=len(test_X_t),
        shuffle=False,
    )

    metrics = model_task.validation_epoch(test_loader, actual_device)
    metrics["train_time_s"] = round(train_time, 2)

    return metrics


# ============================================================
# Main experiment runner
# ============================================================

def run_single_experiment(
    dataset_name: str,
    model_name: str,
    pre_len: int,
    graph_type: int,
    seed: int = 42,
    max_epochs: int = 50,
    device: str = "cuda",
) -> Dict:
    """
    Run a single controlled experiment.

    Returns:
        Dictionary with config, metrics, and graph statistics.
    """
    config = ExperimentConfig(
        dataset_name=dataset_name,
        model_name=model_name,
        pre_len=pre_len,
        graph_type=graph_type,
        seed=seed,
        max_epochs=max_epochs,
    )

    graph_names = {0: "physical", 1: "GSL", 2: "cGSL", 3: "random-sparse"}
    graph_name = graph_names[graph_type]

    print(f"\n  Running: {dataset_name} / {model_name} / PH={pre_len} / {graph_name} / seed={seed}")

    # Load data
    feat, adj_physical = load_data(dataset_name)
    train_X, train_Y, test_X, test_Y = generate_sequences(
        feat, seq_len=config.seq_len, pre_len=pre_len,
        split_ratio=config.split_ratio, normalize=config.normalize,
    )

    # Load graph
    adj, graph_stats = load_graph(
        dataset_name, pre_len, graph_type, feat,
        seq_len=config.seq_len, split_ratio=config.split_ratio,
    )

    # Train and evaluate
    metrics = train_and_evaluate(
        adj=adj,
        model_name=model_name,
        dataset_name=dataset_name,
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        test_Y=test_Y,
        config=config,
        device=device,
    )

    result = {
        "dataset": dataset_name,
        "model": model_name,
        "pre_len": pre_len,
        "graph_type": graph_name,
        "graph_type_id": graph_type,
        "seed": seed,
        "graph_stats": graph_stats,
        "metrics": metrics,
    }

    print(f"    RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, "
          f"R2={metrics['R2']:.4f}, edges={graph_stats['n_edges']}, "
          f"time={metrics['train_time_s']:.1f}s")

    return result


def run_full_experiment(
    datasets: List[str],
    models: List[str],
    pre_lens: List[int],
    graph_types: List[int],
    seeds: List[int],
    max_epochs: int = 50,
    device: str = "cuda",
) -> List[Dict]:
    """
    Run the full experiment matrix.
    """
    results = []
    total = len(datasets) * len(models) * len(pre_lens) * len(graph_types) * len(seeds)
    count = 0

    for ds in datasets:
        for model in models:
            for ph in pre_lens:
                for gt in graph_types:
                    for seed in seeds:
                        count += 1
                        print(f"\n[{count}/{total}]", end="")
                        try:
                            r = run_single_experiment(
                                ds, model, ph, gt, seed, max_epochs, device
                            )
                            results.append(r)
                        except FileNotFoundError as e:
                            print(f"  SKIPPED: {e}")
                            results.append({
                                "dataset": ds, "model": model,
                                "pre_len": ph, "graph_type": gt,
                                "seed": seed, "error": str(e),
                            })
                        except Exception as e:
                            import traceback
                            print(f"  ERROR: {e}")
                            traceback.print_exc()
                            results.append({
                                "dataset": ds, "model": model,
                                "pre_len": ph, "graph_type": gt,
                                "seed": seed, "error": str(e),
                            })

    return results


# ============================================================
# Result formatting
# ============================================================

def format_results_table(results: List[Dict]) -> str:
    """Format results as a readable table."""
    lines = []
    lines.append("=" * 120)
    lines.append("EXPERIMENTAL RESULTS — Clean Reimplementation")
    lines.append("=" * 120)

    # Group by dataset and model
    for ds in ["shenzhen", "losloop"]:
        for model in ["GCN", "TGCN"]:
            ds_results = [r for r in results if r.get("dataset") == ds and r.get("model") == model and "error" not in r]
            if not ds_results:
                continue

            ds_label = "SZ-Taxi" if ds == "shenzhen" else "Los-loop"
            lines.append(f"\n--- {ds_label} / {model} ---")
            lines.append(f"{'PH':>3} {'Graph':>15} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'Edges':>6} {'Isol.':>6} {'Time':>7}")
            lines.append("-" * 70)

            for ph in range(1, 5):
                for gt_name in ["physical", "GSL", "cGSL", "random-sparse"]:
                    matching = [r for r in ds_results if r["pre_len"] == ph and r["graph_type"] == gt_name]
                    if matching:
                        r = matching[0]
                        m = r["metrics"]
                        g = r["graph_stats"]
                        lines.append(
                            f"{ph:>3} {gt_name:>15} {m['RMSE']:>8.4f} {m['MAE']:>8.4f} "
                            f"{m['R2']:>8.4f} {g['n_edges']:>6} {g.get('n_isolated_nodes', 'N/A'):>6} "
                            f"{m.get('train_time_s', 0):>6.1f}s"
                        )
                if ph < 4:
                    lines.append("")

    # Compute improvement table
    lines.append("\n" + "=" * 120)
    lines.append("IMPROVEMENT OVER PHYSICAL BASELINE (RMSE)")
    lines.append("=" * 120)

    for ds in ["shenzhen", "losloop"]:
        for model in ["GCN", "TGCN"]:
            ds_results = [r for r in results if r.get("dataset") == ds and r.get("model") == model and "error" not in r]
            if not ds_results:
                continue

            ds_label = "SZ-Taxi" if ds == "shenzhen" else "Los-loop"
            lines.append(f"\n--- {ds_label} / {model} ---")
            lines.append(f"{'PH':>3} {'GSL vs Phys':>12} {'cGSL vs Phys':>13} {'Rand vs Phys':>13} {'GSL vs Rand':>12}")
            lines.append("-" * 60)

            for ph in range(1, 5):
                phys = [r for r in ds_results if r["pre_len"] == ph and r["graph_type"] == "physical"]
                gsl = [r for r in ds_results if r["pre_len"] == ph and r["graph_type"] == "GSL"]
                cgsl = [r for r in ds_results if r["pre_len"] == ph and r["graph_type"] == "cGSL"]
                rand = [r for r in ds_results if r["pre_len"] == ph and r["graph_type"] == "random-sparse"]

                if phys and gsl and cgsl:
                    p_rmse = phys[0]["metrics"]["RMSE"]
                    g_rmse = gsl[0]["metrics"]["RMSE"]
                    c_rmse = cgsl[0]["metrics"]["RMSE"]

                    gsl_imp = (p_rmse - g_rmse) / p_rmse * 100
                    cgsl_imp = (p_rmse - c_rmse) / p_rmse * 100

                    rand_str = "N/A"
                    gsl_rand_str = "N/A"
                    if rand:
                        r_rmse = rand[0]["metrics"]["RMSE"]
                        rand_imp = (p_rmse - r_rmse) / p_rmse * 100
                        gsl_rand_imp = (r_rmse - g_rmse) / r_rmse * 100
                        rand_str = f"{rand_imp:+.1f}%"
                        gsl_rand_str = f"{gsl_rand_imp:+.1f}%"

                    lines.append(
                        f"{ph:>3} {gsl_imp:>+11.1f}% {cgsl_imp:>+12.1f}% "
                        f"{rand_str:>13} {gsl_rand_str:>12}"
                    )

    return "\n".join(lines)


def save_results(results: List[Dict], output_dir: str):
    """Save results to JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = os.path.join(output_dir, f"experiment_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {json_path}")

    # Save formatted table
    table = format_results_table(results)
    table_path = os.path.join(output_dir, f"experiment_results_{timestamp}.txt")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Table saved to: {table_path}")

    # Save CSV for easy analysis
    csv_rows = []
    for r in results:
        if "error" in r:
            continue
        row = {
            "dataset": r["dataset"],
            "model": r["model"],
            "pre_len": r["pre_len"],
            "graph_type": r["graph_type"],
            "seed": r["seed"],
            "RMSE": r["metrics"]["RMSE"],
            "MAE": r["metrics"]["MAE"],
            "R2": r["metrics"]["R2"],
            "accuracy": r["metrics"]["accuracy"],
            "n_edges": r["graph_stats"]["n_edges"],
            "n_isolated": r["graph_stats"].get("n_isolated_nodes", "N/A"),
            "density": r["graph_stats"]["density"],
            "train_time_s": r["metrics"].get("train_time_s", 0),
        }
        csv_rows.append(row)

    csv_path = os.path.join(output_dir, f"experiment_results_{timestamp}.csv")
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")

    return json_path, table_path, csv_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Clean GSL Experiment Runner")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["shenzhen", "losloop"],
                        help="Dataset to run (default: both)")
    parser.add_argument("--model", type=str, default=None,
                        choices=["GCN", "TGCN"],
                        help="Model to run (default: both)")
    parser.add_argument("--pre_len", type=int, default=None,
                        help="Prediction horizon (default: 1-4)")
    parser.add_argument("--graph_type", type=int, default=None,
                        choices=[0, 1, 2, 3],
                        help="Graph type (0=phys, 1=GSL, 2=cGSL, 3=rand)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42],
                        help="Random seeds (default: [42])")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Max training epochs (default: 50)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--output_dir", type=str,
                        default="results/clean_reimplementation",
                        help="Output directory")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ["shenzhen", "losloop"]
    models = [args.model] if args.model else ["GCN", "TGCN"]
    pre_lens = [args.pre_len] if args.pre_len else [1, 2, 3, 4]
    graph_types = [args.graph_type] if args.graph_type is not None else [0, 1, 2, 3]

    print("=" * 80)
    print("CLEAN-ROOM GSL EXPERIMENT")
    print(f"  Datasets: {datasets}")
    print(f"  Models:   {models}")
    print(f"  PH:       {pre_lens}")
    print(f"  Graphs:   {graph_types}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Epochs:   {args.max_epochs}")
    print(f"  Device:   {args.device}")
    print(f"  Time:     {datetime.now().isoformat()}")
    print("=" * 80)

    start = time.time()
    results = run_full_experiment(
        datasets=datasets,
        models=models,
        pre_lens=pre_lens,
        graph_types=graph_types,
        seeds=args.seeds,
        max_epochs=args.max_epochs,
        device=args.device,
    )
    total_time = time.time() - start

    print(f"\n\nTotal experiment time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Print summary table
    table = format_results_table(results)
    print(table)

    # Save
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
