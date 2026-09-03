#!/usr/bin/env python
"""
Sparsified-Physical-Graph Ablation Study
=========================================
Compares:
  1. Physical graph (baseline)
  2. Sparse-random physical graph (same edge count as GSL, random topology)
  3. GSL (DAGMA learned)
  4. cGSL (symmetrized DAGMA)

For GCN and T-GCN on both SZ-Taxi and Los-loop, PH=1..4.
"""

import subprocess
import sys
import os
import csv
import json
import time
import re

PYTHON = "/data/python-envs/pytorch/bin/python"
REPO_ROOT = "/data/git/mamintoosi/TGCN-GSL-PyTorch"

# All experiment configurations
EXPERIMENTS = []

# Models and their use_gsl values
# use_gsl=0: physical graph
# use_gsl=1: GSL (DAGMA)
# use_gsl=2: cGSL (symmetrized DAGMA)
# use_gsl=3: sparse random physical (ablation)

MODELS = [
    ("GCN", "models.GCN", 100),
    ("TGCN", "models.TGCN", 100),
]

DATASETS = [
    ("shenzhen", "SZ-Taxi"),
    ("losloop", "Los-loop"),
]

PRE_LENS = [1, 2, 3, 4]

# Graph types to test
GRAPH_TYPES = [
    (0, "physical", "Physical graph baseline"),
    (3, "sparse_random", "Sparse random (same edge count as GSL)"),
    (1, "gsl", "DAGMA GSL"),
    (2, "dcg", "DAGMA cGSL (symmetrized)"),
]

# Generate all experiment configs
for dataset_key, dataset_label in DATASETS:
    for model_name, model_class, hidden_dim in MODELS:
        for pre_len in PRE_LENS:
            for gsl_val, gsl_label, gsl_desc in GRAPH_TYPES:
                EXPERIMENTS.append({
                    "dataset": dataset_key,
                    "dataset_label": dataset_label,
                    "model": model_name,
                    "model_class": model_class,
                    "hidden_dim": hidden_dim,
                    "pre_len": pre_len,
                    "gsl": gsl_val,
                    "gsl_label": gsl_label,
                    "gsl_desc": gsl_desc,
                })

print(f"Total experiments to run: {len(EXPERIMENTS)}")
print(f"  {len(DATASETS)} datasets × {len(MODELS)} models × {len(PRE_LENS)} horizons × {len(GRAPH_TYPES)} graph types")


def create_config(exp, seed=42):
    """Create a temporary config YAML for this experiment."""
    config = {
        "fit": {
            "trainer": {"max_epochs": 50, "accelerator": "cuda", "devices": 1},
            "data": {
                "dataset_name": exp["dataset"],
                "batch_size": 64,
                "seq_len": 12,
                "pre_len": exp["pre_len"],
            },
            "model": {
                "model": {
                    "class_path": exp["model_class"],
                    "init_args": {
                        "hidden_dim": exp["hidden_dim"],
                        "use_gsl": exp["gsl"],
                    },
                },
                "learning_rate": 0.001,
                "weight_decay": 0,
                "loss": "mse",
            },
        }
    }

    import yaml
    config_path = f"/tmp/ablation_config_{exp['dataset']}_{exp['model']}_pre{exp['pre_len']}_gsl{exp['gsl']}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return config_path


def run_experiment(exp, seed=42):
    """Run a single experiment and extract metrics."""
    config_path = create_config(exp, seed)

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)

    cmd = [
        PYTHON, "main.py",
        "--config", config_path,
        "--device", "cuda",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max per experiment
            env=env,
        )
        elapsed = time.time() - t0

        # Parse output for metrics
        output = result.stdout + "\n" + result.stderr

        # Find the last epoch's metrics from the log
        metrics = {}
        for line in output.split("\n"):
            if "[Epoch" in line and "RMSE:" in line:
                # Extract RMSE, MAE, Accuracy, R2
                rmse_m = re.search(r"RMSE:\s*([\d.]+)", line)
                mae_m = re.search(r"MAE:\s*([\d.]+)", line)
                acc_m = re.search(r"Accuracy:\s*([\d.]+)", line)
                r2_m = re.search(r"R2:\s*([\d.]+)", line)
                if rmse_m:
                    metrics["RMSE"] = float(rmse_m.group(1))
                if mae_m:
                    metrics["MAE"] = float(mae_m.group(1))
                if acc_m:
                    metrics["Accuracy"] = float(acc_m.group(1))
                if r2_m:
                    metrics["R2"] = float(r2_m.group(1))

        metrics["elapsed"] = elapsed
        metrics["success"] = result.returncode == 0
        metrics["output_tail"] = "\n".join(output.strip().split("\n")[-20:])

        return metrics

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "elapsed": time.time() - t0}
    except Exception as e:
        return {"success": False, "error": str(e), "elapsed": time.time() - t0}
    finally:
        # Clean up temp config
        if os.path.exists(config_path):
            os.remove(config_path)


def main():
    results = []
    total = len(EXPERIMENTS)

    for i, exp in enumerate(EXPERIMENTS):
        label = f"{exp['dataset_label']} {exp['model']} PH={exp['pre_len']} [{exp['gsl_label']}]"
        print(f"\n[{i+1}/{total}] {label}")
        print(f"  {exp['gsl_desc']}")

        metrics = run_experiment(exp)

        row = {
            "dataset": exp["dataset_label"],
            "model": exp["model"],
            "pre_len": exp["pre_len"],
            "graph_type": exp["gsl_label"],
            "graph_desc": exp["gsl_desc"],
            **metrics,
        }
        results.append(row)

        if metrics.get("success"):
            print(f"  RMSE={metrics.get('RMSE', 'N/A'):.4f}  "
                  f"MAE={metrics.get('MAE', 'N/A'):.4f}  "
                  f"Acc={metrics.get('Accuracy', 'N/A'):.4f}  "
                  f"R2={metrics.get('R2', 'N/A'):.4f}  "
                  f"({metrics['elapsed']:.1f}s)")
        else:
            print(f"  FAILED: {metrics.get('error', 'unknown')}")

    # Save results
    results_path = "doc/ablation/ablation_results.json"
    os.makedirs("doc/ablation", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save as CSV
    csv_path = "doc/ablation/ablation_results.csv"
    if results:
        keys = results[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    # Print summary table
    print("\n" + "=" * 100)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Dataset':<12} {'Model':<6} {'PH':<4} {'Graph':<16} {'RMSE':>8} {'MAE':>8} {'Acc':>8} {'R2':>8}")
    print("-" * 100)
    for row in results:
        if row.get("success"):
            print(f"{row['dataset']:<12} {row['model']:<6} {row['pre_len']:<4} "
                  f"{row['graph_type']:<16} {row['RMSE']:>8.4f} {row['MAE']:>8.4f} "
                  f"{row['Accuracy']:>8.4f} {row['R2']:>8.4f}")
        else:
            print(f"{row['dataset']:<12} {row['model']:<6} {row['pre_len']:<4} "
                  f"{row['graph_type']:<16} {'FAILED':>8}")

    print(f"\nResults saved to: {results_path}")
    print(f"Results saved to: {csv_path}")
    return results


if __name__ == "__main__":
    main()
