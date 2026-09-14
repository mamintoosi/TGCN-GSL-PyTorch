#!/bin/bash
# ============================================================
# Graph Structure Learning for Traffic Prediction — experiment runner
# ============================================================
# Usage:
#   bash run_experiments.sh canonical   # Stage 40 12-method matrix
#   bash run_experiments.sh aggregate   # Stage 41 mean±std from JSONs
#   bash run_experiments.sh dagma       # multi-lag DAGMA fit (long)
#   bash run_experiments.sh gsl         # contemporaneous GSL/cGSL graphs
#   bash run_experiments.sh los15       # 15-min Los-loop variant
#   bash run_experiments.sh sparse      # Stage 32 matched-30-edge controls
#   bash run_experiments.sh sweep       # Stage 58 edge-budget sweep
#   bash run_experiments.sh physical    # Physical T-GCN preds for figures
#   bash run_experiments.sh figures     # regenerate paper figures
#
# Python: set PYTHON, or use the pth env on Windows / pytorch env on Linux.
# ============================================================

set -e

if [ -z "${REPO:-}" ]; then
  REPO="$(cd "$(dirname "$0")" && pwd)"
fi
if [ -z "${PYTHON:-}" ]; then
  if [ -x "C:/programs/anaconda3/envs/pth/python.exe" ]; then
    PYTHON="C:/programs/anaconda3/envs/pth/python.exe"
  elif [ -x "/data/python-envs/pytorch/bin/python" ]; then
    PYTHON="/data/python-envs/pytorch/bin/python"
  else
    PYTHON="python"
  fi
fi

cd "$REPO"
CMD="${1:-help}"

run() {
  echo ">>> $*"
  "$PYTHON" "$@"
}

case "$CMD" in
  canonical)
    run src/run_canonical_matrix.py "${@:2}"
    ;;
  aggregate)
    run src/aggregate_stage40_results.py "${@:2}"
    ;;
  dagma)
    run src/run_multilag_dagma.py "${@:2}"
    ;;
  gsl)
    run src/run_gsl_canonical.py "${@:2}"
    ;;
  los15)
    run src/run_los15_resolution.py "${@:2}"
    ;;
  sparse)
    run src/run_sparse_controls.py "${@:2}"
    ;;
  sweep)
    run src/run_sparsity_sweep.py "${@:2}"
    ;;
  physical)
    run src/train_physical_for_figures.py "${@:2}"
    ;;
  figures)
    run src/make_results_figures.py
    run src/make_pred_vs_actual.py
    run src/analyze_sparsity_sweep.py
    ;;
  help|*)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
