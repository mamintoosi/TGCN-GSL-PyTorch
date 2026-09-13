#!/bin/bash
# ============================================================
# Train T-GCN-Physical (PH1, seed 42) for manuscript figures
# ============================================================
# Saves under results/stage26_checkpoint/:
#   y_true.npy, y_pred.npy, train_loss_history.json
# for los_ph1_seed42_physical and sz_ph1_seed42_physical.
#
# Then regenerates:
#   pred_vs_actual_* (Physical vs Mix)
#   train_loss_curves_los_ph1 (Physical vs Mix)
#
# Runtime: tens of minutes per dataset on CPU (dense physical T-GCN).
# Usage:
#   bash run_physical_for_figures.sh              # Los-loop only
#   DATASET=both bash run_physical_for_figures.sh # Los + SZ
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
DATASET="${DATASET:-losloop}"
LOG_DIR="results/stage26_checkpoint"
mkdir -p "$LOG_DIR"

echo "========================================="
echo "Physical T-GCN for figures | dataset=$DATASET"
echo "Python: $PYTHON"
echo "Start: $(date)"
echo "========================================="

"$PYTHON" paper/revised_version/scripts/train_physical_for_figures.py \
  --dataset "$DATASET" 2>&1 | tee "$LOG_DIR/train_physical_${DATASET}.log"

echo "Regenerating figures..."
"$PYTHON" paper/revised_version/scripts/make_pred_vs_actual.py
"$PYTHON" paper/revised_version/scripts/make_results_figures.py

echo "Done: $(date)"
echo "If Physical histories exist, train-loss figure is now Physical vs Mix."
echo "Recompile the manuscript if needed."
