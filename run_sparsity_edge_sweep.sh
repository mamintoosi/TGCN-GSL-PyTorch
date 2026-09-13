#!/bin/bash
# ============================================================
# Stage 58: matched-edge sparsity sweep (Los-loop PH1)
# ============================================================
# Pattern follows run_tgcn_gcn_audit.sh
#
# Modes:
#   smoke — 2 epochs, K=30, seed 42 only (quick canary)
#   full  — 50 epochs, K in {10,20,30,50,80}, seeds 42-46
#
# Usage:
#   bash run_sparsity_edge_sweep.sh              # smoke
#   MODE=full bash run_sparsity_edge_sweep.sh    # full grid (hours)
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
SCRIPT="gsl_stage58_sparsity/sparsity_edge_sweep.py"
LOG_DIR="results/stage58_sparsity_sweep"
mkdir -p "$LOG_DIR"

CPU_CORES="${CPU_CORES:-2,3}"
COOLDOWN_TIME="${COOLDOWN_TIME:-5}"
MODE="${MODE:-smoke}"

run_with_limits() {
    local cmd="$1"
    local log_file="$2"
    echo "========================================="
    echo "Running: $cmd"
    echo "Log: $log_file"
    echo "Start: $(date)"
    echo "========================================="
    if command -v taskset >/dev/null 2>&1; then
        taskset -c $CPU_CORES $cmd 2>&1 | tee "$log_file"
    else
        $cmd 2>&1 | tee "$log_file"
    fi
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: exit $exit_code"
        exit $exit_code
    fi
    echo "Finished: $(date)"
    [ "$COOLDOWN_TIME" -gt 0 ] && sleep "$COOLDOWN_TIME"
    return 0
}

echo "Stage 58 sparsity sweep | Mode=$MODE | Python=$PYTHON"

if [ "$MODE" = "smoke" ]; then
    run_with_limits "$PYTHON $SCRIPT --budgets 30 --seeds 42 --epochs 2 --out-name smoke" \
        "$LOG_DIR/smoke.log"
elif [ "$MODE" = "full" ]; then
    # NoSpatial + 5 budgets × 2 methods × 5 seeds = 1 + 50 runs
    run_with_limits "$PYTHON $SCRIPT --budgets 10 20 30 50 80 --seeds 42 43 44 45 46 --epochs 50 --out-name full" \
        "$LOG_DIR/full.log"
else
    echo "Unknown MODE=$MODE (smoke|full)"
    exit 1
fi

echo "Done. Results: $LOG_DIR/full.csv or smoke.csv"
echo "Interpretation: if CorrTopK ≈ NoSpatial at all K but DAGMA MultiGSL < NoSpatial at K=30,"
echo "then sparsity alone does not explain the multi-lag gain."
