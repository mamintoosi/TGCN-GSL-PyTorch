#!/bin/bash
# ============================================================
# Stage 29: Los-Loop as a 15-Minute Dataset
# ============================================================
# Treats Los-Loop at 15-min resolution, independent of original
# 5-min version. Uses exact same pipeline as Stage 26.
#
# Estimated runtime:
#   Phase DAGMA:  ~2-3 hours (828x828 matrix)
#   Phase Forecast: ~2-3 hours (3 methods × 5 seeds × 4 PHs)
#   Phase Analyze:  ~1 minute
# ============================================================

set -e

REPO="/data/git/mamintoosi/TGCN-GSL-PyTorch"
PYTHON="/data/python-envs/pytorch/bin/python"
SCRIPT="gsl_stage26/stage29_los15min.py"
LOG_DIR="results/stage29_los15min"

cd "$REPO"
mkdir -p "$LOG_DIR"

CPU_CORES="2,3"
COOLDOWN_TIME=10

run_with_limits() {
    local cmd="$1"
    local log_file="$2"

    echo "========================================="
    echo "Running: $cmd"
    echo "Log: $log_file"
    echo "Start time: $(date)"
    echo "========================================="

    taskset -c $CPU_CORES $cmd 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Command failed with exit code $exit_code"
        exit $exit_code
    fi

    echo "Finished at: $(date)"
    if [ $COOLDOWN_TIME -gt 0 ]; then
        echo "Cooling down for $COOLDOWN_TIME seconds..."
        sleep $COOLDOWN_TIME
    fi
}

echo "========================================"
echo "Stage 29: Los-Loop as 15-Minute Dataset"
echo "Start: $(date)"
echo "========================================"

# Phase 1: DAGMA on Los-Loop-15min
# DAGMA is PH-independent (computed once, reused for all PHs).
# Stage 27 already has results for the same data; script reuses them.
echo ""
echo "--- PHASE 1: DAGMA ---"
if [ -f "$LOG_DIR/los15_ph1_seed42_L3_W_full.npy" ]; then
    echo "DAGMA results already exist. Skipping computation."
else
    echo "Computing DAGMA (reuses Stage 27 if available, otherwise ~2-3 hours)..."
    run_with_limits "$PYTHON $SCRIPT --phase dagma --seed 42" "$LOG_DIR/phase1_dagma.log"
fi
echo "Phase 1 finished: $(date)"

# Phase 2: Multi-seed forecasting (seeds 42-46, PH 1-4)
echo ""
echo "--- PHASE 2: Forecasting (5 seeds × 4 PHs × 3 methods) ---"
echo "This will take ~2-3 hours..."
run_with_limits "$PYTHON $SCRIPT --phase forecast --seeds 42 43 44 45 46 --phs 1 2 3 4" "$LOG_DIR/phase2_forecast.log"
echo "Phase 2 finished: $(date)"

# Phase 3: Analysis
echo ""
echo "--- PHASE 3: Analysis ---"
run_with_limits "$PYTHON $SCRIPT --phase analyze --phs 1 2 3 4 --seeds 42 43 44 45 46" "$LOG_DIR/phase3_analyze.log"
echo "Phase 3 finished: $(date)"

echo ""
echo "========================================"
echo "ALL PHASES COMPLETE"
echo "End: $(date)"
echo "Results: $LOG_DIR/"
echo "========================================"
