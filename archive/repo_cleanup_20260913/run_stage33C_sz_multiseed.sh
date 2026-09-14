#!/bin/bash
# ============================================================
# Stage 33 C: SZ-Taxi multi-seed validation (canonical pipeline)
# ============================================================
# Purpose: bring the main-text SZ-Taxi table to the same five-seed standard
# as the Los-loop tables. Reuses the EXISTING SZ DAGMA lag blocks
# (results/stage26_validation/sz_ph*_seed42_L3_*.npy) — NO DAGMA recomputation.
#
# Answers: is the marginal SZ-Taxi improvement (<=0.3% PH1-3, -0.02% PH4,
# single seed) stable across seeds, or is the PH=4 dip seed noise?
# Supports the manuscript's dataset-dependence claim with mean+-std.
#
# Methods: T-GCN-NoSpatial, T-GCN-MultiGSL, T-GCN-MultiGSL-Mix
# (canonical pipeline: batch 128, wd 1e-4, 50 epochs, feat_max = train max).
#
# Runtime: ~30-40 min on GPU (3 methods x 5 seeds x 4 PHs = 60 trainings,
# SZ-Taxi is the smaller dataset).
# ============================================================

set -e

REPO="/data/git/mamintoosi/TGCN-GSL-PyTorch"
PYTHON="/data/python-envs/pytorch/bin/python"
SCRIPT="gsl_stage26/stage33_sz_multiseed.py"
LOG_DIR="results/stage33_sz_multiseed"

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
echo "Stage 33 C: SZ-Taxi multi-seed validation"
echo "Start: $(date)"
echo "========================================"

# Sanity check: SZ DAGMA lag blocks must exist (no DAGMA will be recomputed)
if [ ! -f "results/stage26_validation/sz_ph1_seed42_L3_lag_1.npy" ]; then
    echo "ERROR: SZ DAGMA lag blocks missing in results/stage26_validation/"
    echo "       (sz_ph1_seed42_L3_lag_{1,2,3}.npy). Copy them from the"
    echo "       machine that ran Stage 26 first."
    exit 1
fi

run_with_limits "$PYTHON $SCRIPT --seeds 42 43 44 45 46 --phs 1 2 3 4" \
    "$LOG_DIR/run.log"

echo ""
echo "========================================"
echo "STAGE 33 C COMPLETE"
echo "End: $(date)"
echo "Results: $LOG_DIR/stage33_sz_multiseed_results.json"
echo "========================================"
echo ""
echo "Next: copy results/stage33_sz_multiseed/ back to the Windows machine;"
echo "the SZ-Taxi table can then report five-seed mean+-std."
