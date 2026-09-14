#!/bin/bash
# ============================================================
# Stage 27: Temporal Resolution Experiment (Re-run Phase 2 & 3)
# ============================================================
# Tests whether 15-min vs 5-min resolution explains the
# marginal SZ-Taxi improvement.
#
# Resamples Los-loop from 5-min to 15-min, runs DAGMA,
# and compares edge structures and forecasting performance.
#
# Estimated runtime:
#   Phase DAGMA: ~2-3 hours (828x828 matrix) - SKIPPED (already done)
#   Phase Evaluate: ~5 minutes
#   Phase Analyze: ~1 minute
# ============================================================

set -e

REPO="/data/git/mamintoosi/TGCN-GSL-PyTorch"
PYTHON="/data/python-envs/pytorch/bin/python"
SCRIPT="gsl_stage26/stage26_resolution_experiment.py"
LOG_DIR="results/stage27_resolution"

cd "$REPO"
# mkdir -p "$LOG_DIR"

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
echo "Stage 27: Temporal Resolution Experiment"
echo "Start: $(date)"
echo "========================================"

# Phase 1: DAGMA on resampled Los-loop (long run) - SKIPPED (already completed)
# echo ""
# echo "--- PHASE 1: DAGMA (resampled Los-loop 15-min) ---"
# echo "This will take ~2-3 hours..."
# run_with_limits "$PYTHON $SCRIPT --phase dagma --ph 1 --seed 42" "$LOG_DIR/phase1_dagma.log"
# echo "Phase 1 finished: $(date)"

# Phase 2: Forecasting comparison
echo ""
echo "--- PHASE 2: Forecasting ---"
run_with_limits "$PYTHON $SCRIPT --phase evaluate --ph 1 --seed 42" "$LOG_DIR/phase2_evaluate.log"
echo "Phase 2 finished: $(date)"

# Phase 3: Edge structure analysis
echo ""
echo "--- PHASE 3: Edge Analysis ---"
run_with_limits "$PYTHON $SCRIPT --phase analyze --ph 1 --seed 42" "$LOG_DIR/phase3_analyze.log"
echo "Phase 3 finished: $(date)"

echo ""
echo "========================================"
echo "ALL PHASES COMPLETE"
echo "End: $(date)"
echo "Results: $LOG_DIR/"
echo "========================================"