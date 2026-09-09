#!/bin/bash
# ============================================================
# Stage 33 A: Sparse-Control Experiment (Stage 32 script, canonical names)
# ============================================================
# Answers Reviewer 1, Weakness 5: "is the MultiGSL-Mix gain just sparsity?"
#
#   CorrTop30 : top-30 |Pearson| edges from TRAINING data only, static TGCN
#   RandTop30 : 30 random directed edges (re-drawn per seed), static TGCN
#
# Matched edge budget: 30 directed edges = the sum over the three lag graphs
# used by T-GCN-MultiGSL/Mix (lag_1 12 + lag_2 3 + lag_3 15). The union of
# those edges is 28; the controls' budget of 30 is slightly GENEROUS to the
# controls, which is conservative in favour of the method.
#
# Canonical pipeline: SupervisedForecastTask(loss="mse_with_regularizer"),
# set_seed, Adam(lr=0.001, wd=0.0001), batch 128, 50 epochs, full-batch test
# evaluation, feat_max from training data only.
#
# Runtime: ~5 min on GPU, ~30 min on CPU (taskset-limited).
# ============================================================

set -e

REPO="/data/git/mamintoosi/TGCN-GSL-PyTorch"
PYTHON="/data/python-envs/pytorch/bin/python"
SCRIPT="gsl_stage26/stage32_sparse_control.py"
LOG_DIR="results/stage32_sparse_control"

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
echo "Stage 33 A: Sparse-Control Experiment"
echo "Start: $(date)"
echo "========================================"

# Sanity check: DAGMA lag blocks must exist (needed for the 30-edge reference)
if [ ! -f "results/stage26_validation/los_ph1_seed42_L3_lag_1.npy" ]; then
    echo "ERROR: DAGMA lag blocks missing in results/stage26_validation/"
    echo "       (los_ph1_seed42_L3_lag_{1,2,3}.npy). Copy them from the"
    echo "       machine that ran Stage 26 first."
    exit 1
fi

# Single run: seeds 42-46, PH=1, both controls (10 trainings)
run_with_limits "$PYTHON $SCRIPT --seeds 42 43 44 45 46 --ph 1" \
    "$LOG_DIR/run.log"

echo ""
echo "========================================"
echo "STAGE 33 A COMPLETE"
echo "End: $(date)"
echo "Results: $LOG_DIR/stage32_sparse_control.json"
echo "========================================"
echo ""
echo "Next: copy results/stage32_sparse_control/ back to the Windows"
echo "machine so the manuscript table can cite the verified numbers."
