#!/bin/bash
# ============================================================
# Stage 32: Sparse-Control Experiment (Reviewer 1, Weakness 5)
# ============================================================
# Matched-edge-count sparse baselines at the SAME 30-edge budget
# as T-GCN-MultiGSL-Mix (sum over lag graphs = 12+3+15):
#
#   CorrTop30  : top-30 |Pearson| edges from TRAINING data only,
#                trained with plain standard TGCN. Strongest
#                non-DAGMA heuristic (112-experiment archive).
#   RandTop30  : 30 random off-diagonal directed edges (re-drawn
#                per seed), trained with plain TGCN. Floor control.
#
# Both go through the EXACT canonical Stage 26 pipeline:
#   SupervisedForecastTask(loss="mse_with_regularizer"), set_seed,
#   Adam(lr=0.001, wd=0.0001), batch 128, 50 epochs, full-batch
#   test evaluation, feat_max from training data only.
#
# Answer to R1-W5 is "learned structure": if CorrTop30/RandTop30
# land near/below NoGraph (5.143) and far from MultiGSL-Mix
# (4.452+-0.143), the gains are NOT explained by sparsity alone.
#
# Estimated runtime: ~5 min total on GPU, ~30 min on CPU.
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
echo "Stage 32: Sparse-Control Experiment"
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
echo "STAGE 32 COMPLETE"
echo "End: $(date)"
echo "Results: $LOG_DIR/stage32_sparse_control.json"
echo "========================================"
echo ""
echo "Next: copy results/stage32_sparse_control/ back to the Windows"
echo "machine so the manuscript table can cite the verified numbers."
