#!/bin/bash
# ============================================================
# Stage 33 B: Canonical rerun of the ORIGINAL single-graph GSL baselines
# ============================================================
# Purpose: put T-GCN-GSL (and optionally GCN-GSL) into a main-text comparison
# under the SAME canonical protocol as every revised result:
#   batch 128, weight_decay 1e-4, feat_max from the train split only,
#   5 seeds (42-46), 50 epochs, SupervisedForecastTask evaluation.
#
# Why needed: the appendix GSL tables come from the ORIGINAL protocol
# (batch 64, wd 0, global-normalization code path, single seed, committed
# W_est artifacts whose generation is not reproducible). They are kept in the
# appendix as clearly-labelled historical results; this rerun supplies the
# canonical numbers for the main text.
#
# What is re-learned: DAGMA on contemporaneous training snapshots subsampled
# at every PH-th row (train[0::PH], the original input construction),
# lambda1 = 0.02 Los / 0.01 SZ, one graph per PH, DAGMA w_threshold=0.3 (the
# original protocol relied on this library default), then the Stage 36
# canonical support rule A = 1(|W| > 0) (absolute-magnitude support, negative
# survivors retained; empirically none occur at |W|>=0.3), self-loops
# removed. TRAINING DATA ONLY.
#
# Runtime estimate (CPU for DAGMA, GPU for forecasting):
#   DAGMA: ~15-40 min per PH (207x207, library-default iterations)
#          -> ~1.5-3 h for PH 1-4, computed once and cached as .npy
#   Forecasting: T-GCN, 2 variants x 5 seeds x 4 PHs x 50 epochs ~ 15-25 min
#   With --cyclic and both backbones roughly doubles the forecasting time.
# ============================================================

set -e

REPO="/data/git/mamintoosi/TGCN-GSL-PyTorch"
PYTHON="/data/python-envs/pytorch/bin/python"
SCRIPT="gsl_stage26/stage33_gsl_canonical.py"
LOG_DIR="results/stage33_gsl_canonical"

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
echo "Stage 33 B: Canonical GSL baseline rerun"
echo "Start: $(date)"
echo "========================================"

# --- Los-loop: T-GCN with physical / GSL graphs, 5 seeds, PH 1-4 ------------
# (add --cyclic to include the symmetrized cGSL appendix variant)
run_with_limits "$PYTHON $SCRIPT --dataset losloop --models tgcn --phs 1 2 3 4" \
    "$LOG_DIR/run_losloop_tgcn.log"

# --- Optional: GCN backbone (appendix comparison; comment out if not needed) -
# run_with_limits "$PYTHON $SCRIPT --dataset losloop --models gcn --phs 1 2 3 4" \
#     "$LOG_DIR/run_losloop_gcn.log"

# --- Optional: SZ-Taxi GSL rerun (~1.5-2.5 h DAGMA for PH 1-4) ---------------
# run_with_limits "$PYTHON $SCRIPT --dataset shenzhen --models tgcn --phs 1 2 3 4" \
#     "$LOG_DIR/run_shenzhen_tgcn.log"

echo ""
echo "========================================"
echo "STAGE 33 B COMPLETE"
echo "End: $(date)"
echo "Results: $LOG_DIR/stage33_gsl_canonical_results.json"
echo "========================================"
echo ""
echo "Next: copy results/stage33_gsl_canonical/ back to the Windows machine;"
echo "the report will map the numbers into the manuscript story."
