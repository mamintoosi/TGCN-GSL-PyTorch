#!/bin/bash
# =============================================================================
# Stage 40.3 — Fit SZ-Taxi Contemporaneous DAGMA (PH=1-4)
#
# Runs each PH separately with CPU pinning and cooldown between PHs.
# Estimated runtime: ~1.5-2.5 hours per PH, ~6-10 hours total.
#
# Output:
#   results/stage33_gsl_canonical/sz_gsl_ph{1-4}_seed42_W_est.npy
#   results/stage33_gsl_canonical/sz_gsl_ph{1-4}_seed42_A_binary.npy
#
# Usage:
#   bash run_stage40_3_sz_dagma.sh           # fit all PHs sequentially
#   bash run_stage40_3_sz_dagma.sh --phs 1   # fit PH=1 only
# =============================================================================

set -e

REPO="/data/git/mamintoosi/TGCN-GSL-PyTorch"
PYTHON="/data/python-envs/pytorch/bin/python"
SCRIPT="gsl_stage26/stage40_3_fit_sz_contemporaneous.py"
LOG_DIR="results/stage33_gsl_canonical"

cd "$REPO"
mkdir -p "$LOG_DIR"

CPU_CORES="2,3"
COOLDOWN_TIME=60

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
echo "Stage 40.3 — SZ-Taxi Contemporaneous DAGMA"
echo "Start: $(date)"
echo "CPU cores: $CPU_CORES"
echo "Cooldown: ${COOLDOWN_TIME}s between PHs"
echo "========================================"

# Determine which PHs to run
PHS="${*}"
if [ -z "$PHS" ] || [ "$PHS" = "" ]; then
    PHS="--phs 1 2 3 4"
fi

# Run each PH sequentially with cooldown
for PH in 1 2 3 4; do
    # Check if this PH should be run
    if [[ "$*" == *"--phs"* ]]; then
        if [[ ! "$*" =~ [[:space:]]${PH}($|[[:space:]]) ]]; then
            continue
        fi
    fi

    # Check if already fitted
    A_PATH="$LOG_DIR/sz_gsl_ph${PH}_seed42_A_binary.npy"
    if [ -f "$A_PATH" ]; then
        echo ""
        echo "--- PH=$PH: ALREADY EXISTS, skipping ---"
        continue
    fi

    echo ""
    echo "--- PH=$PH ---"
    run_with_limits "$PYTHON $SCRIPT --phs $PH" "$LOG_DIR/sz_dagma_ph${PH}.log"
    echo "PH=$PH finished: $(date)"
done

echo ""
echo "========================================"
echo "ALL PHASES COMPLETE"
echo "End: $(date)"
echo "Results: $LOG_DIR/sz_gsl_ph*_seed42_*.npy"
echo "========================================"
