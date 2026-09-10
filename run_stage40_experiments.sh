#!/bin/bash
# ============================================================
# Stage 40: Canonical Experiment Pipeline Launcher
# ============================================================
# Runs the full experiment matrix: T-GCN + GCN families,
# both datasets (losloop, shenzhen), PH 1-4, 5 seeds.
#
# Runs per-PH with CPU pinning and cooldown between PHs.
# The Python script is resumable: it detects existing results
# and skips them.  Safe to re-run after interruption.
#
# Usage:
#   bash run_stage40_experiments.sh                    # full run
#   bash run_stage40_experiments.sh --dry-run          # preview only
#   bash run_stage40_experiments.sh --backbone gcn     # GCN only
#   bash run_stage40_experiments.sh --variants gsl cgsl --datasets losloop  # subset
# ============================================================

set -e

REPO="/data/git/mamintoosi/TGCN-GSL-PyTorch"
PYTHON="/data/python-envs/pytorch/bin/python"
SCRIPT="gsl_stage40/scripts/stage40_run_all.py"
LOG_DIR="results/stage40_canonical/logs"

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
echo "Stage 40: Canonical Experiment Pipeline"
echo "Start: $(date)"
echo "CPU cores: $CPU_CORES"
echo "Cooldown: ${COOLDOWN_TIME}s between PHs"
echo "========================================"

# If --dry-run, just run it and exit
if [[ "$*" == *"--dry-run"* ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/dry_run_${TIMESTAMP}.log"
    $PYTHON "$SCRIPT" "$@" 2>&1 | tee "$LOG_FILE"
    exit ${PIPESTATUS[0]}
fi

# Run each PH sequentially with cooldown
for PH in 1 2 3 4; do
    echo ""
    echo "--- PH=$PH ---"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/ph${PH}_${TIMESTAMP}.log"

    # Build command: pass all args plus --phs $PH
    CMD="$PYTHON $SCRIPT --phs $PH"

    # Forward extra args (excluding --phs which we override)
    EXTRA_ARGS=""
    SKIP_NEXT=false
    for arg in "$@"; do
        if $SKIP_NEXT; then
            SKIP_NEXT=false
            continue
        fi
        if [ "$arg" = "--phs" ]; then
            SKIP_NEXT=true
            continue
        fi
        EXTRA_ARGS="$EXTRA_ARGS $arg"
    done
    CMD="$CMD$EXTRA_ARGS"

    run_with_limits "$CMD" "$LOG_FILE"
    echo "PH=$PH finished: $(date)"
done

echo ""
echo "========================================"
echo "ALL PHASES COMPLETE"
echo "End: $(date)"
echo "Results: results/stage40_canonical/training/"
echo "Logs: $LOG_DIR/"
echo "========================================"
