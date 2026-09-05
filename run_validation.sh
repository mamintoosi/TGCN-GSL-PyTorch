#!/bin/bash
# run_validation.sh — Stage 26 Validation Experiments ONLY
# Uses existing DAGMA matrices. No recomputation needed.
# Estimated total: ~4-6 hours

cd /data/git/mamintoosi/TGCN-GSL-PyTorch || exit 1

PYTHON=/data/python-envs/pytorch/bin/python
RESULTS=results/stage26_validation
COOLDOWN=60

run_with_limits() {
    local cmd="$1"
    local log_file="$2"
    echo "========================================="
    echo "Running: $cmd"
    echo "Log: $log_file"
    echo "Start time: $(date)"
    echo "========================================="
    taskset -c 0,1 $cmd 2>&1 | tee "$log_file"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Command failed with exit code $exit_code"
        exit $exit_code
    fi
    echo "Finished at: $(date)"
    echo "Cooling down for $COOLDOWN seconds..."
    sleep $COOLDOWN
}

mkdir -p $RESULTS

# ============================================================
# Experiment A: Multi-seed validation (seeds 42-46)
# ~2.5-4 hrs
# ============================================================
echo "=== Experiment A: Multi-Seed Validation ==="
run_with_limits \
    "$PYTHON gsl_stage26/stage26_validation.py --experiment A --dataset losloop --ph 1 --seeds 42,43,44,45,46 --max-epochs 50" \
    "$RESULTS/validation_A_multiseed_log.txt"

# ============================================================
# Experiment B: Parameter-matched NoGraph control
# ~30-45 min
# ============================================================
echo "=== Experiment B: Parameter-Matched Control ==="
run_with_limits \
    "$PYTHON gsl_stage26/stage26_validation.py --experiment B --dataset losloop --ph 1 --seed 42 --max-epochs 50" \
    "$RESULTS/validation_B_param_match_log.txt"

# ============================================================
# Experiment C: Lag ablation
# ~70-105 min
# ============================================================
echo "=== Experiment C: Lag Ablation ==="
run_with_limits \
    "$PYTHON gsl_stage26/stage26_validation.py --experiment C --dataset losloop --ph 1 --seed 42 --max-epochs 50" \
    "$RESULTS/validation_C_lag_ablation_log.txt"

echo "========================================="
echo "ALL VALIDATION EXPERIMENTS COMPLETED!"
echo "End time: $(date)"
echo "========================================="
