#!/bin/bash
# run_all_experiments_stage25d_only.sh
# Re-runs ONLY Stage 25D (Multi-Lag Pilot) — the only stage that failed.
# All other Stage 25 stages (A, B, C) completed successfully.
#
# Bug fixed: DagDagmaLinear -> DagmaLinear (dagma 1.1.1 API)

cd /data/git/mamintoosi/TGCN-GSL-PyTorch || exit 1

CPU_CORES="0,1"
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
    
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Command failed with exit code $exit_code"
        echo "Continuing to next experiment..."
    fi
    
    echo "Finished at: $(date)"
    echo "Cooling down for $COOLDOWN_TIME seconds..."
    sleep $COOLDOWN_TIME
}

# ============================================================
# Stage 25D: Multi-Lag DAGMA Pilot (Family G)
# Small-scale test with N=20 sensors, 3 lags
# ~5-10 min per dataset
# ============================================================
echo "=== Stage 25D: Multi-Lag Pilot ==="
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage25/stage25_multilag_pilot.py --dataset shenzhen --n-sensors 20 --lags 3 --max-iter 30000 --warm-iter 15000" \
    "results/stage25_validation/stage25D_sz_multilag_pilot.txt"

run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage25/stage25_multilag_pilot.py --dataset losloop --n-sensors 20 --lags 3 --max-iter 30000 --warm-iter 15000" \
    "results/stage25_validation/stage25D_los_multilag_pilot.txt"

echo "========================================="
echo "STAGE 25D EXPERIMENTS COMPLETED!"
echo "End time: $(date)"
echo "========================================="
