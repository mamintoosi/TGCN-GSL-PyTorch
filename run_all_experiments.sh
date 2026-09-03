#!/bin/bash
# run_all_experiments.sh - Complete corrected experiment runner
# Fixed: multi-seed support, Los-loop support, seed-aware filenames

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
        exit $exit_code
    fi
    
    echo "Finished at: $(date)"
    echo "Cooling down for $COOLDOWN_TIME seconds..."
    sleep $COOLDOWN_TIME
}

# ==========================================
# Phase 1: Multi-Seed SZ-Taxi DAGMA (seeds 43-46)
# PH=1 only, ~90 min each = ~6 hrs total
# PH=2,3,4 already done (seed=42)
# ==========================================
for seed in 43 44 45 46; do
    run_with_limits \
        "/data/python-envs/pytorch/bin/python gsl_stage24/stage24_run_dagma.py --ph 1 --seed $seed" \
        "results/stage24_validation/sz_ph1_seed${seed}_dagma_log.txt"
done

# ==========================================
# Phase 2: Multi-Seed SZ-Taxi Evaluation
# ==========================================
for seed in 42 43 44 45 46; do
    run_with_limits \
        "/data/python-envs/pytorch/bin/python gsl_stage24/stage24_evaluate.py --dataset shenzhen --seed $seed" \
        "results/stage24_validation/stage24_eval_sz_seed${seed}.txt"
done

# ==========================================
# Phase 3: Los-loop DAGMA (PH=1-4)
# N=207, 2N=414 (larger than SZ-Taxi)
# ==========================================
for ph in 1 2 3 4; do
    run_with_limits \
        "/data/python-envs/pytorch/bin/python gsl_stage24/stage24_run_dagma.py --ph $ph --dataset losloop --seed 42" \
        "results/stage24_validation/los_ph${ph}_seed42_dagma_log.txt"
done

# ==========================================
# Phase 4: Los-loop Evaluation
# ==========================================
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage24/stage24_evaluate.py --dataset losloop --seed 42" \
    "results/stage24_validation/stage24_eval_los_seed42.txt"

echo "========================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "End time: $(date)"
echo "========================================="
