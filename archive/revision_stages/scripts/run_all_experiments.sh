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

# ============================================================
# STAGE 25: TEMPORAL FUNCTIONAL GRAPH ANALYSIS & EXPERIMENTS
# ============================================================
# All DAGMA matrices already computed (Stage 24).
# These experiments use existing matrices only.

mkdir -p results/stage25_validation

# ============================================================
# Stage 25A: Graph Structural Analysis (Family A+B+H+I)
# Fast: no training, pure numpy analysis
# ============================================================
echo "=== Stage 25A: Graph Structural Analysis ==="
/data/python-envs/pytorch/bin/python gsl_stage25/stage25_graph_analysis.py --dataset shenzhen \
    2>&1 | tee results/stage25_validation/stage25A_sz_analysis.txt

/data/python-envs/pytorch/bin/python gsl_stage25/stage25_graph_analysis.py --dataset losloop \
    2>&1 | tee results/stage25_validation/stage25A_los_analysis.txt

# ============================================================
# Stage 25B: Graph Ensembles & Physical-DAGMA Fusion (Family C+D)
# Evaluates: ensembles, intersections, weighted fusion
# ~20-40 min per dataset per PH (all baselines + new constructions)
# ============================================================
echo "=== Stage 25B: Graph Ensembles & Fusion ==="
for ph in 1 2 3 4; do
    run_with_limits \
        "/data/python-envs/pytorch/bin/python gsl_stage25/stage25_graph_ensembles.py --dataset shenzhen --ph $ph --seed 42 --max-epochs 50" \
        "results/stage25_validation/stage25B_sz_ph${ph}_ensembles.txt"
done

for ph in 1 2 3 4; do
    run_with_limits \
        "/data/python-envs/pytorch/bin/python gsl_stage25/stage25_graph_ensembles.py --dataset losloop --ph $ph --seed 42 --max-epochs 50" \
        "results/stage25_validation/stage25B_los_ph${ph}_ensembles.txt"
done

# ============================================================
# Stage 25C: Dual-Graph & Warm-Up Refinement (Family E+F)
# Introduces new model architectures (DualGCN, DualTGCN)
# ~20-30 min per dataset
# ============================================================
echo "=== Stage 25C: Dual-Graph & Warm-Up ==="
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage25/stage25_dual_graph.py --dataset shenzhen --ph 1 --seed 42 --max-epochs 50" \
    "results/stage25_validation/stage25C_sz_dual_warmup.txt"

run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage25/stage25_dual_graph.py --dataset losloop --ph 1 --seed 42 --max-epochs 50" \
    "results/stage25_validation/stage25C_los_dual_warmup.txt"

# ============================================================
# Stage 25D: Multi-Lag DAGMA Pilot (Family G)
# Small-scale test with N=20 sensors, 3 lags
# ~5-10 min
# ============================================================
echo "=== Stage 25D: Multi-Lag Pilot ==="
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage25/stage25_multilag_pilot.py --dataset shenzhen --n-sensors 20 --lags 3 --max-iter 30000 --warm-iter 15000" \
    "results/stage25_validation/stage25D_sz_multilag_pilot.txt"

run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage25/stage25_multilag_pilot.py --dataset losloop --n-sensors 20 --lags 3 --max-iter 30000 --warm-iter 15000" \
    "results/stage25_validation/stage25D_los_multilag_pilot.txt"

echo "========================================="
echo "STAGE 25 EXPERIMENTS COMPLETED!"
echo "End time: $(date)"
echo "========================================="

# ============================================================
# STAGE 26 — FULL-SENSOR MULTI-LAG DAGMA + FORECASTING
# ============================================================
# Multi-lag DAGMA with L=3 lags on ALL sensors.
# SZ-Taxi: (L+1)*N = 4*156 = 624 variables -> 624x624 matrix
# Los-loop: (L+1)*N = 4*207 = 828 variables -> 828x828 matrix
# Each DAGMA run: estimated 60-180 min depending on dataset size.
#
# Multi-lag block interpretation:
#   lag_3: W[0:N, L*N:(L+1)*N] = sensor_i(t-3) -> sensor_j(t)
#   lag_2: W[N:2N, L*N:(L+1)*N] = sensor_i(t-2) -> sensor_j(t)
#   lag_1: W[2N:3N, L*N:(L+1)*N] = sensor_i(t-1) -> sensor_j(t)
#   current: W[3N:4N, 3N:4N] = contemporaneous

# ============================================================
# STAGE 26A: DAGMA EXTRACTION (expensive, ~1-3 hrs per run)
# ============================================================

# --- SZ-Taxi PH=1 (624x624 matrix, ~60-120 min) ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py --ph 1 --dataset shenzhen --lags 3 --seed 42" \
    "results/stage26_validation/stage26A_sz_ph1_dagma_log.txt"

# --- SZ-Taxi PH=2 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py --ph 2 --dataset shenzhen --lags 3 --seed 42" \
    "results/stage26_validation/stage26A_sz_ph2_dagma_log.txt"

# --- SZ-Taxi PH=3 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py --ph 3 --dataset shenzhen --lags 3 --seed 42" \
    "results/stage26_validation/stage26A_sz_ph3_dagma_log.txt"

# --- SZ-Taxi PH=4 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py --ph 4 --dataset shenzhen --lags 3 --seed 42" \
    "results/stage26_validation/stage26A_sz_ph4_dagma_log.txt"

# --- Los-loop PH=1 (828x828 matrix, ~90-180 min) ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py --ph 1 --dataset losloop --lags 3 --seed 42" \
    "results/stage26_validation/stage26A_los_ph1_dagma_log.txt"

# --- Los-loop PH=2 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py --ph 2 --dataset losloop --lags 3 --seed 42" \
    "results/stage26_validation/stage26A_los_ph2_dagma_log.txt"

# --- Los-loop PH=3 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py --ph 3 --dataset losloop --lags 3 --seed 42" \
    "results/stage26_validation/stage26A_los_ph3_dagma_log.txt"

# --- Los-loop PH=4 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_run_dagma.py --ph 4 --dataset losloop --lags 3 --seed 42" \
    "results/stage26_validation/stage26A_los_ph4_dagma_log.txt"

# ============================================================
# STAGE 26B: FORECASTING EVALUATION
# (requires Stage 26A outputs)
# Each evaluation: ~15-30 min per (dataset, PH) pair
# ============================================================

# --- SZ-Taxi PH=1 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_evaluate.py --dataset shenzhen --ph 1 --seed 42 --max-epochs 50 --n-lags 3 --threshold 0.1" \
    "results/stage26_validation/stage26B_sz_ph1_eval_log.txt"

# --- SZ-Taxi PH=2 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_evaluate.py --dataset shenzhen --ph 2 --seed 42 --max-epochs 50 --n-lags 3 --threshold 0.1" \
    "results/stage26_validation/stage26B_sz_ph2_eval_log.txt"

# --- SZ-Taxi PH=3 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_evaluate.py --dataset shenzhen --ph 3 --seed 42 --max-epochs 50 --n-lags 3 --threshold 0.1" \
    "results/stage26_validation/stage26B_sz_ph3_eval_log.txt"

# --- SZ-Taxi PH=4 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_evaluate.py --dataset shenzhen --ph 4 --seed 42 --max-epochs 50 --n-lags 3 --threshold 0.1" \
    "results/stage26_validation/stage26B_sz_ph4_eval_log.txt"

# --- Los-loop PH=1 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_evaluate.py --dataset losloop --ph 1 --seed 42 --max-epochs 50 --n-lags 3 --threshold 0.1" \
    "results/stage26_validation/stage26B_los_ph1_eval_log.txt"

# --- Los-loop PH=2 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_evaluate.py --dataset losloop --ph 2 --seed 42 --max-epochs 50 --n-lags 3 --threshold 0.1" \
    "results/stage26_validation/stage26B_los_ph2_eval_log.txt"

# --- Los-loop PH=3 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_evaluate.py --dataset losloop --ph 3 --seed 42 --max-epochs 50 --n-lags 3 --threshold 0.1" \
    "results/stage26_validation/stage26B_los_ph3_eval_log.txt"

# --- Los-loop PH=4 ---
run_with_limits \
    "/data/python-envs/pytorch/bin/python gsl_stage26/stage26_evaluate.py --dataset losloop --ph 4 --seed 42 --max-epochs 50 --n-lags 3 --threshold 0.1" \
    "results/stage26_validation/stage26B_los_ph4_eval_log.txt"

echo "========================================="
echo "STAGE 26 EXPERIMENTS COMPLETED!"
echo "End time: $(date)"
echo "========================================="

# ============================================================
