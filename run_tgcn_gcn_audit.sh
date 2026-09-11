#!/bin/bash
# ============================================================
# Stage 57: T-GCN vs GCN fairness audit (loss isolation)
# ============================================================
# Pattern follows run_resolution_experiment.sh
# Does NOT run full multi-seed grid by default — use MODE=full.
#
# Modes:
#   smoke  — 2 epochs, Los-loop PH1 seed 42, all arms, both adj kinds
#   quick  — 50 epochs, Los PH1 seed 42, arms A–D
#   full   — 50 epochs, both datasets, PH1-4, seeds 42-46, arms A–D
#
# Usage:
#   bash run_tgcn_gcn_audit.sh          # smoke
#   MODE=quick bash run_tgcn_gcn_audit.sh
#   MODE=full  bash run_tgcn_gcn_audit.sh
# ============================================================

set -e

# --- environment (edit if needed; same spirit as run_resolution_experiment.sh) ---
if [ -z "${REPO:-}" ]; then
  REPO="$(cd "$(dirname "$0")" && pwd)"
fi
if [ -z "${PYTHON:-}" ]; then
  if [ -x "C:/programs/anaconda3/envs/pth/python.exe" ]; then
    PYTHON="C:/programs/anaconda3/envs/pth/python.exe"
  elif [ -x "/data/python-envs/pytorch/bin/python" ]; then
    PYTHON="/data/python-envs/pytorch/bin/python"
  else
    PYTHON="python"
  fi
fi

cd "$REPO"

SCRIPT="gsl_stage57_tgcn_gcn_audit/audit_tgcn_gcn.py"
LOG_DIR="results/stage57_tgcn_gcn_audit"
mkdir -p "$LOG_DIR"

CPU_CORES="${CPU_CORES:-2,3}"
COOLDOWN_TIME="${COOLDOWN_TIME:-5}"
MODE="${MODE:-smoke}"

run_with_limits() {
    local cmd="$1"
    local log_file="$2"

    echo "========================================="
    echo "Running: $cmd"
    echo "Log: $log_file"
    echo "Start time: $(date)"
    echo "========================================="

    if command -v taskset >/dev/null 2>&1; then
        taskset -c $CPU_CORES $cmd 2>&1 | tee "$log_file"
    else
        $cmd 2>&1 | tee "$log_file"
    fi
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        echo "ERROR: Command failed with exit code $exit_code"
        exit $exit_code
    fi

    echo "Finished at: $(date)"
    if [ "$COOLDOWN_TIME" -gt 0 ]; then
        sleep "$COOLDOWN_TIME"
    fi
}

echo "========================================"
echo "Stage 57: T-GCN vs GCN audit"
echo "Mode: $MODE"
echo "Python: $PYTHON"
echo "Start: $(date)"
echo "========================================"

if [ "$MODE" = "smoke" ]; then
    run_with_limits "$PYTHON $SCRIPT --smoke --out-name smoke.jsonl" \
        "$LOG_DIR/smoke.log"
elif [ "$MODE" = "quick" ]; then
    run_with_limits "$PYTHON $SCRIPT \
        --arms A_gcn_mse B_tgcn_reg C_tgcn_mse D_gcn_reg \
        --datasets losloop \
        --phs 1 \
        --seeds 42 \
        --adj-kinds identity physical \
        --max-epochs 50 \
        --out-name quick_los_ph1_seed42.jsonl" \
        "$LOG_DIR/quick.log"
elif [ "$MODE" = "full" ]; then
    # Full grid — USER RUNS THIS (hours on CPU).
    run_with_limits "$PYTHON $SCRIPT \
        --arms A_gcn_mse B_tgcn_reg C_tgcn_mse D_gcn_reg \
        --datasets losloop shenzhen \
        --phs 1 2 3 4 \
        --seeds 42 43 44 45 46 \
        --adj-kinds identity physical \
        --max-epochs 50 \
        --out-name full.jsonl" \
        "$LOG_DIR/full.log"
else
    echo "Unknown MODE=$MODE (use smoke|quick|full)"
    exit 1
fi

echo ""
echo "========================================"
echo "Done: $(date)"
echo "Results under $LOG_DIR/"
echo "Compare arms: A=GCN+mse, B=T-GCN+reg (canonical), C=T-GCN+mse, D=GCN+reg"
echo "If C ≈ B and D ≈ A, loss is not the main confound."
echo "If C improves vs B (or D worsens vs A), loss asymmetry explains part of GCN>T-GCN."
echo "========================================"
