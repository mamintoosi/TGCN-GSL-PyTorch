#!/bin/bash
# ============================================================
# Stage 40: Canonical Experiment Pipeline Launcher
# ============================================================
# Runs the full experiment matrix: T-GCN + GCN families,
# both datasets (losloop, shenzhen), PH 1-4, 5 seeds.
#
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

REPO="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
SCRIPT="gsl_stage40/scripts/stage40_run_all.py"
LOG_DIR="results/stage40_canonical/logs"

cd "$REPO"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"

echo "========================================"
echo "Stage 40: Canonical Experiment Pipeline"
echo "Start: $(date)"
echo "Log: $LOG_FILE"
echo "========================================"

# Forward all arguments to the Python script
$PYTHON "$SCRIPT" "$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "STAGE 40 COMPLETE"
else
    echo "STAGE 40 FAILED (exit code $EXIT_CODE)"
fi
echo "End: $(date)"
echo "Log: $LOG_FILE"
echo "Results: results/stage40_canonical/training/"
echo "========================================"

exit $EXIT_CODE
