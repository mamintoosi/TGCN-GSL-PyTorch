#!/bin/bash
# ============================================================================
# Wrapper for tools/check_original_dagma_runtime.py
# Diagnostic: reproduce + time the ORIGINAL DAGMA structure-learning protocol
# (submitted manuscript's "approximately 2 minutes" claim, line 519).
#
# NOT executed automatically — run it manually when you are ready:
#     bash tools/check_original_dagma_runtime.sh [options...]
#
# Options are forwarded verbatim to the Python script, e.g.:
#     bash tools/check_original_dagma_runtime.sh --dataset losloop --pre-len 1
#     bash tools/check_original_dagma_runtime.sh --dataset losloop shenzhen --pre-len 1 2 3 4
#     bash tools/check_original_dagma_runtime.sh --dataset shenzhen --pre-len 1 --threads 8
#     bash tools/check_original_dagma_runtime.sh --help
#
# COST WARNING: each DAGMA-linear fit at N ~ 156-207 can take tens of
# minutes on CPU (see results/stage33_gsl_canonical runtimes). Budget
# accordingly before launching.
# ============================================================================

set -euo pipefail

# --- resolve paths relative to the repository root (robust to cwd) -----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHONBIN="${PYTHONBIN:-python3}"

# --- CLI passthrough ----------------------------------------------------------
PY_ARGS=("$@")

# --- timestamped, never-overwriting log --------------------------------------
OUT_DIR="${REPO_ROOT}/results/diag_original_dagma_runtime"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/original_dagma_runtime_${TIMESTAMP}.log"
# belt-and-braces against same-second collisions / pre-existing names
SUFFIX=0
while [[ -e "${LOG_FILE}" ]]; do
    SUFFIX=$((SUFFIX + 1))
    LOG_FILE="${LOG_DIR}/original_dagma_runtime_${TIMESTAMP}_${SUFFIX}.log"
done

CMD=("${PYTHONBIN}" "${REPO_ROOT}/tools/check_original_dagma_runtime.py" "${PY_ARGS[@]}")

echo "==================================================================="
echo "Diagnostic: ORIGINAL DAGMA structure-learning runtime"
echo "Repo root : ${REPO_ROOT}"
echo "Log file  : ${LOG_FILE}"
echo "Command   : ${CMD[*]}"
echo "==================================================================="

# --- run: stdout+stderr to console AND to the log -----------------------------
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

# `tee` in a pipeline would mask the Python exit status without pipefail;
# `set -o pipefail` (above) makes the wrapper stop on a Python failure.

echo "==================================================================="
echo "Done. Log saved to: ${LOG_FILE}"
echo "JSON timing reports (if written): ${OUT_DIR}/original_dagma_timing_*.json"
echo "==================================================================="
