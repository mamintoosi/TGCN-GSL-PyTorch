#!/bin/bash
# ============================================================================
# Wrapper for src/run_dagma_lowiter_smoke.py (todo6)
#
# SMOKE TEST ONLY: refits DAGMA-linear on Los-loop with REDUCED iteration
# budgets (e.g. warm 300 / max 600 = 1800 total iterations = 1% of the
# library default 180000) and compares the resulting graphs against the
# EXISTING canonical PH5/PH6 artifacts (copied read-only with provenance;
# canonical results are NEVER modified).
#
# NOT executed automatically. Run manually when you are ready:
#     bash run_dagma_lowiter_smoke.sh                 # default: PH 5 6, budgets 300:600 and 3000:6000
#     bash run_dagma_lowiter_smoke.sh --help          # see all options
#     bash run_dagma_lowiter_smoke.sh --budgets 300:600 3000:6000 30000:60000
#     bash run_dagma_lowiter_smoke.sh --phs 5 --budgets 300:600 --no-copy-ref
#
# COST WARNING: budgets are scaled versions of the default protocol.
#   300:600   (~1800 iters, 1% of default)  ~ 1-2 min per (PH, budget)
#   3000:6000 (~18000 iters, 10%)           ~ 10-15 min per (PH, budget)
#   30000:60000 (default, 180000 iters)     ~ 15-20 min per (PH, budget)
# Default run = 2 PHs x 2 budgets = 4 fits, roughly 15-30 minutes total.
#
# All outputs (copies, low-iteration matrices, markdown + JSON reports,
# timestamped log) go to results/diag_dagma_lowiter_smoke/ — a NEW isolated
# directory. Nothing in results/stage*, the paper/, or the reviewer-response
# files is touched.
# ============================================================================

set -euo pipefail

# --- resolve paths relative to the repo root (robust to cwd) ---------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
cd "${REPO_ROOT}"

PYTHONBIN="${PYTHONBIN:-python3}"

# --- optional fast smoke of the machinery (no DAGMA fit): ------------------
#     bash run_dagma_lowiter_smoke.sh --selftest
if [[ "${1:-}" == "--selftest" ]]; then
    echo "[selftest] compile + CLI check only (no DAGMA fit):"
    "${PYTHONBIN}" -m py_compile src/run_dagma_lowiter_smoke.py
    "${PYTHONBIN}" src/run_dagma_lowiter_smoke.py --help >/dev/null
    echo "[selftest] OK"
    exit 0
fi

# --- timestamped, never-overwriting log ------------------------------------
OUT_DIR="${REPO_ROOT}/results/diag_dagma_lowiter_smoke"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/lowiter_smoke_${TIMESTAMP}.log"
SUFFIX=0
while [[ -e "${LOG_FILE}" ]]; do
    SUFFIX=$((SUFFIX + 1))
    LOG_FILE="${LOG_DIR}/lowiter_smoke_${TIMESTAMP}_${SUFFIX}.log"
done

CMD=("${PYTHONBIN}" src/run_dagma_lowiter_smoke.py "$@")

echo "==================================================================="
echo "todo6 smoke test: DAGMA-linear at reduced iteration budgets"
echo "Repo root : ${REPO_ROOT}"
echo "Log file  : ${LOG_FILE}"
echo "Command   : ${CMD[*]}"
echo "==================================================================="

"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"

echo "==================================================================="
echo "Done. Log saved to: ${LOG_FILE}"
echo "Reports      : ${OUT_DIR}/lowiter_smoke_report_*.md and .json"
echo "Reference    : ${OUT_DIR}/reference_copies/ (copied, provenance sidecars)"
echo "==================================================================="
