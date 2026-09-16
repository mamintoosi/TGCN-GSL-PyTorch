#!/bin/bash
# ============================================================================
# Supplementary PH5/PH6 experiments — rev2 staged runner (todo5 policy)
# ============================================================================
# Two experiment kinds, each in its OWN isolated root:
#   canonical_completion -> results/supplementary_ph56/
#       lambda_1 = canonical per dataset (losloop 0.02, shenzhen 0.01)
#       10-variant matrix (T-GCN-MultiGSL-Weighted EXCLUDED per todo5)
#       Los-loop canonical T-GCN cells are COPIED from canonical results
#       (not re-trained); Los-loop gsl graphs are COPIED from canonical
#       stage33/59 artifacts; SZ gsl needs 2 fresh DAGMA fits.
#   lambda_sensitivity   -> results/supplementary_ph56_lambda001/
#       Los-loop only, lambda_1 = 0.01, everything freshly computed.
#
# Canonical result trees (stage26/stage33/stage40/stage59) are NEVER written.
#
# Stages ($1; default: plan):
#   plan      print both plans and exit                        [SAFE]
#   graphs    graph preparation for both kinds
#   train     training for both kinds
#   report    aggregate both kinds
#   all       graphs -> train -> report
#
# Usage: bash run_supplementary_ph5_ph6.sh [plan|graphs|train|report|all]
# ============================================================================
set -euo pipefail

# This wrapper lives in the REPO ROOT, so its directory IS the repo root.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

PYTHONBIN="${PYTHONBIN:-python3}"
STAGE="${1:-plan}"

# --- timestamped, never-overwriting log ---------------------------------------
OUT_ROOT="results/supplementary_ph56"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/supplementary_${STAGE}_${TS}.log"
SUFFIX=0
while [[ -e "${LOG_FILE}" ]]; do
    SUFFIX=$((SUFFIX + 1))
    LOG_FILE="${LOG_DIR}/supplementary_${STAGE}_${TS}_${SUFFIX}.log"
done

run_logged() {
    echo "-------------------------------------------------------------------"
    echo "[CMD] $*"
    echo "[LOG] ${LOG_FILE}"
    echo "-------------------------------------------------------------------"
    "$@" 2>&1 | tee -a "${LOG_FILE}"
}

echo "===================================================================="
echo "Supplementary PH5/PH6 experiments (rev2) — stage: ${STAGE}"
echo "Repo root       : ${REPO_ROOT}"
echo "Completion root : ${OUT_ROOT}"
echo "Sensitivity root: ${OUT_ROOT}_lambda001"
echo "Log file        : ${LOG_FILE}"
echo "===================================================================="

case "${STAGE}" in
    plan)
        run_logged "${PYTHONBIN}" src/run_supplementary_graphs.py \
            --experiment-kind canonical_completion --print-plan-only
        run_logged "${PYTHONBIN}" src/run_supplementary_graphs.py \
            --experiment-kind lambda_sensitivity --print-plan-only
        run_logged "${PYTHONBIN}" src/run_supplementary_training.py \
            --experiment-kind canonical_completion --print-plan-only
        run_logged "${PYTHONBIN}" src/run_supplementary_training.py \
            --experiment-kind lambda_sensitivity --print-plan-only
        ;;
    graphs)
        # 1) completion: copies los-loop canonical graphs + multilag blocks
        #    (provenance sidecars), derives cGSL, fits the 2 missing SZ graphs
        run_logged "${PYTHONBIN}" src/run_supplementary_graphs.py \
            --experiment-kind canonical_completion
        # 2) sensitivity: 2 fresh los-loop fits at lambda_1 = 0.01
        run_logged "${PYTHONBIN}" src/run_supplementary_graphs.py \
            --experiment-kind lambda_sensitivity
        ;;
    train)
        # 1) completion matrix (200 cells: 110 newly trained + 50 copied +
        #    40 SZ gsl/cgsl cells after their graphs exist)
        run_logged "${PYTHONBIN}" src/run_supplementary_training.py \
            --experiment-kind canonical_completion
        # 2) sensitivity matrix (100 cells, losloop lambda001)
        run_logged "${PYTHONBIN}" src/run_supplementary_training.py \
            --experiment-kind lambda_sensitivity
        ;;
    report)
        run_logged "${PYTHONBIN}" src/aggregate_supplementary_results.py \
            --experiment-kind canonical_completion
        run_logged "${PYTHONBIN}" src/aggregate_supplementary_results.py \
            --experiment-kind lambda_sensitivity
        ;;
    all)
        "${PYTHONBIN}" src/run_supplementary_graphs.py \
            --experiment-kind canonical_completion 2>&1 | tee -a "${LOG_FILE}"
        "${PYTHONBIN}" src/run_supplementary_graphs.py \
            --experiment-kind lambda_sensitivity 2>&1 | tee -a "${LOG_FILE}"
        "${PYTHONBIN}" src/run_supplementary_training.py \
            --experiment-kind canonical_completion 2>&1 | tee -a "${LOG_FILE}"
        "${PYTHONBIN}" src/run_supplementary_training.py \
            --experiment-kind lambda_sensitivity 2>&1 | tee -a "${LOG_FILE}"
        "${PYTHONBIN}" src/aggregate_supplementary_results.py \
            --experiment-kind canonical_completion 2>&1 | tee -a "${LOG_FILE}"
        "${PYTHONBIN}" src/aggregate_supplementary_results.py \
            --experiment-kind lambda_sensitivity 2>&1 | tee -a "${LOG_FILE}"
        ;;
    *)
        echo "Unknown stage: ${STAGE}" >&2
        echo "Usage: bash run_supplementary_ph5_ph6.sh [plan|graphs|train|report|all]" >&2
        exit 2
        ;;
esac

echo "===================================================================="
echo "Done. Log: ${LOG_FILE}"
echo "===================================================================="
