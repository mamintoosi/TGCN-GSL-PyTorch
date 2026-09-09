#!/bin/bash
# =============================================================================
# Stage 40.3 — Fit SZ-Taxi Contemporaneous DAGMA (PH=1-4)
#
# This script fits the missing contemporaneous DAGMA graphs for SZ-Taxi.
# It ONLY fits DAGMA — no model training.
#
# Estimated runtime: ~1.5-2.5 hours per PH, ~6-10 hours total.
#
# Output:
#   results/stage33_gsl_canonical/sz_gsl_ph{1-4}_seed42_W_est.npy
#   results/stage33_gsl_canonical/sz_gsl_ph{1-4}_seed42_A_binary.npy
#
# Usage:
#   bash run_stage40_3_sz_dagma.sh           # fit all PHs sequentially
#   bash run_stage40_3_sz_dagma.sh --phs 1   # fit PH=1 only
# =============================================================================

set -e

SCRIPT="gsl_stage26/stage40_3_fit_sz_contemporaneous.py"

echo "============================================================"
echo "Stage 40.3 — SZ-Taxi Contemporaneous DAGMA"
echo "Start: $(date)"
echo "============================================================"

# Run the DAGMA-only fitting script
# Pass all arguments to the script (e.g., --phs 1 2 3 4)
conda run -n pth python "$SCRIPT" "$@"

echo ""
echo "============================================================"
echo "Stage 40.3 COMPLETE: $(date)"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run validation:  conda run -n pth python run_stage40_3_validate.py"
echo "  2. Run full Stage 40: python gsl_stage40/scripts/stage40_run_all.py"
