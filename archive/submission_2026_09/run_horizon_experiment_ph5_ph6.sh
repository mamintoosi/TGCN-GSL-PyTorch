#!/bin/bash
# ============================================================
# Long-Horizon Extension: PH=5 and PH=6 (Reviewer 1, Weakness 7)
# ============================================================
# Direct extension of the canonical Stage 40 protocol to two longer
# prediction horizons on the native 5-minute Los-loop grid:
#   PH=5 -> 25 minutes ahead,  PH=6 -> 30 minutes ahead.
#
# Protocol is unchanged from the main experiments:
#   - datasets, 80/20 chronological split, train-max normalization,
#   - forecasting models, hyperparameters, seeds {42..46},
#   - multi-lag DAGMA artifacts (PH-independent Stage 26 fit, reused),
#   - GSL/cGSL: one contemporaneous DAGMA fit per PH (Stage 33 protocol).
#
# Order of operations:
#   Phase 1 (graph prep):   fit ONLY the missing contemporaneous DAGMA
#                           graphs for PH=5/6 (one per PH, CPU-heavy,
#                           ~15-25 min each based on PH1-4 logs). Existing
#                           artifacts are reused; nothing is overwritten.
#   Phase 2 (forecasting):  run the canonical matrix for the selected
#                           variants/PHs/seeds (resumable; completed cells
#                           are skipped automatically).
#   Phase 3 (report):       aggregate the five-seed results into
#                           results/stage59_ph56_horizon/ph56_horizon_summary.json
#
# Estimated runtime (from PH1-4 logs, GPU machine):
#   Phase 1: ~0.5-1 h per PH (skipped if graphs already exist)
#   Phase 2: GPU ~35 min for 6 variants x 2 PHs x 5 seeds; CPU proportionally longer
#   Phase 3: seconds
#
# Results: results/stage40_canonical/training/{dataset}_ph{5,6}_seed{42..46}_{variant}.json
#          results/stage59_ph56_horizon/ph56_horizon_summary.json
# Logs:    results/stage59_ph56_horizon/logs/
#
# Existing PH1-PH4 results are never touched; completed cells are skipped
# by the runner, so re-running this script resumes instead of overwriting.
# ============================================================

set -euo pipefail

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

# ----------------------------- configuration ------------------------------
DATASETS="${DATASETS:-losloop}"          # space-separated; e.g. "losloop shenzhen"
PHS="${PHS:-5 6}"                        # space-separated horizons
VARIANTS="${VARIANTS:-physical no_spatial gsl cgsl multi_gsl multi_gsl_mix}"
SEEDS="${SEEDS:-42 43 44 45 46}"
LOG_DIR="results/stage59_ph56_horizon/logs"
mkdir -p "$LOG_DIR"

run() {
  echo ""
  echo ">>> $*"
  "$PYTHON" "$@"
}

echo "============================================================"
echo "PH=5/6 LONG-HORIZON EXPERIMENT"
echo "Start:      $(date)"
echo "Repo:       $REPO"
echo "Python:     $PYTHON"
echo "Datasets:   $DATASETS"
echo "Horizons:   $PHS"
echo "Variants:   $VARIANTS"
echo "Seeds:      $SEEDS"
echo "============================================================"

# ---------------------------------------------------------------------------
# Phase 1: contemporaneous DAGMA graphs for the selected horizons (GSL/cGSL).
# Long but resumable: existing artifacts are reused, never recomputed.
# Multi-lag graphs (multi_gsl / multi_gsl_mix) need no new fit for any PH.
# ---------------------------------------------------------------------------
run src/run_ph56_horizon.py --phase graphs --datasets $DATASETS --phs $PHS \
  2>&1 | tee "$LOG_DIR/phase1_graphs.log"

# ---------------------------------------------------------------------------
# Phase 2: canonical forecasting matrix at the selected horizons.
# ---------------------------------------------------------------------------
run src/run_canonical_matrix.py \
  --variants $VARIANTS \
  --datasets $DATASETS \
  --phs $PHS \
  --seeds $SEEDS \
  2>&1 | tee "$LOG_DIR/phase2_forecasting.log"

# ---------------------------------------------------------------------------
# Phase 3: five-seed summary report.
# ---------------------------------------------------------------------------
run src/run_ph56_horizon.py --phase report --datasets $DATASETS --phs $PHS \
  2>&1 | tee "$LOG_DIR/phase3_report.log"

echo ""
echo "============================================================"
echo "ALL PHASES COMPLETE"
echo "End:     $(date)"
echo "Results: results/stage40_canonical/training/ (per-cell JSONs)"
echo "         results/stage59_ph56_horizon/ph56_horizon_summary.json"
echo "Logs:    $LOG_DIR/"
echo "============================================================"
