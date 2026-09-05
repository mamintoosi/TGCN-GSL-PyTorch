#!/bin/bash
# ============================================================
# run_checkpoint_training.sh
#
# Train models WITH checkpoint saving and loss logging,
# then generate convergence curves and predicted vs actual plots.
#
# Existing DAGMA matrices are reused — no DAGMA recomputation.
#
# Estimated total runtime: ~4-6 hours (all methods × both datasets × PH=1)
#   - Each training run: ~10-15 min (50 epochs)
#   - 3 methods × 1 seed × 2 datasets = 6 runs
#   - Plus figure generation: ~1 min
#
# Usage:
#   chmod +x run_checkpoint_training.sh
#   ./run_checkpoint_training.sh
# ============================================================

PYTHON="/data/python-envs/pytorch/bin/python"
SCRIPT="gsl_stage26/stage26_train_with_logging.py"
PLOT_SCRIPT="paper/generate_figures_extra.py"
LOG_DIR="results/stage26_checkpoint"
COOL_DOWN=30  # seconds between runs (optional, prevents GPU thermal throttling)

mkdir -p "$LOG_DIR"
mkdir -p paper/figures

 TIMESTAMP() { date "+%Y-%m-%d %H:%M:%S"; }

run_one() {
    local METHOD=$1
    local DATASET=$2
    local PH=$3
    local SEED=$4
    local EXTRA_ARGS=$5

    local LABEL="${DATASET}_ph${PH}_seed${SEED}_${METHOD}"
    local LOG="${LOG_DIR}/${LABEL}_train.log"

    echo ""
    echo "=========================================="
    echo "Running: $PYTHON $SCRIPT --method $METHOD --dataset $DATASET --ph $PH --seed $SEED $EXTRA_ARGS"
    echo "Log: $LOG"
    echo "Start time: $(TIMESTAMP)"
    echo "=========================================="

    $PYTHON $SCRIPT \
        --method "$METHOD" \
        --dataset "$DATASET" \
        --ph "$PH" \
        --seed "$SEED" \
        $EXTRA_ARGS \
        2>&1 | tee "$LOG"

    echo "Finished at: $(TIMESTAMP)"

    if [ "$COOL_DOWN" -gt 0 ]; then
        echo "Cooling down for ${COOL_DOWN} seconds..."
        sleep $COOL_DOWN
    fi
}

# ============================================================
# PHASE 1: Los-loop (primary dataset — strongest results)
# ============================================================
echo "============================================================"
echo "PHASE 1: Los-loop PH=1 (3 methods, seed=42)"
echo "Estimated time: ~30-45 min"
echo "============================================================"

run_one "nograph"          "losloop" 1 42
run_one "multi_graph_fixed" "losloop" 1 42
run_one "gated_multi"      "losloop" 1 42

# ============================================================
# PHASE 2: SZ-Taxi (secondary dataset — for comparison)
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 2: SZ-Taxi PH=1 (3 methods, seed=42)"
echo "Estimated time: ~30-45 min"
echo "============================================================"

run_one "nograph"          "shenzhen" 1 42
run_one "multi_graph_fixed" "shenzhen" 1 42
run_one "gated_multi"      "shenzhen" 1 42

# ============================================================
# PHASE 3: Multi-seed for GatedMulti on Los-loop (best result)
# Seeds 43-46 (seed=42 already done in Phase 1)
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 3: Los-loop GatedMulti multi-seed (seeds 43-46)"
echo "Estimated time: ~40-60 min"
echo "============================================================"

for SEED in 43 44 45 46; do
    run_one "gated_multi" "losloop" 1 $SEED
done

# Also do NoGraph multi-seed for direct comparison
for SEED in 43 44 45 46; do
    run_one "nograph" "losloop" 1 $SEED
done

# ============================================================
# PHASE 4: Los-loop PH=2,3,4 for GatedMulti (multi-horizon)
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 4: Los-loop GatedMulti PH=2,3,4 (seed=42)"
echo "Estimated time: ~30-45 min"
echo "============================================================"

for PH in 2 3 4; do
    run_one "gated_multi" "losloop" $PH 42
    run_one "nograph"     "losloop" $PH 42
done

# ============================================================
# PHASE 5: Generate convergence + predicted vs actual figures
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 5: Generating extra figures"
echo "============================================================"

$PYTHON "$PLOT_SCRIPT"

echo ""
echo "============================================================"
echo "ALL TRAINING AND FIGURE GENERATION COMPLETE!"
echo "End time: $(TIMESTAMP)"
echo ""
echo "Output locations:"
echo "  Checkpoints:  results/stage26_checkpoint/"
echo "  Figures:      paper/figures/fig8_convergence.pdf"
echo "                paper/figures/fig9_predicted_vs_actual.pdf"
echo ""
echo "To regenerate figures after all training:"
echo "  $PYTHON $PLOT_SCRIPT"
echo "============================================================"
