#!/bin/bash
# =============================================================================
# EEEM068: Industrial Waste Classification
# run_swin_t_phase1_grid.sh — Linux/Mac equivalent
# =============================================================================
# Runs all 12 Phase 1 grid combinations for swin_t sequentially.
#
# USAGE (from project root):
#   chmod +x scripts/run_swin_t_phase1_grid.sh   # only needed once
#   ./scripts/run_swin_t_phase1_grid.sh
#
# To run overnight on HPC / Colab:
#   nohup ./scripts/run_swin_t_phase1_grid.sh > logs/swin_t_grid.log 2>&1 &
#   tail -f logs/swin_t_grid.log
# =============================================================================

set -e

START_TIME=$(date +%s)
RUN_COUNT=0
FAIL_COUNT=0

echo "======================================================"
echo " EEEM068 - swin_t Phase 1 Grid Search"
echo " 12 runs | lr x [1e-5, 5e-5, 1e-4, 3e-4] | bs x [32, 64, 128]"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

run_experiment() {
    local config=$1
    local label=$2
    echo ""
    echo "------------------------------------------------------"
    echo " Running: $label"
    echo " Config:  $config"
    echo " Time:    $(date '+%H:%M:%S')"
    echo "------------------------------------------------------"
    if python train.py --config "$config"; then
        echo " DONE: $label"
        return 0
    else
        echo " FAILED: $label"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 1
    fi
}

run_experiment "configs/experiments/swin_t/phase1_lr1e-4_bs64.yaml" "R1  | lr=1e-4 | bs= 64  [REFERENCE]"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr1e-4_bs32.yaml" "R2  | lr=1e-4 | bs= 32"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr1e-4_bs128.yaml" "R3  | lr=1e-4 | bs=128"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr5e-5_bs64.yaml" "R4  | lr=5e-5 | bs= 64"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr5e-5_bs32.yaml" "R5  | lr=5e-5 | bs= 32"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr5e-5_bs128.yaml" "R6  | lr=5e-5 | bs=128"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr3e-4_bs64.yaml" "R7  | lr=3e-4 | bs= 64"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr3e-4_bs32.yaml" "R8  | lr=3e-4 | bs= 32"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr3e-4_bs128.yaml" "R9  | lr=3e-4 | bs=128"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr1e-5_bs64.yaml" "R10 | lr=1e-5 | bs= 64"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr1e-5_bs32.yaml" "R11 | lr=1e-5 | bs= 32"; RUN_COUNT=$((RUN_COUNT + 1))
run_experiment "configs/experiments/swin_t/phase1_lr1e-5_bs128.yaml" "R12 | lr=1e-5 | bs=128"; RUN_COUNT=$((RUN_COUNT + 1))

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))
SECONDS=$((ELAPSED % 60))

echo ""
echo "======================================================"
echo " GRID COMPLETE - swin_t Phase 1"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Elapsed:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo " Runs:     $RUN_COUNT total | $((RUN_COUNT - FAIL_COUNT)) succeeded | $FAIL_COUNT failed"
echo " Results:  experiments/results/swin_t/"
echo "======================================================"

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo " WARNING: $FAIL_COUNT run(s) failed. Check output above."
    exit 1
fi
