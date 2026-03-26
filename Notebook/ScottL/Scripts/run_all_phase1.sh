#!/bin/bash
# =============================================================================
# EEEM068: Industrial Waste Classification
# run_all_phase1.sh — Master runner for all models (Linux/Mac)
# =============================================================================
# Runs the full Phase 1 grid for ALL four models sequentially.
# This is the "start it and go to bed" script.
#
# USAGE (from project root):
#   chmod +x scripts/run_all_phase1.sh   # only needed once
#   ./scripts/run_all_phase1.sh
#
# To run overnight on HPC / Colab:
#   nohup ./scripts/run_all_phase1.sh > logs/all_phase1.log 2>&1 &
#   tail -f logs/all_phase1.log
#
# To run a single model instead:
#   ./scripts/run_resnet50_phase1_grid.sh
#   ./scripts/run_efficientnet_b3_phase1_grid.sh
#   ./scripts/run_swin_t_phase1_grid.sh
#   ./scripts/run_convnext_t_phase1_grid.sh
# =============================================================================

set -e

START_TIME=$(date +%s)

echo "======================================================"
echo " EEEM068 - Full Phase 1 Grid Search (All Models)"
echo " 48 total runs across 4 models"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

run_model_grid() {
    local script=$1
    local model=$2
    echo ""
    echo "======================================================"
    echo " Starting: $model"
    echo " $(date '+%Y-%m-%d %H:%M:%S')"
    echo "======================================================"
    bash "$script"
    echo " Completed: $model at $(date '+%H:%M:%S')"
}

run_model_grid "scripts/run_resnet50_phase1_grid.sh" "resnet50"
run_model_grid "scripts/run_efficientnet_b3_phase1_grid.sh" "efficientnet_b3"
run_model_grid "scripts/run_swin_t_phase1_grid.sh" "swin_t"
run_model_grid "scripts/run_convnext_t_phase1_grid.sh" "convnext_t"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "======================================================"
echo " ALL PHASE 1 RUNS COMPLETE"
echo " Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo " Total elapsed: ${HOURS}h ${MINUTES}m"
echo " Results: experiments/results/"
echo "======================================================"
