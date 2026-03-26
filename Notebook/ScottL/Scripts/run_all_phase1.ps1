# =============================================================================
# EEEM068: Industrial Waste Classification
# run_all_phase1.ps1 — Master runner for all models
# =============================================================================
# Runs the full Phase 1 grid for ALL four models sequentially.
# This is the "start it and go to bed" script.
#
# USAGE (from project root):
#   .\scripts\run_all_phase1.ps1
#
# To run a single model instead:
#   .\scripts\run_resnet50_phase1_grid.ps1
#   .\scripts\run_efficientnet_b3_phase1_grid.ps1
#   .\scripts\run_swin_t_phase1_grid.ps1
#   .\scripts\run_convnext_t_phase1_grid.ps1
# =============================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " EEEM068 - Full Phase 1 Grid Search (All Models)" -ForegroundColor Cyan
Write-Host " 48 total runs across 4 models" -ForegroundColor Cyan
Write-Host " Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$startTime = Get-Date

$modelScripts = @(
    @{ script = ".\scripts\run_resnet50_phase1_grid.ps1"; model = "resnet50" }
    @{ script = ".\scripts\run_efficientnet_b3_phase1_grid.ps1"; model = "efficientnet_b3" }
    @{ script = ".\scripts\run_swin_t_phase1_grid.ps1"; model = "swin_t" }
    @{ script = ".\scripts\run_convnext_t_phase1_grid.ps1"; model = "convnext_t" }
)

foreach ($item in $modelScripts) {
    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Magenta
    Write-Host " Starting: $($item.model)" -ForegroundColor Magenta
    Write-Host "======================================================" -ForegroundColor Magenta
    & $item.script
}

$elapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host " ALL PHASE 1 RUNS COMPLETE" -ForegroundColor Green
Write-Host " Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host " Total elapsed: $($elapsed.Hours)h $($elapsed.Minutes)m" -ForegroundColor Green
Write-Host " Results: experiments/results/" -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
