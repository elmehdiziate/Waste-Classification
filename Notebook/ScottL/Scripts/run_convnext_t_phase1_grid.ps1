# =============================================================================
# EEEM068: Industrial Waste Classification
# run_convnext_t_phase1_grid.ps1
# =============================================================================
# Runs all 12 Phase 1 grid combinations for convnext_t sequentially.
#
# USAGE (from project root):
#   .\scripts\run_convnext_t_phase1_grid.ps1
#
# To run overnight (output to log file):
#   .\scripts\run_convnext_t_phase1_grid.ps1 | Tee-Object -FilePath logs\convnext_t_grid.log
# =============================================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " EEEM068 - convnext_t Phase 1 Grid Search" -ForegroundColor Cyan
Write-Host " 12 runs | lr x [1e-5, 5e-5, 1e-4, 3e-4] | bs x [32, 64, 128]" -ForegroundColor Cyan
Write-Host " Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

$ErrorActionPreference = "Stop"
$startTime = Get-Date
$runCount  = 0
$failCount = 0

function Run-Experiment {
    param($configPath, $label)
    Write-Host "" 
    Write-Host "------------------------------------------------------" -ForegroundColor Yellow
    Write-Host " Running: $label" -ForegroundColor Yellow
    Write-Host " Config:  $configPath" -ForegroundColor Yellow
    Write-Host " Time:    $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Yellow
    Write-Host "------------------------------------------------------" -ForegroundColor Yellow
    try {
        python train.py --config $configPath
        Write-Host " DONE: $label" -ForegroundColor Green
        return $true
    } catch {
        Write-Host " FAILED: $label - $_" -ForegroundColor Red
        return $false
    }
}

$runs = @(
    @{ config = "configs/experiments/convnext_t/phase1_lr1e-4_bs64.yaml"; label = "R1  | lr=1e-4 | bs= 64  [REFERENCE]" },
    @{ config = "configs/experiments/convnext_t/phase1_lr1e-4_bs32.yaml"; label = "R2  | lr=1e-4 | bs= 32" },
    @{ config = "configs/experiments/convnext_t/phase1_lr1e-4_bs128.yaml"; label = "R3  | lr=1e-4 | bs=128" },
    @{ config = "configs/experiments/convnext_t/phase1_lr5e-5_bs64.yaml"; label = "R4  | lr=5e-5 | bs= 64" },
    @{ config = "configs/experiments/convnext_t/phase1_lr5e-5_bs32.yaml"; label = "R5  | lr=5e-5 | bs= 32" },
    @{ config = "configs/experiments/convnext_t/phase1_lr5e-5_bs128.yaml"; label = "R6  | lr=5e-5 | bs=128" },
    @{ config = "configs/experiments/convnext_t/phase1_lr3e-4_bs64.yaml"; label = "R7  | lr=3e-4 | bs= 64" },
    @{ config = "configs/experiments/convnext_t/phase1_lr3e-4_bs32.yaml"; label = "R8  | lr=3e-4 | bs= 32" },
    @{ config = "configs/experiments/convnext_t/phase1_lr3e-4_bs128.yaml"; label = "R9  | lr=3e-4 | bs=128" },
    @{ config = "configs/experiments/convnext_t/phase1_lr1e-5_bs64.yaml"; label = "R10 | lr=1e-5 | bs= 64" },
    @{ config = "configs/experiments/convnext_t/phase1_lr1e-5_bs32.yaml"; label = "R11 | lr=1e-5 | bs= 32" },
    @{ config = "configs/experiments/convnext_t/phase1_lr1e-5_bs128.yaml"; label = "R12 | lr=1e-5 | bs=128" }
)

foreach ($run in $runs) {
    $runCount++
    $success = Run-Experiment -configPath $run.config -label $run.label
    if (-not $success) { $failCount++ }
    Write-Host " Progress: $runCount / $($runs.Count) runs complete" -ForegroundColor Cyan
}

$elapsed = (Get-Date) - $startTime

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " GRID COMPLETE - convnext_t Phase 1" -ForegroundColor Cyan
Write-Host " Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
Write-Host " Elapsed:  $($elapsed.Hours)h $($elapsed.Minutes)m $($elapsed.Seconds)s" -ForegroundColor Cyan
Write-Host " Runs:     $runCount total | $($runCount - $failCount) succeeded | $failCount failed" -ForegroundColor Cyan
Write-Host " Results:  experiments/results/convnext_t/" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

if ($failCount -gt 0) {
    Write-Host ""
    Write-Host " WARNING: $failCount run(s) failed. Check logs above." -ForegroundColor Red
    exit 1
}
