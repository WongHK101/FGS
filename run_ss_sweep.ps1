# ========= User config =========
$DATA_ROOT = "F:\databackup\GeoTGS-TC\input\PVpanel"
$OUT_BASE  = "F:\databackup\GeoTGS-TC\output\ss_run"

# Use the python from your conda env (recommended).
# If you prefer, run: python -c "import sys; print(sys.executable)" and paste here.
$PY = "D:\anaconda\envs\fgs\python.exe"
# =================================

$COMMON = @(
  "--iterations","200",
  "--densify_from_iter","0",
  "--densify_until_iter","200",
  "--densification_interval","50",
  "--disable_viewer"
)

New-Item -ItemType Directory -Force -Path $OUT_BASE | Out-Null

function Run-One($name, $extraArgs) {
  $outDir = Join-Path $OUT_BASE $name
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $logPath = Join-Path $OUT_BASE ($name + ".log")
  Write-Host "===== RUN $name ====="
  Write-Host "OUT: $outDir"
  Write-Host "LOG: $logPath"

  & $PY train.py `
    -s $DATA_ROOT `
    -m $outDir `
    --ss_enable --ss_source colmap_sparse --ss_aabb_margin 0.0 `
    $extraArgs `
    $COMMON 2>&1 | Tee-Object -FilePath $logPath

  Write-Host "===== DONE $name ====="
  Write-Host ""
}

# 1) Baseline: AABB-only
Run-One "baseline_aabb" @()

# 2) SSNN loose (verified effective)
Run-One "ssnn_02_04" @("--ss_voxel_size","0.2","--ss_nn_dist_thr","0.4")

# 3) SSNN mid
Run-One "ssnn_02_03" @("--ss_voxel_size","0.2","--ss_nn_dist_thr","0.3")

# 4) SSNN mid-tight
Run-One "ssnn_02_025" @("--ss_voxel_size","0.2","--ss_nn_dist_thr","0.25")

Write-Host "All runs finished. Logs in $OUT_BASE"
