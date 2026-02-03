# run_render_sweep.ps1
$ErrorActionPreference = "Stop"

$PY="D:\anaconda\envs\fgs\python.exe"
$DATA_ROOT="F:\databackup\GeoTGS-TC\input\PVpanel"
$OUT_ROOT="F:\databackup\GeoTGS-TC\output\ss_run"

# 统一渲染设置
$ITER=200
$RES=8
$LLFF=8

$MODELS=@(
  "baseline_aabb",
  "ssnn_02_025",
  "ssnn_02_03",
  "ssnn_02_04"
)

# torch 自检（可选但强烈建议）
& $PY -c "import torch; print('torch=', torch.__version__)"

foreach ($name in $MODELS) {
  $MODEL = Join-Path $OUT_ROOT $name
  Write-Host "===== RENDER TEST: $name ====="

  # 关键：--eval 才有 test；--llffhold 8 才会像你之前那样切出 130 张
  & $PY render.py `
    -m $MODEL `
    -s $DATA_ROOT `
    -r $RES `
    --iteration $ITER `
    --eval `
    --llffhold $LLFF `
    --skip_train 2>&1 | Tee-Object -FilePath (Join-Path $MODEL "render_test.log")

  Write-Host "===== METRICS: $name ====="
  & $PY metrics.py -m $MODEL 2>&1 | Tee-Object -FilePath (Join-Path $MODEL "metrics_test.log")
}

Write-Host "All done."
Write-Host "You can also run one-shot metrics over all models:"
Write-Host "  $PY metrics.py -m $($MODELS | ForEach-Object { Join-Path $OUT_ROOT $_ })"
