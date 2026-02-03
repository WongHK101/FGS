$DATA_ROOT = "F:\databackup\GeoTGS-TC\input\PVpanel"
$OUT_BASE  = "F:\databackup\GeoTGS-TC\output\ss_verify"

$COMMON = @(
  "--iterations","200",
  "--densify_from_iter","0",
  "--densify_until_iter","200",
  "--densification_interval","50",
  "--disable_viewer"
)

# Conservative SSNN defaults (adjust if your scene scale is very different)
$VOX_LOOSE = 0.20
$NN_LOOSE  = 0.40
$VOX_TIGHT = 0.10
$NN_TIGHT  = 0.20

New-Item -ItemType Directory -Force -Path $OUT_BASE | Out-Null

# A) colmap_sparse + AABB-only
$OUT_DIR = Join-Path $OUT_BASE "ss_aabb"
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
python train.py `
  -s $DATA_ROOT `
  -m $OUT_DIR `
  --ss_enable --ss_source colmap_sparse --ss_aabb_margin 0.0 `
  $COMMON 2>&1 | Tee-Object -FilePath (Join-Path $OUT_BASE "ss_aabb.log")

# B) colmap_sparse + SSNN (loose)
$OUT_DIR = Join-Path $OUT_BASE "ssnn_loose"
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
python train.py `
  -s $DATA_ROOT `
  -m $OUT_DIR `
  --ss_enable --ss_source colmap_sparse --ss_aabb_margin 0.0 `
  --ss_voxel_size $VOX_LOOSE --ss_nn_dist_thr $NN_LOOSE `
  $COMMON 2>&1 | Tee-Object -FilePath (Join-Path $OUT_BASE "ssnn_loose.log")

# C) colmap_sparse + SSNN (tight)
$OUT_DIR = Join-Path $OUT_BASE "ssnn_tight"
New-Item -ItemType Directory -Force -Path $OUT_DIR | Out-Null
python train.py `
  -s $DATA_ROOT `
  -m $OUT_DIR `
  --ss_enable --ss_source colmap_sparse --ss_aabb_margin 0.0 `
  --ss_voxel_size $VOX_TIGHT --ss_nn_dist_thr $NN_TIGHT `
  $COMMON 2>&1 | Tee-Object -FilePath (Join-Path $OUT_BASE "ssnn_tight.log")
