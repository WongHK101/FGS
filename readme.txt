完整流程如下：
01_cfr：<data_root>/fit/image/image-fit 和 .../image-exif 都存在且有图
02_eval_crop：<data_root>/fit/metrics/summary_all.json 存在且非空
03_prepare_input：<data_root>/input 存在且有图
04_convert_gtgs：<data_root>/distorted/sparse_aligned 下能找到 cameras.bin/.txt
05_train_rgb：<out_root>/Model_RGB/chkpnt{rgb_iter}.pth 存在
06_render_rgb：<out_root>/Model_RGB/test 下存在渲染图
07_metrics_rgb：<out_root>/Model_RGB/results.json 或 results.txt 存在
08_undistort_thermal：<data_root>/thermal_UD/images 有图，且 thermal_UD/sparse 存在
09_normalize_sparse_ud：<data_root>/thermal_UD/sparse/0 存在且含 cameras.*
10_train_thermal：<out_root>/Model_T/chkpnt{t_iter}.pth 存在
11_render_thermal：<out_root>/Model_T/test 下存在渲染图
12_metrics_thermal：<out_root>/Model_T/results.json 或 results.txt 存在
13_blend：<out_root>/Model_F 存在且下面有子文件夹（不同 alpha/method 输出）
14_eval_sweep：<out_root>/eval/summary.csv 存在且非空
# 一步到位命令：
python run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\GeoTGS\input\PVpanel" `
  --out_root  "F:\databackup\GeoTGS\output\PVpanel-r1" `
  --rgb_res 1 `
  --t_res 1 `
  --from_step 5 `
  --to_step 14
如只需要第8-14步
python run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\GeoTGS-TC\input\PVpanel" `
  --out_root  "F:\databackup\GeoTGS-TC\output\PVpanel-r1" `
  --rgb_res 1 `
  --t_res 1 `
  --from_step 5 `
  --to_step 14

# 逐步流程：
# 第一步，处理RGB和T的FOV、分辨率不一致的问题：
python cfr.py `
  --rgb_dir "F:\databackup\GeoTGS\input\PV-r4\RGB" `
  --th_dir  "F:\databackup\GeoTGS\input\PV-r4\thermal" `
  --out_dir "F:\databackup\GeoTGS\input\PV-r4\fit" `
  --comparison
# 第二步，对fit和exif的结果进行评价：
python eval_crop_metrics.py `
  --th_dir   "F:\databackup\GeoTGS\input\PV-r4\thermal" `
  --rgb_dir  "F:\databackup\GeoTGS\input\PV-r4\fit\image\image-fit" `
  --rgb_dir  "F:\databackup\GeoTGS\input\PV-r4\fit\image\image-exif" `
  --tag fit `
  --tag exif `
  --out_dir "F:\databackup\GeoTGS\input\PV-r4\fit\metrics"
# 第三步，整理训练目录：
# 依据第二步结果，自动比较fit和exif两种方法哪种更好，将对应图片拷贝到F:\databackup\GeoTGS\input\PV-r4\input中
$root = "F:\databackup\GeoTGS\input\PV-r4"
$summary = Join-Path $root "fit\metrics\summary_all.json"
if (!(Test-Path $summary)) { throw "summary_all.json not found: $summary (请先跑第2步)" }
$obj = Get-Content $summary -Raw | ConvertFrom-Json
$cands = @($obj.candidates)
if (!$cands -or $cands.Count -lt 1) { throw "No candidates in summary_all.json" }
$cands = $cands | Where-Object { $_.count -gt 0 -and (Test-Path $_.rgb_dir) }
if (!$cands -or $cands.Count -lt 1) { throw "All candidates invalid (count==0 or rgb_dir missing)" }
function Score($v) {
  if ($null -eq $v) { return -1e30 }
  return [double]$v
}
$best = $cands | Sort-Object `
  @{Expression={ Score $_.mean.edge_f1   }; Descending=$true}, `
  @{Expression={ Score $_.mean.grad_ncc  }; Descending=$true}, `
  @{Expression={ Score $_.mean.nmi       }; Descending=$true}, `
  @{Expression={ Score $_.mean.edge_dice }; Descending=$true}, `
  @{Expression={ Score $_.mean.mi        }; Descending=$true} `
  | Select-Object -First 1
$src = $best.rgb_dir
$dst = Join-Path $root "input"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Get-ChildItem -Path $dst -File -Recurse -Include *.jpg,*.jpeg,*.png,*.tif,*.tiff,*.JPG,*.JPEG,*.PNG,*.TIF,*.TIFF `
  -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Copy-Item -Path (Join-Path $src "*") -Destination $dst -Force
Write-Host ("[OK] Selected tag: {0} | src: {1} -> dst: {2}" -f $best.tag, $src, $dst)
# 第四步，COLMAP：
新参数测试
python convert-gtgs.py `
  -s "F:\databackup\GeoTGS\input\NighttimeBuilding" `
  --mapper_multiple_models 1 `
  --min_model_size 5 `
  --init_min_num_inliers 30 `
  --abs_pose_min_num_inliers 6 `
  --camera SIMPLE_RADIAL `
  --image_reader_single_camera 1 `
  --feature_args "--SiftExtraction.max_num_features=16384 --SiftExtraction.peak_threshold=0.0035 --SiftExtraction.domain_size_pooling=1" `
  --matching spatial `
  --matcher_args "--SpatialMatching.max_num_neighbors=200 --SpatialMatching.max_distance=800 --SiftMatching.guided_matching=1 --SiftMatching.cross_check=1 --SiftMatching.max_ratio=0.85 --SiftMatching.max_distance=0.75 --TwoViewGeometry.min_num_inliers=15 --TwoViewGeometry.max_error=4" `
  --mapper_args "--Mapper.max_reg_trials=10 --Mapper.min_num_matches=20 --Mapper.filter_max_reproj_error=3 --Mapper.filter_min_tri_angle=2" `
  --use_model_aligner `
  --model_aligner_args "--ref_is_gps=1 --alignment_type=enu --alignment_max_error=30.0" `
  --prior_position_std_m 1.0
# 第五步，一阶段训练：
python train.py `
  -s "F:\databackup\GeoTGS\input\PV-r4" `
  --images images `
  -m "F:\databackup\GeoTGS\output\PV-r4\Model_RGB" `
  -r 4 `
  --iterations 30000 `
  --checkpoint_iterations 30000 `
  --data_device cuda `
  --eval `
  --densify_from_iter 1500 `
  --densify_until_iter 10000 `
  --densification_interval 300 `
  --densify_grad_threshold 0.001 `
  --lambda_dssim 0.3
python render.py -m "F:\databackup\GeoTGS\output\PV-r4\Model_RGB" -s "F:\databackup\GeoTGS\input\PV-r4"
python metrics.py -m "F:\databackup\GeoTGS\output\PV-r4\Model_RGB"
# 第六步，用第四步得到的稀疏模型去undistort thermal image：
colmap image_undistorter `
  --image_path "F:\databackup\GeoTGS\input\PV-r4\thermal" `
  --input_path "F:\databackup\GeoTGS\input\PV-r4\distorted\sparse_aligned" `
  --output_path "F:\databackup\GeoTGS\input\PV-r4\thermal_UD" `
  --output_type COLMAP
# 第七步：将thermal_UD\sparse内的文件移至0文件夹内:
$src="F:\databackup\GeoTGS\input\PV-r4\thermal_UD\sparse"
$dst=Join-Path $src "0"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
Get-ChildItem -Path $src -File | Move-Item -Destination $dst -Force
# 第八步，二阶段训练：
python train.py `
  -s "F:\databackup\GeoTGS\input\PV-r4\thermal_UD" `
  -images images `
  -m "F:\databackup\GeoTGS\output\PV-r4\Model_T" `
  --start_checkpoint "F:\databackup\GeoTGS\output\PV-r4\Model_RGB\chkpnt30000.pth" `
  -r 4 `
  --iterations 40000 `
  --checkpoint_iterations 40000 `
  --position_lr_init 0 --position_lr_final 0 `
  --scaling_lr 0 --rotation_lr 0 `
  --opacity_lr 0 `
  --feature_lr 0.001 `
  --densify_from_iter 999999 --densify_until_iter 0 `
  --densification_interval 999999 --opacity_reset_interval 999999 `
  --lambda_dssim 0.05 `
  --eval
python render.py -m "F:\databackup\GeoTGS\output\PV-r4\Model_T" -s "F:\databackup\GeoTGS\input\PV-r4\thermal_UD"
python metrics.py -m "F:\databackup\GeoTGS\output\PV-r4\Model_T"
# 第九步，模型融合：
python blend_model_strict_endpoints.py `
  --rgb_model_dir "F:\databackup\GeoTGS\output\PV-r4\Model_RGB" --rgb_iter 30000 `
  --t_model_dir   "F:\databackup\GeoTGS\output\PV-r4\Model_T" --t_iter 40000 `
  --alphas "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1" `
  --out_root "F:\databackup\GeoTGS\output\PV-r4\Model_F" `
  --out_iter 40000 `
  --methods sh_only sh_opacity sh_opacity_scale sh_opacity_geom all_float dc_ycc_only sh_opacity_dc_ycc `
  --clean_out --verify_endpoints
# 第十步，批量渲染+评价：
python eval_blend_sweep.py `
  --sweep_root "F:\databackup\GeoTGS\output\PV-r4\Model_F" `
  --rgb_render "F:\databackup\GeoTGS\output\PV-r4\Model_RGB" `
  --t_render   "F:\databackup\GeoTGS\output\PV-r4\Model_T" `
  --out_dir "F:\databackup\GeoTGS\output\PV-r4\eval" `
  --auto_render
# 查看模型：
E:\3DGS\GS\gaussian-splatting\SIBR_viewers\install\bin\SIBR_gaussianViewer_app.exe `
-m F:\databackup\GeoTGS-TC\output\PVpanel_maximprove_ssAABB_tstruct_0.2\Model_T
============================================================
新增：Sparse Support gating（改进1，可选，默认关闭）
============================================================
用途：
- 在“增密/分裂/克隆”前，对候选高斯做 Sparse Support 过滤（AABB / 预留 NN 距离阈值），用于抑制空中雾/半透明伪影、避免无支撑区域被错误增密。
默认：
- 不启用（不传 --ss_enable），行为与旧版本完全一致。

启用方式（pipeline）：
- 在 run_gtgs_full_pipeline.py 中添加：--ss_enable
- 支撑点来源：--ss_source {colmap_sparse, init_pcd}
  - colmap_sparse：读取 COLMAP sparse points3D（优先 points3D.bin，其次 points3D.txt）
  - init_pcd：回退到初始化 point cloud（当 sparse points 不可用/为空时也会自动回退）
- AABB：自动根据 support 点计算，额外 margin：--ss_aabb_margin
- 体素索引（可选）：--ss_voxel_size
  - 若为 None：仅做 AABB 过滤（AABB-only）
  - 若设置数值：构建 VoxelHashNN（用于后续 NN gating / 查询加速）
- NN gating（预留）：--ss_nn_dist_thr（默认 None）

失败回退：
- 若 support 点读取失败或为空：会打印 1 行 [WARN] 并自动关闭 sparse support（不影响训练继续进行）。

------------------------------------------------------------
新增：伪彩热像“结构梯度损失”（改进4，可选，默认关闭）
------------------------------------------------------------
用途：
- 针对 3 通道 8bit JPG 伪彩热像，提取“结构通道”并对其梯度做一致性约束，缓解 T 模型纹理崩溃/过度平滑。
默认：
- 不启用（t_struct_grad_w=0），行为与旧版本完全一致。

启用方式（pipeline）：
- 仅在 Thermal 阶段追加到 train.py：
  - --t_struct_grad_w <float>   （>0 才启用）
  - --t_struct_grad_norm        （默认 True）
建议起步：
- t_struct_grad_w：0.05 ~ 0.20（按数据/噪声可适当调整）

兼容性：
- 不改变任何旧参数名与默认值；不开启时不改变任何 train 命令字符串。

------------------------------------------------------------
新增参数速查表（均为可选，默认不启用）
------------------------------------------------------------
参数名 | 默认值 | 作用范围 | 说明 | 是否会改变默认命令
----- | ------ | -------- | ---- | ----------------
--ss_enable | False | RGB+T | 启用 Sparse Support gating | 否（默认不追加）
--ss_source | colmap_sparse | RGB+T | support 点来源（不可用时自动回退 init_pcd） | 否
--ss_aabb_margin | 0.0 | RGB+T | support AABB 的边界扩张（米/世界单位） | 否
--ss_voxel_size | None | RGB+T | 体素大小；None=仅AABB | 否
--ss_nn_dist_thr | None | RGB+T | 预留：NN 距离阈值过滤 | 否
--t_struct_grad_w | 0.0 | 仅T | 伪彩热像结构梯度损失权重（>0启用） | 否（默认不追加）
--t_struct_grad_norm | True | 仅T | 结构梯度归一化开关 | 否

------------------------------------------------------------
示例命令（仅追加参数；其余照旧）
------------------------------------------------------------
1) 仅开启 Sparse Support（AABB-only）
python run_gtgs_full_pipeline.py ^
  --data_root "<DATA_ROOT>" ^
  --out_root  "<OUT_ROOT>" ^
  --rgb_res 8 --t_res 8 ^
  --train_preset baseline3dgs ^
  --from_step 5 --to_step 14 ^
  --device cuda ^
  --ss_enable ^
  --ss_source colmap_sparse ^
  --ss_aabb_margin 0.0

2) 同时开启 Sparse Support + Thermal 结构梯度损失（仅T）
python run_gtgs_full_pipeline.py ^
  --data_root "<DATA_ROOT>" ^
  --out_root  "<OUT_ROOT>" ^
  --rgb_res 8 --t_res 8 ^
  --train_preset baseline3dgs ^
  --from_step 5 --to_step 14 ^
  --device cuda ^
  --ss_enable ^
  --ss_source colmap_sparse ^
  --ss_aabb_margin 0.0 ^
  --t_struct_grad_w 0.1 ^
  --t_struct_grad_norm
