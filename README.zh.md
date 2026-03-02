# FGS-0202v1（中文版）  
两阶段 RGB→Thermal Gaussian Splatting 管线说明

本仓库是对 3D Gaussian Splatting 的工程化扩展，目标是稳定完成：

- Stage-1：RGB 重建（几何+外观）
- Stage-2：Thermal 迁移训练（几何稳定、热纹理学习）
- 端到端可恢复执行（1-14 步）
- 扩展评测（结构、伪影、novel-view 稳定性）

---

## 1. 目录与核心脚本

- `run_gtgs_full_pipeline.py`：主管线（步骤编排、断点续跑、参数透传）
- `train.py`：训练入口（含 SGF / SS / clamp / thermal reset / t_struct_grad）
- `scene/gaussian_model.py`：高斯参数与 SS 逻辑（NN、adaptive、island、prune）
- `metrics.py`：基础指标（PSNR/SSIM/LPIPS）
- `metrics_plus.py`：有 GT 的扩展指标
- `novel_view_metrics.py`：无 GT novel-view 指标
- `blend_model_strict_endpoints.py` + `eval_blend_sweep.py`：RGB-T 融合与评估

---

## 2. 环境要求

- OS：Windows（PowerShell）
- Python：conda 环境（建议 `fgs`）
- CUDA：建议可用（训练和部分 IQA 可走 GPU）
- 主要依赖：仓库现有依赖（不需要额外改训练代码）

快速自检：

```powershell
python -c "import torch; print(torch.__version__)"
python -c "import py_compile; py_compile.compile('run_gtgs_full_pipeline.py', doraise=True); py_compile.compile('train.py', doraise=True); print('OK')"
```

---

## 3. 管线步骤（1-14）

1. CFR 对齐裁剪（`cfr.py`）
2. 裁剪质量评估（`eval_crop_metrics.py`）
3. 准备 COLMAP 输入
4. COLMAP 重建（`convert-gtgs.py`）
5. RGB 训练
6. RGB 渲染
7. RGB 指标评估
8. Thermal 去畸变（`thermal_UD`）
9. sparse 规范化
10. Thermal 训练
11. Thermal 渲染
12. Thermal 指标评估
13. RGB-T 融合
14. 融合 sweep 评估

状态文件：`<data_root>/_pipeline_state/*.json`  
可通过 `--from_step/--to_step` 局部运行。

---

## 4. 当前主线改进（默认参数）

以下为当前代码默认主线（可直接复现）：

### 4.1 SGF（热阶段几何稳定）

- 默认开启（`--sgf_disable` 不传）
- Thermal 几何学习率冻结（位置/缩放/旋转）
- restore 后重设 optimizer LR，避免 checkpoint 覆盖

### 4.2 SS（SparseSupport，默认在 RGB 侧执行）

- `ss_enable_rgb=True`
- `ss_enable_t=False`
- `ss_prune_after_rgb=True`
- `ss_prune_before_thermal=False`
- `ss_use_aabb=false`
- `ss_voxel_size=1.5`
- `ss_nn_dist_thr=3.5`
- `ss_adaptive_nn=true`
- `ss_adaptive_alpha=1.2`
- `ss_adaptive_beta=0.2`
- `ss_adaptive_max_scale=2.0`
- `ss_drop_small_islands=10`
- `ss_island_radius=10.0`

说明：当前 island 是在 NN 过滤后的保留集合上做连通域小岛删除。

### 4.3 Thermal 稳定化

- `clamp_scale_max_t=10.0`
- `thermal_reset_features=True`
- `t_struct_grad_w=0.006`
- `t_struct_grad_norm=true`
- `t_opacity_lr=2e-4`

### 4.4 评测默认

- `run_metrics_plus=True`
- `run_novel_view_metrics=True`
- novel-view 默认：
  - `mode=grid72`
  - `grid_pitch_list=15,30,60`
  - `grid_distance_factors=0.5,1,1.5`
- SIBR 椭球导出默认关闭：
  - `novel_dump_sibr_ellipsoid=false`
  - `novel_dump_ellipsoid_proxy=false`

---

## 5. 快速运行

## 5.1 一次完整流程（推荐）

```powershell
python run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr5\input\PVpanel" `
  --out_root  "F:\databackup\xr5\output\PVpanel_full_default" `
  --rgb_res 4 --t_res 4 `
  --rgb_iter 30000 --t_iter 60000
```

## 5.2 仅跑 Thermal（复用已有 RGB）

```powershell
python run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr5\input\PVpanel" `
  --out_root  "F:\databackup\xr5\output\PVpanel_thermal_only" `
  --from_step 10 --to_step 12 `
  --rgb_res 4 --t_res 4 `
  --rgb_iter 30000 --t_iter 60000
```

---

## 6. 常用排错

- `thermal_UD seems incomplete`：
  - 检查 `step8` 是否成功，`thermal_UD/images` 和 `thermal_UD/sparse/0` 是否存在。
- `start_checkpoint not found`：
  - 先跑到 step5，确认 `Model_RGB/chkpnt30000.pth` 存在。
- 指标看起来与主观视觉不一致：
  - 建议联合看 `metrics_plus`（结构）和 `novel_view_metrics`（无 GT 稳定性/伪影）。

---

## 7. 论文复现建议

- 首先固定主线默认参数，做完整 baseline。
- 做逐模块去除（Ablation remove-one）而不是全排列。
- 对每组保留：
  - 命令行（`--save_cmds`）
  - `results.json / results_plus.json / novel_view_metrics_grid.json`
  - 关键可视化（normal + ellipsoid）

---

## 8. 说明

- 本文档只覆盖当前主线有效改进。
- 已弃用/验证失败方案不作为默认流程说明。

