# FGS-0202v1（中文）

面向 **RGB -> Thermal 两阶段 3DGS** 的工程化管线。  
核心能力：断点续跑、几何稳定、SS 裁剪、热阶段稳态化、扩展评测。

> 论文写作提纲请看：`README-paper.zh.md`（中文）或 `README-paper.md`（英文）。

---

## 1）当前主线改进（有效且启用）

- **SGF（Stable Geometry Freezing）**：热阶段恢复后重置 LR，避免优化器状态覆盖导致几何漂移。
- **SparseSupport（SS）**：默认在 RGB 训练结束后做一次 prune（NN + adaptive + island）。
- **热阶段稳定化三件套**：`clamp_scale_max_t`、`thermal_reset_features`、`t_struct_grad`。
- **扩展评测**：
  - `metrics.py`：PSNR/SSIM/LPIPS
  - `metrics_plus.py`：结构/对齐/额外 IQA
  - `novel_view_metrics.py`：无 GT 新视角稳定性与伪影
  - `eval_blend_sweep.py`：RGB-T 融合扫权评测

说明：废弃/失败分支不作为默认主线描述。

---

## 2）代码入口（建议阅读顺序）

- `run_gtgs_full_pipeline.py`：1-14 全流程编排、参数默认值、断点续跑。
- `train.py`：训练主逻辑，含 SGF、SS prune 触发、热阶段损失。
- `scene/gaussian_model.py`：SS 核心过滤与 prune、clamp 实现。
- `metrics_plus.py`：有 GT 的扩展指标。
- `novel_view_metrics.py`：无 GT 新视角指标。
- `eval_blend_sweep.py`：最终融合模型评测。
- `summarize_ablation_excel.py` / `summarize_output2_excel.py`：汇总出表。

---

## 3）环境配置

环境文件：

- `environment.fgs.yml`
- `requirements.txt`

创建环境：

```powershell
conda env create -f environment.fgs.yml
conda activate fgs
```

如果不使用 `environment.fgs.yml`，手动安装依赖：

```powershell
pip install -r requirements.txt
```

安装 3DGS 扩展：

```powershell
pip install .\submodules\diff-gaussian-rasterization
pip install .\submodules\simple-knn
```

环境自检：

```powershell
python -c "import torch, numpy, cv2, PIL, plyfile, openpyxl; import diff_gaussian_rasterization, simple_knn; print('ENV_OK')"
python -c "import pyiqa, piq, flip_evaluator; print('IQA_OK')"
```

---

## 4）数据目录约定

每套数据（`--data_root`）至少包含：

- `RGB/`
- `thermal/`

管线会自动生成：

- `fit/`
- `input/`
- `distorted/`
- `thermal_UD/`
- `_pipeline_state/`（断点 marker）

每个实验输出（`--out_root`）常见为：

- `Model_RGB/`
- `Model_T/`
- `Model_F/`
- `eval/`

---

## 5）管线步骤（1-14）

1. CFR 对齐/裁剪（`cfr.py`）
2. 裁剪质量评估（`eval_crop_metrics.py`）
3. 准备 COLMAP 输入
4. COLMAP 重建（`convert-gtgs.py`）
5. RGB 训练
6. RGB 渲染
7. RGB 评测（`metrics.py` + `metrics_plus.py`）
8. thermal 去畸变（依赖 step4 sparse）
9. thermal sparse 规范化
10. thermal 训练
11. thermal 渲染
12. thermal 评测（`metrics.py` + `metrics_plus.py` + `novel_view_metrics.py`）
13. RGB/T 融合
14. 融合扫权评测

marker 在：`<data_root>/_pipeline_state/*.json`

---

## 6）当前默认参数（来自 argparse）

### 6.1 全局

- `align=fit`
- `comparison=true`
- `rgb_iter=30000`，`t_iter=60000`
- `rgb_res=4`，`t_res=4`

### 6.2 SS 默认策略（当前主线）

- `ss_enable_rgb=true`
- `ss_enable_t=false`
- `ss_prune_after_rgb=true`
- `ss_prune_before_thermal=false`
- `ss_use_aabb=false`
- `ss_voxel_size=1.5`
- `ss_nn_dist_thr=3.5`
- `ss_adaptive_nn=true`
- `ss_adaptive_alpha=1.2`
- `ss_adaptive_beta=0.2`
- `ss_adaptive_max_scale=2.0`
- `ss_drop_small_islands=10`
- `ss_island_radius=10.0`

重点：默认是 **RGB 训练后一次性 prune**，不是训练中 densify gating。

### 6.3 热阶段默认

- SGF 开（`sgf_disable=false`）
- `t_opacity_lr=2e-4`
- `clamp_scale_max_t=10.0`
- `thermal_reset_features=true`
- `t_struct_grad_w=0.006`
- `t_struct_grad_norm=true`

### 6.4 评测默认

- `run_metrics_plus=true`
- `run_novel_view_metrics=true`
- `metrics_plus_extra_iqa=flip,dists,fsim,vif,ms-ssim,gmsd,haarpsi,niqe,brisque,piqe,hdrvdp3`
- `metrics_plus_extra_iqa_space=y`
- `metrics_plus_extra_iqa_device=cuda`
- `novel_view_mode=grid72`
- `novel_grid_azimuth_count=8`
- `novel_grid_pitch_list=15,30,60`
- `novel_grid_distance_factors=0.5,1,1.5`

SIBR/椭球导出默认关闭：

- `novel_dump_sibr_ellipsoid=false`
- `novel_dump_ellipsoid_proxy=false`

---

## 7）常用命令

### 7.1 跑完整 1-14

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr6\input\PVpanel" `
  --out_root "F:\databackup\xr6\output\PVpanel_full"
```

### 7.2 只跑 thermal（10-12）

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr6\input\PVpanel" `
  --out_root "F:\databackup\xr6\output\PVpanel_t_only" `
  --from_step 10 --to_step 12
```

### 7.3 单模块消融示例（关 SGF）

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr6\input\PVpanel" `
  --out_root "F:\databackup\xr6\output\PVpanel_no_sgf" `
  --sgf_disable
```

---

## 8）常见问题

### 8.1 `thermal_UD seems incomplete`

说明 step8 输出不完整。检查：

- `<data_root>/thermal_UD/images/` 是否存在且非空
- `<data_root>/thermal_UD/sparse/` 是否存在

脚本已支持按文件 stem 做 `.jpg/.png` 兼容匹配。

### 8.2 额外 IQA 列全是 `NaN`

表示后端包缺失或不可用：

- 安装 `pyiqa` / `piq` / `flip-evaluator`
- 重新执行 step7/step12 评测

### 8.3 断点续跑异常

- 检查 `<data_root>/_pipeline_state/`
- 用 `--from_step --to_step` 定位重跑
- marker 与实际输出不一致时加 `--force`

---

## 9）复现建议

- 每轮论文实验固定一个 conda 环境并冻结版本。
- 保留 `--save_cmds` 与 profile/debug dump 便于追溯。
- 消融时一次只改一个模块，避免结论混淆。

---

## 10）许可证

本项目基于 Inria Gaussian Splatting 代码，许可条款见 `LICENSE.md`。
