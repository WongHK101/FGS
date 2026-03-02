# README-paper（中文论文提纲版）

本文件用于给论文写作（含 Web GPT）提供结构化上下文。  
仅覆盖**当前主线有效改进**，不写已弃用分支。

---

## 1）一句话方法概述

我们构建了一个可复现的 RGB->Thermal 两阶段 3DGS 工程管线，通过 SGF 几何冻结、RGB 后 SS 一次性裁剪、热阶段稳态化（clamp/reset/结构梯度损失）和扩展评测体系，提升 thermal 重建的结构稳定性与伪影可控性。

---

## 2）方法模块与代码锚点

### 2.1 两阶段管线

- Stage-1：RGB 重建
- Stage-2：从 RGB checkpoint 恢复并做 thermal 迁移
- 末端：RGB-T 融合与 sweep 评估

代码锚点：

- `run_gtgs_full_pipeline.py`
- `train.py`
- `blend_model_strict_endpoints.py`
- `eval_blend_sweep.py`

### 2.2 SGF（Stable Geometry Freezing）

作用：

- 防止 restore 后 optimizer state 覆盖 CLI 学习率
- 保持热阶段几何稳定，避免结构漂移

代码锚点：

- `train.py`：restore 流程 + `_reapply_lrs_after_restore()`
- 开关：`--sgf_disable`

### 2.3 SparseSupport（SS）后处理裁剪

作用：

- 在 RGB 训练完成后，基于支持域删除离群/漂浮高斯

机制：

- NN 距离门控（`ss_nn_dist_thr`）
- adaptive 动态阈值（`ss_adaptive_*`）
- island 小连通域剔除（`ss_drop_small_islands`, `ss_island_radius`）

代码锚点：

- `scene/gaussian_model.py`：
  - `set_sparse_support`
  - `_ss_gate_selected_mask`
  - `_ss_query_nn_d1_d2`
  - `_ss_filter_small_islands`
  - `prune_outside_sparse_support`
- `train.py`：`ss_prune_after_rgb` / `ss_prune_before_thermal`

### 2.4 热阶段稳态化

- `clamp_scale_max_t`：限制热阶段 scale 上界
- `thermal_reset_features`：restore 后清理 SH 残留
- `t_struct_grad_w`：热伪彩结构梯度约束

代码锚点：

- `scene/gaussian_model.py`：`clamp_scaling_max_`
- `train.py`：clamp/reset 注入点
- `utils/loss_utils.py`：`structure_grad_loss`

---

## 3）当前可复现实验默认配置

以 `run_gtgs_full_pipeline.py` 当前默认值为准：

- `rgb_iter=30000`, `t_iter=60000`
- `rgb_res=4`, `t_res=4`
- `align=fit`

主线默认模块：

- SGF：开
- SS：默认只在 RGB 启用，且 `ss_prune_after_rgb=true`
- 热阶段 SS：默认关
- `ss_use_aabb=false`
- `ss_voxel_size=1.5`, `ss_nn_dist_thr=3.5`
- `ss_adaptive_nn=true`, `alpha=1.2`, `beta=0.2`, `max_scale=2.0`
- `ss_drop_small_islands=10`, `ss_island_radius=10.0`
- `clamp_scale_max_t=10.0`
- `thermal_reset_features=true`
- `t_struct_grad_w=0.006`
- `t_opacity_lr=2e-4`

---

## 4）评测体系（论文建议）

### 4.1 有 GT 指标

- 基础：PSNR / SSIM / LPIPS（`metrics.py`）
- 扩展：`metrics_plus.py`
  - 对齐鲁棒结构指标（如 `AlignedGradientCorr`, `AlignedEdgeF1_best`）
  - 纹理/边缘质量代理指标
  - 额外 IQA（FLIP/FSIM/DISTS/VIF/MS-SSIM 等，后端可选）

### 4.2 无 GT 新视角指标

`novel_view_metrics.py`：

- 局部时序 flicker
- 空域伪影/尖刺代理
- 背景敏感性
- 聚合新视角质量评分

### 4.3 融合评估

`eval_blend_sweep.py`：

- 多 alpha、多方法扫权
- 输出融合模型评测汇总

---

## 5）论文章节建议（实验部分）

### A. 主实验（Main Results）

- 所有数据集：本方法 vs baseline
- 同时报传统指标 + 结构/新视角指标
- 提供典型可视化（normal，必要时加 ellipsoid 诊断）

### B. 单模块去除消融（remove-one）

以 full-config 为基线，每次只去掉一个模块：

- no SGF
- no SS
- no adaptive
- no island
- no clamp
- no thermal reset
- no t_struct_grad
- opacity LR 对照

### C. 关键参数敏感性

控制实验量，重点扫：

- island 参数（drop/radius）
- opacity LR（围绕默认值）
- 可选 adaptive 参数（alpha/beta）

### D. 轻量化与效率

建议汇报：

- 高斯数量
- ckpt/ply 大小
- stage2 时间（或 step 区间时间）
- 可选渲染时间代理

---

## 6）“指标与主观不一致”写作建议

若出现“视觉更好但 PSNR/SSIM 不升”，建议在文中解释：

- 像素误差与结构稳定/伪影抑制是不同目标
- 几何冻结减少了对像素误差的过拟合式补偿
- 结构指标与无 GT 指标更能体现工程可用性

---

## 7）环境复现要求（简写）

- conda + Python 3.10
- 与 CUDA 匹配的 PyTorch
- 3DGS 扩展：`diff_gaussian_rasterization`、`simple_knn`
- 额外 IQA 可选：`pyiqa`、`piq`、`flip-evaluator`

说明：IQA 后端缺失时应输出 NaN，不影响主流程运行。

---

## 8）给 Web GPT 的输入材料

建议一起提供：

- 本文件 `README-paper.zh.md`
- 主 README（`README.zh.md`/`README.md`）
- 实验汇总表（xlsx/csv）
- 命令与 profile/debug 记录（`cmd_*.txt`, `pipeline_profile.json`, `pipeline_debug.json`）

并要求生成：

1. 论文结构化初稿
2. 与代码锚点一致的方法章节
3. 仅基于真实结果的实验章节
4. remove-one 消融章节
5. 局限性与失败案例

---

## 9）范围声明

本提纲仅描述当前可复现主线方案。  
历史分支仅在必要时作为补充材料，不进入主方法叙事。
