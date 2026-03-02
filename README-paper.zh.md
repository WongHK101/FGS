# README-PAPER（中文提纲版）  
面向论文撰写：方法-实验-消融-局限

本文件用于快速给 Web GPT/写作助手提供结构化上下文，帮助生成论文初稿。  
内容只覆盖当前主线有效方案，不包含已弃用分支。

---

## 1. 研究问题与目标

在两阶段 RGB→Thermal 3DGS 中，常见问题包括：

- Thermal 阶段几何漂移/巨型高斯
- 空域漂浮点与伪影
- 热图纹理迁移不稳定（RGB 残留、透明带）
- 指标与主观视觉不一致

目标：在不重构 3DGS 主干前提下，构建可复现的工程主线，提升“几何稳定 + 伪影抑制 + 热纹理可用性”。

---

## 2. 方法主线（建议论文中的 Method）

## 2.1 两阶段训练框架

- Stage-1（RGB）：学习基础几何与外观
- Stage-2（Thermal）：从 RGB checkpoint restore，专注热外观学习

代码锚点：

- `run_gtgs_full_pipeline.py`（步骤编排）
- `train.py`（训练主循环）

## 2.2 SGF（Stable Geometry Freezing）

核心：

- thermal 阶段冻结几何相关学习率
- restore 后重新施加 CLI 学习率，避免 optimizer state 覆盖

代码锚点：

- `train.py`：`_reapply_lrs_after_restore()`、`--sgf_disable`

## 2.3 SparseSupport（SS）一阶段后裁剪

核心：

- 使用 sparse support 约束高斯可保留区域
- 默认在 RGB 训练后做一次 prune（`ss_prune_after_rgb`）
- 机制包含：
  - NN 距离门控
  - adaptive 阈值放宽
  - island 连通域小岛剔除

代码锚点：

- `scene/gaussian_model.py`：
  - `set_sparse_support(...)`
  - `_ss_gate_selected_mask(...)`
  - `_ss_query_nn_d1_d2(...)`
  - `_ss_filter_small_islands(...)`
  - `prune_outside_sparse_support(...)`
- `train.py`：`ss_prune_after_rgb` / `ss_prune_before_thermal`

## 2.4 Thermal 稳定化三件套

- `clamp_scale_max_t`：限制 thermal 阶段 scale 上界
- `thermal_reset_features`：restore 后清理颜色残留
- `t_struct_grad_w`：结构梯度约束

代码锚点：

- `scene/gaussian_model.py`：`clamp_scaling_max_(...)`
- `train.py`：thermal reset、t_struct_grad 注入
- `utils/loss_utils.py`：结构梯度损失

---

## 3. 默认参数（论文主线可复现配置）

建议在论文“Implementation Details”中直接列出：

- 分辨率与迭代：`rgb_res=4, t_res=4, rgb_iter=30000, t_iter=60000`
- SGF：开启（`sgf_disable=False`）
- SS（RGB侧）：
  - `ss_enable_rgb=True`
  - `ss_enable_t=False`
  - `ss_prune_after_rgb=True`
  - `ss_use_aabb=false`
  - `ss_voxel_size=1.5`
  - `ss_nn_dist_thr=3.5`
  - `ss_adaptive_nn=true`
  - `ss_adaptive_alpha=1.2`
  - `ss_adaptive_beta=0.2`
  - `ss_adaptive_max_scale=2.0`
  - `ss_drop_small_islands=10`
  - `ss_island_radius=10.0`
- Thermal 稳定化：
  - `clamp_scale_max_t=10`
  - `thermal_reset_features=True`
  - `t_struct_grad_w=0.006`
  - `t_opacity_lr=2e-4`

---

## 4. 评测体系（建议论文中的 Evaluation）

## 4.1 指标层级

1) 传统指标（有 GT）  
- PSNR / SSIM / LPIPS

2) 结构与纹理（有 GT）  
- `metrics_plus.py` 输出：
  - EdgePSNR
  - GradientCorr / AlignedGradientCorr
  - EdgeF1_best / AlignedEdgeF1_best
  - IQA_flip / IQA_fsim / IQA_dists（若后端可用）

3) 无 GT 新视角稳定性  
- `novel_view_metrics.py` 输出：
  - TemporalFlicker_local_mean
  - AirArtifactScore_mean
  - BgSensitivity_mean

## 4.2 指标分化说明（论文必须解释）

SGF/SS 可能出现：

- 像素级指标不升或下降
- 但主观几何与伪影明显改善

需在文中解释：像素误差与“几何合理性/伪影清洁度”不完全同目标，二者可分化。

---

## 5. 消融建议（写作模板）

## 5.1 逐模块去除（推荐主表）

基线：`P6-00_full`  
逐项去除：

- `P6-01_no_ss`
- `P6-02_no_adaptive`
- `P6-03_no_island`
- `P6-04_no_sgf`
- `P6-05_no_clamp_t`
- `P6-06_no_thermal_reset`
- `P6-07_no_t_struct`
- `P6-08_t_opacity_0p025`（参数对照）

## 5.2 结果呈现建议

- 表1：主指标 + 结构指标（每数据集）
- 表2：novel-view 指标（无 GT）
- 表3：逐模块去除相对增量（Δ 相对 full）
- 图：关键机位 normal/ellipsoid 可视化

---

## 6. 局限与未来工作（可直接写入论文）

- 指标与视觉存在目标不一致，需更任务导向评价
- SS 参数在跨场景尺度上仍需调参
- Thermal 颜色空间与伪彩映射可能影响统一量化

可延展方向：

- 数据集自适应 SS 超参估计
- 更稳健的跨视角结构一致性损失
- 融合阶段学习型 alpha/method 选择器

---

## 7. 可直接喂给 Web GPT 的提示模板

```text
请基于 README-paper.zh.md 生成论文初稿，要求：
1) 方法章节严格按“SGF + SS + Thermal稳定化三件套 + 两阶段框架”组织；
2) 实验章节先给总体设置，再给逐模块去除消融；
3) 讨论章节必须解释像素指标与结构/伪影指标分化；
4) 不编造实验数值，仅使用我提供的 CSV/Excel；
5) 同时输出：主文精简版 + 补充材料扩展版。
```

