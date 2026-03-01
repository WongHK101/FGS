# README-paper（论文写作专用提纲）

> 目标：本文件用于**快速喂给 Web GPT 生成论文初稿**。  
> 范围：只写当前主线改进（不写已弃用方案），并给出可复现的实验与消融框架。

---

## 0. 一键使用方式（给 Web GPT 的输入模板）

把下面整段复制给 Web GPT，再把你的最新结果表（CSV/Excel）一起上传：

```text
你是计算机视觉顶会论文写作助手。请基于我提供的 README-paper 与实验表，生成论文初稿（中文+英文摘要），要求：
1) 先给论文结构（标题、摘要、引言、相关工作、方法、实验、消融、局限、结论）；
2) 方法部分严格按 README-paper 的模块与代码锚点描述，不要发明模块；
3) 实验部分优先用我提供的真实指标，不要臆造数值；
4) 消融按“先单模块、再组合”的逻辑写，强调可复现命令与默认参数；
5) 如果某些指标与主观视觉不一致，明确解释“像素指标与结构/伪影指标分化”的原因；
6) 输出时给一版“主文精简稿”和一版“补充材料扩展稿（附更多图表）”。
```

---

## 1. 论文主线（一句话）

提出一套两阶段 RGB→Thermal 的高斯重建流程，在不改动基础 3DGS 主体框架的前提下，通过**几何稳定化 + 支持域裁剪 + 热阶段结构约束 + 扩展评测**，降低漂浮伪影/透明带并提升热场景可用性。

---

## 2. 方法总览（主线模块）

## 2.1 两阶段训练框架（RGB Stage-1 + Thermal Stage-2）

- Stage-1：RGB 训练得到几何与外观基础模型；
- Stage-2：从 RGB checkpoint 恢复，在冻结/弱化几何更新前提下学习 thermal 外观；
- 最后进行 RGB/T 融合与扫混合权重评估。

代码锚点：
- `run_gtgs_full_pipeline.py`（1-14 步编排）
- `train.py`（阶段训练逻辑）
- `blend_model_strict_endpoints.py`、`eval_blend_sweep.py`

## 2.2 SGF（Stable Geometry Freezing）

核心作用：
- 解决 restore 后 optimizer state 覆盖 CLI 学习率的问题；
- 在 thermal 阶段保持几何结构稳定，减少异常高斯膨胀。

代码锚点：
- `train.py`：`_reapply_lrs_after_restore()`、`--sgf_disable`

## 2.3 SparseSupport（SS）支持域裁剪

核心作用：
- 利用稀疏支持（COLMAP sparse 或 init_pcd）剔除离群高斯；
- 以一次性 prune 方式减少漂浮点和远空域伪影。

子机制：
- NN 距离阈值（`ss_nn_dist_thr`）
- adaptive 阈值放宽（`ss_adaptive_*`）
- island 小连通域剔除（`ss_drop_small_islands` + `ss_island_radius`）

代码锚点：
- `scene/gaussian_model.py`：
  - `set_sparse_support(...)`
  - `_ss_gate_selected_mask(...)`
  - `_ss_query_nn_d1_d2(...)`
  - `_ss_filter_small_islands(...)`
  - `prune_outside_sparse_support(...)`
- `train.py`：
  - `ss_prune_after_rgb`
  - `ss_prune_before_thermal`

## 2.4 Thermal 阶段的稳定化增强

- `clamp_scale_max_t`：限制 thermal 阶段 scale 上界，抑制巨型椭球；
- `thermal_reset_features`：restore 后清空 SH 颜色残留；
- `t_struct_grad_w`：结构梯度损失，强化 thermal 纹理结构保持。

代码锚点：
- `scene/gaussian_model.py`：`clamp_scaling_max_(...)`
- `train.py`：thermal reset、结构损失注入
- `utils/loss_utils.py`：`structure_grad_loss(...)`

## 2.5 CFR 数据对齐入口

- 管线支持 fit/exif/auto，当前主线默认 fit；
- 对齐质量通过 step2 评估并进入后续重建。

代码锚点：
- `cfr.py`
- `eval_crop_metrics.py`
- `run_gtgs_full_pipeline.py`（step1-3）

---

## 3. 当前建议默认配置（论文主方案候选）

> 注意：以你最终代码中的 argparse 默认值为准；论文中建议写“最终采用”而不是“始终最优”。

建议写法（可复现实验默认）：

- 分辨率/迭代：
  - `rgb_res=4`, `t_res=4`
  - `rgb_iter=30000`, `t_iter=60000`
- Thermal 稳定化：
  - SGF 开启（`sgf_disable=False`）
  - `t_opacity_lr=2e-4`
  - `clamp_scale_max_t=10`
  - `thermal_reset_features=True`
  - `t_struct_grad_w=0.006`
- SS（当前主线）：
  - `ss_enable_rgb=True`
  - `ss_prune_after_rgb=True`
  - `ss_use_aabb=false`
  - `ss_voxel_size=1.5`
  - `ss_nn_dist_thr=3.5`
  - `ss_adaptive_nn=true`
  - `ss_adaptive_alpha=1.2`
  - `ss_adaptive_beta=0.2`
  - `ss_adaptive_max_scale=2.0`
  - `ss_drop_small_islands=5`
  - `ss_island_radius=5.0`

---

## 4. 评测体系（论文建议呈现）

## 4.1 三类指标并行汇报

1) 像素保真（传统）  
- PSNR / SSIM / LPIPS

2) 结构与纹理（有 GT）  
- `metrics_plus.py`：
  - AlignedGradientCorr
  - AlignedEdgeF1 / AlignedEdgeF1_best
  - EdgePR_AUC / EdgeF1_mean（若已启用）
  - TextureLCN_TenengradRatio

3) 无 GT 伪影与稳定性（novel-view）  
- `novel_view_metrics.py`：
  - AirArtifactScore
  - SpikeScore_air
  - BgSensitivityRatio
  - TemporalFlicker_local
  - SGF_NovelQualityScore

## 4.2 指标分化解释（建议论文中必须写）

当“视觉更干净”但 PSNR/SSIM 不升时，应解释：
- 像素指标对微小位移/曝光差异敏感；
- 几何稳定化减少了“过拟合式像素补偿”；
- 结构/伪影指标更能反映工程可用性。

---

## 5. 消融写作框架（推荐）

## 5.1 单模块有效性（先做）

- SGF on/off
- SS on/off（或 prune timing 对比）
- clamp on/off
- thermal_reset_features on/off
- t_struct_grad_w 0 vs 0.006
- t_opacity_lr（如 2e-4 vs 0.025）

## 5.2 参数扫描（再做）

- NN 阈值：`ss_nn_dist_thr`
- adaptive：`alpha/beta/max_scale`
- island：`drop/radius`

## 5.3 组合消融（最后做）

- 以主方案为 anchor，逐项移除（remove-one）：
  - `Full`
  - `-SGF`
  - `-SS`
  - `-clamp`
  - `-reset`
  - `-t_struct_grad`

---

## 6. 图表组织建议（可直接写进论文）

主文建议 4 张表 + 2 组图：

1. 主结果表（5 套数据，主方案 vs baseline）  
2. 单模块消融表（remove-one）  
3. 参数扫描表（SS/island/adaptive 关键阈值）  
4. 速度/轻量化表（高斯数量、模型体积、阶段时长）

图像建议：
- 同机位 novel-view 对比（normal + ellipsoid）
- 透明带/漂浮物典型局部放大图（失败模式 vs 改进后）

---

## 7. 局限与风险（建议如实写）

- SS prune 可能在低纹理连续区域产生局部空洞；
- island 参数对不同场景尺度敏感，需要数据集级调参；
- 指标体系仍可能出现“主观提升 vs 像素指标下降”的分化；
- 跨数据集泛化需要更多样场景（夜景、地面视角、高遮挡）验证。

---

## 8. 复现命令模板（论文附录可用）

## 8.1 全流程

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "<DATA_ROOT>" `
  --out_root "<OUT_ROOT>"
```

## 8.2 断点续跑（仅训练+评测）

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "<DATA_ROOT>" `
  --out_root "<OUT_ROOT>" `
  --from_step 5 --to_step 14
```

## 8.3 thermal opacity 对比

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "<DATA_ROOT>" `
  --out_root "<OUT_ROOT_OP025>" `
  --from_step 10 --to_step 14 `
  --t_opacity_lr 0.025
```

---

## 9. 给 Web GPT 的“写作注意事项”

请在提示词中明确要求 Web GPT：

1) 不要臆造数值，必须使用你上传的真实表格；  
2) 不要把未启用/弃用模块写成主贡献；  
3) 每个贡献都给出代码锚点（文件 + 函数名）；  
4) 讨论中承认失败案例与权衡（透明带、空洞、指标分化）；  
5) 结论强调工程可用性与可复现，而不仅是单一像素指标。

---

## 10. 最终写作前检查清单（Checklist）

- [ ] 主方案参数与代码默认值一致  
- [ ] 所有表格来自同一版评测脚本  
- [ ] 训练/评测是否同分辨率（`r=4`）  
- [ ] novel-view 是否同机位、同参数  
- [ ] 失败案例是否给出解释与对应改进方向  
- [ ] 附录中给出完整 CLI 与版本信息（环境、commit、时间）

