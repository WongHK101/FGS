# GeoTGS/FGS — ADPP_COMPAT_v2 跑通指南（baseline3DGS vs ADPP）

本 README 对应你已替换的 ADPP_COMPAT_v2 文件集，目标是**一套管线**同时支持：
- **baseline3dgs**：两阶段都用原版 3DGS train 默认参数（真实还原原版对 T 的糟糕表现）
- **adpp**：ADP++ 自适应调参（同一套方法同时作用于 Stage-1/Stage-2），并加入“抗雾伪影 + T 边缘变硬”的闭环策略

> 重要：baseline 与 adpp 的对比，推荐只改 `--train_preset` 和 `--out_root`，其他保持一致。


---

## 0) 你需要替换到位的文件（必须一致）

请确认本地最终路径是这些（均为覆盖替换）：

1. `FGS/train.py`
2. `FGS/scene/gaussian_model.py`
3. `FGS/utils/loss_utils.py`
4. `FGS/utils/adpp_controller.py`
5. `FGS/utils/adpp_signals.py`
6. `FGS/run_gtgs_full_pipeline.py`  （用 `run_gtgs_full_pipeline_ADPP_BASELINE.py` 覆盖）

---

## 1) 数据目录约定（与现有管线一致）

你的数据根目录形如：

`F:\databackup\GeoTGS-TC\input\PVpanel\`
- RGB：`RGB/`
- Thermal：`thermal/`
- CFR 输出：`fit/`（脚本会用）
- COLMAP 输入：`input/`（脚本会准备或复用）
- COLMAP 输出：`distorted/sparse_aligned/`（脚本会用）
- thermal_UD：`thermal_UD/`（脚本会生成/复用）

输出目录（你自行指定）形如：

`F:\databackup\GeoTGS-TC\output\PVpanel-r4_xxx\`
- `Model_RGB/`
- `Model_T/`
- `Model_F/`
- `eval/`

---

## 2) 最推荐的两条命令（只改 preset + out_root 做消融）

> Windows PowerShell 的换行符是反引号：`（注意：反引号必须在行尾，后面不能有空格）

### A) baseline3dgs（两阶段都用 train 默认参数）——强烈推荐作为 baseline
```powershell
python run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\GeoTGS-TC\input\PVpanel" `
  --out_root  "F:\databackup\GeoTGS-TC\output\PVpanel-r4_baseline3dgs" `
  --colmap "colmap" `
  --exiftool "exiftool" `
  --rgb_res 4 --t_res 4 `
  --train_preset baseline3dgs `
  --blend_endpoint_mode blend `
  --auto_render
```

### B) adpp（ADP++ 自适应调参，Stage-1/2 同时启用）——与你的论文主方法一致
```powershell
python run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\GeoTGS-TC\input\PVpanel" `
  --out_root  "F:\databackup\GeoTGS-TC\output\PVpanel-r4_adpp" `
  --colmap "colmap" `
  --exiftool "exiftool" `
  --rgb_res 4 --t_res 4 `
  --train_preset adpp `
  --adpp_trigger cycle `
  --adpp_decision_interval 50 `
  --rgb_adp_log `
  --rgb_adp_plot_paper `
  --blend_endpoint_mode blend `
  --auto_render
```

---

## 3) 常用调试/复现实验参数

### 3.1 只跑到某一步（例如只跑到 COLMAP）
```powershell
python run_gtgs_full_pipeline.py `
  --data_root "..." `
  --out_root  "..." `
  --from_step 1 --to_step 4
```

### 3.2 强制重跑（不使用 resume 跳过）
```powershell
python run_gtgs_full_pipeline.py --data_root "..." --out_root "..." --force
```

### 3.3 清理某些中间产物（更干净的对比）
- `--clean_fit`：重跑 CFR
- `--clean_input`：重建 COLMAP input/
- `--clean_thermal_ud`：重做 thermal_UD
- `--clean_blend_out`：重做 Model_F

---

## 4) 关于“ADPP vs 原版”对齐的注意事项（你最关心的点）

1) **不要再用“同一套手工超参，关掉ADP”的对比**  
   - 你的对比应当是：**baseline3dgs（两阶段默认） vs adpp（闭环自适应）**  
   - 手工参数（你之前那套）只保留作“上界/ sanity check”，不作为 baseline。

2) **ADPP 不是只影响 Stage-1**  
   - `--train_preset adpp` 会让 Stage-1/2 都进入闭环：  
     - Stage-2 会启用 “Thermal 边缘锐化（梯度/拉普拉斯一致性）” 的 loss（触发式增权）
     - 同时启用 anti-fog 的动作钩子（主要在 densify/prune 附近抑制离 sparse 太远的伪影）

3) **分辨率参数仍然支持：`--rgb_res 4 --t_res 4`**  
   - 管线内部会把 `-r` 传给 train/render。

---

## 5) 如果你看到“空中雾”仍明显（快速建议）

优先只动一个旋钮（便于消融）：

- `--adpp_anti_fog_strength_max 1.0`（更强的远离 sparse 抑制）
- 或 `--adpp_fog_prune_frac_max 0.10`（更激进地 prune fog）

建议你一次只改一个参数，跑 5k~10k iter 先看趋势，再决定是否全程跑完。

---

## 6) 如果你看到“T 边缘仍明显糊”（快速建议）

同样一次只动一个旋钮：

- 增大上限：`--adpp_max_edge_loss_weight 0.20`（默认较保守）
- 或提高梯度项占比：`--adpp_edge_grad_weight 1.0 --adpp_edge_lap_weight 0.3`

> 建议你先看 `eval/summary.csv` 里的 “fused vs Thermal GT” 指标变化，再决定是否继续加权。

---

## 7) 一条命令不换行版本（避免 PowerShell 反引号粘贴问题）

baseline3dgs：
```powershell
python run_gtgs_full_pipeline.py --data_root "F:\databackup\GeoTGS-TC\input\PVpanel" --out_root "F:\databackup\GeoTGS-TC\output\PVpanel-r4_baseline3dgs" --colmap "colmap" --exiftool "exiftool" --rgb_res 4 --t_res 4 --train_preset baseline3dgs --blend_endpoint_mode blend --auto_render
```

adpp：
```powershell
python run_gtgs_full_pipeline.py --data_root "F:\databackup\GeoTGS-TC\input\PVpanel" --out_root "F:\databackup\GeoTGS-TC\output\PVpanel-r4_adpp" --colmap "colmap" --exiftool "exiftool" --rgb_res 4 --t_res 4 --train_preset adpp --adpp_trigger cycle --adpp_decision_interval 50 --rgb_adp_log --rgb_adp_plot_paper --blend_endpoint_mode blend --auto_render
```

---

如果你下一步想要我把 **run_gtgs_full_pipeline.py** 也做成“更严格的可复现实验模板”（比如自动在 out_root 写入本次所有参数快照、git hash、依赖版本 telling），你继续跟我说“继续”即可。
