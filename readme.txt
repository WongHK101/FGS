GeoTGS / FGS 一键全流程（Resumable Pipeline + ADP 版）
================================================

本仓库（脚本集合）用于从 **RGB + Thermal** 原始图像出发，一条命令跑通：
CFR 裁剪/对齐 → 裁剪评价 → COLMAP(GPS先验) → 3DGS Stage-1(RGB) → Thermal 去畸变 → Stage-2(Thermal) → 融合(blend sweep) → 批量评估。
并可在 Stage-1 启用 **ADP-Texture（Artifact-aware Densification & Pruning）**，自动抑制天空/漂浮点等伪影，同时输出论文级曲线（CSV + PNG/PDF）。

----------------------------------------------------------------------
0. 前置依赖
----------------------------------------------------------------------
1) Python 环境：与你能跑通 gaussian-splatting (3DGS) 的环境一致
   - torch / torchvision
   - diff-gaussian-rasterization 已编译
2) COLMAP（命令行可用）：
   - Windows：colmap.exe 或 colmap.bat/cmd
   - Linux：colmap 在 PATH
3) ExifTool（命令行可用）：用于 CFR 中同步/修正 EXIF/XMP（尤其是 GPS/焦距/尺寸）
4) （可选）TensorBoard：用于实时曲线；没有也不影响训练

----------------------------------------------------------------------
1. 代码/文件放置（非常重要）
----------------------------------------------------------------------
把以下脚本放在 **graphdeco-inria/gaussian-splatting 仓库根目录**（与 train.py / render.py / metrics.py 同级）：

- run_gtgs_full_pipeline.py            （本仓库的一键脚本，建议用“UPDATED版本”）
- cfr.py
- eval_crop_metrics.py
- convert-gtgs.py
- blend_model_strict_endpoints.py
- eval_blend_sweep.py

同时，为启用 ADP（Stage-1 RGB）与日志/CSV/绘图，请确保你已经替换/新增了以下文件：
- scene/gaussian_model.py              （你已经拿到的 ADP 版）
- train.py                             （你已经拿到的 ADP+日志+CSV 版）
- utils/adp_logger.py                  （周期级 CSV：adp_cycle.csv）
- utils/adp_iter_logger.py             （迭代级 CSV：adp_iter.csv）
- tools/plot_adp_paper_fig.py          （论文友好图：PNG+PDF）
（tools/plot_adp_logs.py 可选，不影响一键流程）

> 说明：run_gtgs_full_pipeline.py 默认假设 cwd=gaussian-splatting repo 根目录。

----------------------------------------------------------------------
2. 数据目录结构（data_root）
----------------------------------------------------------------------
你的 data_root 需要至少包含两个目录，并且 **图片直接放在目录下（不要嵌套子文件夹）**：

<data_root>/
  RGB/                # RGB 原始图（jpg/png 等），直接放文件
  thermal/            # Thermal 原始图（jpg/png 等），直接放文件

脚本会在 data_root 下自动生成/使用：
  fit/                # CFR 输出（image-fit / image-exif 等）
  input/              # 准备给 COLMAP 的输入（images/ + priors 等）
  distorted/          # COLMAP 重建输出（sparse_aligned 等）
  thermal_UD/         # Thermal 去畸变后的数据集（images/ + sparse/0）

----------------------------------------------------------------------
3. 一条命令跑通整个管线（推荐）
----------------------------------------------------------------------
（1）Windows PowerShell 示例：
python run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\GeoTGS\input\PV-r4" `
  --out_root  "F:\databackup\GeoTGS\output\PV-r4" `
  --colmap "colmap" `
  --exiftool "exiftool" `
  --rgb_adp_profile default `
  --rgb_adp_log `
  --rgb_adp_plot_paper `
  --auto_render `
  --verify_endpoints

（2）Linux/macOS bash 示例：
python run_gtgs_full_pipeline.py \
  --data_root "/data/GeoTGS/input/PV-r4" \
  --out_root  "/data/GeoTGS/output/PV-r4" \
  --colmap "colmap" \
  --exiftool "exiftool" \
  --rgb_adp_profile default \
  --rgb_adp_log \
  --rgb_adp_plot_paper \
  --auto_render \
  --verify_endpoints

这条命令会按顺序执行 1-14 步（默认可断点续跑）。

----------------------------------------------------------------------
4. 断点续跑 / 清理 / 强制重跑
----------------------------------------------------------------------
- 默认 **Resumable**：若检测到某一步的“关键输出文件”已存在，会自动跳过。
- 每一步会在 <data_root>/_pipeline_state/ 写入 marker json。

常用参数：
- --from_step N --to_step M     # 只跑指定步骤范围（1-14）
- --force                       # 无视现有输出，强制重跑（关闭跳过逻辑）
- --dry_run                     # 只打印命令，不执行
- --clean_fit                   # 清理 <data_root>/fit
- --clean_input                 # 清理 <data_root>/input
- --clean_thermal_ud            # 清理 <data_root>/thermal_UD
- --clean_blend_out             # 清理 <out_root>/Model_F（融合输出）
- --skip_train                  # 跳过 Stage-1/2 训练 + render/metrics（用于调试前半段）
- --skip_blend                  # 跳过融合 + sweep 评估（用于只做重建/训练）

----------------------------------------------------------------------
5. ADP（Stage-1 RGB）参数怎么选（在一键脚本里）
----------------------------------------------------------------------
你只需要设置：
- --rgb_adp_profile  off | default | aggressive | conservative
- --rgb_adp_log                  # 开启 CSV：Model_RGB/adp_cycle.csv + adp_iter.csv
- --rgb_adp_plot_paper           # 训练后自动导出论文图到 Model_RGB/adp_plots_paper/

建议：
- default：先跑通、通用最稳
- aggressive：天空/漂浮点很多时更强抑制
- conservative：担心误杀平滑真实表面时

额外频率参数（可选）：
- --rgb_adp_log_interval 50
- --rgb_adp_iter_csv_interval 50
- --rgb_adp_tex_interval 4

----------------------------------------------------------------------
6. 关键输出检查（每一步“判定成功”的典型输出）
----------------------------------------------------------------------
01_cfr：
  <data_root>/fit/image/image-fit/    有图
  <data_root>/fit/image/image-exif/   有图

02_eval_crop：
  <data_root>/fit/metrics/summary_all.json  存在且非空

03_prepare_input：
  <data_root>/input/  存在且有图（images/ 等）

04_convert_gtgs：
  <data_root>/distorted/sparse_aligned/  能找到 cameras.bin/.txt（对齐后稀疏模型）

05_train_rgb：
  <out_root>/Model_RGB/chkpnt{rgb_iter}.pth 存在
  （若启用 ADP+日志）<out_root>/Model_RGB/adp_cycle.csv、adp_iter.csv

06_render_rgb：
  <out_root>/Model_RGB/test/  下存在渲染图

07_metrics_rgb：
  <out_root>/Model_RGB/results.json 或 results.txt

08_undistort_thermal：
  <data_root>/thermal_UD/images/ 有图 + thermal_UD/sparse 存在

09_normalize_sparse_ud：
  <data_root>/thermal_UD/sparse/0  存在且含 cameras.*

10_train_thermal：
  <out_root>/Model_T/chkpnt{t_iter}.pth 存在（脚本默认 stage-2 不做 densify）

11_render_thermal：
  <out_root>/Model_T/test/ 下存在渲染图

12_metrics_thermal：
  <out_root>/Model_T/results.json 或 results.txt

13_blend：
  <out_root>/Model_F/  存在且下面有各 alpha/method 子目录输出

14_eval_sweep：
  <out_root>/eval/summary.csv 存在且非空

----------------------------------------------------------------------
7. 常见问题（快速排雷）
----------------------------------------------------------------------
- CFR 找不到图片：请确认 RGB/thermal 目录下图片是“直接放文件”，不是嵌套子目录。
- COLMAP 找不到：用 --colmap 指定可执行文件路径（Windows 上可用 colmap.bat/cmd）
- ExifTool 找不到：用 --exiftool 指定 exiftool 路径。
- 想只跑到某一步：用 --to_step，比如只跑到 04_convert_gtgs：--to_step 4
- 训练想改迭代次数：--rgb_iter / --t_iter
- 需要 baseline（不启用 ADP）：--rgb_adp_profile off（或不传该参数）

----------------------------------------------------------------------
8. 版本说明
----------------------------------------------------------------------
若你使用的是“UPDATED 的 run_gtgs_full_pipeline.py”，它支持：
- --rgb_adp_profile / --rgb_adp_log / --rgb_adp_plot_paper 等 Stage-1 ADP 选项
- 自动把 ADP 参数透传给 train.py

如果你当前目录里还是旧版 run_gtgs_full_pipeline.py，请用 UPDATED 版本替换即可。
