【项目目的】
- 在 3DGS 基础上做 GeoTGS/双模态（RGB + Thermal伪彩）重建与融合评估。
- 当前重点：抑制 RGB 空域雾/半透明伪影（Sparse Support gating），缓解 Thermal 纹理崩溃（伪彩结构约束等）。

【当前代码状态】
- 基于原版 3DGS + 6 个管线文件（convert-gtgs.py / blend_model_strict_endpoints.py / eval_blend_sweep.py / eval_crop_metrics.py / run_gtgs_full_pipeline.py / cfr.py）。
- 训练入口：train.py（保持原版参数兼容）。
- 数据：DJI Thermal 伪彩 JPG，3 通道 8bit；已做辐射一致处理。

【当前问题】
- RGB: 空域出现雾/半透明伪影，怀疑 densify 在无支撑区域生长。
- Thermal: 边缘“尖刺”、空域出现大块异常高斯；PSNR/SSIM 对改进不敏感。
- 需要证据链诊断 + 最小可复现实验来定位根因，再做模块级改动。

【硬约束】
- backward-compatible：默认行为/旧命令不变；新参数默认关闭；一次只改一个文件；不加新依赖。
- 每次交付必须带 summary + 兼容性检查点 + 最小自检命令。
