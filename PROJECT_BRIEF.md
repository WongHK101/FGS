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

【本轮改动概览（2026-02 / FGS-0202v1）】
目标：在原版 3DGS + 6 个管线脚本基础上，增加两项可选能力（默认关闭，保证 backward-compatible）：
A) 改进1：Sparse Support gating —— 抑制空域雾/半透明伪影，防止无支撑 densify 生长
B) 改进4：Thermal 伪彩结构梯度损失 —— 缓解 T 纹理崩溃（默认关闭，仅 T 显式开启生效）

【已落地的功能点】
1) Sparse Support gating
- CLI：train.py/run_gtgs_full_pipeline.py 支持 --ss_enable --ss_source {colmap_sparse,init_pcd} --ss_aabb_margin --ss_voxel_size --ss_nn_dist_thr（默认全关）
- colmap_sparse：读取 COLMAP sparse points (points3D.bin/txt) -> support xyz -> AABB（含 margin）
- gaussians.set_sparse_support(aabb,index,nn_dist_thr)；densify_and_prune 中对 split/clone/new 候选做 AABB/NN gating
- 默认不启用时，训练/渲染/指标输出应与原版一致

2) SS 性能修复（卡顿）
- VoxelHashNN 改为纯 PyTorch（支持 CUDA），query 走 27-neighborhood voxel + min dist
- 引入 index.to(device)；要求 index 搬运只发生一次，禁止在 densify 热路径 .cpu/.numpy/.to()

3) Thermal 结构梯度损失（伪彩不转单通道）
- utils/loss_utils.py 新增 pseudo_color_structure(x) 与 structure_grad_loss(pred,gt,mask,normalize)
- train.py 新增 --t_struct_grad_w(默认0) --t_struct_grad_norm(默认True)
- 仅当 t_struct_grad_w>0 且输入为3通道时计算并加入总 loss；否则不影响旧路径

【当前已验证现状】
- 编译/导入自检通过；run_gtgs_full_pipeline.py -h 能看到 ss/t_struct 参数
- SS AABB-only 已能正常启用（日志显示 SparseSupport enabled...）
- 指标：RGB/T PSNR 对 SS 与 t_struct 变化不敏感；T 仍有“尖刺”边缘与空域巨大异常高斯

【当前核心问题（待诊断）】
- T 伪彩建模出现：边缘尖刺、空域黄色大片高斯；怀疑是高斯形变/不透明度/颜色拟合与几何错配导致
- SS gating 可能只限制 densify 生长，但无法解决已有高斯的颜色/形变异常；需要更贴近根因的约束或训练策略调整

【硬约束】
- backward-compatible：默认行为/旧命令完全不变；新参数默认关闭
- 一次只改一个文件；不做无关重构/格式化；不加新依赖
- 每次交付：改动点摘要 + 兼容性检查点 + 最小自检命令
