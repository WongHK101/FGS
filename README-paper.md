# README-paper (Paper-Writing Brief)

This file is a structured brief for drafting papers from this repository.
It focuses on **active method components only** (no deprecated branches).

---

## 1) One-Sentence Method Summary

We build a reproducible two-stage RGB->Thermal Gaussian Splatting pipeline that improves thermal-stage robustness by combining geometry freezing (SGF), sparse-support post-RGB pruning (SS), thermal stabilization (clamp + feature reset + structure-grad), and expanded evaluation beyond PSNR/SSIM/LPIPS.

---

## 2) Method Modules and Code Anchors

### 2.1 Two-stage pipeline

- Stage-1 RGB reconstruction
- Stage-2 thermal adaptation from RGB checkpoint
- Optional RGB-T blending and sweep evaluation

Code anchors:

- `run_gtgs_full_pipeline.py` (steps 1-14 orchestration)
- `train.py` (stage-dependent training logic)
- `blend_model_strict_endpoints.py`, `eval_blend_sweep.py` (final blend evaluation)

### 2.2 SGF (Stable Geometry Freezing)

Purpose:

- prevent restore-time optimizer-state LR carryover from violating stage-2 freeze intent
- keep stage-2 geometry stable while adapting appearance

Code anchors:

- `train.py`: restore flow + `_reapply_lrs_after_restore()`
- switch: `--sgf_disable`

### 2.3 SparseSupport (SS) post-RGB prune

Purpose:

- remove unsupported/floating gaussians before stage-2

Mechanisms:

- NN gate (`ss_nn_dist_thr` + voxel index)
- adaptive threshold (`ss_adaptive_*`)
- island removal (`ss_drop_small_islands`, `ss_island_radius`)

Code anchors:

- `scene/gaussian_model.py`: `set_sparse_support`, `_ss_gate_selected_mask`, `_ss_query_nn_d1_d2`, `_ss_filter_small_islands`, `prune_outside_sparse_support`
- `train.py`: `ss_prune_after_rgb`, `ss_prune_before_thermal`

### 2.4 Thermal-stage stabilizers

- scale clamp (`clamp_scale_max_t`)
- feature reset after restore (`thermal_reset_features`)
- thermal structure gradient loss (`t_struct_grad_w`, `t_struct_grad_norm`)

Code anchors:

- `scene/gaussian_model.py`: `clamp_scaling_max_`
- `train.py`: clamp/reset integration points
- `utils/loss_utils.py`: `structure_grad_loss`

---

## 3) Reproducible Default Recipe (Current)

Read directly from `run_gtgs_full_pipeline.py` defaults:

- `rgb_iter=30000`, `t_iter=60000`
- `rgb_res=4`, `t_res=4`
- `align=fit`

Mainline module defaults:

- SGF: on
- SS: enabled on RGB only, with `ss_prune_after_rgb=true`
- Thermal SS: off by default
- `ss_use_aabb=false`
- `ss_voxel_size=1.5`, `ss_nn_dist_thr=3.5`
- `ss_adaptive_nn=true`, `alpha=1.2`, `beta=0.2`, `max_scale=2.0`
- `ss_drop_small_islands=10`, `ss_island_radius=10.0`
- `clamp_scale_max_t=10.0`
- `thermal_reset_features=true`
- `t_struct_grad_w=0.006`
- `t_opacity_lr=2e-4`

---

## 4) Evaluation Stack (Paper-Oriented)

### 4.1 Reference metrics (with GT)

- Base: `PSNR`, `SSIM`, `LPIPS` (`metrics.py`)
- Extended (`metrics_plus.py`):
  - alignment-robust structure (`AlignedGradientCorr`, `AlignedEdgeF1_best`, etc.)
  - texture/edge quality proxies
  - optional extra IQA (FLIP/FSIM/DISTS/VIF/MS-SSIM/...)

### 4.2 No-reference novel-view metrics

`novel_view_metrics.py` outputs stability/cleanliness indicators, including (depending on mode/config):

- local temporal flicker
- air-region artifact proxies
- spike/edge clutter proxies
- background sensitivity proxies
- aggregated novel quality score

### 4.3 Blend evaluation

`eval_blend_sweep.py` evaluates multiple blend weights and methods and writes summary tables for final fused model selection.

---

## 5) Suggested Experiment Chapters (for paper)

### Chapter A: Main comparison

- Compare final method vs baseline(s) on all datasets.
- Report both traditional and structure/novel-view metrics.
- Provide representative visual comparisons (normal + ellipsoid diagnostics if used).

### Chapter B: Remove-one ablation (module validity)

Use a full-config baseline and remove one module per run:

- no SGF
- no SS
- no adaptive
- no island
- no clamp
- no thermal reset
- no t_struct_grad
- opacity LR variant

### Chapter C: Parameter sensitivity (lightweight)

Recommended to keep focused and bounded:

- island sensitivity (`drop_small_islands`, `island_radius`)
- opacity LR around selected default
- optional adaptive NN neighborhood (alpha/beta)

### Chapter D: Efficiency & lightweight

Report:

- gaussian count
- checkpoint / ply size
- stage-2 time (or step-window time)
- optional render-time proxy from novel metrics

---

## 6) Notes on Metric Divergence (important for writing)

If visuals improve while PSNR/SSIM do not, explicitly explain:

- pixel-wise fidelity and structural/artifact quality are different objectives
- geometry stabilization reduces overfitting-style pixel compensation
- structure/no-ref metrics are needed to capture practical visual cleanliness

---

## 7) Environment Requirements for Reproducibility

Use the same environment policy as `README.md`:

- conda `python=3.10`
- PyTorch matching CUDA
- 3DGS extensions: `diff_gaussian_rasterization`, `simple_knn`
- optional IQA backends: `pyiqa`, `piq`, `flip-evaluator`

Missing IQA backend should produce `NaN` columns, not pipeline failure.

---

## 8) What to Feed Web GPT for Drafting

Provide these artifacts together:

- this file (`README-paper.md`)
- `README.md`
- experiment summary tables (`summary.xlsx` / csv)
- key config/command dumps (`cmd_*.txt`, profile/debug json)

Then ask for:

1. full paper skeleton
2. method section aligned to code anchors
3. experiment section using only provided numeric results
4. ablation section with remove-one logic
5. limitation/failure cases

---

## 9) Scope Boundary

This brief only covers currently active and reproducible mainline modules.
Do not include discarded branches in the final paper narrative unless explicitly needed as historical notes.
