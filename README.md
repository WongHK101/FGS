# FGS-0202v1

Two-stage RGB -> Thermal Gaussian Splatting pipeline with resumable orchestration, geometry stabilization, sparse-support pruning, and extended evaluation.

Language: [English](README.md) | [中文](README.zh.md)

This README is the engineering/usage entry point.  
For paper writing structure, see `README-paper.md` / `README-paper.zh.md`.

---

## 1) What Is Included (Mainline Features)

Current mainline (active) improvements:

- **SGF (Stable Geometry Freezing)** for stage-2 (thermal) restore-time LR correction and geometry stability.
- **SparseSupport (SS)** with NN/adaptive/island filtering, used as **post-RGB one-shot prune** by default.
- **Thermal stabilizers**: `clamp_scale_max_t`, `thermal_reset_features`, `t_struct_grad`.
- **Expanded metrics**:
  - `metrics.py` (PSNR/SSIM/LPIPS)
  - `metrics_plus.py` (structure/alignment/extra IQA)
  - `novel_view_metrics.py` (no-ref novel-view quality/stability)
  - blend sweep evaluation (`eval_blend_sweep.py`).

Deprecated/abandoned branches are intentionally not documented as default workflow.

---

## 2) Repository Map (Read Order)

- `run_gtgs_full_pipeline.py`: full 1-14 pipeline, resume/skip, default recipe, experiment control.
- `train.py`: training loop, restore flow, SGF, SS prune triggers, thermal loss injection.
- `scene/gaussian_model.py`: SS gating/prune internals + clamp implementation.
- `metrics_plus.py`: GT-based extended metrics and extra IQA backend handling.
- `novel_view_metrics.py`: novel-view rendering path + no-reference metrics.
- `eval_blend_sweep.py`: RGB-T blend scoring and final model selection.
- `summarize_ablation_excel.py`, `summarize_output2_excel.py`: experiment aggregation to Excel.

---

## 3) Environment Setup

Use the pinned files:

- `environment.fgs.yml`
- `requirements.txt`

Create environment:

```powershell
conda env create -f environment.fgs.yml
conda activate fgs
```

If you install manually instead of `environment.fgs.yml`:

```powershell
pip install -r requirements.txt
```

Install 3DGS extensions:

```powershell
pip install .\submodules\diff-gaussian-rasterization
pip install .\submodules\simple-knn
```

Environment check:

```powershell
python -c "import torch, numpy, cv2, PIL, plyfile, openpyxl; import diff_gaussian_rasterization, simple_knn; print('ENV_OK')"
python -c "import pyiqa, piq, flip_evaluator; print('IQA_OK')"
```

---

## 4) Data Layout

Per dataset root (`--data_root`), expected inputs:

- `RGB/`
- `thermal/`

Pipeline-generated folders include:

- `fit/`
- `input/`
- `distorted/`
- `thermal_UD/`
- `_pipeline_state/` (resume markers)

Per experiment output (`--out_root`), typical folders:

- `Model_RGB/`
- `Model_T/`
- `Model_F/` (blend result)
- `eval/`

---

## 5) Pipeline Steps (1-14)

1. CFR alignment/cropping (`cfr.py`)
2. crop quality eval (`eval_crop_metrics.py`)
3. prepare COLMAP input
4. COLMAP conversion/reconstruction (`convert-gtgs.py`)
5. RGB training (`train.py`)
6. RGB render (`render.py`)
7. RGB metrics (`metrics.py`, `metrics_plus.py`)
8. thermal undistort from aligned sparse model
9. thermal sparse normalization
10. thermal training (`train.py`)
11. thermal render
12. thermal metrics (`metrics.py`, `metrics_plus.py`, `novel_view_metrics.py`)
13. RGB/T blend (`blend_model_strict_endpoints.py`)
14. blend sweep eval (`eval_blend_sweep.py`)

Resume markers are stored under `<data_root>/_pipeline_state/*.json`.

---

## 6) Current Default Recipe (from argparse defaults)

### 6.1 Global

- `align=fit`
- `comparison=true`
- `rgb_iter=30000`, `t_iter=60000`
- `rgb_res=4`, `t_res=4`

### 6.2 SparseSupport default behavior

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

Important: default strategy is **one-shot prune after RGB**.  
Training-time densify gating is **not enabled by default**.

### 6.3 Thermal stage defaults

- SGF enabled (`sgf_disable=false`)
- `t_opacity_lr=2e-4`
- `clamp_scale_max_t=10.0`
- `thermal_reset_features=true`
- `t_struct_grad_w=0.006`
- `t_struct_grad_norm=true`

### 6.4 Evaluation defaults

- `run_metrics_plus=true`
- `run_novel_view_metrics=true`
- `metrics_plus_extra_iqa=flip,dists,fsim,vif,ms-ssim,gmsd,haarpsi,niqe,brisque,piqe,hdrvdp3`
- `metrics_plus_extra_iqa_space=y`
- `metrics_plus_extra_iqa_device=cuda`
- `novel_view_mode=grid72`
- `novel_grid_azimuth_count=8`
- `novel_grid_pitch_list=15,30,60`
- `novel_grid_distance_factors=0.5,1,1.5`

SIBR/ellipsoid dump defaults:

- `novel_dump_sibr_ellipsoid=false`
- `novel_dump_ellipsoid_proxy=false`

---

## 7) Common Commands

### 7.1 Full run (1-14)

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr6\input\PVpanel" `
  --out_root "F:\databackup\xr6\output\PVpanel_full"
```

### 7.2 Resume only thermal stage (10-12)

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr6\input\PVpanel" `
  --out_root "F:\databackup\xr6\output\PVpanel_t_only" `
  --from_step 10 --to_step 12
```

### 7.3 Toggle one module (example: disable SGF)

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr6\input\PVpanel" `
  --out_root "F:\databackup\xr6\output\PVpanel_no_sgf" `
  --sgf_disable
```

---

## 8) Troubleshooting

### 8.1 `thermal_UD seems incomplete`

Step-8 output validation failed. Check:

- `<data_root>/thermal_UD/images/` exists and non-empty
- `<data_root>/thermal_UD/sparse/` exists

The pipeline includes extension fallback by stem matching (`.jpg`/`.png`) for thermal image names.

### 8.2 Extra IQA columns are `NaN`

Your IQA backend is missing/unavailable in current env.

- install optional packages (`pyiqa`, `piq`, `flip-evaluator`)
- rerun step 7/12 metrics

### 8.3 Resume behaves unexpectedly

- inspect `<data_root>/_pipeline_state/`
- rerun with `--from_step ... --to_step ...`
- add `--force` when marker/output status is inconsistent

---

## 9) Notes for Reproducibility

- Keep one conda env per paper run (`fgs`) and freeze package versions.
- Keep `--save_cmds` + profile/debug dumps for traceability.
- For ablation comparability, avoid changing multiple defaults at once.

---

## 10) License

Built on top of Inria Gaussian Splatting; see `LICENSE.md` for terms.
