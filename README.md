# FGS-0202v1

Language: [English](README.md) | [??](README.zh.md)  
Paper brief: [README-paper.md](README-paper.md) | [README-paper.zh.md](README-paper.zh.md)

FGS-0202v1 is a two-stage RGB -> Thermal Gaussian Splatting pipeline with resumable orchestration, geometry stabilization, sparse-support pruning, thermal-stage regularization, and extended evaluation.

## Overview

Active method components:
- `SGF` (Stable Geometry Freezing) for stage-2 restore-time LR correction and geometry stability.
- `SparseSupport` (`SS`) as a one-shot prune after RGB by default.
- Thermal stabilizers: `clamp_scale_max_t`, `thermal_reset_features`, `t_struct_grad`.
- Extended evaluation: `metrics.py`, `metrics_plus.py`, `novel_view_metrics.py`, `eval_blend_sweep.py`.

Main code entry points:
- `run_gtgs_full_pipeline.py`: full 1-14 pipeline, defaults, resume/skip.
- `train.py`: stage-aware training logic, SGF, SS prune triggers, thermal loss injection.
- `scene/gaussian_model.py`: SS filtering/prune internals and clamp.
- `metrics_plus.py`: GT-based extended metrics and extra IQA.
- `novel_view_metrics.py`: no-reference novel-view metrics.
- `eval_blend_sweep.py`: RGB-T blend sweep evaluation.
- `summarize_output2_excel.py`, `summarize_sota_comparison.py`, `summarize_paper_final.py`: experiment aggregation.

## Environment

Pinned environment files:
- `environment.fgs.yml`
- `requirements.txt`

Create the environment:

```powershell
conda env create -f environment.fgs.yml
conda activate fgs
```

Manual install:

```powershell
pip install -r requirements.txt
pip install .\submodules\diff-gaussian-rasterization
pip install .\submodules\simple-knn
```

Locked core package versions:
- `python==3.10.18`
- `torch==2.0.1`
- `torchvision==0.15.2`
- `numpy==1.26.4`
- `Pillow==11.1.0`
- `opencv-python==4.10.0.84`
- `scipy==1.15.3`
- `scikit-image==0.25.2`
- `matplotlib==3.10.5`
- `tqdm==4.67.1`
- `plyfile==1.1.2`
- `pandas==2.3.3`
- `piexif==1.1.3`
- `openpyxl==3.1.5`
- `pyiqa==0.1.14.1`
- `piq==0.8.0`
- `flip-evaluator==1.7`

Quick checks:

```powershell
python -c "import torch, numpy, cv2, PIL, plyfile, openpyxl; import diff_gaussian_rasterization, simple_knn; print('ENV_OK')"
python -c "import pyiqa, piq, flip_evaluator; print('IQA_OK')"
```

## Data Layout

Per dataset root (`--data_root`) the raw inputs are:
- `RGB/`
- `thermal/`

Pipeline-generated folders include:
- `fit/`
- `input/`
- `distorted/`
- `thermal_UD/`
- `_pipeline_state/`

Per experiment output (`--out_root`) the main folders are:
- `Model_RGB/`
- `Model_T/`
- `Model_F/`
- `eval/`

## Pipeline Steps

1. CFR alignment/cropping (`cfr.py`)
2. crop quality evaluation (`eval_crop_metrics.py`)
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
14. blend sweep evaluation (`eval_blend_sweep.py`)

Resume markers are stored under `<data_root>/_pipeline_state/*.json`.

## Current Default Recipe

The defaults below are taken from `run_gtgs_full_pipeline.py`.

Global:
- `align=fit`
- `comparison=true`
- `rgb_iter=30000`, `t_iter=60000`
- `rgb_res=4`, `t_res=4`

SparseSupport default behavior:
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

Important:
- the default SS path is a one-shot prune after RGB
- training-time densify gating is off by default

Thermal stage defaults:
- SGF enabled
- `t_opacity_lr=2e-4`
- `clamp_scale_max_t=10.0`
- `thermal_reset_features=true`
- `t_struct_grad_w=0.006`
- `t_struct_grad_norm=true`

Evaluation defaults:
- `run_metrics_plus=true`
- `run_novel_view_metrics=true`
- `metrics_plus_extra_iqa=flip,dists,fsim,vif,ms-ssim,gmsd,haarpsi,niqe,brisque,piqe,hdrvdp3`
- `metrics_plus_extra_iqa_space=y`
- `metrics_plus_extra_iqa_device=cuda`
- `novel_view_mode=grid72`
- `novel_grid_azimuth_count=8`
- `novel_grid_pitch_list=15,30,60`
- `novel_grid_distance_factors=0.5,1,1.5`
- `novel_dump_ellipsoid_proxy=false`
- `novel_dump_sibr_ellipsoid=false`

## Common Commands

Full pipeline (1-14):

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr6\input\PVpanel" `
  --out_root "F:\databackup\xr6\output\PVpanel_full"
```

Thermal-only rerun (10-12):

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr6\input\PVpanel" `
  --out_root "F:\databackup\xr6\output\PVpanel_t_only" `
  --from_step 10 --to_step 12
```

Single-module ablation example:

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr6\input\PVpanel" `
  --out_root "F:\databackup\xr6\output\PVpanel_no_sgf" `
  --sgf_disable
```

## Summaries

XR6 paper summaries:
- `F:\databackup\xr6\output\Summaries\Paper_Final.xlsx`
- `F:\databackup\xr6\output\Summaries\Paper_Final_Source.csv`
- `F:\databackup\xr6\output\Summaries\Paper_Final_QA.json`

SOTA comparison summaries:
- `F:\databackup\xr6\output\SOTA_Comparison\Summaries\SOTA_Unified_Main.xlsx`
- `F:\databackup\xr6\output\SOTA_Comparison\Summaries\SOTA_Unified_Efficiency.xlsx`
- `F:\databackup\xr6\output\SOTA_Comparison\Summaries\SOTA_Unified_QA.json`

Use `F:\databackup\xr6\output\Summaries\README_SUMMARIES.md` for the XR6 table map.  
Use `F:\databackup\xr6\output\SOTA_Comparison\Summaries\README_SOTA_SUMMARIES.md` for the external comparison table map.

Efficiency notes:
- for all-method comparison, prefer `duration_s`, `core_model_mb`, and `artifact_mb`
- `gaussian_count` and `ply_mb` only apply to Gaussian-representation methods

## Troubleshooting

`thermal_UD seems incomplete`
- check `<data_root>/thermal_UD/images/`
- check `<data_root>/thermal_UD/sparse/`
- the pipeline supports `.jpg` / `.png` stem matching

Extra IQA columns are `NaN`
- install `pyiqa`, `piq`, `flip-evaluator`
- rerun the metrics steps

Resume behaves unexpectedly
- inspect `<data_root>/_pipeline_state/`
- rerun with `--from_step ... --to_step ...`
- add `--force` if markers and outputs disagree

## License

Built on top of Inria Gaussian Splatting. See `LICENSE.md`.

