# README-paper

This file is the paper-writing brief for the current mainline only.
It is meant to be fed together with the summary workbooks to drafting tools or collaborators.

## Method Summary

FGS is a reproducible two-stage RGB -> Thermal Gaussian Splatting pipeline. The active method consists of four parts:
- `SGF`: restore-time LR correction and stage-2 geometry freezing
- `SS`: post-RGB sparse-support pruning
- thermal stabilizers: `clamp_scale_max_t`, `thermal_reset_features`, `t_struct_grad`
- expanded evaluation: reference, structure, IQA, and no-reference novel-view metrics

## Code Anchors

- `run_gtgs_full_pipeline.py`: 1-14 pipeline orchestration and defaults
- `train.py`: stage-aware training, SGF, prune timing, thermal losses
- `scene/gaussian_model.py`: SS filtering/pruning and scale clamp
- `metrics_plus.py`: GT-based extended metrics and extra IQA
- `novel_view_metrics.py`: no-reference novel-view metrics
- `eval_blend_sweep.py`: blend sweep evaluation

## Current Reproducible Defaults

These are the defaults to cite unless a specific experiment overrides them.

Global:
- `align=fit`
- `comparison=true`
- `rgb_iter=30000`, `t_iter=60000`
- `rgb_res=4`, `t_res=4`

SparseSupport:
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

Thermal stage:
- SGF on
- `t_opacity_lr=2e-4`
- `clamp_scale_max_t=10.0`
- `thermal_reset_features=true`
- `t_struct_grad_w=0.006`
- `t_struct_grad_norm=true`

Evaluation:
- `run_metrics_plus=true`
- `run_novel_view_metrics=true`
- extra IQA on `Y` with `cuda`
- grid novel-view protocol: `azimuth=8`, `pitch=15,30,60`, `distance_factors=0.5,1,1.5`

## What the Method Is Claiming

The current results support the following claim:
- competitive reconstruction quality
- cleaner outputs and lower halo/background leakage
- much lighter Gaussian representation than heavier Gaussian baselines

The current results do not support the claim that FGS is uniformly best on every traditional test-view metric against every baseline.

## Tables to Use in the Paper

Primary entry:
- `F:\databackup\xr6\output\Summaries\Paper_Final.xlsx`
- `F:\databackup\xr6\output\Summaries\Paper_Final_QA.json`

Main-text recommendation:
- `SOTA_Main`
- `SOTA_Efficiency`
- `Ablation_RemoveOne`
- `Ablation_SS`

Appendix recommendation:
- `Ablation_Combinations`
- `Ablation_Efficiency`
- `Master_All`
- `QA`

## Recommended Main-Text Metrics

Primary comparison metrics:
- `PSNR`
- `SSIM`
- `LPIPS`
- `EdgeF1_best`
- `AlignedGradientCorr`
- `IQA_flip`
- `IQA_fsim`

Cleanliness/lightweight support metrics:
- `BgLeakRatio_band`
- `EdgeHaloScore`
- `gaussian_count`
- `ply_mb`
- `core_model_mb`

Use `core_model_mb` and `artifact_mb` for all-method efficiency comparison.  
Use `gaussian_count` and `ply_mb` only for Gaussian-representation methods.

## Notes on Interpretation

If visuals improve while PSNR/SSIM do not, the paper should explicitly explain that:
- pixel fidelity and structural cleanliness are different objectives
- SGF and SS reduce unstable compensation behavior and floating artifacts
- structure/no-reference metrics are needed to capture practical visual quality

## Inputs to Give a Writing Model

Provide these files together:
- `README.md`
- `README-paper.md`
- `F:\databackup\xr6\output\Summaries\Paper_Final.xlsx`
- `F:\databackup\xr6\output\Summaries\Paper_Final_QA.json`
- `F:\databackup\xr6\output\Summaries\README_SUMMARIES.md`
- representative visual comparisons from XR6 and SOTA runs

## Scope Boundary

Do not describe discarded branches or one-off debugging paths as final method components.  
Do not claim full average-metric dominance over every Gaussian thermal baseline.  
Center the paper narrative on the quality-cleanliness-lightweight trade-off.

