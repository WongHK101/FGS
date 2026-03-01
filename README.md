# FGS-0202v1: Two-Stage RGB/Thermal Gaussian Splatting Pipeline

This repository is a production-style extension of Gaussian Splatting for two-stage RGB -> Thermal reconstruction, with:

- resumable end-to-end pipeline (`run_gtgs_full_pipeline.py`)
- geometry stabilization for thermal fine-tuning (SGF)
- SparseSupport pruning for outlier suppression
- thermal-specific regularization and initialization
- expanded evaluation stack (`metrics_plus.py`, `novel_view_metrics.py`, blend sweep scoring)

The goal of this README is to make the codebase immediately usable for:

- engineering usage (training/evaluation)
- ablation/reproducibility
- paper writing (what to read, where each claim is implemented)

Deprecated or abandoned ideas are intentionally not documented as "mainline features".

---

## 1. Repository Scope

The pipeline is organized as 14 resumable steps:

1. CFR alignment/cropping (`cfr.py`)
2. Crop quality evaluation (`eval_crop_metrics.py`)
3. Prepare COLMAP input images
4. COLMAP conversion/reconstruction (`convert-gtgs.py`)
5. Stage-1 RGB training (`train.py`)
6. RGB render (`render.py`)
7. RGB metrics (`metrics.py`, `metrics_plus.py`)
8. Thermal undistortion using aligned sparse model
9. Thermal sparse layout normalization
10. Stage-2 Thermal training (`train.py`)
11. Thermal render
12. Thermal metrics (`metrics.py`, `metrics_plus.py`, `novel_view_metrics.py`)
13. RGB/T blend (`blend_model_strict_endpoints.py`)
14. Blend sweep evaluation (`eval_blend_sweep.py`)

Resumability is marker-based and stored at:

- `<data_root>/_pipeline_state/*.json`

Important implication:

- Running multiple experiments on the same `data_root` shares step markers.
- Use `--force` when needed, or isolate `data_root` copies for fully independent runs.

---

## 2. Key Improvements (Active Mainline)

This section lists improvements currently used in mainline flow and where they live in code.

### 2.1 SGF (Stable Geometry Freezing) for Thermal Stage

Purpose:

- prevent optimizer-state carryover after restore from silently changing intended LRs
- keep thermal stage geometry stable while still learning thermal appearance

Implementation anchors:

- `train.py`
  - `gaussians.restore(...)`
  - `_reapply_lrs_after_restore()` (gated by `--sgf_disable`)
  - thermal LR freeze setup in stage-2 command (pipeline side)

Default behavior:

- enabled by default (set `--sgf_disable` to disable)

---

### 2.2 SparseSupport (SS) Pruning

Purpose:

- remove geometry unsupported by sparse structure (mainly floating/outlier gaussians)

Implementation anchors:

- `scene/gaussian_model.py`
  - `set_sparse_support(...)`
  - `_ss_gate_selected_mask(...)`
  - `_ss_query_nn_d1_d2(...)`
  - `_ss_filter_small_islands(...)`
  - `prune_outside_sparse_support(...)`
- `train.py`
  - support source init and forwarding into `gaussians.set_sparse_support(...)`
  - `ss_prune_after_rgb` and `ss_prune_before_thermal` trigger points
- `run_gtgs_full_pipeline.py`
  - stage-specific SS forwarding

Current mainline default strategy:

- SS enabled for RGB stage
- one-shot prune after RGB (`--ss_prune_after_rgb`)
- thermal stage SS disabled by default unless explicitly enabled

Current default SS knobs (pipeline defaults):

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

### 2.3 Thermal Clamp and Thermal Feature Reset

Purpose:

- suppress giant gaussians in thermal stage
- remove RGB color residue at thermal start

Implementation anchors:

- `scene/gaussian_model.py`
  - `clamp_scaling_max_(max_scale)`
- `train.py`
  - clamp after restore/prune for thermal stage
  - `thermal_reset_features`: zero SH feature tensors + clear Adam moments for SH groups

Current defaults:

- `clamp_scale_max_t = 10.0`
- `thermal_reset_features = on`

---

### 2.4 Thermal Structure Gradient Loss (`t_struct_grad`)

Purpose:

- preserve pseudo-color thermal structure/edge consistency in stage-2

Implementation anchors:

- `utils/loss_utils.py`
  - `structure_grad_loss(...)`
- `train.py`
  - thermal loss injection controlled by `--t_struct_grad_w`, `--t_struct_grad_norm`
- `run_gtgs_full_pipeline.py`
  - thermal forwarding through `_build_tstruct_train_args(...)`

Current default:

- `t_struct_grad_w = 0.006`
- `t_struct_grad_norm = true`

---

### 2.5 Expanded Evaluation Stack

Purpose:

- move beyond only PSNR/SSIM/LPIPS
- report structure, artifact, stability, and no-reference novel-view quality

Implementation anchors:

- `metrics_plus.py` (reference-based extensions)
- `novel_view_metrics.py` (novel-view/no-ref diagnostics)
- `eval_blend_sweep.py` (final blended model selection and score integration)
- `summarize_ablation_excel.py` (multi-sheet ablation summary)

Current pipeline defaults:

- `--run_metrics_plus` on
- `--run_novel_view_metrics` on
- `metrics_plus` extra IQA enabled by default
  - `--extra_iqa flip,dists,fsim,vif,ms-ssim,gmsd,haarpsi,niqe,brisque,piqe,hdrvdp3`
  - `--extra_iqa_space y`
  - `--extra_iqa_device cuda`
- novel-view mode default:
  - `mode=grid72`
  - `grid_azimuth_count=8`
  - `grid_pitch_list=15,30,60`
  - `grid_distance_factors=0.5,1,1.5`

SIBR ellipsoid dump:

- pipeline default is off
- legacy proxy ellipsoid dump also default off

---

## 3. Code Map for Paper Drafting (What to Read First)

If you are using an LLM (or human reviewer workflow) to draft a paper, start from this map.

### 3.1 Pipeline and Experiment Logic

- `run_gtgs_full_pipeline.py`
  - argparse defaults define the effective algorithm recipe
  - stage command construction (`train1_cmd`, `train2_cmd`)
  - step resume/skip markers and output checks

Focus for claims:

- default method definition
- ablation switch mapping
- evaluation invocation

### 3.2 Core Training Behavior

- `train.py`
  - sparse support initialization and stage-dependent prune timing
  - thermal restore -> SGF LR reapply
  - thermal clamp/reset/structure-loss integration points

Focus for claims:

- "geometry frozen in stage-2" and "why"
- "what is optimized in stage-2"
- exact trigger timing of prune/clamp/reset

### 3.3 SparseSupport Mechanics

- `scene/gaussian_model.py`
  - support config and enable/disable state
  - NN/adaptive/island gating internals
  - one-shot prune path

Focus for claims:

- support-space definition
- threshold and adaptive policy semantics
- island post-filter behavior

### 3.4 Evaluation and Final Scores

- `metrics_plus.py`
- `novel_view_metrics.py`
- `eval_blend_sweep.py`
- `summarize_ablation_excel.py`

Focus for claims:

- why score families are used (structure/cleanliness/stability/fidelity)
- exact metric naming used in tables
- which metrics are "higher is better" vs "lower is better"

---

## 4. Environment Setup

## 4.1 Recommended Environment

- OS: Windows + PowerShell
- Python: conda env (project commonly uses `fgs`)
- Verified Python: `3.10.x`

Example:

```powershell
conda create -n fgs python=3.10 -y
conda activate fgs
```

Install dependencies according to your Gaussian Splatting base setup (CUDA/PyTorch, rasterization extension, etc.).

Additional scripts in this repo may require:

- `openpyxl` (Excel summaries)
- optional IQA backends (`pyiqa`, `piq`, or other metric providers). Missing backends degrade to `NaN` for unavailable IQAs, without breaking pipeline.

SIBR is optional and not required for default training/evaluation.

---

## 5. Data Layout

Each dataset root should minimally contain:

- `RGB/`
- `thermal/`

Pipeline generated dirs include (examples):

- `fit/`
- `input/`
- `distorted/`
- `thermal_UD/`
- `_pipeline_state/`

Output root includes:

- `Model_RGB/`
- `Model_T/`
- `Model_F/`
- `eval/`

---

## 6. Default Behavior Summary (Current Mainline)

From current `run_gtgs_full_pipeline.py` defaults:

- alignment:
  - `align=fit` (fixed fit for downstream)
  - `comparison=true`
- stage-1 RGB:
  - `rgb_iter=30000`
  - `rgb_res=4`
  - densify enabled in standard RGB window
  - SS on by default for RGB
  - prune after RGB on by default
- stage-2 Thermal:
  - `t_iter=60000`
  - `t_res=4`
  - geometry-related LRs frozen in command construction
  - `t_opacity_lr=2e-4`
  - SGF on by default
  - thermal reset on by default
  - thermal clamp default `10.0`
  - `t_struct_grad_w=0.006`
  - thermal densify disabled by default command design
- metrics:
  - `metrics_plus` on by default
  - `novel_view_metrics` on by default
  - blend sweep evaluation (step 14) enabled by default

---

## 7. Quick Start Commands

## 7.1 Full 1-14 Run

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr5\input\PVpanel" `
  --out_root "F:\databackup\xr5\output\PVpanel_full_default"
```

## 7.2 Resume Mid-Pipeline

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr5\input\PVpanel" `
  --out_root "F:\databackup\xr5\output\PVpanel_step5_12" `
  --from_step 5 --to_step 12
```

## 7.3 Force Rerun Existing Steps

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr5\input\PVpanel" `
  --out_root "F:\databackup\xr5\output\PVpanel_force" `
  --force
```

## 7.4 Change Thermal Opacity LR (Ablation)

```powershell
D:\anaconda\envs\fgs\python.exe run_gtgs_full_pipeline.py `
  --data_root "F:\databackup\xr5\input\PVpanel" `
  --out_root "F:\databackup\xr5\output\PVpanel_topacity025" `
  --from_step 10 --to_step 14 `
  --t_opacity_lr 0.025
```

---

## 8. Practical Ablation Switches

Common toggles:

- disable RGB SS:
  - `--no_ss_enable_rgb`
- disable RGB post-train prune:
  - `--no_ss_prune_after_rgb`
- enable thermal SS:
  - `--ss_enable_t`
- enable thermal pre-train prune:
  - `--ss_prune_before_thermal`
- disable thermal reset:
  - `--no_thermal_reset_features`
- disable SGF:
  - `--sgf_disable`
- disable thermal structure loss:
  - `--t_struct_grad_w 0`

Clamp notes:

- thermal clamp uses `--clamp_scale_max_t` (default 10.0)
- to make clamp effectively no-op, pass a non-positive value (e.g. `--clamp_scale_max_t -1`)

---

## 9. Output Artifacts

Typical files:

- model checkpoints:
  - `Model_RGB/chkpnt30000.pth`
  - `Model_T/chkpnt60000.pth`
- standard metrics:
  - `Model_*/results.json`
- extended metrics:
  - `Model_*/results_plus.json`
  - `Model_*/novel_view_metrics.json`
- blend sweep:
  - `eval/summary.csv`

Optional pipeline diagnostics:

- command dumps: `cmd_train1.txt`, `cmd_train2.txt`, `cmd_render.txt`, `cmd_metrics.txt` (`--save_cmds`)
- debug dump JSON (`--debug_dump`)
- profile JSON (`--profile_pipeline`)

---

## 10. Troubleshooting

## 10.1 "thermal_UD seems incomplete"

This means step 8 output check failed.

Check:

- `<data_root>/thermal_UD/images/` exists and non-empty
- `<data_root>/thermal_UD/sparse/` exists

The pipeline includes thermal image name alias fallback by stem matching to improve JPG/PNG compatibility when sparse image names differ by extension.

## 10.2 Missing step outputs after interruption

- inspect `<data_root>/_pipeline_state`
- rerun with appropriate `--from_step`
- use `--force` if outputs/markers are inconsistent

## 10.3 IQA columns are NaN

- extra IQA backend is missing for those metrics in current environment
- install optional backend package(s), then rerun metrics

---

## 11. Recommended Reading Order for New Contributors

1. `run_gtgs_full_pipeline.py`
2. `train.py`
3. `scene/gaussian_model.py`
4. `utils/sparse_support.py`
5. `metrics_plus.py`
6. `novel_view_metrics.py`
7. `eval_blend_sweep.py`
8. `summarize_ablation_excel.py`

This order maps from "high-level experiment orchestration" to "core model behavior" to "evaluation and reporting".

---

## 12. License and Base Project

This project is built on top of the Inria Gaussian Splatting codebase and follows the corresponding licensing terms in `LICENSE.md`.

