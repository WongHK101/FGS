# -*- coding: utf-8 -*-
"""
train.py (FGS / GeoTGS)

This file is based on the official graphdeco-inria/gaussian-splatting train.py,
with **optional** extensions for ADP++ (Adaptive Densification & Pruning / plus):

- Self-adaptive controller (adpp_controller.py) driven by signals (adpp_signals.py)
- Optional edge-aware thermal refinement losses (utils/loss_utils.py)
- Optional CSV logging (adp_logger.py / adp_iter_logger.py)

Default behavior (no ADP/ADP++ flags) remains identical to the upstream training loop.
"""

from __future__ import annotations

import os
import sys
import uuid
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
from random import randint
from tqdm import tqdm

from utils.loss_utils import l1_loss, ssim
# Optional extra losses (added in updated utils/loss_utils.py)
try:
    from utils.loss_utils import edge_aware_grad_loss, edge_aware_laplacian_loss
    _HAS_EDGE_LOSSES = True
except Exception:
    _HAS_EDGE_LOSSES = False

from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
try:
    from utils.graphics_utils import patch_depth  # type: ignore
except Exception:
    patch_depth = None  # type: ignore

from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    TENSORBOARD_FOUND = True
except Exception:
    TENSORBOARD_FOUND = False

# Warn only once when depth supervision is requested but patch_depth/GT is unavailable
_DEPTH_SUP_WARNED = False

# ---- ADP++ optional modules ----
try:
    try:
        from utils.adpp_controller import ADPPController, ADPPConfig  # type: ignore
        from utils.adpp_signals import edge_score_grad_ncc, fog_score_outside_aabb  # type: ignore
    except Exception:
        from utils.adpp_controller import ADPPController, ADPPConfig  # type: ignore
        from adpp_signals import edge_score_grad_ncc, fog_score_outside_aabb  # type: ignore
    _HAS_ADPP = True
except Exception:
    ADPPController = None  # type: ignore
    ADPPConfig = None      # type: ignore
    edge_score_grad_ncc = None  # type: ignore
    fog_score_outside_aabb = None  # type: ignore
    _HAS_ADPP = False

# ---- Optional CSV loggers ----
try:
    from adp_logger import ADPCSVLogger  # type: ignore
except Exception:
    ADPCSVLogger = None  # type: ignore
try:
    from adp_iter_logger import ADPIterCSVLogger  # type: ignore
except Exception:
    ADPIterCSVLogger = None  # type: ignore


def _to_gray(img: torch.Tensor) -> torch.Tensor:
    """
    Convert [C,H,W] image tensor to [H,W] grayscale for signal computation.
    """
    if img.dim() == 2:
        return img
    if img.dim() == 3:
        if img.shape[0] == 1:
            return img[0]
        # simple luminance proxy; avoid hard-coded color weights
        return img.mean(dim=0)
    raise ValueError(f"Unexpected img shape: {tuple(img.shape)}")


def _is_thermal_dataset(source_path: str) -> bool:
    sp = source_path.lower()
    return ("thermal" in sp) or ("thermal_ud" in sp) or ("thermal-ud" in sp)


def _get_xyz_opacity(gaussians: GaussianModel) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Robustly obtain xyz and (sigmoid) opacity tensors across slightly different forks.

    Returns:
      xyz: [N,3]
      opacity: [N,1] in [0,1]
    """
    # xyz
    xyz = getattr(gaussians, "get_xyz", None)
    if callable(xyz):
        xyz_t = xyz()
    else:
        xyz_t = xyz
    if xyz_t is None:
        xyz_t = getattr(gaussians, "_xyz", None)
    if xyz_t is None:
        raise AttributeError("Cannot locate gaussians xyz tensor (get_xyz / _xyz).")

    # opacity (prefer post-sigmoid accessor if available)
    op = getattr(gaussians, "get_opacity", None)
    if callable(op):
        op_t = op()
    else:
        op_t = op
    if op_t is None:
        raw = getattr(gaussians, "_opacity", None)
        if raw is None:
            raise AttributeError("Cannot locate gaussians opacity tensor (get_opacity / _opacity).")
        op_t = torch.sigmoid(raw)
    return xyz_t, op_t


def _update_optimizer_lr_mult(optimizer: torch.optim.Optimizer, group_name: str, mult: float, allow: bool = True) -> None:
    """
    Update lr of a param group by name, preserving its base_lr the first time we touch it.
    """
    if optimizer is None:
        return
    for g in optimizer.param_groups:
        if g.get("name", "") == group_name:
            if "base_lr" not in g:
                g["base_lr"] = g.get("lr", 0.0)
            base = float(g.get("base_lr", g.get("lr", 0.0)))
            g["lr"] = (base * float(mult)) if allow else 0.0


def training(dataset, opt, pipe, test_iterations, save_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # --- ADPP/compat: when resuming from a checkpoint, ensure requested exports exist ---
    if checkpoint is not None:
        try:
            if first_iter in set(save_iterations):
                pc_dir = os.path.join(scene.model_path, "point_cloud", f"iteration_{first_iter}")
                if not os.path.isdir(pc_dir):
                    print(f"[ITER {first_iter}] Saving Gaussians (resume export)")
                    scene.save(first_iter)
        except Exception as _e:
            print(f"[WARN] Failed to export point_cloud for resume checkpoint at iter {first_iter}: {_e}")


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # -----------------
    # ADP++ initialization (optional)
    # -----------------
    adp_enabled = bool(getattr(opt, "adp_enabled", False))
    adpp_enabled = bool(getattr(opt, "adpp_enabled", False))
    use_adpp = (adp_enabled or adpp_enabled) and _HAS_ADPP
    if (adp_enabled or adpp_enabled) and not _HAS_ADPP:
        print("[WARN] ADP/ADP++ flags set but adpp_controller.py / adpp_signals.py not importable. Running vanilla training.")

    is_thermal = _is_thermal_dataset(str(dataset.source_path))

    # Trust region (AABB) from the *initial* gaussian positions (sparse-ish support)
    aabb_min = None
    aabb_max = None
    if use_adpp:
        try:
            xyz0, _ = _get_xyz_opacity(gaussians)
            xyz0 = xyz0.detach()
            aabb_min = xyz0.min(dim=0).values
            aabb_max = xyz0.max(dim=0).values
            # Expand AABB slightly (configurable)
            margin = float(getattr(opt, "adpp_aabb_margin", 0.10))
            extent = (aabb_max - aabb_min).clamp_min(1e-6)
            aabb_min = aabb_min - margin * extent
            aabb_max = aabb_max + margin * extent
        except Exception as e:
            print(f"[WARN] ADP++ could not compute trust AABB from initial gaussians: {e}")
            aabb_min, aabb_max = None, None

    controller: Optional[ADPPController] = None  # type: ignore
    last_action: Dict[str, Any] = {
        "edge_loss_weight": 0.0,
        "allow_scaling": True,
        "scaling_lr_mult": 1.0,
        "densify_interval_mult": 1.0,
        "densify_grad_thr_mult": 1.0,
        "anti_fog_strength": 0.0,
        "fog_prune_frac": 0.0,
    }

    # Optional CSV logging
    cycle_logger = None
    iter_logger = None

    if use_adpp and ADPPConfig is not None:
        cfg = ADPPConfig(
            trigger=str(getattr(opt, "adpp_trigger", "cycle")),
            decision_interval=int(getattr(opt, "adp_tex_interval", 4)),
            q_edge_init=float(getattr(opt, "adp_q_tex_gate_init", 0.55)),
            q_edge_final=float(getattr(opt, "adp_q_tex_gate_final", 0.85)),
            q_fog_init=float(getattr(opt, "adpp_q_fog_init", 0.55)),
            q_fog_final=float(getattr(opt, "adpp_q_fog_final", 0.85)),
            bad_streak_kill=int(getattr(opt, "adp_bad_streak_kill", 3)),
            max_edge_loss_weight=float(getattr(opt, "adpp_max_edge_loss_weight", 0.25)),
            edge_grad_weight=float(getattr(opt, "adpp_edge_grad_weight", 1.0)),
            edge_lap_weight=float(getattr(opt, "adpp_edge_lap_weight", 1.0)),
            densify_interval_mult_max=float(getattr(opt, "adpp_densify_interval_mult_max", 2.0)),
            densify_grad_thr_mult_max=float(getattr(opt, "adpp_densify_grad_thr_mult_max", 2.0)),
            anti_fog_strength_max=float(getattr(opt, "adpp_anti_fog_strength_max", 0.75)),
            fog_prune_frac_max=float(getattr(opt, "adpp_fog_prune_frac_max", 0.10)),
            loss_spike_factor=float(getattr(opt, "adpp_loss_spike_factor", 2.5)),
        )
        controller = ADPPController(cfg)

        # CSV loggers (optional)
        if bool(getattr(opt, "adp_csv_log", False)) and ADPCSVLogger is not None:
            cycle_logger = ADPCSVLogger(os.path.join(dataset.model_path, "adp_cycle.csv"))
            print(f"[ADP] Cycle CSV logging enabled: {os.path.join(dataset.model_path, 'adp_cycle.csv')}")
        if bool(getattr(opt, "adp_iter_csv_log", False)) and ADPIterCSVLogger is not None:
            iter_logger = ADPIterCSVLogger(os.path.join(dataset.model_path, "adp_iter.csv"))
            print(f"[ADP] Iter CSV logging enabled: {os.path.join(dataset.model_path, 'adp_iter.csv')}")

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # For health monitoring (loss spikes)
    ema_loss = None

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_mod = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_mod)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, 0, 1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (iteration < opt.iterations):
                    break
            except Exception:
                network_gui.conn = None

        gaussians.update_learning_rate(iteration)

        # Update scaling LR multiplier if controller wants it
        if use_adpp and controller is not None:
            _update_optimizer_lr_mult(
                gaussians.optimizer,
                "scaling",
                float(last_action.get("scaling_lr_mult", 1.0)),
                allow=bool(last_action.get("allow_scaling", True)),
            )

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Random camera
        if not hasattr(scene, "viewpoint_stack") or len(scene.viewpoint_stack) == 0:
            scene.viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = scene.viewpoint_stack.pop(randint(0, len(scene.viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        depth = render_pkg.get("depth", None)

        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # Optional depth loss (guarded for compatibility across forks)
        depth_l1_weight_init = float(getattr(opt, "depth_l1_weight_init", 0.0))
        depth_l1_weight_final = float(getattr(opt, "depth_l1_weight_final", 0.0))
        if depth is not None and depth_l1_weight_final > 0.0:
            # Optional delay multiplier (some forks use this; others don't)
            depth_l1_weight_delay_mult = float(getattr(opt, "depth_l1_weight_delay_mult", 1.0))
            depth_l1_weight = min(
                depth_l1_weight_final,
                depth_l1_weight_init + depth_l1_weight_delay_mult * depth_l1_weight_final * iteration / max(opt.iterations, 1),
            )
            global _DEPTH_SUP_WARNED
            depth_gt = None
            if callable(patch_depth):
                try:
                    depth_gt = patch_depth(viewpoint_cam)
                except Exception:
                    depth_gt = None
            if depth_gt is None:
                if not _DEPTH_SUP_WARNED:
                    print("[WARN] Depth supervision requested (depth_l1_weight_final>0) but depth GT / patch_depth is unavailable. Skipping depth loss.")
                    _DEPTH_SUP_WARNED = True
            else:
                depth_gt = depth_gt.cuda() if hasattr(depth_gt, 'cuda') else depth_gt
                loss += depth_l1_weight * torch.abs(depth - depth_gt).mean()

        # -----------------
        # ADP++: edge-aware thermal refinement loss (optional)
        # -----------------
        if use_adpp and is_thermal and _HAS_EDGE_LOSSES:
            w_edge = float(last_action.get("edge_loss_weight", 0.0))
            if w_edge > 0.0:
                # Optional alpha mask support (if dataset provides it; otherwise None)
                alpha_mask = getattr(viewpoint_cam, "alpha_mask", None)
                if alpha_mask is not None:
                    alpha_mask = alpha_mask.cuda()

                # Two complementary sharpness terms
                loss_g = edge_aware_grad_loss(image, gt_image, mask=alpha_mask)
                loss_l = edge_aware_laplacian_loss(image, gt_image, mask=alpha_mask)
                # Controller may want to weight grad vs lap differently
                wg = float(last_action.get("edge_grad_weight", 1.0))
                wl = float(last_action.get("edge_lap_weight", 1.0))
                loss += w_edge * (wg * loss_g + wl * loss_l)

        loss.backward()

        # -----------------
        # ADP++: compute signals + controller step (cheap, in-loop)
        # -----------------
        if use_adpp and controller is not None:
            # loss health monitoring (EMA)
            loss_val = float(loss.detach().item())
            if ema_loss is None:
                ema_loss = loss_val
            else:
                ema_loss = 0.99 * float(ema_loss) + 0.01 * loss_val
            health_bad = (not np.isfinite(loss_val)) or (ema_loss is not None and loss_val > float(ema_loss) * float(controller.cfg.loss_spike_factor))

            edge_score = None
            fog_score = None

            # Compute edge score every adp_log_interval (default 50)
            log_int = int(getattr(opt, "adp_log_interval", 50))
            if log_int > 0 and (iteration % log_int == 0):
                try:
                    pred_g = _to_gray(torch.clamp(image.detach(), 0, 1))
                    gt_g = _to_gray(torch.clamp(gt_image.detach(), 0, 1))
                    # edge_score_grad_ncc expects [H,W] or [1,H,W]
                    edge_score = float(edge_score_grad_ncc(pred_g, gt_g).item())
                except Exception:
                    edge_score = None

            # Compute fog score (requires trust AABB)
            fog_int = int(getattr(opt, "adpp_fog_interval", log_int))
            if aabb_min is not None and aabb_max is not None and fog_int > 0 and (iteration % fog_int == 0):
                try:
                    xyz_t, op_t = _get_xyz_opacity(gaussians)
                    fog_score = float(fog_score_outside_aabb(xyz_t.detach(), op_t.detach(), aabb_min, aabb_max).item())
                except Exception:
                    fog_score = None

            # Controller step returns an action dict
            last_action = controller.step(iteration, edge_score=edge_score, fog_score=fog_score, health_bad=health_bad)

            # Provide action to gaussian model (if it implements hooks)
            if hasattr(gaussians, "adpp_set_action") and callable(getattr(gaussians, "adpp_set_action")):
                try:
                    gaussians.adpp_set_action(last_action)  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                # Fallback: attach to object for other modules to read
                try:
                    gaussians._adpp_action = last_action  # type: ignore[attr-defined]
                except Exception:
                    pass

            # Iter-level CSV log (optional)
            if iter_logger is not None:
                it_int = int(getattr(opt, "adp_iter_csv_interval", log_int))
                if it_int > 0 and (iteration % it_int == 0):
                    d = {}
                    d.update({"iteration": iteration, "loss": loss_val})
                    d.update(controller.get_log_dict(prefix="adp_"))
                    iter_logger.log(d)

        with torch.no_grad():
            # Progress bar update
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.7f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Save
            if (iteration in save_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification + pruning
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # Dynamic schedule from controller
                densify_interval = int(opt.densification_interval)
                densify_thr = float(opt.densify_grad_threshold)
                if use_adpp:
                    densify_interval = max(1, int(round(densify_interval * float(last_action.get("densify_interval_mult", 1.0)))))
                    densify_thr = densify_thr * float(last_action.get("densify_grad_thr_mult", 1.0))

                if iteration > opt.densify_from_iter and (iteration % densify_interval == 0):
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    # Optional: anti-fog pruning hook (prefers gaussian_model implementation)
                    if use_adpp and float(last_action.get("anti_fog_strength", 0.0)) > 0.0 and aabb_min is not None and aabb_max is not None:
                        if hasattr(gaussians, "adpp_pre_densify_hook") and callable(getattr(gaussians, "adpp_pre_densify_hook")):
                            try:
                                gaussians.adpp_pre_densify_hook(aabb_min, aabb_max, last_action)  # type: ignore[attr-defined]
                            except Exception:
                                pass

                                        # Densify/prune API differs across GaussianModel variants.
                    # Our ADPP-compatible GaussianModel expects `radii` (and may optionally use iteration bounds).
                    try:
                        gaussians.densify_and_prune(
                            densify_thr,
                            0.005,
                            scene.cameras_extent,
                            size_threshold,
                            radii,
                            iteration=iteration,
                            densify_from_iter=getattr(opt, 'densify_from_iter', None),
                            densify_until_iter=getattr(opt, 'densify_until_iter', None),
                        )
                    except TypeError:
                        try:
                            gaussians.densify_and_prune(densify_thr, 0.005, scene.cameras_extent, size_threshold, radii)
                        except TypeError:
                            gaussians.densify_and_prune(densify_thr, 0.005, scene.cameras_extent, size_threshold)


                    if use_adpp and float(last_action.get("anti_fog_strength", 0.0)) > 0.0 and aabb_min is not None and aabb_max is not None:
                        if hasattr(gaussians, "adpp_post_densify_hook") and callable(getattr(gaussians, "adpp_post_densify_hook")):
                            try:
                                gaussians.adpp_post_densify_hook(aabb_min, aabb_max, last_action)  # type: ignore[attr-defined]
                            except Exception:
                                pass

                    # Cycle CSV log (optional)
                    if cycle_logger is not None and controller is not None:
                        d = {"iteration": iteration}
                        d.update(controller.get_log_dict(prefix="adp_"))
                        cycle_logger.log(d)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # TensorBoard scalars
            if tb_writer is not None:
                tb_writer.add_scalar("train/loss_l1", Ll1.item(), iteration)
                tb_writer.add_scalar("train/loss_total", loss.item(), iteration)
                tb_writer.add_scalar("train/psnr", psnr(image, gt_image).mean().item(), iteration)
                if use_adpp and controller is not None:
                    for k, v in controller.get_log_dict(prefix="adp/").items():
                        if isinstance(v, (int, float)):
                            tb_writer.add_scalar(k, float(v), iteration)

        # Evaluation on test set (unchanged)
        if iteration in test_iterations:
            # Validation can be memory-hungry; keep it no-grad and optionally subsample cameras.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            validation_configs = (
                {"name": "test", "cameras": scene.getTestCameras()},
                {"name": "train", "cameras": scene.getTrainCameras()},
            )

            # Subsample cameras for validation to stabilize memory on large datasets.
            max_eval_cams = int(getattr(args, "eval_max_cams", 32) or 32)
            if max_eval_cams > 0:
                for cfg in validation_configs:
                    cams = cfg.get("cameras", None)
                    if cams is None:
                        continue
                    try:
                        n = len(cams)
                    except Exception:
                        continue
                    if n > max_eval_cams:
                        step = max(1, n // max_eval_cams)
                        cfg["cameras"] = cams[::step][:max_eval_cams]

            for config in validation_configs:
                cams = config.get("cameras", None)
                if cams is None:
                    continue
                try:
                    num_cams = len(cams)
                except Exception:
                    num_cams = 0
                if num_cams <= 0:
                    continue

                l1_test = 0.0
                psnr_test = 0.0
                try:
                    with torch.no_grad():
                        for idx, viewpoint in enumerate(cams):
                            image_val = torch.clamp(render(viewpoint, gaussians, pipe, background)["render"], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                            l1_test += l1_loss(image_val, gt_image).mean().double()
                            psnr_test += psnr(image_val, gt_image).mean().double()

                    l1_test /= num_cams
                    psnr_test /= num_cams
                    print("\n[ITER {}] Evaluating {} ({} cams): L1 {:.6f} PSNR {:.2f}".format(
                        iteration, config["name"], num_cams, l1_test, psnr_test
                    ))
                    if tb_writer is not None:
                        tb_writer.add_scalar(config["name"] + "/loss_l1", l1_test, iteration)
                        tb_writer.add_scalar(config["name"] + "/psnr", psnr_test, iteration)

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("[WARN] CUDA OOM during {} eval at iter {} — skipping this eval round.".format(
                            config.get("name", "val"), iteration
                        ))
                        torch.cuda.empty_cache()
                    else:
                        raise

            torch.cuda.empty_cache()


def prepare_output_and_logger(args):
    if not args.model_path:
        args.model_path = os.path.join("./output/", str(uuid.uuid4()))
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting training (with optional ADP++).")

    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    # -----------------
    # Pipeline compatibility flags
    # -----------------
    # Some forks of arguments.py may not expose these options, but the pipeline runner expects them.
    # We add them here if missing to keep train.py compatible with run_gtgs_full_pipeline.py.
    if '--test_iterations' not in parser._option_string_actions:
        parser.add_argument('--test_iterations', nargs='+', type=int, default=[7000, 30000],
                            help='Iteration(s) at which to run evaluation renders during training.')
    if '--save_iterations' not in parser._option_string_actions:
        parser.add_argument('--save_iterations', nargs='+', type=int, default=[7000, 30000],
                            help='Iteration(s) at which to save point clouds during training.')
    if '--checkpoint_iterations' not in parser._option_string_actions:
        parser.add_argument('--checkpoint_iterations', nargs='+', type=int, default=[30000],
                            help='Iteration(s) at which to save training checkpoints (chkpnt*.pth).')
    if '--start_checkpoint' not in parser._option_string_actions:
        parser.add_argument('--start_checkpoint', type=str, default=None,
                            help='Path to a checkpoint to start/continue training from.')
    if '--debug_from' not in parser._option_string_actions:
        parser.add_argument('--debug_from', type=int, default=-1,
                            help='Start debugging visualization from this iteration (disabled if <0).')
    if '--quiet' not in parser._option_string_actions:
        parser.add_argument('--quiet', action='store_true', default=False,
                            help='Reduce console output.')
    if '--detect_anomaly' not in parser._option_string_actions:
        parser.add_argument('--detect_anomaly', action='store_true', default=False,
                            help='Enable torch autograd anomaly detection (slow).')


    # -----------------
    # Extra ADP/ADP++ CLI flags (optional)
    # -----------------
    # Keep names compatible with your pipeline runner.
    parser.add_argument("--adp_enabled", action="store_true", default=False,
                        help="Enable ADP++ controller (self-adaptive training). Default: off.")
    parser.add_argument("--adpp_enabled", action="store_true", default=False,
                        help=argparse.SUPPRESS)  # alias

    parser.add_argument("--adp_tex_interval", type=int, default=4,
                        help="ADP++ decision interval base (iters). (used as decision_interval in controller)")
    parser.add_argument("--adp_log_interval", type=int, default=50,
                        help="ADP++ signal/log interval (iters).")
    parser.add_argument("--adp_no_cycle_print", action="store_true", default=False,
                        help="Disable ADP++ cycle console prints (controller still runs).")

    # Compatibility with older ADP profiles (mapped into controller thresholds)
    parser.add_argument("--adp_bad_streak_kill", type=int, default=3)
    parser.add_argument("--adp_q_tex_gate_init", type=float, default=0.55)
    parser.add_argument("--adp_q_tex_gate_final", type=float, default=0.85)

    # Anti-fog config
    parser.add_argument("--adpp_q_fog_init", type=float, default=0.55)
    parser.add_argument("--adpp_q_fog_final", type=float, default=0.85)
    parser.add_argument("--adpp_aabb_margin", type=float, default=0.10,
                        help="Trust AABB expansion margin used by fog score (fraction of extent).")
    parser.add_argument("--adpp_fog_interval", type=int, default=50,
                        help="Fog score interval (iters). Default follows adp_log_interval if not set.")

    # Edge refinement config
    parser.add_argument("--adpp_max_edge_loss_weight", type=float, default=0.25)
    parser.add_argument("--adpp_edge_grad_weight", type=float, default=1.0)
    parser.add_argument("--adpp_edge_lap_weight", type=float, default=1.0)

    # Densification schedule adaptivity
    parser.add_argument("--adpp_densify_interval_mult_max", type=float, default=2.0)
    parser.add_argument("--adpp_densify_grad_thr_mult_max", type=float, default=2.0)

    # Anti-fog action bounds
    parser.add_argument("--adpp_anti_fog_strength_max", type=float, default=0.75)
    parser.add_argument("--adpp_fog_prune_frac_max", type=float, default=0.10)

    # Trigger rule (you asked me to decide: default='cycle' for stability / paper-friendly)
    parser.add_argument("--adpp_trigger", choices=["cycle", "continuous"], default="cycle",
                        help="ADP++ trigger rule. cycle=update mainly on densify/prune cycles; continuous=update every decision interval.")

    parser.add_argument("--adpp_loss_spike_factor", type=float, default=2.5,
                        help="If instantaneous loss > EMA*factor, treat as 'health_bad' for controller.")

    # CSV logging (optional; requires adp_logger.py/adp_iter_logger.py)
    parser.add_argument("--adp_csv_log", action="store_true", default=False,
                        help="Write cycle-level ADP stats CSV (adp_cycle.csv).")
    parser.add_argument("--adp_iter_csv_log", action="store_true", default=False,
                        help="Write iteration-level ADP stats CSV (adp_iter.csv).")
    parser.add_argument("--adp_iter_csv_interval", type=int, default=50,
                        help="Iteration-level CSV log interval (iters).")

    args = get_combined_args(parser)
    # ------------------------------------------------------------------
    # Compatibility: get_combined_args() drops None-valued defaults.
    # Some pipeline-required options (e.g., --start_checkpoint) default to None,
    # so they may be missing from `args`. Ensure they exist with sane defaults.
    # ------------------------------------------------------------------
    if not hasattr(args, 'start_checkpoint'):
        args.start_checkpoint = None
    print("Optimizing " + args.model_path)

    # Keep downstream extraction unchanged
    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # Attach ADP/ADP++ fields onto opt so downstream modules (GaussianModel) can read them if needed.
    # This also avoids fragile cross-file dependencies on argparse.Namespace.
    for k in [
        "adp_enabled", "adpp_enabled",
        "adp_tex_interval", "adp_log_interval", "adp_no_cycle_print",
        "adp_bad_streak_kill", "adp_q_tex_gate_init", "adp_q_tex_gate_final",
        "adpp_q_fog_init", "adpp_q_fog_final", "adpp_aabb_margin", "adpp_fog_interval",
        "adpp_max_edge_loss_weight", "adpp_edge_grad_weight", "adpp_edge_lap_weight",
        "adpp_densify_interval_mult_max", "adpp_densify_grad_thr_mult_max",
        "adpp_anti_fog_strength_max", "adpp_fog_prune_frac_max",
        "adpp_trigger", "adpp_loss_spike_factor",
        "adp_csv_log", "adp_iter_csv_log", "adp_iter_csv_interval",
    ]:
        setattr(opt, k, getattr(args, k, getattr(opt, k, None)))

    # Alias: --adp_enabled or --adpp_enabled both enable ADP++
    if getattr(opt, "adpp_enabled", False) and not getattr(opt, "adp_enabled", False):
        opt.adp_enabled = True

    prepare_output_and_logger(args)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # --- ADPP/compat: always export final point_cloud for downstream render/metrics ---
    try:
        final_iter = opt.iterations
        pc_dir = os.path.join(scene.model_path, "point_cloud", f"iteration_{final_iter}")
        if not os.path.isdir(pc_dir):
            print(f"[ITER {final_iter}] Saving Gaussians (final export)")
            scene.save(final_iter)
    except Exception as _e:
        print(f"[WARN] Failed to export final point_cloud at iter {final_iter}: {_e}")

    print("\nTraining complete.")