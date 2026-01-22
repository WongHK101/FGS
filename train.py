#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import get_expon_lr_func, safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.adp_logger import ADPCSVLogger
from utils.adp_iter_logger import ADPIterCSVLogger

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def _compute_image_texture_map(gt_image: torch.Tensor, alpha_mask: torch.Tensor = None) -> torch.Tensor:
    """Compute a normalized gradient-magnitude map (H,W) in [0,1] from GT image.
    Used as 'texture support' for ADP-Texture gating.
    """
    gray = gt_image.mean(dim=0)  # (H,W)
    gx = gray[:, 1:] - gray[:, :-1]
    gy = gray[1:, :] - gray[:-1, :]
    gx = F.pad(gx, (0, 1, 0, 0), mode="constant", value=0.0)
    gy = F.pad(gy, (0, 0, 0, 1), mode="constant", value=0.0)
    gmag = torch.sqrt(gx * gx + gy * gy)

    if alpha_mask is not None:
        if alpha_mask.dim() == 3:
            am = alpha_mask[0] if alpha_mask.shape[0] == 1 else alpha_mask.mean(dim=0)
        else:
            am = alpha_mask
        gmag = gmag * am

    scale = gmag.mean() + 2.0 * gmag.std() + 1e-6
    tex_map = torch.clamp(gmag / scale, 0.0, 1.0)
    return tex_map


def _compute_tex_values_for_gaussians(
    gaussians: GaussianModel,
    viewpoint_cam,
    visibility_filter: torch.Tensor,
    gt_image: torch.Tensor,
    alpha_mask: torch.Tensor = None,
) -> torch.Tensor:
    if visibility_filter is None or visibility_filter.numel() == 0:
        return None
    if visibility_filter.dtype == torch.bool and (not bool(visibility_filter.any().item())):
        return None

    tex_map = _compute_image_texture_map(gt_image, alpha_mask=alpha_mask)  # (H,W)
    H, W = tex_map.shape[0], tex_map.shape[1]

    idx = visibility_filter.squeeze(-1) if visibility_filter.dim() > 1 else visibility_filter
    xyz = gaussians.get_xyz[idx]  # (M,3)
    if xyz.numel() == 0 or xyz.shape[0] == 0:
        return None

    ones = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
    xyz1 = torch.cat([xyz, ones], dim=1)  # (M,4)

    full_proj = viewpoint_cam.full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)  # (4,4)
    clip = xyz1 @ full_proj  # (M,4)
    ndc = clip[:, :3] / (clip[:, 3:4] + 1e-7)

    x = (ndc[:, 0] * 0.5 + 0.5) * (W - 1)
    y = (-ndc[:, 1] * 0.5 + 0.5) * (H - 1)

    x_norm = x / max(1.0, (W - 1)) * 2.0 - 1.0
    y_norm = y / max(1.0, (H - 1)) * 2.0 - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)

    tex_in = tex_map.view(1, 1, H, W)
    samples = F.grid_sample(tex_in, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return samples.view(-1).clamp(0.0, 1.0)


def _tb_log_scalars(tb_writer, scalars: dict, iteration: int):
    if tb_writer is None or not isinstance(scalars, dict) or len(scalars) == 0:
        return
    for k, v in scalars.items():
        try:
            if v is None:
                continue
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(k, float(v), iteration)
        except Exception:
            continue


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and getattr(opt, "optimizer_type", "default_adam") == "sparse_adam":
        sys.stderr.write("[Warning] Sparse adam is not available. Switching to default adam.\n")
        opt.optimizer_type = "default_adam"

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # CSV loggers (paper-friendly)
    adp_cycle_logger = None
    adp_iter_logger = None

    if getattr(opt, "adp_enabled", False) and getattr(opt, "adp_csv_log", False):
        csv_path = getattr(opt, "adp_csv_path", None)
        if csv_path is None or str(csv_path).strip() == "":
            csv_path = os.path.join(scene.model_path, "adp_cycle.csv")
        adp_cycle_logger = ADPCSVLogger(str(csv_path))
        print(f"[ADP] Cycle CSV logging enabled: {csv_path}")

    if getattr(opt, "adp_enabled", False) and getattr(opt, "adp_iter_csv_log", False):
        csv_path = getattr(opt, "adp_iter_csv_path", None)
        if csv_path is None or str(csv_path).strip() == "":
            csv_path = os.path.join(scene.model_path, "adp_iter.csv")
        adp_iter_logger = ADPIterCSVLogger(str(csv_path), extra_fields=["total_points"])
        print(f"[ADP] Iter CSV logging enabled: {csv_path}")

    if getattr(opt, "optimizer_type", "default_adam") == "sparse_adam":
        gaussians.optimizer = SparseGaussianAdam(gaussians.optimizer.param_groups, lr=0.0, eps=1e-15)
        gaussians.optimizer.set_state_dict(gaussians.optimizer.state_dict())

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    viewpoint_indices = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    if hasattr(opt, "depth_l1_weight_init") and hasattr(opt, "depth_l1_weight_final"):
        depth_delay_steps = getattr(opt, "depth_l1_weight_delay_steps", 0)
        depth_delay_mult  = getattr(opt, "depth_l1_weight_delay_mult", 1.0)
        depth_max_steps   = getattr(opt, "depth_l1_weight_max_steps", getattr(opt, "iterations", 30000))

        depth_l1_weight = get_expon_lr_func(
            opt.depth_l1_weight_init,
            opt.depth_l1_weight_final,
            lr_delay_steps=depth_delay_steps,
            lr_delay_mult=depth_delay_mult,
            max_steps=depth_max_steps,
        )
    else:
        depth_l1_weight = lambda _step: 0.0

    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifier = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background,
                        scaling_modifier=scaling_modifier,
                        separate_sh=SPARSE_ADAM_AVAILABLE
                    )["render"]
                    net_image_bytes = (torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < opt.iterations) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))

        rand_idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        _ = viewpoint_indices.pop(rand_idx)

        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(
            viewpoint_cam, gaussians, pipe, bg,
            scaling_modifier=1.0,
            use_trained_exp=dataset.train_test_exp,
            separate_sh=SPARSE_ADAM_AVAILABLE
        )
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        alpha_mask = None
        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render, (pipe, background),
                dataset.train_test_exp
            )

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )

                tex_values = None
                if getattr(opt, "adp_enabled", False) and (iteration % int(getattr(opt, "adp_tex_interval", 4)) == 0):
                    tex_values = _compute_tex_values_for_gaussians(
                        gaussians=gaussians,
                        viewpoint_cam=viewpoint_cam,
                        visibility_filter=visibility_filter,
                        gt_image=gt_image,
                        alpha_mask=alpha_mask,
                    )

                gaussians.add_densification_stats(
                    viewspace_point_tensor,
                    visibility_filter,
                    radii=radii,
                    tex_values=tex_values,
                )

                # Iter CSV logging
                if adp_iter_logger is not None:
                    interval = int(getattr(opt, "adp_iter_csv_interval", 50))
                    if interval > 0 and (iteration % interval == 0):
                        stats = gaussians.get_adp_log_dict(prefix="adp_")
                        extra = {"total_points": int(gaussians.get_xyz.shape[0])}
                        adp_iter_logger.log_iter(iteration, stats, extra=extra)

                # TensorBoard: ADP stats (throttled)
                if getattr(opt, "adp_enabled", False) and tb_writer is not None:
                    log_int = int(getattr(opt, "adp_log_interval", 50))
                    if log_int > 0 and (iteration % log_int == 0):
                        _tb_log_scalars(tb_writer, gaussians.get_adp_log_dict(prefix="adp/"), iteration)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        size_threshold,
                        radii,
                        iteration=iteration,
                        densify_from_iter=opt.densify_from_iter,
                        densify_until_iter=opt.densify_until_iter,
                    )

                    if getattr(opt, "adp_enabled", False) and tb_writer is not None:
                        _tb_log_scalars(tb_writer, gaussians.get_adp_log_dict(prefix="adp/"), iteration)

                    if adp_cycle_logger is not None:
                        adp_cycle_logger.log_cycle(iteration, getattr(gaussians, "adp_last_cycle_stats", {}))

                    if getattr(opt, "adp_enabled", False) and (not getattr(opt, "adp_no_cycle_print", False)):
                        s = getattr(gaussians, "adp_last_cycle_stats", {})
                        try:
                            print(
                                f"[ADP][ITER {iteration}] "
                                f"gate_ratio={s.get('gate_ratio', None):.3f} "
                                f"tex_gate={s.get('tex_thr_gate', None):.4f} "
                                f"grad_thr={s.get('grad_thr', None):.6f} "
                                f"adp_prune={int(s.get('adp_prune_count', 0))} "
                                f"prune_total={int(s.get('prune_total_count', 0))} "
                                f"N:{int(s.get('N_before', 0))}->{int(s.get('N_after_densify', 0))}->{int(s.get('N_after_prune', 0))}"
                            )
                        except Exception:
                            print(f"[ADP][ITER {iteration}] cycle stats: {s}")

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") if os.getenv("OAR_JOB_ID") else str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    if not TENSORBOARD_FOUND:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss_fn, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {"name": "train", "cameras": [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for _, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, use_trained_exp=train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"],
                        0.0, 1.0
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if viewpoint.alpha_mask is not None:
                        image *= viewpoint.alpha_mask.to("cuda")
                        gt_image *= viewpoint.alpha_mask.to("cuda")
                    l1_test += l1_loss_fn(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    # --- ADP flags ---
    parser.add_argument("--adp_enabled", action="store_true", default=False)
    parser.add_argument("--adp_tex_interval", type=int, default=4)
    parser.add_argument("--adp_log_interval", type=int, default=50)
    parser.add_argument("--adp_no_cycle_print", action="store_true", default=False)

    # Cycle CSV
    parser.add_argument("--adp_csv_log", action="store_true", default=False)
    parser.add_argument("--adp_csv_path", type=str, default=None)

    # Iter CSV
    parser.add_argument("--adp_iter_csv_log", action="store_true", default=False)
    parser.add_argument("--adp_iter_csv_interval", type=int, default=50)
    parser.add_argument("--adp_iter_csv_path", type=str, default=None)

    # ADP hyperparams
    parser.add_argument("--adp_tex_beta", type=float, default=0.05)
    parser.add_argument("--adp_spark_beta", type=float, default=0.05)
    parser.add_argument("--adp_bad_streak_kill", type=int, default=3)
    parser.add_argument("--adp_bad_streak_decay", type=int, default=1)
    parser.add_argument("--adp_q_tex_gate_init", type=float, default=0.60)
    parser.add_argument("--adp_q_tex_gate_final", type=float, default=0.85)
    parser.add_argument("--adp_q_tex_prune_init", type=float, default=0.20)
    parser.add_argument("--adp_q_tex_prune_final", type=float, default=0.35)
    parser.add_argument("--adp_q_grad_init", type=float, default=0.70)
    parser.add_argument("--adp_q_grad_final", type=float, default=0.90)
    parser.add_argument("--adp_q_spark", type=float, default=0.95)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset = lp.extract(args)
    opt = op.extract(args)
    pipe = pp.extract(args)

    # Copy ADP args into opt
    _adp_keys = [
        "adp_enabled", "adp_tex_interval", "adp_log_interval", "adp_no_cycle_print",
        "adp_csv_log", "adp_csv_path",
        "adp_iter_csv_log", "adp_iter_csv_interval", "adp_iter_csv_path",
        "adp_tex_beta", "adp_spark_beta", "adp_bad_streak_kill", "adp_bad_streak_decay",
        "adp_q_tex_gate_init", "adp_q_tex_gate_final",
        "adp_q_tex_prune_init", "adp_q_tex_prune_final",
        "adp_q_grad_init", "adp_q_grad_final",
        "adp_q_spark",
    ]
    for _k in _adp_keys:
        if hasattr(args, _k):
            setattr(opt, _k, getattr(args, _k))

    training(dataset, opt, pipe, args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")
