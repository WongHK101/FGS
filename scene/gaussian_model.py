# NOTE: This file is generated for ADPP_COMPAT_v2 integration (GeoTGS/FGS).
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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # --- ADP (Artifact-aware Densification & Pruning) ---
        # Buffers are tensors (NOT optimized). They are updated during training to
        # gate densification and to prune persistent floaters (e.g., sky sparkles).
        self.adp_enabled = False
        self.adp_tex_ema = torch.empty(0)       # (N,1) EMA of texture support in [0,1]
        self.adp_spark_ema = torch.empty(0)     # (N,1) EMA of sparkle risk proxy
        self.adp_vis_count = torch.empty(0)     # (N,1) visibility counter (for analysis)
        self.adp_bad_streak = torch.empty(0, dtype=torch.int32)  # (N,1) persistent artifact counter
        self._adp_last_cycle_stats = {}  # ADP cycle stats cache for logging

        # --- ADPP (Adaptive Densification/Pruning Policy) ---
        # Runtime-only action dict set by train.py (not saved in checkpoints).
        self._adpp_action = None
        # Optional densification region constraint mask (N,). When set, densification
        # is allowed only where this mask is True. Cleared after each densify cycle.
        self._adpp_densify_extra_mask = None
        # Cache some ADPP stats for debugging/plots.
        self._adpp_last_stats = {}


        # Hyperparameters (can be overridden via OptimizationParams)
        self.adp_tex_beta = 0.05
        self.adp_spark_beta = 0.05
        self.adp_bad_streak_kill = 3
        self.adp_bad_streak_decay = 1
        self.adp_eps = 1e-6

        # Quantile schedules (paper-friendly, reduces per-scene tuning)
        # quantile q means "q fraction is below threshold".
        # Densify gate becomes stricter over time (higher q).
        self.adp_q_tex_gate_init = 0.60
        self.adp_q_tex_gate_final = 0.85
        # Prune uses a *low* texture threshold (lower q) + high sparkle threshold.
        self.adp_q_tex_prune_init = 0.20
        self.adp_q_tex_prune_final = 0.35
        self.adp_q_grad_init = 0.70
        self.adp_q_grad_final = 0.90
        self.adp_q_spark = 0.95
        self.setup_functions()
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            # ADP state (optional; restore() is backward-compatible)
            self.adp_enabled,
            self.adp_tex_ema,
            self.adp_spark_ema,
            self.adp_vis_count,
            self.adp_bad_streak,
        )
    def restore(self, model_args, training_args):
        # Backward-compatible restore:
        # - older checkpoints: 12-tuple (official 3DGS)
        # - new checkpoints:   17-tuple (adds ADP buffers)
        if len(model_args) == 12:
            (self.active_sh_degree,
             self._xyz,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args

            self.training_setup(training_args)
            self.xyz_gradient_accum = xyz_gradient_accum
            self.denom = denom
            self.optimizer.load_state_dict(opt_dict)

            # Init ADP buffers (disabled unless training_args enables it)
            self._init_adp_buffers(training_args, keep_existing=False)
            return

        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         adp_enabled,
         adp_tex_ema,
         adp_spark_ema,
         adp_vis_count,
         adp_bad_streak) = model_args

        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

        # Restore ADP buffers, then apply current hyperparams from training_args
        self.adp_enabled = bool(adp_enabled)
        self.adp_tex_ema = adp_tex_ema
        self.adp_spark_ema = adp_spark_ema
        self.adp_vis_count = adp_vis_count
        self.adp_bad_streak = adp_bad_streak
        self._init_adp_buffers(training_args, keep_existing=True)

    def _init_adp_buffers(self, training_args, keep_existing: bool = False):
        # Read ADP settings from args if present.
        self.adp_enabled = bool(getattr(training_args, "adp_enabled", self.adp_enabled))
        self.adp_tex_beta = float(getattr(training_args, "adp_tex_beta", self.adp_tex_beta))
        self.adp_spark_beta = float(getattr(training_args, "adp_spark_beta", self.adp_spark_beta))
        self.adp_bad_streak_kill = int(getattr(training_args, "adp_bad_streak_kill", self.adp_bad_streak_kill))
        self.adp_bad_streak_decay = int(getattr(training_args, "adp_bad_streak_decay", self.adp_bad_streak_decay))

        self.adp_q_tex_gate_init = float(getattr(training_args, "adp_q_tex_gate_init", self.adp_q_tex_gate_init))
        self.adp_q_tex_gate_final = float(getattr(training_args, "adp_q_tex_gate_final", self.adp_q_tex_gate_final))
        self.adp_q_tex_prune_init = float(getattr(training_args, "adp_q_tex_prune_init", self.adp_q_tex_prune_init))
        self.adp_q_tex_prune_final = float(getattr(training_args, "adp_q_tex_prune_final", self.adp_q_tex_prune_final))
        self.adp_q_grad_init = float(getattr(training_args, "adp_q_grad_init", self.adp_q_grad_init))
        self.adp_q_grad_final = float(getattr(training_args, "adp_q_grad_final", self.adp_q_grad_final))
        self.adp_q_spark = float(getattr(training_args, "adp_q_spark", self.adp_q_spark))

        N = int(self.get_xyz.shape[0])
        device = self.get_xyz.device

        # If ADP is disabled (baseline runs), keep ADP buffers empty to avoid
        # shape-sync overhead and any indexing issues during densify/prune.
        if not self.adp_enabled:
            self.adp_tex_ema = torch.empty((0, 1), device=device)
            self.adp_spark_ema = torch.empty((0, 1), device=device)
            self.adp_vis_count = torch.empty((0, 1), device=device)
            self.adp_bad_streak = torch.empty((0, 1), device=device, dtype=torch.int32)
            return

        def ensure_tensor(t, dtype=None):
            if (not keep_existing) or (t is None) or (not isinstance(t, torch.Tensor)) or (t.numel() == 0) or (t.shape[0] != N):
                if dtype is None:
                    return torch.zeros((N, 1), device=device)
                return torch.zeros((N, 1), device=device, dtype=dtype)
            return t

        self.adp_tex_ema = ensure_tensor(self.adp_tex_ema)
        self.adp_spark_ema = ensure_tensor(self.adp_spark_ema)
        self.adp_vis_count = ensure_tensor(self.adp_vis_count)
        self.adp_bad_streak = ensure_tensor(self.adp_bad_streak, dtype=torch.int32)
    def get_adp_log_dict(self, prefix: str = "") -> dict:
        """Return a dict of ADP-related scalars for logging.

        Safe to call even when ADP is disabled. Values are Python floats/ints.
        If `prefix` is provided, it is prepended to all keys (e.g., "adp_").
        """
        try:
            import torch
        except Exception:
            torch = None  # type: ignore

        def _to_float(v):
            if v is None:
                return None
            if torch is not None and isinstance(v, torch.Tensor):
                if v.numel() == 0:
                    return None
                if v.numel() == 1:
                    return float(v.detach().item())
                return float(v.detach().mean().item())
            try:
                return float(v)
            except Exception:
                return None

        def _stats_1d(t, name: str):
            out = {}
            if torch is None or t is None or (not isinstance(t, torch.Tensor)) or t.numel() == 0:
                return out
            x = t.detach().view(-1).float()
            # Downsample for quantiles to keep logging cheap on large scenes
            if x.numel() > 65536:
                step = max(1, x.numel() // 65536)
                x = x[::step]

            try:
                out[f"{name}_mean"] = float(x.mean().item())
                out[f"{name}_p50"] = float(torch.quantile(x, 0.50).item()) if x.numel() >= 32 else float(x.median().item())
                if x.numel() >= 64:
                    out[f"{name}_p10"] = float(torch.quantile(x, 0.10).item())
                    out[f"{name}_p90"] = float(torch.quantile(x, 0.90).item())
                out[f"{name}_min"] = float(x.min().item())
                out[f"{name}_max"] = float(x.max().item())
            except Exception:
                # never crash training for logging
                return out
            return out

        d = {}
        # point count
        try:
            d["total_points"] = int(self.get_xyz.shape[0])
        except Exception:
            d["total_points"] = 0

        d["enabled"] = int(bool(getattr(self, "adp_enabled", False)))

        # per-point aggregates
        d.update(_stats_1d(getattr(self, "adp_tex_ema", None), "tex_ema"))
        d.update(_stats_1d(getattr(self, "adp_spark_ema", None), "spark_ema"))
        d.update(_stats_1d(getattr(self, "adp_vis_count", None), "vis_count"))
        d.update(_stats_1d(getattr(self, "adp_bad_streak", None), "bad_streak"))

        # static hyperparams
        for k in [
            "adp_tex_beta", "adp_spark_beta", "adp_bad_streak_kill", "adp_bad_streak_decay",
            "adp_q_tex_gate_init", "adp_q_tex_gate_final",
            "adp_q_tex_prune_init", "adp_q_tex_prune_final",
            "adp_q_grad_init", "adp_q_grad_final",
            "adp_q_spark", "adp_eps",
        ]:
            if hasattr(self, k):
                d[k] = _to_float(getattr(self, k))

        # last cycle stats (populated in densify_and_prune)
        last = getattr(self, "_adp_last_cycle_stats", None)
        if isinstance(last, dict) and last:
            d.update(last)

        if prefix:
            return {f"{prefix}{k}": v for k, v in d.items()}
        return d

    # compat alias (some train.py versions call this)
    def get_adp_iter_log_dict(self, prefix: str = "") -> dict:
        return self.get_adp_log_dict(prefix=prefix)
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # Init/validate ADP buffers
        self._init_adp_buffers(training_args, keep_existing=False)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

        # ADP buffers (only when ADP enabled). Keep them shape-synced with current point count.
        if getattr(self, "adp_enabled", False) and isinstance(self.adp_tex_ema, torch.Tensor) and self.adp_tex_ema.numel() > 0:
            N = int(valid_points_mask.shape[0])

            def _pad_trunc(t: torch.Tensor, fill=0.0, dtype=None):
                if (not isinstance(t, torch.Tensor)) or (t.numel() == 0):
                    if dtype is None:
                        return torch.full((N, 1), float(fill), device=self.get_xyz.device)
                    return torch.full((N, 1), fill, device=self.get_xyz.device, dtype=dtype)

                if t.shape[0] == N:
                    return t
                if t.shape[0] > N:
                    return t[:N]

                pad_n = N - t.shape[0]
                pad = torch.full((pad_n, t.shape[1]), float(fill), device=t.device, dtype=t.dtype)
                return torch.cat((t, pad), dim=0)

            # First align lengths (densify can change point count between ADP updates)
            self.adp_tex_ema = _pad_trunc(self.adp_tex_ema, fill=0.0)
            self.adp_spark_ema = _pad_trunc(self.adp_spark_ema, fill=0.0)
            self.adp_vis_count = _pad_trunc(self.adp_vis_count, fill=0.0)
            self.adp_bad_streak = _pad_trunc(self.adp_bad_streak, fill=0, dtype=torch.int32)

            # Then apply pruning mask
            self.adp_tex_ema = self.adp_tex_ema[valid_points_mask]
            self.adp_spark_ema = self.adp_spark_ema[valid_points_mask]
            self.adp_vis_count = self.adp_vis_count[valid_points_mask]
            self.adp_bad_streak = self.adp_bad_streak[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, adp_new=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))

        # ADP: append inherited stats for the new points
        if adp_new is not None and isinstance(self.adp_tex_ema, torch.Tensor) and self.adp_tex_ema.numel() > 0:
            self.adp_tex_ema = torch.cat((self.adp_tex_ema, adp_new["tex_ema"]), dim=0)
            self.adp_spark_ema = torch.cat((self.adp_spark_ema, adp_new["spark_ema"]), dim=0)
            self.adp_vis_count = torch.cat((self.adp_vis_count, adp_new["vis_count"]), dim=0)
            self.adp_bad_streak = torch.cat((self.adp_bad_streak, adp_new["bad_streak"]), dim=0)

        # Reset densification accumulators (official behavior)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, extra_mask=None):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        if extra_mask is not None:
            # Robust: allow size mismatch (e.g., caller mask computed before a densify op)
            if extra_mask.numel() != selected_pts_mask.numel():
                if extra_mask.numel() < selected_pts_mask.numel():
                    pad = torch.zeros((selected_pts_mask.numel() - extra_mask.numel(),), device=extra_mask.device, dtype=extra_mask.dtype)
                    extra_mask_ = torch.cat((extra_mask, pad), dim=0)
                else:
                    extra_mask_ = extra_mask[: selected_pts_mask.numel()]
            else:
                extra_mask_ = extra_mask
            selected_pts_mask = torch.logical_and(selected_pts_mask, extra_mask_)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if selected_pts_mask.sum() == 0:
            return

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        adp_new = None
        if self.adp_enabled and isinstance(self.adp_tex_ema, torch.Tensor) and self.adp_tex_ema.numel() > 0:
            adp_new = {
                "tex_ema": self.adp_tex_ema[selected_pts_mask].repeat(N,1).detach(),
                "spark_ema": self.adp_spark_ema[selected_pts_mask].repeat(N,1).detach(),
                "vis_count": self.adp_vis_count[selected_pts_mask].repeat(N,1).detach(),
                "bad_streak": torch.zeros((new_xyz.shape[0], 1), device=new_xyz.device, dtype=self.adp_bad_streak.dtype),
            }

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, adp_new=adp_new)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    def densify_and_clone(self, grads, grad_threshold, scene_extent, extra_mask=None):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        if extra_mask is not None:
            # Robust: extra_mask may be computed on pre-densify point count (clone adds points before split)
            if extra_mask.numel() != selected_pts_mask.numel():
                if extra_mask.numel() < selected_pts_mask.numel():
                    pad = torch.zeros((selected_pts_mask.numel() - extra_mask.numel(),), device=extra_mask.device, dtype=extra_mask.dtype)
                    extra_mask_ = torch.cat((extra_mask, pad), dim=0)
                else:
                    extra_mask_ = extra_mask[: selected_pts_mask.numel()]
            else:
                extra_mask_ = extra_mask
            selected_pts_mask = torch.logical_and(selected_pts_mask, extra_mask_)

        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        if selected_pts_mask.sum() == 0:
            return

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        adp_new = None
        if self.adp_enabled and isinstance(self.adp_tex_ema, torch.Tensor) and self.adp_tex_ema.numel() > 0:
            # Inherit texture/sparkle/visibility stats; reset streak for new points.
            adp_new = {
                "tex_ema": self.adp_tex_ema[selected_pts_mask].detach(),
                "spark_ema": self.adp_spark_ema[selected_pts_mask].detach(),
                "vis_count": self.adp_vis_count[selected_pts_mask].detach(),
                "bad_streak": torch.zeros((new_xyz.shape[0], 1), device=new_xyz.device, dtype=self.adp_bad_streak.dtype),
            }

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, adp_new=adp_new)
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii, iteration=None, densify_from_iter=None, densify_until_iter=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        # ADP cycle stats cache
        N_before = int(self.get_xyz.shape[0])

        self.tmp_radii = radii

        # ADP gating / adaptive thresholds
        extra_mask = None
        grad_thr = max_grad
        adp_prune_mask = None
        tex_thr_gate = None
        tex_thr_prune = None
        spark_thr = None

        if self.adp_enabled and isinstance(self.adp_tex_ema, torch.Tensor) and self.adp_tex_ema.numel() == self.get_xyz.shape[0]:
            # Progress in [0,1] for schedules
            if iteration is None or densify_from_iter is None or densify_until_iter is None or densify_until_iter <= densify_from_iter:
                p = 1.0
            else:
                p = float(max(0.0, min(1.0, (iteration - densify_from_iter) / float(max(1, densify_until_iter - densify_from_iter)))))

            def lerp(a, b, t):
                return a + (b - a) * t

            q_tex_gate = lerp(self.adp_q_tex_gate_init, self.adp_q_tex_gate_final, p)
            q_tex_prune = lerp(self.adp_q_tex_prune_init, self.adp_q_tex_prune_final, p)
            q_grad = lerp(self.adp_q_grad_init, self.adp_q_grad_final, p)

            tex = self.adp_tex_ema.squeeze(-1)
            g = grads.squeeze(-1)
            spark = self.adp_spark_ema.squeeze(-1)

            def safe_quantile(x, q, default):
                if x.numel() < 32:
                    return default
                try:
                    return torch.quantile(x, q)
                except Exception:
                    # kthvalue is 1-indexed
                    k = int(q * (x.numel() - 1)) + 1
                    return x.kthvalue(k).values

            # Densify gate: require sufficiently high texture support
            tex_thr_gate = safe_quantile(tex, q_tex_gate, default=torch.tensor(0.0, device=tex.device, dtype=tex.dtype))
            extra_mask = tex >= tex_thr_gate

            # Adaptive grad threshold computed on gated set (fallback to max_grad)
            if extra_mask.any():
                grad_thr_adp = safe_quantile(g[extra_mask], q_grad, default=torch.tensor(float(max_grad), device=g.device, dtype=g.dtype))
                grad_thr_tensor = torch.maximum(torch.tensor(float(max_grad), device=g.device, dtype=g.dtype), grad_thr_adp)
                grad_thr = grad_thr_tensor

            # Persistent prune: low texture + high sparkle (a.k.a. "sky sparkles")
            tex_thr_prune = safe_quantile(tex, q_tex_prune, default=torch.tensor(0.0, device=tex.device, dtype=tex.dtype))
            spark_thr = safe_quantile(spark, self.adp_q_spark, default=torch.tensor(float("inf"), device=spark.device, dtype=spark.dtype))

            bad_now = torch.logical_and(tex < tex_thr_prune, spark > spark_thr)

            # Update streak counters
            bs = self.adp_bad_streak.squeeze(-1)
            bs[bad_now] = bs[bad_now] + 1
            if self.adp_bad_streak_decay > 0:
                bs[~bad_now] = torch.clamp(bs[~bad_now] - self.adp_bad_streak_decay, min=0)
            else:
                bs[~bad_now] = bs[~bad_now] * 0
            self.adp_bad_streak = bs.unsqueeze(-1)
            adp_prune_mask = bs >= int(self.adp_bad_streak_kill)

        # Densification (with optional gate)
        # ADPP: optional densify region constraint (set by adpp_pre_densify_hook).
        adpp_mask = getattr(self, "_adpp_densify_extra_mask", None)
        if adpp_mask is not None:
            try:
                adpp_mask = adpp_mask.to(device=self.get_xyz.device).bool().view(-1)
                if extra_mask is None:
                    extra_mask = adpp_mask
                else:
                    extra_mask = extra_mask.bool().view(-1)
                    if adpp_mask.numel() != extra_mask.numel():
                        # Align lengths conservatively (allow densify on any new tail points).
                        if adpp_mask.numel() < extra_mask.numel():
                            pad = torch.ones((extra_mask.numel() - adpp_mask.numel(),), device=extra_mask.device, dtype=torch.bool)
                            adpp_mask = torch.cat((adpp_mask, pad), dim=0)
                        else:
                            adpp_mask = adpp_mask[: extra_mask.numel()]
                    extra_mask = torch.logical_and(extra_mask, adpp_mask)
            except Exception:
                # If anything goes wrong, ignore ADPP mask for safety.
                pass
        self.densify_and_clone(grads, grad_thr, extent, extra_mask=extra_mask)
        self.densify_and_split(grads, grad_thr, extent, extra_mask=extra_mask)

        N_after_densify = int(self.get_xyz.shape[0])
        # Approx densify candidate counts (based on pre-densify grads)
        try:
            g1 = grads.squeeze(-1)
            thr_val = float(grad_thr.item()) if hasattr(grad_thr, 'item') else float(grad_thr)
            densify_candidates = int((g1 >= thr_val).sum().item())
            if extra_mask is not None and extra_mask.numel() == g1.numel():
                densify_candidates_gated = int(((g1 >= thr_val) & extra_mask).sum().item())
            else:
                densify_candidates_gated = densify_candidates
        except Exception:
            densify_candidates = None
            densify_candidates_gated = None


        # Official pruning rules
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        prune_official_count = int(prune_mask.sum().item())
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        # ADP persistent prune
        if adp_prune_mask is not None:
            # Robust: densification may have changed point count (new points should not be pruned by ADP streak)
            if adp_prune_mask.numel() != prune_mask.numel():
                if adp_prune_mask.numel() < prune_mask.numel():
                    pad = torch.zeros((prune_mask.numel() - adp_prune_mask.numel(),), device=adp_prune_mask.device, dtype=adp_prune_mask.dtype)
                    adp_prune_mask = torch.cat((adp_prune_mask, pad), dim=0)
                else:
                    adp_prune_mask = adp_prune_mask[: prune_mask.numel()]
            prune_mask = torch.logical_or(prune_mask, adp_prune_mask)

        prune_total_count = int(prune_mask.sum().item())
        self.prune_points(prune_mask)
        N_after_prune = int(self.get_xyz.shape[0])

        # Update ADP cycle stats cache (used by get_adp_log_dict and CSV logging).
        try:
            gate_total = int(N_before)
            gate_pass = int(extra_mask.sum().item()) if (extra_mask is not None and hasattr(extra_mask, 'sum')) else int(N_before)
            gate_ratio = float(gate_pass) / float(gate_total) if gate_total > 0 else 0.0
        except Exception:
            gate_total, gate_pass, gate_ratio = None, None, None

        try:
            adp_prune_count = int(adp_prune_mask.sum().item()) if (adp_prune_mask is not None and hasattr(adp_prune_mask, 'sum')) else None
        except Exception:
            adp_prune_count = None

        def _tfloat(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    return float(x.detach().item()) if x.numel()==1 else float(x.detach().mean().item())
            except Exception:
                pass
            try:
                return float(x)
            except Exception:
                return None

        self._adp_last_cycle_stats = {
            'N_before': N_before,
            'N_after_densify': N_after_densify,
            'N_after_prune': N_after_prune,
            'gate_total': gate_total,
            'gate_pass': gate_pass,
            'gate_ratio': gate_ratio,
            'densify_candidates': densify_candidates,
            'densify_candidates_gated': densify_candidates_gated,
            'tex_thr_gate': _tfloat(tex_thr_gate),
            'tex_thr_prune': _tfloat(tex_thr_prune),
            'spark_thr': _tfloat(spark_thr),
            'grad_thr': _tfloat(grad_thr),
            'prune_official_count': prune_official_count,
            'adp_prune_count': adp_prune_count,
            'prune_total_count': prune_total_count,
        }
        self.tmp_radii = None
        # Clear ADPP densify mask to avoid accidental reuse on later cycles.
        self._adpp_densify_extra_mask = None

        torch.cuda.empty_cache()
    def add_densification_stats(self, viewspace_point_tensor, update_filter, radii=None, tex_values=None):
        # update_filter is typically a boolean mask for visible points
        if update_filter is None or update_filter.numel() == 0:
            return
        idx = update_filter.squeeze(-1) if update_filter.dim() > 1 else update_filter

        self.xyz_gradient_accum[idx] += torch.norm(viewspace_point_tensor.grad[idx, :2], dim=-1, keepdim=True)
        self.denom[idx] += 1

        if not self.adp_enabled:
            return

        # Texture support (EMA in [0,1])
        if tex_values is not None:
            tv = tex_values.detach().view(-1, 1).clamp(0.0, 1.0)
            self.adp_tex_ema[idx] = (1.0 - self.adp_tex_beta) * self.adp_tex_ema[idx] + self.adp_tex_beta * tv
            self.adp_vis_count[idx] += 1

        # Sparkle risk (opacity / (radii + eps))
        if radii is not None:
            r = radii[idx].detach().view(-1, 1)
            op = self.get_opacity[idx].detach()
            spark = op / (r + self.adp_eps)
            self.adp_spark_ema[idx] = (1.0 - self.adp_spark_beta) * self.adp_spark_ema[idx] + self.adp_spark_beta * spark


    # ---------------------------------------------------------------------
    # ADPP hooks (called from train.py, runtime-only, safe defaults)
    # ---------------------------------------------------------------------
    def adpp_set_action(self, action):
        """Store the latest ADPP action dict (runtime only)."""
        if action is None:
            self._adpp_action = None
        else:
            try:
                self._adpp_action = dict(action)
            except Exception:
                self._adpp_action = action

    def _adpp_inside_aabb_mask(self, aabb_min, aabb_max):
        """Return (N,) bool mask for points inside the axis-aligned bounding box."""
        xyz = self.get_xyz
        if xyz.numel() == 0:
            return None
        try:
            amin = aabb_min.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
            amax = aabb_max.to(device=xyz.device, dtype=xyz.dtype).view(1, 3)
        except Exception:
            amin = torch.tensor(aabb_min, device=xyz.device, dtype=xyz.dtype).view(1, 3)
            amax = torch.tensor(aabb_max, device=xyz.device, dtype=xyz.dtype).view(1, 3)
        inside = torch.logical_and(xyz >= amin, xyz <= amax).all(dim=1)
        return inside

    def adpp_pre_densify_hook(self, aabb_min, aabb_max, action=None):
        """
        Called right before densify_and_prune (under torch.no_grad()).
        - Sets an internal mask to prevent densification outside the trusted AABB.
        - Optionally decays opacity outside the AABB to suppress far-away fog/floaters.
        """
        if action is None:
            action = getattr(self, "_adpp_action", None)
        if action is None:
            return
        inside = self._adpp_inside_aabb_mask(aabb_min, aabb_max)
        if inside is None:
            return
        # Constrain densification to trusted region (prevents 'fog growth' in empty space).
        self._adpp_densify_extra_mask = inside.detach()

        # Anti-fog: decay opacity outside trusted region (logit-space, no gradients).
        try:
            strength = float(action.get("anti_fog_strength", 0.0) or 0.0)
        except Exception:
            strength = 0.0
        if strength <= 0.0:
            return
        outside = ~inside
        if not outside.any():
            return
        # A modest logit shift works well and is stable across scenes.
        # strength in [0,1] -> delta in [0, 1.5]
        delta = max(0.0, min(1.5, 1.5 * strength))
        try:
            self._opacity.data[outside] = self._opacity.data[outside] - delta
        except Exception:
            pass

    def adpp_post_densify_hook(self, aabb_min, aabb_max, action=None):
        """
        Called right after densify_and_prune (under torch.no_grad()).
        - Optional hard prune of a small fraction of high-opacity points outside AABB.
        - Decays opacity outside AABB once more (helps converge to clean geometry).
        """
        if action is None:
            action = getattr(self, "_adpp_action", None)
        if action is None:
            return
        inside = self._adpp_inside_aabb_mask(aabb_min, aabb_max)
        if inside is None:
            return
        outside = ~inside
        if not outside.any():
            return

        try:
            strength = float(action.get("anti_fog_strength", 0.0) or 0.0)
        except Exception:
            strength = 0.0
        try:
            prune_frac = float(action.get("fog_prune_frac", 0.0) or 0.0)
        except Exception:
            prune_frac = 0.0
        prune_frac = max(0.0, min(0.5, prune_frac))

        # Optional hard prune: remove top-k most opaque points outside the trusted region.
        if prune_frac > 0.0:
            try:
                outside_idx = outside.nonzero(as_tuple=False).view(-1)
                op = self.get_opacity[outside].detach().view(-1)
                k = int(op.numel() * prune_frac)
                if k > 0 and outside_idx.numel() == op.numel():
                    topk = torch.topk(op, k, largest=True).indices
                    prune_mask = torch.zeros((self.get_xyz.shape[0],), device=self.get_xyz.device, dtype=torch.bool)
                    prune_mask[outside_idx[topk]] = True
                    self.prune_points(prune_mask)
                    self._adpp_last_stats["fog_pruned"] = int(k)
            except Exception:
                pass

        # Secondary opacity decay (even if prune_frac==0).
        if strength > 0.0:
            delta = max(0.0, min(1.0, 1.0 * strength))
            try:
                outside2 = self._adpp_inside_aabb_mask(aabb_min, aabb_max)
                if outside2 is not None:
                    outside2 = ~outside2
                    if outside2.any():
                        self._opacity.data[outside2] = self._opacity.data[outside2] - delta
            except Exception:
                pass

        # Best-effort: keep ADPP mask cleared.
        self._adpp_densify_extra_mask = None
