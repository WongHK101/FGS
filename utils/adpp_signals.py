# -*- coding: utf-8 -*-
"""
utils/adpp_signals.py

Signal computation helpers for ADP++ (rule-based closed-loop controller).

**Compatibility note**
- train_ADPP_COMPAT_v2.py expects:
    from utils.adpp_signals import edge_score_grad_ncc, fog_score_outside_aabb
  and calls:
    edge_score_grad_ncc(...).item()
    fog_score_outside_aabb(...).item()
  Therefore these functions MUST return a 0-dim torch.Tensor (scalar), not Python float.

Provided signals:
- edge_score_grad_ncc: gradient-structure similarity score in ~[0,1] (higher is better)
- fog_score_outside_aabb: "fog mass" outside a trusted AABB in [0,1] (higher is worse)

Design goals:
- Stable across datasets (avoid brittle hard thresholds).
- Torch-only implementation (no OpenCV/scipy).
- Works on CPU or CUDA. Overhead is low when called sparsely (e.g., every 50 iters).
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F

__all__ = [
    "sobel_grad_mag",
    "edge_score_grad_ncc",
    "edge_score_f1",
    "compute_aabb",
    "fog_score_outside_aabb",
    "fog_score_low_visibility",
]


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _to_4d(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor to shape [B=1, C=1, H, W]."""
    if x.dim() == 2:
        x = x[None, None, ...]
    elif x.dim() == 3:
        # [1,H,W] or [B,H,W]
        if x.shape[0] == 1:
            x = x[None, ...]          # [1,1,H,W]
        else:
            x = x[:, None, ...]       # [B,1,H,W]
    elif x.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported image tensor shape: {tuple(x.shape)}")
    return x


def _to_mask_4d(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Convert mask to [B,1,H,W] float mask (0/1) on like's device/dtype."""
    m = _to_4d(mask)
    m = m.to(device=like.device, dtype=like.dtype)
    return (m > 0.5).to(dtype=like.dtype)


def _normalize01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize to [0,1] per-batch for stability (min-max)."""
    x4 = _to_4d(x)
    B = x4.shape[0]
    flat = x4.view(B, -1)
    x_min = flat.min(dim=1).values.view(B, 1, 1, 1)
    x_max = flat.max(dim=1).values.view(B, 1, 1, 1)
    return (x4 - x_min) / (x_max - x_min + eps)


# Cache Sobel kernels per (dtype, device)
_SOBEL_CACHE: Dict[Tuple[torch.dtype, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _sobel_kernels(dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (dtype, str(device))
    if key in _SOBEL_CACHE:
        return _SOBEL_CACHE[key]
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[1,  2,  1],
                       [0,  0,  0],
                       [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    _SOBEL_CACHE[key] = (kx, ky)
    return kx, ky


# -----------------------------------------------------------------------------
# Edge score (Action A driver)
# -----------------------------------------------------------------------------

def sobel_grad_mag(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute Sobel gradient magnitude. Returns [B,1,H,W]."""
    x = _to_4d(img)
    kx, ky = _sobel_kernels(x.dtype, x.device)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + eps)


def edge_score_grad_ncc(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gradient-NCC edge similarity score in ~[0,1]. Returns torch scalar."""
    gp = _normalize01(sobel_grad_mag(pred, eps=eps), eps=eps)
    gg = _normalize01(sobel_grad_mag(gt, eps=eps), eps=eps)

    if valid_mask is not None:
        m = _to_mask_4d(valid_mask, like=gp)
        denom = m.sum().clamp_min(1.0)
        mp = (gp * m).sum() / denom
        mg = (gg * m).sum() / denom
        vp = ((gp - mp) * m).pow(2).sum() / denom
        vg = ((gg - mg) * m).pow(2).sum() / denom
        cov = (((gp - mp) * (gg - mg)) * m).sum() / denom
    else:
        mp = gp.mean()
        mg = gg.mean()
        vp = (gp - mp).pow(2).mean()
        vg = (gg - mg).pow(2).mean()
        cov = ((gp - mp) * (gg - mg)).mean()

    ncc = cov / (torch.sqrt(vp * vg) + eps)
    ncc = ncc.clamp(-1.0, 1.0)
    score01 = (ncc + 1.0) * 0.5
    return score01


def edge_score_f1(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    gt_quantile: float = 0.90,
    pred_quantile: float = 0.90,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Edge F1 score (quantile threshold on gradient magnitudes). Returns torch scalar."""
    gp = sobel_grad_mag(pred, eps=eps).flatten()
    gg = sobel_grad_mag(gt, eps=eps).flatten()

    if valid_mask is not None:
        # Build boolean mask on same device
        m = _to_mask_4d(valid_mask, like=gp).flatten().to(dtype=torch.bool)
        gp = gp[m]
        gg = gg[m]

    if gp.numel() < 16 or gg.numel() < 16:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)

    thr_g = torch.quantile(gg, gt_quantile).clamp_min(eps)
    thr_p = torch.quantile(gp, pred_quantile).clamp_min(eps)

    eg = gg >= thr_g
    ep = gp >= thr_p

    tp = (eg & ep).sum().to(dtype=pred.dtype)
    fp = (~eg & ep).sum().to(dtype=pred.dtype)
    fn = (eg & ~ep).sum().to(dtype=pred.dtype)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = (2.0 * precision * recall) / (precision + recall + eps)
    return f1.clamp(0.0, 1.0)


# -----------------------------------------------------------------------------
# Fog score (Action B driver)
# -----------------------------------------------------------------------------

def compute_aabb(points_xyz: torch.Tensor, margin: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute axis-aligned bounding box for points_xyz [N,3]."""
    if points_xyz.numel() == 0:
        raise ValueError("compute_aabb(): empty points")
    aabb_min = points_xyz.min(dim=0).values
    aabb_max = points_xyz.max(dim=0).values
    if margin > 0:
        m = torch.tensor([margin, margin, margin], device=points_xyz.device, dtype=points_xyz.dtype)
        aabb_min = aabb_min - m
        aabb_max = aabb_max + m
    return aabb_min, aabb_max


def fog_score_outside_aabb(
    xyz: torch.Tensor,
    opacity: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
    scales: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fog score = mass(outside trust AABB) / mass(total). Returns torch scalar in [0,1]."""
    if opacity.dim() == 2 and opacity.shape[1] == 1:
        op = opacity[:, 0]
    else:
        op = opacity
    op = op.clamp(0.0, 1.0)

    inside = (xyz[:, 0] >= aabb_min[0]) & (xyz[:, 0] <= aabb_max[0]) & \
             (xyz[:, 1] >= aabb_min[1]) & (xyz[:, 1] <= aabb_max[1]) & \
             (xyz[:, 2] >= aabb_min[2]) & (xyz[:, 2] <= aabb_max[2])

    w = op
    if scales is not None:
        if scales.dim() == 2 and scales.shape[1] == 3:
            s = scales.abs().mean(dim=1)
        else:
            s = scales.abs()
        s = s / (s.mean().clamp_min(eps))
        w = w * s

    total = w.sum().clamp_min(eps)
    outside = w[~inside].sum()
    return (outside / total).clamp(0.0, 1.0)


def fog_score_low_visibility(
    opacity: torch.Tensor,
    visibility: torch.Tensor,
    vis_thr: float = 0.02,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Alternative fog proxy when per-Gaussian visibility is available. Returns torch scalar."""
    if opacity.dim() == 2 and opacity.shape[1] == 1:
        op = opacity[:, 0]
    else:
        op = opacity
    op = op.clamp(0.0, 1.0)

    vis = visibility.clamp_min(0.0)
    mask = vis < vis_thr

    total = op.sum().clamp_min(eps)
    bad = op[mask].sum()
    return (bad / total).clamp(0.0, 1.0)
