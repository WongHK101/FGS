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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from typing import Optional
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

# -----------------------------------------------------------------------------
# ADP++ extensions: edge-aware / gradient-based losses (paper-grade, stable)
# -----------------------------------------------------------------------------
# These utilities are designed to be:
#  - Safe: no effect on baseline unless explicitly used by train.py
#  - Stable: robust normalization + optional masking
#  - Dependency-free: pure PyTorch (no OpenCV)
#
# Typical thermal usage (single-channel):
#   pred_t = rendered_image[:, :1, ...] or rendered_image.mean(1, keepdim=True)
#   gt_t   = gt_image[:, :1, ...]        (already normalized to [0,1])
#   L_edge = edge_aware_grad_loss(pred_t, gt_t, edge_quantile=0.90, edge_boost=4.0)
#
# If you have a valid_mask (0/1) for pixels (e.g. alpha>0):
#   L_edge = edge_aware_grad_loss(pred_t, gt_t, valid_mask=mask)
# -----------------------------------------------------------------------------

def _to_nchw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an image tensor to NCHW.
    Accepts:
      - HxW
      - 1xHxW / CxHxW
      - NxHxW
      - NxCxHxW
    Returns: NxCxHxW
    """
    if x.dim() == 2:
        return x[None, None, ...]
    if x.dim() == 3:
        # [C,H,W] or [N,H,W]
        if x.shape[0] in (1, 3):
            return x[None, ...]
        return x[:, None, ...]
    if x.dim() == 4:
        return x
    raise ValueError(f"Unsupported tensor shape: {tuple(x.shape)}")


def _to_mask_nchw(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    Convert mask to Nx1xHxW float mask in {0,1}, broadcastable to 'like'.
    """
    m = _to_nchw(mask)
    if m.shape[1] != 1:
        # if user passes NxCxHxW, reduce to 1 channel
        m = m[:, :1, ...]
    m = m.to(dtype=like.dtype, device=like.device)
    return (m > 0.5).to(dtype=like.dtype)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    Charbonnier penalty: sqrt(x^2 + eps^2).
    More robust than L1 for noisy thermal.
    """
    return torch.sqrt(x * x + (eps * eps))


def masked_mean(x: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Mean of x under mask. Mask is Nx1xHxW or broadcastable.
    """
    m = valid_mask
    denom = m.sum().clamp_min(eps)
    return (x * m).sum() / denom


def masked_l1_loss(network_output: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Masked L1 mean. Safe for any C.
    """
    x = _to_nchw(network_output)
    y = _to_nchw(gt).to(dtype=x.dtype, device=x.device)
    m = _to_mask_nchw(valid_mask, like=x)
    # broadcast mask to channels
    m = m.expand(x.shape[0], 1, x.shape[2], x.shape[3])
    return masked_mean(torch.abs(x - y), m, eps=eps)


def masked_l2_loss(network_output: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Masked L2 mean. Safe for any C.
    """
    x = _to_nchw(network_output)
    y = _to_nchw(gt).to(dtype=x.dtype, device=x.device)
    m = _to_mask_nchw(valid_mask, like=x)
    return masked_mean((x - y) ** 2, m, eps=eps)


def _sobel_kernels(dtype: torch.dtype, device: torch.device):
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[1,  2,  1],
                       [0,  0,  0],
                       [-1, -2, -1]], dtype=dtype, device=device).view(1, 1, 3, 3)
    return kx, ky


def image_gradients_sobel(img: torch.Tensor, eps: float = 1e-6):
    """
    Sobel gradients for each channel separately.
    Args:
        img: NxCxHxW (or broadcastable; will be converted)
    Returns:
        gx, gy: NxCxHxW
    """
    x = _to_nchw(img)
    N, C, H, W = x.shape
    kx, ky = _sobel_kernels(x.dtype, x.device)
    # apply per-channel with groups
    kx = kx.expand(C, 1, 3, 3)
    ky = ky.expand(C, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1, groups=C)
    gy = F.conv2d(x, ky, padding=1, groups=C)
    return gx, gy


def grad_magnitude(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gx, gy = image_gradients_sobel(img, eps=eps)
    return torch.sqrt(gx * gx + gy * gy + eps)


def _normalize_per_image(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize tensor per-image and per-channel to [0,1] using min-max.
    Works for NxCxHxW.
    """
    t = _to_nchw(x)
    N, C, _, _ = t.shape
    flat = t.view(N, C, -1)
    t_min = flat.min(dim=-1).values.view(N, C, 1, 1)
    t_max = flat.max(dim=-1).values.view(N, C, 1, 1)
    return (t - t_min) / (t_max - t_min + eps)


def gradient_consistency_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor = None,
    robust_eps: float = 1e-3,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize gradient mismatch between pred and gt (Sobel).
    If normalize=True, compare gradients after per-image normalization to reduce exposure-scale sensitivity.
    """
    p = _to_nchw(pred)
    g = _to_nchw(gt).to(dtype=p.dtype, device=p.device)

    if normalize:
        p = _normalize_per_image(p, eps=eps)
        g = _normalize_per_image(g, eps=eps)

    pgx, pgy = image_gradients_sobel(p, eps=eps)
    ggx, ggy = image_gradients_sobel(g, eps=eps)

    diff = charbonnier(pgx - ggx, eps=robust_eps) + charbonnier(pgy - ggy, eps=robust_eps)

    if valid_mask is None:
        return diff.mean()
    m = _to_mask_nchw(valid_mask, like=p)
    return masked_mean(diff, m, eps=eps)


def laplacian_map(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Laplacian response per-channel.
    """
    x = _to_nchw(img)
    N, C, H, W = x.shape
    k = torch.tensor([[0,  1, 0],
                      [1, -4, 1],
                      [0,  1, 0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    k = k.expand(C, 1, 3, 3)
    lap = F.conv2d(x, k, padding=1, groups=C)
    return lap


def laplacian_consistency_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor = None,
    robust_eps: float = 1e-3,
    normalize: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalize Laplacian mismatch (captures edge crispness, reduces blur).
    """
    p = _to_nchw(pred)
    g = _to_nchw(gt).to(dtype=p.dtype, device=p.device)

    if normalize:
        p = _normalize_per_image(p, eps=eps)
        g = _normalize_per_image(g, eps=eps)

    lp = laplacian_map(p, eps=eps)
    lg = laplacian_map(g, eps=eps)

    diff = charbonnier(lp - lg, eps=robust_eps)
    if valid_mask is None:
        return diff.mean()
    m = _to_mask_nchw(valid_mask, like=p)
    return masked_mean(diff, m, eps=eps)



def edge_aware_grad_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor = None,
    robust_eps: float = 1e-3,
    normalize: bool = True,
    edge_quantile: float = 0.85,
    edge_boost: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Edge-aware gradient consistency loss.

    This loss is designed to harden edges by giving higher weight to pixels
    where the *GT* has strong gradients.

    Args:
        pred, gt: (H,W), (C,H,W), or (N,C,H,W).
        mask: optional alpha/valid mask (broadcastable to N,1,H,W).
        robust_eps: Charbonnier epsilon.
        normalize: if True, per-image normalize before computing gradients.
        edge_quantile: quantile used to estimate a robust edge scale.
        edge_boost: edge weight gain (0 disables edge weighting).
        eps: numeric stability epsilon.
    """
    p = _to_nchw(pred)
    g = _to_nchw(gt).to(dtype=p.dtype, device=p.device)

    if normalize:
        p = _normalize_per_image(p, eps=eps)
        g = _normalize_per_image(g, eps=eps)

    pgx, pgy = image_gradients_sobel(p, eps=eps)
    ggx, ggy = image_gradients_sobel(g, eps=eps)

    # Edge strength from GT gradients
    gmag = torch.sqrt(ggx * ggx + ggy * ggy + eps).mean(dim=1, keepdim=True)  # (N,1,H,W)

    # Robust per-image threshold
    if mask is None:
        thr = torch.quantile(gmag.view(gmag.shape[0], -1), q=edge_quantile, dim=1, keepdim=True).view(-1, 1, 1, 1)
    else:
        m = _to_mask_nchw(mask, like=p)
        valid = gmag[m > 0.5]
        if valid.numel() == 0:
            thr = torch.quantile(gmag.view(gmag.shape[0], -1), q=edge_quantile, dim=1, keepdim=True).view(-1, 1, 1, 1)
        else:
            thr = torch.quantile(valid, q=edge_quantile).view(1, 1, 1, 1)
    thr = torch.clamp(thr, min=eps)

    edge_w = 1.0 + float(edge_boost) * torch.clamp(gmag / thr, 0.0, 1.0)

    diff = charbonnier(pgx - ggx, eps=robust_eps) + charbonnier(pgy - ggy, eps=robust_eps)
    diff = diff.mean(dim=1, keepdim=True) * edge_w  # (N,1,H,W)

    if mask is None:
        return diff.mean()
    m = _to_mask_nchw(mask, like=p)
    return masked_mean(diff, m, eps=eps)


def edge_aware_laplacian_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor = None,
    robust_eps: float = 1e-3,
    normalize: bool = True,
    edge_quantile: float = 0.85,
    edge_boost: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Edge-aware Laplacian consistency loss.

    A second-order edge loss that encourages crisper boundaries.
    Edge weights are derived from the magnitude of the GT Laplacian response.
    """
    p = _to_nchw(pred)
    g = _to_nchw(gt).to(dtype=p.dtype, device=p.device)

    if normalize:
        p = _normalize_per_image(p, eps=eps)
        g = _normalize_per_image(g, eps=eps)

    pl = laplacian_map(p, eps=eps)
    gl = laplacian_map(g, eps=eps)

    gmag = torch.abs(gl).mean(dim=1, keepdim=True)  # (N,1,H,W)

    if mask is None:
        thr = torch.quantile(gmag.view(gmag.shape[0], -1), q=edge_quantile, dim=1, keepdim=True).view(-1, 1, 1, 1)
    else:
        m = _to_mask_nchw(mask, like=p)
        valid = gmag[m > 0.5]
        if valid.numel() == 0:
            thr = torch.quantile(gmag.view(gmag.shape[0], -1), q=edge_quantile, dim=1, keepdim=True).view(-1, 1, 1, 1)
        else:
            thr = torch.quantile(valid, q=edge_quantile).view(1, 1, 1, 1)
    thr = torch.clamp(thr, min=eps)

    edge_w = 1.0 + float(edge_boost) * torch.clamp(gmag / thr, 0.0, 1.0)

    diff = charbonnier(pl - gl, eps=robust_eps).mean(dim=1, keepdim=True) * edge_w

    if mask is None:
        return diff.mean()
    m = _to_mask_nchw(mask, like=p)
    return masked_mean(diff, m, eps=eps)

