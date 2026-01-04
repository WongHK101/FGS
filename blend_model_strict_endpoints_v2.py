import argparse
import csv
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Core metrics deps
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Image IO helpers
# ---------------------------

def _imread_rgb(path: Path) -> np.ndarray:
    """Read image as float32 RGB in [0,1]."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    pil = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    return (np.asarray(pil).astype(np.float32) / 255.0)


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    """Luma Y from RGB in [0,1]."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


# ---------------------------
# Thermal scalar extraction
# ---------------------------

# hue extraction adapted from standard RGB->HSV formulas

def rgb_to_hsv_np(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Hue
    h = np.zeros_like(cmax)
    mask = delta > 1e-12

    # where cmax==r
    idx = mask & (cmax == r)
    h[idx] = ((g[idx] - b[idx]) / delta[idx]) % 6.0
    # where cmax==g
    idx = mask & (cmax == g)
    h[idx] = ((b[idx] - r[idx]) / delta[idx]) + 2.0
    # where cmax==b
    idx = mask & (cmax == b)
    h[idx] = ((r[idx] - g[idx]) / delta[idx]) + 4.0

    h = (h / 6.0) % 1.0

    # Saturation
    s = np.zeros_like(cmax)
    nonzero = cmax > 1e-12
    s[nonzero] = delta[nonzero] / cmax[nonzero]

    v = cmax
    return h, s, v


def thermal_scalar_from_rgb(img: np.ndarray, mode: str) -> np.ndarray:
    """Extract a thermal scalar map S from an RGB pseudo-colored image.

    Supported:
      - "luma"  : Y
      - "hue"   : hue only
      - "hue_y" : hue * Y   (helps disambiguate low-saturation hue)
      - "sat"   : saturation
      - "value" : HSV value

    Returns float32 array in [0,1] (not necessarily perceptually linear).
    """
    mode = mode.lower()
    if mode == "luma":
        return rgb_to_y(img).astype(np.float32)
    h, s, v = rgb_to_hsv_np(img)
    if mode == "hue":
        return h.astype(np.float32)
    if mode == "sat":
        return s.astype(np.float32)
    if mode == "value":
        return v.astype(np.float32)
    if mode == "hue_y":
        return (h * rgb_to_y(img)).astype(np.float32)
    raise ValueError(f"Unknown thermal_scalar mode: {mode}")


def align_scalar(pred: np.ndarray, ref: np.ndarray, mode: str) -> np.ndarray:
    """Align scalar map pred to ref (for fair similarity metrics)."""
    mode = mode.lower()
    if mode == "none":
        return pred
    p = pred.reshape(-1).astype(np.float64)
    r = ref.reshape(-1).astype(np.float64)

    if mode == "linear":
        # least squares: a*p + b ~ r
        A = np.vstack([p, np.ones_like(p)]).T
        try:
            a, b = np.linalg.lstsq(A, r, rcond=None)[0]
        except Exception:
            a, b = 1.0, 0.0
        out = (a * pred + b).astype(np.float32)
        return out

    if mode == "rank":
        # monotonic mapping using ranks
        order = np.argsort(p)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(p))
        # map ranks to ref quantiles
        r_sorted = np.sort(r)
        out = r_sorted[ranks].reshape(pred.shape).astype(np.float32)
        return out

    raise ValueError(f"Unknown align mode: {mode}")


# ---------------------------
# Structure metrics
# ---------------------------

def edge_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of Sobel edge magnitude between two 2D arrays."""
    # simple sobel via finite diffs (fast, dependency-light)
    ax = np.diff(a, axis=1, append=a[:, -1:])
    ay = np.diff(a, axis=0, append=a[-1:, :])
    bx = np.diff(b, axis=1, append=b[:, -1:])
    by = np.diff(b, axis=0, append=b[-1:, :])
    amag = np.sqrt(ax * ax + ay * ay)
    bmag = np.sqrt(bx * bx + by * by)

    va = amag.reshape(-1)
    vb = bmag.reshape(-1)
    va = va - va.mean()
    vb = vb - vb.mean()
    denom = (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)
    return float(np.dot(va, vb) / denom)


# ---------------------------
# 2D fusion metrics (no GT)
# ---------------------------

def entropy_8bit(x: np.ndarray) -> float:
    """Shannon entropy on 8-bit quantized grayscale."""
    x8 = np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
    hist = np.bincount(x8.reshape(-1), minlength=256).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def mutual_information_8bit(a: np.ndarray, b: np.ndarray) -> float:
    """Mutual information between two grayscale maps (8-bit quantized)."""
    a8 = np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8).reshape(-1)
    b8 = np.clip(np.round(b * 255.0), 0, 255).astype(np.uint8).reshape(-1)
    joint = np.bincount(a8 * 256 + b8, minlength=256 * 256).astype(np.float64)
    joint = joint.reshape(256, 256)
    pxy = joint / (joint.sum() + 1e-12)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    nz = pxy > 0
    mi = (pxy[nz] * (np.log2(pxy[nz] + 1e-12) - np.log2(px[:, None][nz] + 1e-12) - np.log2(py[None, :][nz] + 1e-12))).sum()
    return float(mi)


def spatial_frequency(x: np.ndarray) -> float:
    """Spatial Frequency (SF) metric for grayscale image."""
    # Row frequency
    rf = np.sqrt(np.mean(np.diff(x, axis=0) ** 2))
    # Column frequency
    cf = np.sqrt(np.mean(np.diff(x, axis=1) ** 2))
    return float(np.sqrt(rf * rf + cf * cf))


def avg_gradient(x: np.ndarray) -> float:
    gx = np.diff(x, axis=1, append=x[:, -1:])
    gy = np.diff(x, axis=0, append=x[-1:, :])
    g = np.sqrt(gx * gx + gy * gy)
    return float(np.mean(g))


def qabf_xydeas(A: np.ndarray, B: np.ndarray, F: np.ndarray) -> float:
    """Q_AB/F edge-based fusion quality metric (Xydeas & Petrovic).

    Implementation follows common open-source reimplementations.
    Inputs are grayscale float arrays in [0,1].
    """
    # Sobel-ish gradients
    def grad(img):
        gx = np.diff(img, axis=1, append=img[:, -1:])
        gy = np.diff(img, axis=0, append=img[-1:, :])
        g = np.sqrt(gx * gx + gy * gy) + 1e-12
        ang = np.arctan2(gy, gx)  # [-pi, pi]
        return g, ang

    gA, aA = grad(A)
    gB, aB = grad(B)
    gF, aF = grad(F)

    # Relative gradient strength preservation
    GA = np.minimum(gF / gA, gA / gF)
    GB = np.minimum(gF / gB, gB / gF)
    GA = np.clip(GA, 0, 1)
    GB = np.clip(GB, 0, 1)

    # Orientation preservation (normalized to [0,1])
    def ang_pres(aS):
        d = np.abs(aF - aS)
        d = np.minimum(d, 2 * np.pi - d)
        # map [0, pi] -> [1, 0]
        return np.clip(1.0 - d / (np.pi / 2.0), 0, 1)

    AA = ang_pres(aA)
    AB = ang_pres(aB)

    # Logistic mapping (params used in many repos)
    Kg, Dg = -15.0, 0.5
    Ka, Da = -22.0, 0.8

    QgA = 1.0 / (1.0 + np.exp(Kg * (GA - Dg)))
    QgB = 1.0 / (1.0 + np.exp(Kg * (GB - Dg)))
    QaA = 1.0 / (1.0 + np.exp(Ka * (AA - Da)))
    QaB = 1.0 / (1.0 + np.exp(Ka * (AB - Da)))

    QA = QgA * QaA
    QB = QgB * QaB

    wA = gA
    wB = gB
    num = (QA * wA + QB * wB).sum()
    den = (wA + wB).sum() + 1e-12
    return float(num / den)


def vifp(ref: np.ndarray, dist: np.ndarray, sigma_nsq: float = 2.0) -> float:
    """VIFp (pixel-domain) reference metric for grayscale images.

    Adapted from common implementations (multi-scale, Gaussian filtering).
    Inputs should be float in [0,1].
    """
    ref = ref.astype(np.float64)
    dist = dist.astype(np.float64)

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if scale > 1:
            ref = gaussian_filter(ref, sd)
            dist = gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = gaussian_filter(ref, sd)
        mu2 = gaussian_filter(dist, sd)
        sigma1_sq = gaussian_filter(ref * ref, sd) - mu1 * mu1
        sigma2_sq = gaussian_filter(dist * dist, sd) - mu2 * mu2
        sigma12 = gaussian_filter(ref * dist, sd) - mu1 * mu2

        sigma1_sq = np.maximum(0.0, sigma1_sq)
        sigma2_sq = np.maximum(0.0, sigma2_sq)

        g = sigma12 / (sigma1_sq + 1e-12)
        sv_sq = sigma2_sq - g * sigma12

        g = np.where(sigma1_sq < 1e-12, 0.0, g)
        sv_sq = np.where(sigma1_sq < 1e-12, sigma2_sq, sv_sq)

        g = np.where(sigma2_sq < 1e-12, 0.0, g)
        sv_sq = np.where(sigma2_sq < 1e-12, 0.0, sv_sq)

        sv_sq = np.maximum(1e-12, sv_sq)

        num += np.sum(np.log10(1.0 + (g * g) * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq))

    return float(num / (den + 1e-12))


# ---------------------------
# LPIPS (optional)
# ---------------------------

class _LPIPSWrapper:
    def __init__(self, net: str = "alex", device: str = "cuda"):
        self.available = False
        self.device = device
        self.model = None
        try:
            import torch
            import lpips as lpips_pkg
            self.torch = torch
            self.model = lpips_pkg.LPIPS(net=net)
            self.model = self.model.to(device)
            self.model.eval()
            self.available = True
        except Exception:
            self.available = False

    def _to_tensor(self, img01: np.ndarray):
        t = self.torch.from_numpy(img01.transpose(2, 0, 1)).float()[None, ...]
        # [0,1] -> [-1,1]
        t = t * 2.0 - 1.0
        return t.to(self.device)

    def __call__(self, a01: np.ndarray, b01: np.ndarray) -> float:
        if not self.available:
            return float("nan")
        with self.torch.no_grad():
            ta = self._to_tensor(a01)
            tb = self._to_tensor(b01)
            v = self.model(ta, tb)
            return float(v.item())


# ---------------------------
# Parsing + grouping
# ---------------------------

def parse_methods(items: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad --methods entry (expect Name=PATH): {it}")
        name, p = it.split("=", 1)
        out.append((name.strip(), Path(p.strip())))
    return out


_alpha_pat = re.compile(r"-a([0-9]+(?:\.[0-9]+)?)")


def parse_group_alpha(name: str) -> Tuple[str, Optional[float]]:
    """Return (group, alpha). Group is prefix before '-a...'."""
    m = _alpha_pat.search(name)
    if not m:
        return name, None
    alpha = float(m.group(1))
    group = name[: m.start()].rstrip("-_ ")
    if not group:
        group = name
    return group, alpha


def _safe_float(x: Optional[float]) -> float:
    return float("nan") if x is None else float(x)


# ---------------------------
# Montage
# ---------------------------

def make_montage_pages(
    rgb_ref: Dict[str, Path],
    t_ref: Dict[str, Path],
    methods: List[Tuple[str, Dict[str, Path]]],
    out_dir: Path,
    frames: List[str],
    max_cols: int = 8,
    thumb_w: int = 320,
):
    """Create multi-page montage.

    Layout per page (for each frame row):
      [RGB_ref] [methods ...] [T_ref]
    We paginate columns when too many methods.
    """
    # Pre-load and resize into thumbnails
    def thumb(path: Path) -> Image.Image:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        scale = thumb_w / float(w)
        nh = int(round(h * scale))
        return im.resize((thumb_w, nh), Image.BILINEAR)

    method_names = [n for n, _ in methods]
    # paginate methods columns
    for page_start in range(0, len(methods), max_cols):
        chunk = methods[page_start : page_start + max_cols]
        cols = 2 + len(chunk)  # RGB + methods + T
        rows = len(frames)

        thumbs = []
        row_heights = []
        for fn in frames:
            row_imgs = [thumb(rgb_ref[fn])]
            for _, m_map in chunk:
                row_imgs.append(thumb(m_map[fn]))
            row_imgs.append(thumb(t_ref[fn]))
            thumbs.append(row_imgs)
            row_heights.append(max(im.size[1] for im in row_imgs))

        W = cols * thumb_w
        H = sum(row_heights)
        canvas = Image.new("RGB", (W, H), (0, 0, 0))

        y = 0
        for r in range(rows):
            x = 0
            for c in range(cols):
                im = thumbs[r][c]
                canvas.paste(im, (x, y))
                x += thumb_w
            y += row_heights[r]

        page_idx = page_start // max_cols
        out_path = out_dir / f"montage_page_{page_idx:02d}.png"
        canvas.save(out_path)


# ---------------------------
# Main evaluation
# ---------------------------

@dataclass
class MetricRow:
    method: str
    frame: str

    ssim_y: float
    edgecorr_y: float

    spearman_s: float
    ssim_s: float

    # NVS vs GT (optional)
    psnr_rgb: float
    ssim_rgb: float
    lpips_rgb: float

    psnr_s: float
    ssim_s_gt: float
    lpips_s: float

    # Fusion metrics (optional)
    en_y: float
    mi_total: float
    qabf: float
    vifp_total: float
    sf_y: float
    ag_y: float


def compute_metrics_for_frame(
    rgb_ref_img: np.ndarray,
    t_ref_img: np.ndarray,
    meth_img: np.ndarray,
    thermal_scalar: str,
    thermal_align: str,
    do_fusion_metrics: bool,
    do_gt_metrics: bool,
    rgb_gt_img: Optional[np.ndarray],
    t_gt_img: Optional[np.ndarray],
    lpips_fn: Optional[_LPIPSWrapper],
) -> Dict[str, float]:
    """Compute all metrics for one frame."""

    # --- structure vs RGB ref (render reference) ---
    y_ref = rgb_to_y(rgb_ref_img)
    y_m = rgb_to_y(meth_img)

    ssim_y = float(ssim(y_ref, y_m, data_range=1.0))
    edgecorr_y = edge_correlation(y_ref, y_m)

    # --- thermal vs T ref (render reference) ---
    s_ref = thermal_scalar_from_rgb(t_ref_img, thermal_scalar)
    s_m = thermal_scalar_from_rgb(meth_img, thermal_scalar)
    s_m_al = align_scalar(s_m, s_ref, thermal_align)

    spearman_s = float(spearmanr(s_ref.reshape(-1), s_m_al.reshape(-1)).correlation)
    if math.isnan(spearman_s):
        spearman_s = 0.0

    ssim_s = float(ssim(s_ref, s_m_al, data_range=float(np.max(s_ref) - np.min(s_ref) + 1e-6)))

    # --- NVS vs GT (optional) ---
    psnr_rgb = ssim_rgb = lpips_rgb = float("nan")
    psnr_s = ssim_s_gt = lpips_s = float("nan")

    if do_gt_metrics and rgb_gt_img is not None:
        # Full RGB against RGB GT
        psnr_rgb = float(psnr(rgb_gt_img, meth_img, data_range=1.0))
        ssim_rgb = float(ssim(rgb_gt_img, meth_img, channel_axis=2, data_range=1.0))
        if lpips_fn is not None:
            lpips_rgb = lpips_fn(rgb_gt_img, meth_img)

    if do_gt_metrics and t_gt_img is not None:
        # Scalar against Thermal GT scalar
        s_gt = thermal_scalar_from_rgb(t_gt_img, thermal_scalar)
        s_m_gt_al = align_scalar(s_m, s_gt, thermal_align)

        # Normalize scalar to [0,1] for PSNR/SSIM stability
        def norm01(x):
            lo, hi = float(np.min(x)), float(np.max(x))
            if hi - lo < 1e-12:
                return np.zeros_like(x, dtype=np.float32)
            return ((x - lo) / (hi - lo)).astype(np.float32)

        s_gt_n = norm01(s_gt)
        s_m_n = norm01(s_m_gt_al)

        psnr_s = float(psnr(s_gt_n, s_m_n, data_range=1.0))
        ssim_s_gt = float(ssim(s_gt_n, s_m_n, data_range=1.0))

        if lpips_fn is not None:
            # LPIPS expects RGB; replicate scalar
            a3 = np.repeat(s_gt_n[..., None], 3, axis=2)
            b3 = np.repeat(s_m_n[..., None], 3, axis=2)
            lpips_s = lpips_fn(a3, b3)

    # --- Fusion metrics (optional) ---
    en_y = mi_total = qabf = vifp_total = sf_y = ag_y = float("nan")
    if do_fusion_metrics:
        # treat sources as RGB luma + thermal scalar; fused as fused luma
        # normalize scalar sources to [0,1] for fusion metrics
        def norm01(x):
            lo, hi = float(np.min(x)), float(np.max(x))
            if hi - lo < 1e-12:
                return np.zeros_like(x, dtype=np.float32)
            return ((x - lo) / (hi - lo)).astype(np.float32)

        A = np.clip(y_ref, 0, 1)
        B = norm01(s_ref)
        F = np.clip(y_m, 0, 1)

        en_y = entropy_8bit(F)
        mi_total = mutual_information_8bit(F, A) + mutual_information_8bit(F, B)
        qabf = qabf_xydeas(A, B, F)

        # VIFp can be slow; still useful for fusion papers
        try:
            vifp_total = 0.5 * (vifp(A, F) + vifp(B, F))
        except Exception:
            vifp_total = float("nan")

        sf_y = spatial_frequency(F)
        ag_y = avg_gradient(F)

    return dict(
        ssim_y=ssim_y,
        edgecorr_y=edgecorr_y,
        spearman_s=spearman_s,
        ssim_s=ssim_s,
        psnr_rgb=psnr_rgb,
        ssim_rgb=ssim_rgb,
        lpips_rgb=lpips_rgb,
        psnr_s=psnr_s,
        ssim_s_gt=ssim_s_gt,
        lpips_s=lpips_s,
        en_y=en_y,
        mi_total=mi_total,
        qabf=qabf,
        vifp_total=vifp_total,
        sf_y=sf_y,
        ag_y=ag_y,
    )


def mean_ignore_nan(vals: List[float]) -> float:
    arr = np.array(vals, dtype=np.float64)
    if np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def plot_alpha_trajectories(
    per_method_means: Dict[str, Dict[str, float]],
    out_path: Path,
    x_key: str = "SSIM_Y_mean",
    y_key: str = "Spearman_S_mean",
):
    # group -> list of (alpha, x, y, name)
    groups: Dict[str, List[Tuple[float, float, float, str]]] = {}
    baselines: List[Tuple[float, float, str]] = []

    for name, m in per_method_means.items():
        group, alpha = parse_group_alpha(name)
        x = m.get(x_key, float("nan"))
        y = m.get(y_key, float("nan"))
        if alpha is None:
            baselines.append((x, y, name))
        else:
            groups.setdefault(group, []).append((alpha, x, y, name))

    plt.figure(figsize=(11, 8))
    plt.title("Structure vs Thermal (alpha-trajectories)")
    plt.xlabel(x_key.replace("_mean", ""))
    plt.ylabel(y_key.replace("_mean", ""))
    plt.grid(True, alpha=0.3)

    # trajectories (only groups with >=2 alpha points)
    for group, pts in groups.items():
        pts = sorted(pts, key=lambda t: t[0])
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=2, label=group)
        for a, x, y, _ in pts:
            plt.text(x, y, f"{a:g}", fontsize=9)

    # baselines as scatter only (no connecting line)
    if baselines:
        bx = [b[0] for b in baselines]
        by = [b[1] for b in baselines]
        plt.scatter(bx, by, marker="x", s=80, c="k", label="baselines")

    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_metric_vs_alpha(
    per_method_means: Dict[str, Dict[str, float]],
    out_path: Path,
    metric_key: str,
    title: str,
    y_label: str,
):
    # group -> list of (alpha, value)
    groups: Dict[str, List[Tuple[float, float]]] = {}
    for name, m in per_method_means.items():
        group, alpha = parse_group_alpha(name)
        if alpha is None:
            continue
        v = m.get(metric_key, float("nan"))
        groups.setdefault(group, []).append((alpha, v))

    plt.figure(figsize=(11, 6))
    plt.title(title)
    plt.xlabel("alpha")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)

    for group, pts in groups.items():
        pts = sorted(pts, key=lambda t: t[0])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        plt.plot(xs, ys, marker="o", linewidth=2, label=group)
        for a, v in pts:
            if not (math.isnan(v) or math.isinf(v)):
                plt.text(a, v, f"{a:g}", fontsize=8)

    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_render", required=True, help="RGB reference renders dir")
    ap.add_argument("--t_render", required=True, help="Thermal reference renders dir")
    ap.add_argument(
        "--methods",
        required=True,
        nargs="+",
        help="One or more entries Name=DIR (DIR is renders folder)",
    )
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--thermal_scalar", default="hue_y", choices=["luma", "hue", "hue_y", "sat", "value"])
    ap.add_argument("--thermal_align", default="linear", choices=["none", "linear", "rank"])

    ap.add_argument("--sample_frames", type=int, default=0, help="0 means all frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_side", type=int, default=960, help="Resize max side for speed (0=no resize)")

    # GT/NVS
    ap.add_argument("--rgb_gt", default="", help="Optional RGB GT dir. If empty, try ../gt next to rgb_render")
    ap.add_argument("--t_gt", default="", help="Optional Thermal GT dir. If empty, try ../gt next to t_render")
    ap.add_argument("--with_gt_metrics", action="store_true", help="Enable PSNR/SSIM/LPIPS vs GT")

    # Fusion
    ap.add_argument("--with_fusion_metrics", action="store_true", help="Enable EN/MI/QABF/VIFp/SF/AG")

    # LPIPS
    ap.add_argument("--lpips_net", default="alex", choices=["alex", "vgg", "squeeze"])
    ap.add_argument("--lpips_device", default="cuda", help="cuda or cpu")

    # Montage
    ap.add_argument("--montage_cols", type=int, default=8)
    ap.add_argument("--montage_thumb_w", type=int, default=320)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_render = Path(args.rgb_render)
    t_render = Path(args.t_render)
    methods = parse_methods(args.methods)

    # Resolve GT folders
    rgb_gt = Path(args.rgb_gt) if args.rgb_gt else (rgb_render.parent / "gt")
    t_gt = Path(args.t_gt) if args.t_gt else (t_render.parent / "gt")
    have_rgb_gt = rgb_gt.exists() and any(rgb_gt.glob("*.png"))
    have_t_gt = t_gt.exists() and any(t_gt.glob("*.png"))
    do_gt = bool(args.with_gt_metrics) and (have_rgb_gt or have_t_gt)

    if args.with_gt_metrics and not do_gt:
        print("[WARN] --with_gt_metrics requested but GT folders not found. GT metrics will be NaN.")

    # LPIPS setup
    lpips_fn = None
    if do_gt:
        lp = _LPIPSWrapper(net=args.lpips_net, device=args.lpips_device)
        if lp.available:
            lpips_fn = lp
        else:
            print("[WARN] LPIPS not available. Install 'lpips' (pip install lpips) and ensure torch works.")

    # Collect filenames
    def list_imgs(d: Path) -> Dict[str, Path]:
        exts = ["*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG"]
        files: List[Path] = []
        for e in exts:
            files.extend(d.glob(e))
        # unique by name
        m = {}
        for p in files:
            m[p.name] = p
        return m

    rgb_map = list_imgs(rgb_render)
    t_map = list_imgs(t_render)

    if not rgb_map:
        raise FileNotFoundError(f"No images in rgb_render: {rgb_render}")
    if not t_map:
        raise FileNotFoundError(f"No images in t_render: {t_render}")

    # method maps
    method_maps: List[Tuple[str, Dict[str, Path]]] = []
    for name, p in methods:
        mp = list_imgs(p)
        if not mp:
            raise FileNotFoundError(f"No images in method renders: {name} -> {p}")
        method_maps.append((name, mp))

    # common filenames intersection
    common = set(rgb_map.keys()) & set(t_map.keys())
    for _, mp in method_maps:
        common &= set(mp.keys())

    common = sorted(common)
    if not common:
        raise RuntimeError("No common filenames across rgb/t/method renders")

    # sample frames
    frames = common
    if args.sample_frames and args.sample_frames > 0 and args.sample_frames < len(common):
        rng = random.Random(args.seed)
        frames = rng.sample(common, args.sample_frames)
        frames = sorted(frames)

    print(f"[INFO] frames used: {len(frames)}")

    # Prepare montage inputs (paths only)
    rgb_ref_paths = {fn: rgb_map[fn] for fn in frames}
    t_ref_paths = {fn: t_map[fn] for fn in frames}
    method_paths = [(name, {fn: mp[fn] for fn in frames}) for name, mp in method_maps]

    # Compute metrics
    rows: List[MetricRow] = []

    for fn in frames:
        rgb_img = _resize_max_side(_imread_rgb(rgb_map[fn]), args.max_side)
        t_img = _resize_max_side(_imread_rgb(t_map[fn]), args.max_side)

        rgb_gt_img = None
        t_gt_img = None
        if do_gt and have_rgb_gt and (rgb_gt / fn).exists():
            rgb_gt_img = _resize_max_side(_imread_rgb(rgb_gt / fn), args.max_side)
        if do_gt and have_t_gt and (t_gt / fn).exists():
            t_gt_img = _resize_max_side(_imread_rgb(t_gt / fn), args.max_side)

        for name, mp in method_maps:
            m_img = _resize_max_side(_imread_rgb(mp[fn]), args.max_side)

            mets = compute_metrics_for_frame(
                rgb_ref_img=rgb_img,
                t_ref_img=t_img,
                meth_img=m_img,
                thermal_scalar=args.thermal_scalar,
                thermal_align=args.thermal_align,
                do_fusion_metrics=args.with_fusion_metrics,
                do_gt_metrics=do_gt,
                rgb_gt_img=rgb_gt_img,
                t_gt_img=t_gt_img,
                lpips_fn=lpips_fn,
            )

            rows.append(
                MetricRow(
                    method=name,
                    frame=fn,
                    ssim_y=mets["ssim_y"],
                    edgecorr_y=mets["edgecorr_y"],
                    spearman_s=mets["spearman_s"],
                    ssim_s=mets["ssim_s"],
                    psnr_rgb=mets["psnr_rgb"],
                    ssim_rgb=mets["ssim_rgb"],
                    lpips_rgb=mets["lpips_rgb"],
                    psnr_s=mets["psnr_s"],
                    ssim_s_gt=mets["ssim_s_gt"],
                    lpips_s=mets["lpips_s"],
                    en_y=mets["en_y"],
                    mi_total=mets["mi_total"],
                    qabf=mets["qabf"],
                    vifp_total=mets["vifp_total"],
                    sf_y=mets["sf_y"],
                    ag_y=mets["ag_y"],
                )
            )

    # Write per_frame.csv
    per_frame_path = out_dir / "per_frame.csv"
    with per_frame_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "frame",
                "SSIM_Y",
                "EdgeCorr_Y",
                "Spearman_S",
                "SSIM_S",
                "PSNR_RGB",
                "SSIM_RGB",
                "LPIPS_RGB",
                "PSNR_S",
                "SSIM_S_GT",
                "LPIPS_S",
                "EN_Y",
                "MI_total",
                "QABF",
                "VIFp_total",
                "SF_Y",
                "AG_Y",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.method,
                    r.frame,
                    r.ssim_y,
                    r.edgecorr_y,
                    r.spearman_s,
                    r.ssim_s,
                    r.psnr_rgb,
                    r.ssim_rgb,
                    r.lpips_rgb,
                    r.psnr_s,
                    r.ssim_s_gt,
                    r.lpips_s,
                    r.en_y,
                    r.mi_total,
                    r.qabf,
                    r.vifp_total,
                    r.sf_y,
                    r.ag_y,
                ]
            )

    # Summaries
    by_method: Dict[str, List[MetricRow]] = {}
    for r in rows:
        by_method.setdefault(r.method, []).append(r)

    per_method_means: Dict[str, Dict[str, float]] = {}
    for m, rs in by_method.items():
        per_method_means[m] = {
            "SSIM_Y_mean": mean_ignore_nan([x.ssim_y for x in rs]),
            "EdgeCorr_Y_mean": mean_ignore_nan([x.edgecorr_y for x in rs]),
            "Spearman_S_mean": mean_ignore_nan([x.spearman_s for x in rs]),
            "SSIM_S_mean": mean_ignore_nan([x.ssim_s for x in rs]),
            "PSNR_RGB_mean": mean_ignore_nan([x.psnr_rgb for x in rs]),
            "SSIM_RGB_mean": mean_ignore_nan([x.ssim_rgb for x in rs]),
            "LPIPS_RGB_mean": mean_ignore_nan([x.lpips_rgb for x in rs]),
            "PSNR_S_mean": mean_ignore_nan([x.psnr_s for x in rs]),
            "SSIM_S_GT_mean": mean_ignore_nan([x.ssim_s_gt for x in rs]),
            "LPIPS_S_mean": mean_ignore_nan([x.lpips_s for x in rs]),
            "EN_Y_mean": mean_ignore_nan([x.en_y for x in rs]),
            "MI_total_mean": mean_ignore_nan([x.mi_total for x in rs]),
            "QABF_mean": mean_ignore_nan([x.qabf for x in rs]),
            "VIFp_total_mean": mean_ignore_nan([x.vifp_total for x in rs]),
            "SF_Y_mean": mean_ignore_nan([x.sf_y for x in rs]),
            "AG_Y_mean": mean_ignore_nan([x.ag_y for x in rs]),
        }

    summary_path = out_dir / "summary.csv"
    header = ["method"] + list(next(iter(per_method_means.values())).keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in sorted(per_method_means.keys()):
            w.writerow([m] + [per_method_means[m].get(k, float("nan")) for k in header[1:]])

    # Plots
    plot_alpha_trajectories(per_method_means, out_dir / "pareto_lines.png")

    # Metric-vs-alpha key plots
    plot_metric_vs_alpha(per_method_means, out_dir / "curve_ssimY.png", "SSIM_Y_mean", "SSIM_Y vs alpha", "SSIM_Y")
    plot_metric_vs_alpha(per_method_means, out_dir / "curve_edgecorrY.png", "EdgeCorr_Y_mean", "EdgeCorr_Y vs alpha", "EdgeCorr_Y")
    plot_metric_vs_alpha(per_method_means, out_dir / "curve_spearmanS.png", "Spearman_S_mean", "Spearman_S vs alpha", "Spearman_S")
    plot_metric_vs_alpha(per_method_means, out_dir / "curve_ssimS.png", "SSIM_S_mean", "SSIM_S vs alpha", "SSIM_S")

    if do_gt:
        plot_metric_vs_alpha(per_method_means, out_dir / "curve_psnrRGB.png", "PSNR_RGB_mean", "PSNR (RGB GT) vs alpha", "PSNR")
        plot_metric_vs_alpha(per_method_means, out_dir / "curve_lpipsRGB.png", "LPIPS_RGB_mean", "LPIPS (RGB GT) vs alpha", "LPIPS")
        plot_metric_vs_alpha(per_method_means, out_dir / "curve_psnrS.png", "PSNR_S_mean", "PSNR (Thermal scalar GT) vs alpha", "PSNR")
        plot_metric_vs_alpha(per_method_means, out_dir / "curve_lpipsS.png", "LPIPS_S_mean", "LPIPS (Thermal scalar GT) vs alpha", "LPIPS")

    if args.with_fusion_metrics:
        plot_metric_vs_alpha(per_method_means, out_dir / "curve_MI.png", "MI_total_mean", "MI_total vs alpha", "MI")
        plot_metric_vs_alpha(per_method_means, out_dir / "curve_QABF.png", "QABF_mean", "Q_AB/F vs alpha", "Q_AB/F")
        plot_metric_vs_alpha(per_method_means, out_dir / "curve_EN.png", "EN_Y_mean", "Entropy (Y) vs alpha", "Entropy")

    # Montage pages
    make_montage_pages(
        rgb_ref_paths,
        t_ref_paths,
        method_paths,
        out_dir,
        frames,
        max_cols=args.montage_cols,
        thumb_w=args.montage_thumb_w,
    )

    print(f"[OK] wrote: {summary_path}")
    print(f"[OK] wrote: {per_frame_path}")
    print(f"[OK] wrote: {out_dir / 'pareto_lines.png'}")
    print(f"[OK] wrote: {out_dir / 'curve_ssimY.png'} (and other curve_*.png)")
    print(f"[OK] wrote: montage_page_*.png")

    # Console summary
    print("\nSummary (mean over frames):")
    keys_print = [
        "SSIM_Y_mean",
        "EdgeCorr_Y_mean",
        "Spearman_S_mean",
        "SSIM_S_mean",
    ]
    if do_gt:
        keys_print += ["PSNR_RGB_mean", "SSIM_RGB_mean", "LPIPS_RGB_mean", "PSNR_S_mean", "SSIM_S_GT_mean", "LPIPS_S_mean"]
    if args.with_fusion_metrics:
        keys_print += ["MI_total_mean", "QABF_mean", "EN_Y_mean", "VIFp_total_mean"]

    for m in sorted(per_method_means.keys()):
        print(f"== {m} ==")
        for k in keys_print:
            print(f"  {k:16s}: {per_method_means[m].get(k, float('nan')):.4f}")


if __name__ == "__main__":
    main()