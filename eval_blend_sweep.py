# eval_blend_sweep.py
# ------------------------------------------------------------
# Evaluate a sweep folder:
#   sweep_root/
#     strategy_A/
#       0/   (alpha folder)
#       0.1/
#       ...
#     strategy_B/
#       ...
#
# For each alpha folder, we auto-find:
#   alpha_dir/test/ours_*/renders
# and pick the max iteration.
#
# Metrics:
# - RGB fidelity vs RGB reference: PSNR / SSIM / LPIPS (if available)
# - Structure vs RGB reference: SSIM on luma-Y, EdgeCorr on Y edges
# - Thermal fidelity vs T reference: Spearman on thermal scalar, SSIM on thermal scalar
#
# Plots:
# - lines_psnr.png / lines_ssim.png / lines_lpips.png
# - lines_spearmanS.png / lines_ssimS.png
# - pareto.png: SSIM_RGB vs Spearman_S with alpha annotations
# - montage_page_*.png: per-strategy rows, columns are [RGB_ref | alphas window | T_ref]
#
# Place this script in 3DGS root (same level as render.py / metrics.py).
# ------------------------------------------------------------

from __future__ import annotations

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

# Matplotlib headless safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

# Optional: use 3DGS-native torch metrics if available
HAS_TORCH = False
HAS_GS_METRICS = False
HAS_LPIPS_PYTORCH = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None  # type: ignore

# Try import 3DGS psnr/ssim helpers (preferred when script is in 3DGS root)
# If not present, fall back to numpy implementations.
try:
    # Ensure 3DGS root is in sys.path
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))

    # 3DGS repo commonly has these:
    # - utils/image_utils.py: psnr
    # - utils/loss_utils.py: ssim
    from utils.image_utils import psnr as gs_psnr  # type: ignore
    from utils.loss_utils import ssim as gs_ssim  # type: ignore

    HAS_GS_METRICS = True
except Exception:
    gs_psnr = None  # type: ignore
    gs_ssim = None  # type: ignore
    HAS_GS_METRICS = False

# LPIPS: you have lpipsPyTorch, not lpips
try:
    # Common API: from lpipsPyTorch import lpips
    from lpipsPyTorch import lpips as lpips_fn  # type: ignore
    HAS_LPIPS_PYTORCH = True
except Exception:
    lpips_fn = None  # type: ignore
    HAS_LPIPS_PYTORCH = False


# -----------------------------
# Utilities
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS


def list_images(d: Path) -> List[str]:
    if not d.exists():
        return []
    files = [p.name for p in d.iterdir() if p.is_file() and is_image(p)]
    files.sort(key=natural_key)
    return files


def load_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr


def maybe_resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    im = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
    im = im.resize((nw, nh), resample=Image.BILINEAR)
    return np.asarray(im).astype(np.float32) / 255.0


def rgb_to_luma_y(rgb: np.ndarray) -> np.ndarray:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def sobel_like_edge(y: np.ndarray) -> np.ndarray:
    # Pure numpy edge (no cv2 dependency)
    y = y.astype(np.float32)
    # Simple gradients
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, 1:-1] = (y[:, 2:] - y[:, :-2]) * 0.5
    gy[1:-1, :] = (y[2:, :] - y[:-2, :]) * 0.5
    e = np.sqrt(gx * gx + gy * gy)
    # normalize robustly
    p = np.percentile(e, 99.0)
    if p > 1e-12:
        e = np.clip(e / p, 0.0, 1.0)
    return e


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).mean()) * np.sqrt((b * b).mean()) + 1e-12)
    return float((a * b).mean() / denom)


def rankdata(x: np.ndarray) -> np.ndarray:
    # average ranks for ties
    x = x.reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    # tie handling
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a.astype(np.float64))
    rb = rankdata(b.astype(np.float64))
    return pearson_corr(ra, rb)


def ssim_simple(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = ((x - mu_x) ** 2).mean()
    var_y = ((y - mu_y) ** 2).mean()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    return float(num / (den + 1e-12))


def ssim_gray(x: np.ndarray, y: np.ndarray) -> float:
    # Try skimage, else simple
    try:
        from skimage.metrics import structural_similarity as ssim_sk  # type: ignore
        return float(ssim_sk(x.astype(np.float64), y.astype(np.float64), data_range=1.0))
    except Exception:
        return ssim_simple(x, y)


def torch_from_rgb01(img: np.ndarray) -> "torch.Tensor":
    # img: H,W,3 in [0,1]
    t = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # 3,H,W
    t = t.unsqueeze(0).float()  # 1,3,H,W
    return t


def psnr_rgb(a: np.ndarray, b: np.ndarray) -> float:
    # Prefer 3DGS psnr if available; else numpy
    if HAS_GS_METRICS and HAS_TORCH:
        ta = torch_from_rgb01(a)
        tb = torch_from_rgb01(b)
        v = gs_psnr(ta, tb)  # type: ignore
        return float(v.item() if hasattr(v, "item") else v)
    # numpy fallback
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 100.0
    return float(10.0 * math.log10(1.0 / mse))


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    # Prefer 3DGS ssim if available; else average channel SSIM
    if HAS_GS_METRICS and HAS_TORCH:
        ta = torch_from_rgb01(a)
        tb = torch_from_rgb01(b)
        v = gs_ssim(ta, tb)  # type: ignore
        return float(v.item() if hasattr(v, "item") else v)
    # fallback: mean of per-channel SSIM (cheap approximation)
    return float(np.mean([ssim_gray(a[..., c], b[..., c]) for c in range(3)]))


def lpips_rgb(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not (HAS_LPIPS_PYTORCH and HAS_TORCH):
        return None
    try:
        ta = torch_from_rgb01(a)
        tb = torch_from_rgb01(b)
        # lpipsPyTorch typically expects [0,1] tensors (it normalizes internally),
        # but some variants expect [-1,1]. We'll try [0,1] first; if fails, use [-1,1].
        v = lpips_fn(ta, tb)
        return float(v.item() if hasattr(v, "item") else v)
    except Exception:
        try:
            ta = torch_from_rgb01(a) * 2.0 - 1.0
            tb = torch_from_rgb01(b) * 2.0 - 1.0
            v = lpips_fn(ta, tb)
            return float(v.item() if hasattr(v, "item") else v)
        except Exception:
            return None


# -----------------------------
# Thermal scalar
# -----------------------------
def rgb_to_hsv(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-12
    idx = mask & (mx == r)
    h[idx] = ((g[idx] - b[idx]) / (diff[idx] + 1e-12)) % 6.0
    idx = mask & (mx == g)
    h[idx] = ((b[idx] - r[idx]) / (diff[idx] + 1e-12)) + 2.0
    idx = mask & (mx == b)
    h[idx] = ((r[idx] - g[idx]) / (diff[idx] + 1e-12)) + 4.0
    h = (h / 6.0) % 1.0

    s = np.zeros_like(mx)
    s[mx > 1e-12] = diff[mx > 1e-12] / (mx[mx > 1e-12] + 1e-12)
    v = mx
    return h, s, v


def thermal_scalar_from_rgb(rgb: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode in ("y", "gray", "luma"):
        return rgb_to_luma_y(rgb)
    h, s, v = rgb_to_hsv(rgb)
    if mode == "hue":
        return h
    if mode == "hue_y":
        y = rgb_to_luma_y(rgb)
        out = h.copy()
        out[s < 0.10] = y[s < 0.10]
        return out
    if mode == "sat":
        return s
    if mode == "val":
        return v
    raise ValueError(f"Unknown thermal_scalar mode: {mode}")


def align_scalar(s_method: np.ndarray, s_ref: np.ndarray, mode: str) -> np.ndarray:
    mode = mode.lower()
    if mode in ("none", "off"):
        return s_method.astype(np.float32)
    if mode == "linear":
        sm = s_method.astype(np.float64).reshape(-1)
        sr = s_ref.astype(np.float64).reshape(-1)
        A = np.stack([sm, np.ones_like(sm)], axis=1)
        x, *_ = np.linalg.lstsq(A, sr, rcond=None)
        a, b = float(x[0]), float(x[1])
        out = a * s_method + b
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    raise ValueError(f"Unknown thermal_align mode: {mode}")


# -----------------------------
# Sweep discovery
# -----------------------------
RE_OURS = re.compile(r"ours_(\d+)$")


def find_renders_dir(alpha_dir: Path) -> Optional[Path]:
    """
    Prefer: alpha_dir/test/ours_*/renders (max iter).
    Fallback: any 'renders' directory under alpha_dir (shallow-ish).
    """
    test_dir = alpha_dir / "test"
    candidates: List[Tuple[int, Path]] = []
    if test_dir.exists():
        for ours in test_dir.iterdir():
            if not ours.is_dir():
                continue
            m = RE_OURS.match(ours.name)
            if not m:
                continue
            it = int(m.group(1))
            r = ours / "renders"
            if r.exists() and r.is_dir():
                candidates.append((it, r))

    if candidates:
        candidates.sort(key=lambda t: t[0])
        return candidates[-1][1]

    # fallback: search for renders directories (limited depth)
    # To avoid scanning huge trees, cap the number of hits.
    hits = []
    try:
        for p in alpha_dir.rglob("renders"):
            if p.is_dir():
                hits.append(p)
                if len(hits) >= 20:
                    break
    except Exception:
        pass

    if not hits:
        return None

    # pick the one that contains "ours_" in its parents if possible
    def score(p: Path) -> Tuple[int, int]:
        s1 = 1 if any(RE_OURS.match(x.name) for x in p.parents) else 0
        # deeper is usually more specific
        return (s1, len(p.parts))

    hits.sort(key=score)
    return hits[-1]


def parse_alpha_dirname(name: str) -> Optional[float]:
    try:
        a = float(name)
        if 0.0 <= a <= 1.0:
            return a
        return None
    except Exception:
        return None


@dataclass
class MethodEntry:
    strategy: str
    alpha: float
    label: str
    renders: Path


def discover_sweep_methods(sweep_root: Path) -> List[MethodEntry]:
    out: List[MethodEntry] = []
    if not sweep_root.exists():
        raise FileNotFoundError(sweep_root)

    for strategy_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        strategy = strategy_dir.name
        for alpha_dir in sorted([p for p in strategy_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            a = parse_alpha_dirname(alpha_dir.name)
            if a is None:
                continue
            renders = find_renders_dir(alpha_dir)
            if renders is None:
                print(f"[WARN] No renders found under: {alpha_dir}")
                continue
            label = f"{strategy}-a{alpha_dir.name}"
            out.append(MethodEntry(strategy=strategy, alpha=a, label=label, renders=renders))

    # sort by strategy then alpha
    out.sort(key=lambda e: (e.strategy.lower(), e.alpha))
    return out


# -----------------------------
# Montage
# -----------------------------
def load_font(size: int = 14):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def thumb(img: np.ndarray, w: int, h: int) -> Image.Image:
    im = Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8))
    im = im.resize((w, h), resample=Image.BILINEAR)
    return im


def make_montage_pages(
    out_dir: Path,
    rgb_dir: Path,
    t_dir: Path,
    by_strategy: Dict[str, List[MethodEntry]],
    frame_names: List[str],
    thumb_w: int,
    thumb_h: int,
    max_cols_per_page: int,
    max_frames: int,
):
    if not frame_names:
        return

    font = load_font(14)
    # use first N frames for montage pages
    frames = frame_names[:max_frames]

    for frame_idx, fn in enumerate(frames):
        rgb_img = load_rgb(rgb_dir / fn)
        t_img = load_rgb(t_dir / fn)
        rgb_th = thumb(rgb_img, thumb_w, thumb_h)
        t_th = thumb(t_img, thumb_w, thumb_h)

        # per strategy: list sorted by alpha
        strategies = sorted(by_strategy.keys(), key=str.lower)

        # determine max alpha cols across strategies
        max_mid_cols = 0
        cols_per_strategy: Dict[str, List[MethodEntry]] = {}
        for s in strategies:
            cols = sorted(by_strategy[s], key=lambda e: e.alpha)
            cols_per_strategy[s] = cols
            max_mid_cols = max(max_mid_cols, len(cols))

        if max_mid_cols == 0:
            return

        mid_window = max(1, max_cols_per_page - 2)  # left RGB + right T
        num_pages = int(math.ceil(max_mid_cols / mid_window))

        row_label_w = 320
        pad = 8
        header_h = 30
        row_h = thumb_h + header_h + pad * 2

        for page_idx in range(num_pages):
            start = page_idx * mid_window
            end = min(max_mid_cols, (page_idx + 1) * mid_window)
            mid_len = end - start
            cols_total = 2 + mid_len

            W = row_label_w + cols_total * thumb_w + (cols_total + 2) * pad
            H = len(strategies) * row_h + pad * 2 + 70

            canvas = Image.new("RGB", (W, H), (255, 255, 255))
            draw = ImageDraw.Draw(canvas)

            title = f"Montage frame={fn} ({frame_idx+1}/{len(frames)})  page {page_idx+1}/{num_pages}  mid[{start}:{end})"
            draw.text((pad, pad), title, fill=(0, 0, 0), font=font)

            y0 = 60
            x0 = row_label_w + pad

            draw.text((x0 + 2, y0), "RGB_ref", fill=(0, 0, 0), font=font)
            tx = x0 + (1 + mid_len) * (thumb_w + pad)
            draw.text((tx + 2, y0), "T_ref", fill=(0, 0, 0), font=font)

            row_i = 0
            for s in strategies:
                row_top = y0 + header_h + pad + row_i * row_h

                draw.text((pad, row_top), s, fill=(0, 0, 0), font=font)

                canvas.paste(rgb_th, (x0, row_top))

                cols = cols_per_strategy[s]
                for j in range(mid_len):
                    idx = start + j
                    cx = x0 + (1 + j) * (thumb_w + pad)
                    if idx < len(cols):
                        entry = cols[idx]
                        p = entry.renders / fn
                        img = load_rgb(p) if p.exists() else np.zeros_like(rgb_img)
                        im_th = thumb(img, thumb_w, thumb_h)
                        canvas.paste(im_th, (cx, row_top))
                        draw.text((cx + 2, row_top + thumb_h + 2), f"{entry.alpha:g}", fill=(0, 0, 0), font=font)
                    else:
                        draw.rectangle([cx, row_top, cx + thumb_w, row_top + thumb_h], outline=(180, 180, 180))

                canvas.paste(t_th, (tx, row_top))
                row_i += 1

            out_path = out_dir / f"montage_page_f{frame_idx:02d}_{page_idx:03d}.png"
            canvas.save(out_path)
            print("[OK] wrote:", out_path)


# -----------------------------
# Plotting
# -----------------------------
def plot_lines(
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    series: Dict[str, List[Tuple[float, float]]],
    annotate: bool = True,
    invert_y: bool = False,
):
    plt.figure()
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if invert_y:
        ax.invert_yaxis()

    groups = sorted(series.keys(), key=str.lower)
    cmap = plt.get_cmap("tab20")
    for i, g in enumerate(groups):
        pts = sorted(series[g], key=lambda t: t[0])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=2, color=cmap(i % 20), label=g)
        if annotate:
            for a, y in pts:
                ax.text(a, y, f"{a:g}", fontsize=9)

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[OK] wrote:", out_png)


def plot_pareto(
    out_png: Path,
    title: str,
    series: Dict[str, List[Tuple[float, float, float]]],
    xlabel: str,
    ylabel: str,
):
    # series[group] = [(alpha, x, y)]
    plt.figure()
    ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    groups = sorted(series.keys(), key=str.lower)
    cmap = plt.get_cmap("tab20")
    for i, g in enumerate(groups):
        pts = sorted(series[g], key=lambda t: t[0])
        if not pts:
            continue
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=2, color=cmap(i % 20), label=g)
        for a, x, y in pts:
            ax.text(x, y, f"{a:g}", fontsize=9)

    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("[OK] wrote:", out_png)


# -----------------------------
# Main eval
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        "Evaluate sweep blends (PSNR/SSIM/LPIPS + structure/thermal curves)."
    )

    # Minimal required
    ap.add_argument("--sweep_root", required=True, help="Top folder: sweep_root/strategy/alpha/...")
    ap.add_argument("--rgb_render", required=True, help="RGB reference renders folder.")
    ap.add_argument("--t_render", required=True, help="T reference renders folder.")

    # Optional
    ap.add_argument("--out_dir", default="", help="Output directory (default: sweep_root/_eval).")
    ap.add_argument("--thermal_scalar", default="hue_y", choices=["y", "gray", "luma", "hue", "hue_y", "sat", "val"])
    ap.add_argument("--thermal_align", default="linear", choices=["none", "linear"])
    ap.add_argument("--sample_frames", type=int, default=8, help="0 means all frames.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_side", type=int, default=0, help="Resize so max(H,W)<=max_side (0 disables).")

    # Montage defaults ON (you asked for montage)
    ap.add_argument("--montage", action="store_true", help="Generate montage pages.")
    ap.add_argument("--montage_thumb_w", type=int, default=220)
    ap.add_argument("--montage_thumb_h", type=int, default=176)
    ap.add_argument("--montage_max_cols", type=int, default=12)
    ap.add_argument("--montage_frames", type=int, default=1, help="How many frames to montage (default 1).")

    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    rgb_dir = Path(args.rgb_render)
    t_dir = Path(args.t_render)

    out_dir = Path(args.out_dir) if args.out_dir else (sweep_root / "_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover sweep
    entries = discover_sweep_methods(sweep_root)
    if not entries:
        raise RuntimeError(f"No methods found under sweep_root: {sweep_root}")

    # Group by strategy
    by_strategy: Dict[str, List[MethodEntry]] = {}
    for e in entries:
        by_strategy.setdefault(e.strategy, []).append(e)

    # Frames: intersection across rgb/t and all method folders
    rgb_frames = set(list_images(rgb_dir))
    t_frames = set(list_images(t_dir))
    if not rgb_frames or not t_frames:
        raise FileNotFoundError("rgb_render or t_render directory has no images.")

    common = rgb_frames & t_frames
    for e in entries:
        mf = set(list_images(e.renders))
        if not mf:
            print(f"[WARN] empty renders for {e.label}: {e.renders}")
            continue
        common &= mf

    common = sorted(list(common), key=natural_key)
    if not common:
        raise RuntimeError("No common frame names across rgb/t and methods.")

    # sampling
    if args.sample_frames and args.sample_frames > 0 and args.sample_frames < len(common):
        rnd = random.Random(args.seed)
        idxs = sorted(rnd.sample(range(len(common)), args.sample_frames))
        frames = [common[i] for i in idxs]
    else:
        frames = common

    # Warn about LPIPS availability
    if not HAS_LPIPS_PYTORCH:
        print("[WARN] LPIPS unavailable (cannot import lpipsPyTorch). Will skip LPIPS metrics.")

    # CSV rows
    per_frame_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    # Evaluate each entry
    for e in entries:
        psnrs: List[float] = []
        ssims: List[float] = []
        lpips_vals: List[float] = []

        ssim_y_vals: List[float] = []
        edge_vals: List[float] = []

        spear_vals: List[float] = []
        ssim_s_vals: List[float] = []

        for fn in frames:
            rgb = load_rgb(rgb_dir / fn)
            t = load_rgb(t_dir / fn)
            m = load_rgb(e.renders / fn)

            if args.max_side and args.max_side > 0:
                rgb = maybe_resize_max_side(rgb, args.max_side)
                t = maybe_resize_max_side(t, args.max_side)
                m = maybe_resize_max_side(m, args.max_side)

            # RGB fidelity
            p = psnr_rgb(m, rgb)
            s = ssim_rgb(m, rgb)
            l = lpips_rgb(m, rgb)

            psnrs.append(p)
            ssims.append(s)
            if l is not None:
                lpips_vals.append(l)

            # Structure on Y
            y_rgb = np.clip(rgb_to_luma_y(rgb), 0.0, 1.0)
            y_m = np.clip(rgb_to_luma_y(m), 0.0, 1.0)
            sy = ssim_gray(y_m, y_rgb)
            e_rgb = sobel_like_edge(y_rgb)
            e_m = sobel_like_edge(y_m)
            ec = pearson_corr(e_m, e_rgb)

            ssim_y_vals.append(sy)
            edge_vals.append(ec)

            # Thermal scalar fidelity vs T
            s_t = np.clip(thermal_scalar_from_rgb(t, args.thermal_scalar), 0.0, 1.0)
            s_m = np.clip(thermal_scalar_from_rgb(m, args.thermal_scalar), 0.0, 1.0)
            s_m_al = align_scalar(s_m, s_t, args.thermal_align)

            sp = spearman_corr(s_m_al, s_t)
            ss = ssim_gray(np.clip(s_m_al, 0.0, 1.0), np.clip(s_t, 0.0, 1.0))

            spear_vals.append(sp)
            ssim_s_vals.append(ss)

            per_frame_rows.append({
                "strategy": e.strategy,
                "alpha": e.alpha,
                "label": e.label,
                "frame": fn,
                "PSNR_RGB": p,
                "SSIM_RGB": s,
                "LPIPS_RGB": (l if l is not None else ""),
                "SSIM_Y": sy,
                "EdgeCorr_Y": ec,
                "Spearman_S": sp,
                "SSIM_S": ss,
            })

        summary_rows.append({
            "strategy": e.strategy,
            "alpha": e.alpha,
            "label": e.label,
            "n_frames": len(frames),
            "PSNR_RGB_mean": float(np.mean(psnrs)) if psnrs else float("nan"),
            "SSIM_RGB_mean": float(np.mean(ssims)) if ssims else float("nan"),
            "LPIPS_RGB_mean": (float(np.mean(lpips_vals)) if lpips_vals else ""),
            "SSIM_Y_mean": float(np.mean(ssim_y_vals)) if ssim_y_vals else float("nan"),
            "EdgeCorr_Y_mean": float(np.mean(edge_vals)) if edge_vals else float("nan"),
            "Spearman_S_mean": float(np.mean(spear_vals)) if spear_vals else float("nan"),
            "SSIM_S_mean": float(np.mean(ssim_s_vals)) if ssim_s_vals else float("nan"),
        })

    # Write CSVs
    summary_csv = out_dir / "summary.csv"
    per_frame_csv = out_dir / "per_frame.csv"

    summary_rows.sort(key=lambda r: (str(r["strategy"]).lower(), float(r["alpha"])))
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "strategy", "alpha", "label", "n_frames",
            "PSNR_RGB_mean", "SSIM_RGB_mean", "LPIPS_RGB_mean",
            "SSIM_Y_mean", "EdgeCorr_Y_mean",
            "Spearman_S_mean", "SSIM_S_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    with open(per_frame_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "strategy", "alpha", "label", "frame",
            "PSNR_RGB", "SSIM_RGB", "LPIPS_RGB",
            "SSIM_Y", "EdgeCorr_Y",
            "Spearman_S", "SSIM_S",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_frame_rows:
            w.writerow(r)

    print("[OK] wrote:", summary_csv)
    print("[OK] wrote:", per_frame_csv)

    # Build line series from summary
    series_psnr: Dict[str, List[Tuple[float, float]]] = {}
    series_ssim: Dict[str, List[Tuple[float, float]]] = {}
    series_lpips: Dict[str, List[Tuple[float, float]]] = {}
    series_spear: Dict[str, List[Tuple[float, float]]] = {}
    series_ssimS: Dict[str, List[Tuple[float, float]]] = {}
    pareto: Dict[str, List[Tuple[float, float, float]]] = {}

    for r in summary_rows:
        strat = str(r["strategy"])
        a = float(r["alpha"])

        psnr_m = float(r["PSNR_RGB_mean"])
        ssim_m = float(r["SSIM_RGB_mean"])
        spear_m = float(r["Spearman_S_mean"])

        series_psnr.setdefault(strat, []).append((a, psnr_m))
        series_ssim.setdefault(strat, []).append((a, ssim_m))
        series_spear.setdefault(strat, []).append((a, spear_m))
        series_ssimS.setdefault(strat, []).append((a, float(r["SSIM_S_mean"])))
        pareto.setdefault(strat, []).append((a, ssim_m, spear_m))

        lp = r["LPIPS_RGB_mean"]
        if lp != "":
            try:
                series_lpips.setdefault(strat, []).append((a, float(lp)))
            except Exception:
                pass

    # Plots (multiple line charts as you requested)
    plot_lines(out_dir / "lines_psnr.png", "RGB fidelity (PSNR vs alpha)", "alpha", "PSNR (dB)", series_psnr)
    plot_lines(out_dir / "lines_ssim.png", "RGB fidelity (SSIM vs alpha)", "alpha", "SSIM", series_ssim)
    if series_lpips:
        plot_lines(out_dir / "lines_lpips.png", "RGB fidelity (LPIPS vs alpha)", "alpha", "LPIPS (lower better)", series_lpips, invert_y=False)

    plot_lines(out_dir / "lines_spearmanS.png", f"Thermal fidelity (Spearman({args.thermal_scalar}) vs alpha)", "alpha", "Spearman", series_spear)
    plot_lines(out_dir / "lines_ssimS.png", f"Thermal fidelity (SSIM({args.thermal_scalar}) vs alpha)", "alpha", "SSIM_S", series_ssimS)

    # Pareto (structure vs thermal)
    plot_pareto(out_dir / "pareto.png", "Trade-off trajectory (SSIM_RGB vs Spearman_S)", pareto, "SSIM_RGB", "Spearman_S")

    # Montage pages
    if args.montage:
        make_montage_pages(
            out_dir=out_dir,
            rgb_dir=rgb_dir,
            t_dir=t_dir,
            by_strategy=by_strategy,
            frame_names=frames,
            thumb_w=args.montage_thumb_w,
            thumb_h=args.montage_thumb_h,
            max_cols_per_page=args.montage_max_cols,
            max_frames=max(1, args.montage_frames),
        )

    print("\nDone.")
    print("Outputs in:", out_dir)


if __name__ == "__main__":
    main()
