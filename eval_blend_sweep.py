#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_blend_sweep.py

Evaluate a sweep of blended 3DGS models laid out as:
  sweep_root/
    <strategy_name>/
      <alpha>/            (a 3DGS model dir; may or may not have test/ours_*/renders yet)
        point_cloud/...
        cfg_args

It produces TWO evaluation sets:
  1) render_ref: compare each blend render against RGB-render (structure) and T-render (thermal scalar)
  2) gt_ref    : compare each blend render against RGB-GT (and optionally T-GT if provided)

It can also auto-render missing renders by calling render.py.
"""

from __future__ import annotations

import os
import re
import ast
import sys
import math
import json
import shlex
import time
import shutil
import argparse
import subprocess
import warnings

# Silence torchvision deprecation warnings triggered inside lpipsPyTorch/torchvision
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*weights.*positional parameter.*deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*Arguments other than a weight enum.*deprecated.*",
)

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# --- optional deps ---
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None

try:
    from PIL import Image  # type: ignore
except Exception as e:
    raise RuntimeError("PIL is required (pip install pillow).") from e

try:
    from skimage.metrics import structural_similarity as sk_ssim  # type: ignore
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr  # type: ignore
except Exception as e:
    raise RuntimeError("scikit-image is required (pip install scikit-image).") from e

try:
    from scipy.stats import spearmanr  # type: ignore
except Exception:
    spearmanr = None

# LPIPS: use lpipsPyTorch (your env has it)
LPIPS_OK = False
_lpips_fn = None
try:
    import torch  # type: ignore
    from lpipsPyTorch import lpips as lpips_fn  # type: ignore

    def _lpips_init() -> Any:
        # lpipsPyTorch expects torch tensors in [0,1] or [-1,1] depending on fn; we use [-1,1]
        # Using alexnet by default
        return "alex"

    _lpips_fn = lpips_fn
    LPIPS_OK = True
except Exception:
    LPIPS_OK = False
    _lpips_fn = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, **kwargs):
        return it


# -----------------------------
# Helpers: images
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    out = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    out.sort(key=lambda p: p.name)
    return out


def read_image_f32(path: Path) -> np.ndarray:
    # returns float32 RGB in [0,1]
    if cv2 is not None:
        im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read image: {path}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return (im.astype(np.float32) / 255.0)
    # PIL fallback
    im = Image.open(path).convert("RGB")
    return (np.asarray(im).astype(np.float32) / 255.0)


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    # img: H,W,3 in [0,1]
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def edge_map_y(img_y: np.ndarray) -> np.ndarray:
    # simple Sobel
    if cv2 is not None:
        gx = cv2.Sobel(img_y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_y, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        return mag
    # numpy sobel
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = kx.T
    from scipy.signal import convolve2d  # type: ignore
    gx = convolve2d(img_y, kx, mode="same", boundary="symm")
    gy = convolve2d(img_y, ky, mode="same", boundary="symm")
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float((a @ b) / denom)


# -----------------------------
# Thermal scalar from pseudo-color images
# -----------------------------
def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    # img in [0,1], returns hsv in [0,1]
    if cv2 is not None:
        hsv = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 0] = hsv[..., 0] / 179.0
        hsv[..., 1] = hsv[..., 1] / 255.0
        hsv[..., 2] = hsv[..., 2] / 255.0
        return hsv
    # PIL / numpy implementation
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn + 1e-12

    h = np.zeros_like(mx)
    mask = (mx == r)
    h[mask] = ((g - b)[mask] / diff[mask]) % 6.0
    mask = (mx == g)
    h[mask] = ((b - r)[mask] / diff[mask]) + 2.0
    mask = (mx == b)
    h[mask] = ((r - g)[mask] / diff[mask]) + 4.0
    h = (h / 6.0) % 1.0

    s = diff / (mx + 1e-12)
    v = mx
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def thermal_scalar(img_rgb: np.ndarray, mode: str = "hue_y") -> np.ndarray:
    """
    Convert pseudo-colored thermal render to a single scalar map S.
    - hue: hue channel (0..1)
    - sat: saturation
    - val: value
    - hue_y: hue weighted by luminance (more robust when saturation is small)
    """
    mode = mode.lower()
    hsv = rgb_to_hsv(img_rgb)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    y = rgb_to_y(img_rgb)
    if mode == "hue":
        return h
    if mode == "sat":
        return s
    if mode == "val":
        return v
    if mode == "hue_y":
        return h * (0.3 + 0.7 * y)
    raise ValueError(f"Unknown thermal_scalar mode: {mode}")


def align_scalar(pred: np.ndarray, ref: np.ndarray, mode: str = "linear") -> np.ndarray:
    """
    Align scalar map pred to ref (global). Useful when a method changes contrast.
    - none: no align
    - linear: pred' = a*pred + b by least squares
    - rank: match mean/std after monotonic rank mapping (approx)
    """
    mode = mode.lower()
    if mode == "none":
        return pred
    p = pred.reshape(-1).astype(np.float64)
    r = ref.reshape(-1).astype(np.float64)
    if mode == "linear":
        A = np.stack([p, np.ones_like(p)], axis=1)
        x, *_ = np.linalg.lstsq(A, r, rcond=None)
        a, b = float(x[0]), float(x[1])
        return (a * pred + b).astype(np.float32)
    if mode == "rank":
        # map to z-score with ref mean/std
        pm, ps = float(p.mean()), float(p.std() + 1e-12)
        rm, rs = float(r.mean()), float(r.std() + 1e-12)
        z = (pred - pm) / ps
        return (z * rs + rm).astype(np.float32)
    raise ValueError(f"Unknown thermal_align mode: {mode}")


# -----------------------------
# Metrics
# -----------------------------
def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    # a,b: H,W float32 [0,1]
    return float(sk_ssim(a, b, data_range=1.0))


def psnr_rgb(a: np.ndarray, b: np.ndarray) -> float:
    return float(sk_psnr(a, b, data_range=1.0))


def psnr_gray(a: np.ndarray, b: np.ndarray) -> float:
    return float(sk_psnr(a, b, data_range=1.0))


def lpips_rgb(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not LPIPS_OK or _lpips_fn is None:
        return None
    # a,b: H,W,3 in [0,1] -> torch [-1,1]
    ta = torch.from_numpy(a.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
    tb = torch.from_numpy(b.transpose(2, 0, 1)).unsqueeze(0).float() * 2.0 - 1.0
    with torch.no_grad():
        v = _lpips_fn(ta, tb, net_type=_lpips_init())
    return float(v.item())


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if spearmanr is None:
        # fallback: pearson on ranks
        ra = a.argsort().argsort().astype(np.float64)
        rb = b.argsort().argsort().astype(np.float64)
        return corrcoef(ra, rb)
    return float(spearmanr(a, b).correlation)


# -----------------------------
# Paths / discovery
# -----------------------------
@dataclass
class MethodEntry:
    strategy: str
    alpha: float
    label: str
    model_dir: Path
    renders_dir: Optional[Path] = None
    gt_dir: Optional[Path] = None
    used_iter: Optional[int] = None


def parse_alpha_dirname(name: str) -> Optional[float]:
    try:
        # allow "0", "0.1", "1"
        return float(name)
    except Exception:
        return None


def find_best_ours_dir(model_dir: Path, force_iter: Optional[int] = None) -> Optional[Path]:
    test = model_dir / "test"
    if not test.exists():
        return None
    ours = []
    for d in test.iterdir():
        if d.is_dir() and d.name.startswith("ours_"):
            suf = d.name.split("_", 1)[-1]
            if suf.isdigit():
                it = int(suf)
                if force_iter is not None and it != force_iter:
                    continue
                ours.append((it, d))
    if not ours:
        return None
    ours.sort(key=lambda x: x[0])
    return ours[-1][1]


def resolve_renders_and_gt(path: Path, force_iter: Optional[int] = None) -> Tuple[Path, Path, Optional[int]]:
    """
    Accept either:
      - a model dir containing test/ours_*/renders, gt
      - a direct renders dir (…/renders) whose sibling gt exists
      - a direct gt dir (…/gt) whose sibling renders exists
    """
    if path.name.lower() == "renders":
        renders = path
        gt = path.parent / "gt"
        it = None
        m = re.search(r"ours_(\d+)", str(path.parent))
        if m:
            it = int(m.group(1))
        return renders, gt, it
    if path.name.lower() == "gt":
        gt = path
        renders = path.parent / "renders"
        it = None
        m = re.search(r"ours_(\d+)", str(path.parent))
        if m:
            it = int(m.group(1))
        return renders, gt, it

    # treat as model dir
    ours_dir = find_best_ours_dir(path, force_iter=force_iter)
    if ours_dir is None:
        raise FileNotFoundError(f"No test/ours_* under model dir: {path}")
    renders = ours_dir / "renders"
    gt = ours_dir / "gt"
    it = None
    m = re.search(r"ours_(\d+)", ours_dir.name)
    if m:
        it = int(m.group(1))
    return renders, gt, it


def parse_cfg_args(model_dir: Path) -> Dict[str, Any]:
    cfg = model_dir / "cfg_args"
    if not cfg.exists():
        return {}
    text = cfg.read_text(encoding="utf-8", errors="ignore").strip()
    m = re.search(r"Namespace\((.*)\)\s*$", text)
    if not m:
        return {}
    body = m.group(1)
    # split by commas not inside quotes/brackets
    parts = []
    buf = []
    depth = 0
    in_str = False
    quote = ""
    esc = False
    for ch in body:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\":
            buf.append(ch)
            esc = True
            continue
        if in_str:
            buf.append(ch)
            if ch == quote:
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            quote = ch
            buf.append(ch)
            continue
        if ch in "([{":
            depth += 1
            buf.append(ch)
            continue
        if ch in ")]}":
            depth -= 1
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    out: Dict[str, Any] = {}
    for part in parts:
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except Exception:
            out[k] = v.strip("'\"")
    return out


def infer_pointcloud_iters(model_dir: Path) -> List[int]:
    pc = model_dir / "point_cloud"
    if not pc.exists():
        return []
    iters = []
    for d in pc.iterdir():
        if d.is_dir() and d.name.startswith("iteration_"):
            suf = d.name.split("_", 1)[-1]
            if suf.isdigit():
                iters.append(int(suf))
    iters.sort()
    return iters


def choose_iteration(model_dir: Path, preferred: int) -> int:
    if (model_dir / "point_cloud" / f"iteration_{preferred}").exists():
        return preferred
    iters = infer_pointcloud_iters(model_dir)
    if iters:
        return iters[-1]
    return preferred


def discover_methods(sweep_root: Path) -> List[MethodEntry]:
    methods: List[MethodEntry] = []
    for strategy_dir in sorted([p for p in sweep_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        strategy = strategy_dir.name
        for alpha_dir in sorted([p for p in strategy_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
            a = parse_alpha_dirname(alpha_dir.name)
            if a is None:
                continue
            methods.append(MethodEntry(strategy=strategy, alpha=a, label=f"{strategy}-a{alpha_dir.name}", model_dir=alpha_dir))
    return methods


# -----------------------------
# Rendering (auto)
# -----------------------------
def mklink_junction(dst: Path, src: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        p = subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], capture_output=True, text=True)
        return p.returncode == 0
    except Exception:
        return False


def ensure_gt_dedup(gt_dir: Path, shared_gt: Path, mode: str) -> None:
    mode = mode.lower()
    if mode == "keep":
        return
    if mode == "delete":
        if gt_dir.exists():
            shutil.rmtree(gt_dir, ignore_errors=True)
        return
    if mode == "link":
        if gt_dir.exists() and gt_dir.is_dir():
            shutil.rmtree(gt_dir, ignore_errors=True)
        ok = mklink_junction(gt_dir, shared_gt)
        if not ok:
            # fallback: do nothing (renders will still exist)
            return
    else:
        raise ValueError(f"Unknown gt_mode: {mode}")


def run_render(
    render_py: Path,
    model_dir: Path,
    source_path: Path,
    images_subdir: str,
    resolution: int,
    iteration: int,
    python_exe: str,
    extra: str = "",
    verbose: bool = True,
) -> int:
    cmd = [
        python_exe,
        str(render_py),
        "-m", str(model_dir),
        "-s", str(source_path),
        "-i", images_subdir,
        "-r", str(int(resolution)),
        "--iteration", str(int(iteration)),
    ]
    if extra:
        cmd += shlex.split(extra)
    if verbose:
        print("[AUTO_RENDER]", " ".join(cmd))
    p = subprocess.run(cmd)
    return int(p.returncode)


def auto_render_methods(
    methods: List[MethodEntry],
    render_py: Path,
    source_path: Path,
    images_subdir: str,
    resolution: int,
    preferred_iter: int,
    python_exe: str,
    gt_mode: str,
    shared_gt: Optional[Path],
    extra: str = "",
    verbose: bool = True,
) -> None:
    missing = []
    for m in methods:
        # decide expected render dir after rendering
        it = choose_iteration(m.model_dir, preferred_iter)
        rdir = m.model_dir / "test" / f"ours_{it}" / "renders"
        if not rdir.exists() or not any(rdir.glob("*")):
            missing.append(m)

    if not missing:
        return

    bar = tqdm(missing, desc="Auto-render", unit="model")
    for m in bar:
        it = choose_iteration(m.model_dir, preferred_iter)
        m.used_iter = it

        if shared_gt is not None and gt_mode.lower() == "link":
            ensure_gt_dedup(m.model_dir / "test" / f"ours_{it}" / "gt", shared_gt, mode="link")

        bar.set_postfix_str(f"{m.strategy} a={m.alpha:g} it={it}")
        ret = run_render(
            render_py=render_py,
            model_dir=m.model_dir,
            source_path=source_path,
            images_subdir=images_subdir,
            resolution=resolution,
            iteration=it,
            python_exe=python_exe,
            extra=extra,
            verbose=verbose,
        )
        if ret != 0:
            print(f"[WARN] render.py failed ({ret}) for: {m.model_dir}")
            continue

        rdir = m.model_dir / "test" / f"ours_{it}" / "renders"
        gdir = m.model_dir / "test" / f"ours_{it}" / "gt"
        m.renders_dir = rdir if rdir.exists() else None
        m.gt_dir = gdir if gdir.exists() else None

        if shared_gt is not None and gt_mode.lower() in ("delete", "link"):
            ensure_gt_dedup(gdir, shared_gt, mode=gt_mode)


# -----------------------------
# Evaluation
# -----------------------------
def pair_by_name(a_dir: Path, b_dir: Path) -> List[Tuple[Path, Path]]:
    a = list_images(a_dir)
    b = list_images(b_dir)
    if not a or not b:
        return []
    a_map = {p.name: p for p in a}
    b_map = {p.name: p for p in b}
    names = sorted(set(a_map.keys()) & set(b_map.keys()))
    if names:
        return [(a_map[n], b_map[n]) for n in names]
    # fallback: by index
    n = min(len(a), len(b))
    return list(zip(a[:n], b[:n]))


def eval_render_ref(
    method_renders: Path,
    rgb_renders: Path,
    t_renders: Path,
    thermal_mode: str,
    thermal_align: str,
    sample_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    pairs_rgb = pair_by_name(method_renders, rgb_renders)
    pairs_t = pair_by_name(method_renders, t_renders)
    if not pairs_rgb or not pairs_t:
        return {}

    # restrict to intersection names if provided
    if sample_names is not None:
        pairs_rgb = [(a, b) for (a, b) in pairs_rgb if a.name in sample_names]
        pairs_t = [(a, b) for (a, b) in pairs_t if a.name in sample_names]

    ssim_y = []
    edge_corr = []
    sp_s = []
    ssim_s = []

    for (m_img_p, rgb_p), (_, t_p) in zip(pairs_rgb, pairs_t):
        m = read_image_f32(m_img_p)
        rgb = read_image_f32(rgb_p)
        t = read_image_f32(t_p)

        my = rgb_to_y(m)
        rgby = rgb_to_y(rgb)
        ssim_y.append(ssim_gray(my, rgby))
        edge_corr.append(corrcoef(edge_map_y(my), edge_map_y(rgby)))

        ms = thermal_scalar(m, thermal_mode)
        ts = thermal_scalar(t, thermal_mode)
        ms = align_scalar(ms, ts, mode=thermal_align)

        sp_s.append(spearman(ms, ts))
        ssim_s.append(ssim_gray(np.clip(ms, 0, 1), np.clip(ts, 0, 1)))

    return {
        "SSIM_Y": float(np.mean(ssim_y)),
        "EdgeCorr_Y": float(np.mean(edge_corr)),
        "Spearman_S": float(np.mean(sp_s)),
        "SSIM_S": float(np.mean(ssim_s)),
    }


def eval_vs_gt(
    method_renders: Path,
    gt_dir: Path,
    sample_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    pairs = pair_by_name(method_renders, gt_dir)
    if not pairs:
        return {}

    if sample_names is not None:
        pairs = [(a, b) for (a, b) in pairs if a.name in sample_names]

    psnr = []
    ssim = []
    lp = []
    psnr_y = []
    ssim_y = []

    for m_p, gt_p in pairs:
        m = read_image_f32(m_p)
        g = read_image_f32(gt_p)
        psnr.append(psnr_rgb(m, g))
        ssim.append(float(sk_ssim(m, g, channel_axis=2, data_range=1.0)))

        y_m = rgb_to_y(m)
        y_g = rgb_to_y(g)
        psnr_y.append(psnr_gray(y_m, y_g))
        ssim_y.append(ssim_gray(y_m, y_g))

        v = lpips_rgb(m, g)
        if v is not None:
            lp.append(v)

    out = {
        "PSNR": float(np.mean(psnr)),
        "SSIM": float(np.mean(ssim)),
        "PSNR_Y": float(np.mean(psnr_y)),
        "SSIM_Y": float(np.mean(ssim_y)),
    }
    if lp:
        out["LPIPS"] = float(np.mean(lp))
    return out


def sample_frame_names(ref_dir: Path, k: int, seed: int) -> List[str]:
    imgs = list_images(ref_dir)
    if not imgs:
        return []
    rng = np.random.RandomState(seed)
    if k <= 0 or k >= len(imgs):
        return [p.name for p in imgs]
    idx = rng.choice(len(imgs), size=k, replace=False)
    idx.sort()
    return [imgs[i].name for i in idx]


# -----------------------------
# Plotting
# -----------------------------
def plot_lines(
    rows: List[Dict[str, Any]],
    out_path: Path,
    x_key: str,
    y_key: str,
    group_key: str = "strategy",
    label_key: str = "alpha",
    title: str = "",
) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # group by strategy
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r[group_key]), []).append(r)

    plt.figure(figsize=(8, 6))
    for g, items in groups.items():
        items = sorted(items, key=lambda rr: float(rr["alpha"]))
        xs = [float(it[x_key]) for it in items]
        ys = [float(it[y_key]) for it in items]
        plt.plot(xs, ys, marker="o", label=g)
        # annotate alpha on each point (no method names on points)
        for it, x, y in zip(items, xs, ys):
            plt.text(x, y, f'{it[label_key]:g}', fontsize=8)

    plt.xlabel(x_key)
    plt.ylabel(y_key)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# Montage
# -----------------------------
def make_montage(
    out_dir: Path,
    methods: List[MethodEntry],
    rgb_renders: Path,
    t_renders: Path,
    sample_name: str,
    cols_per_page: int = 6,
) -> None:
    """
    Create montage pages:
      each row = one strategy
      columns = RGB_ref | alphas... | T_ref
    Split alphas across pages if too many.
    """
    # group by strategy
    strat_map: Dict[str, List[MethodEntry]] = {}
    for m in methods:
        if m.renders_dir is None:
            continue
        strat_map.setdefault(m.strategy, []).append(m)

    # choose alpha order from union
    alphas_all = sorted({round(m.alpha, 6) for m in methods})
    # split alphas into chunks
    chunks = [alphas_all[i:i + cols_per_page] for i in range(0, len(alphas_all), cols_per_page)]

    # load reference images
    rgb_ref_path = rgb_renders / sample_name
    t_ref_path = t_renders / sample_name
    if not rgb_ref_path.exists() or not t_ref_path.exists():
        return
    rgb_ref = Image.fromarray((read_image_f32(rgb_ref_path) * 255).astype(np.uint8))
    t_ref = Image.fromarray((read_image_f32(t_ref_path) * 255).astype(np.uint8))

    # determine cell size
    W, H = rgb_ref.size
    cell_w, cell_h = W, H
    pad = 6
    header_h = 26

    # strategy order
    strategies = sorted(strat_map.keys())

    for pi, alpha_chunk in enumerate(chunks, start=1):
        # columns: RGB + chunk + T
        ncols = 2 + len(alpha_chunk)
        nrows = len(strategies)
        canvas_w = pad + ncols * (cell_w + pad)
        canvas_h = pad + header_h + nrows * (cell_h + pad)

        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

        # paste row by row
        for ri, strat in enumerate(strategies):
            y0 = pad + header_h + ri * (cell_h + pad)
            # left: rgb ref
            x = pad
            canvas.paste(rgb_ref, (x, y0))
            x += cell_w + pad
            # alphas
            entries = {round(m.alpha, 6): m for m in strat_map[strat]}
            for a in alpha_chunk:
                m = entries.get(round(a, 6))
                if m is not None and m.renders_dir is not None:
                    p = m.renders_dir / sample_name
                    if p.exists():
                        im = Image.fromarray((read_image_f32(p) * 255).astype(np.uint8))
                        canvas.paste(im, (x, y0))
                x += cell_w + pad
            # right: t ref
            canvas.paste(t_ref, (x, y0))

        out_path = out_dir / f"montage_page{pi}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Evaluate sweep blends with render-ref + gt-ref metrics.")

    ap.add_argument("--sweep_root", required=True, help="Top folder: sweep_root/strategy/alpha/...")
    ap.add_argument("--rgb_render", required=True, help="RGB model output dir OR explicit renders/gt dir.")
    ap.add_argument("--t_render", required=True, help="T model output dir OR explicit renders/gt dir.")
    ap.add_argument("--out_dir", default=None, help="Output dir (default: sweep_root/eval)")

    # render selection for refs
    ap.add_argument("--rgb_iter", type=int, default=None, help="Force RGB ours_<iter> (optional).")
    ap.add_argument("--t_iter", type=int, default=None, help="Force T ours_<iter> (optional).")

    # auto render
    ap.add_argument("--auto_render", action="store_true", help="Auto run render.py for missing method renders.")
    ap.add_argument("--render_py", default=None, help="Path to render.py (default: next to this script)")
    ap.add_argument("--python", dest="python_exe", default=sys.executable, help="Python executable for render.py")
    ap.add_argument("--render_iter", type=int, default=None, help="Iteration to render (default inferred from T model)")
    ap.add_argument("--render_source", default=None, help="Override -s for render.py (default from RGB cfg_args)")
    ap.add_argument("--render_images", default=None, help="Override -i for render.py (default from RGB cfg_args)")
    ap.add_argument("--render_resolution", type=int, default=None, help="Override -r for render.py (default from RGB cfg_args)")
    ap.add_argument("--render_extra", default="", help="Extra args appended to render.py (string).")
    ap.add_argument("--gt_mode", choices=["keep", "delete", "link"], default="link",
                    help="When auto_render: what to do with per-method gt to save disk. link uses junction to RGB gt.")

    # evaluation
    ap.add_argument("--thermal_scalar", default="hue_y", choices=["hue", "sat", "val", "hue_y"])
    ap.add_argument("--thermal_align", default="linear", choices=["none", "linear", "rank"])
    ap.add_argument("--sample_frames", type=int, default=8, help="Randomly sample K frames for metrics (0=all).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--no_montage", action="store_true", help="Disable montage output.")
    ap.add_argument("--montage_cols", type=int, default=6, help="How many alpha columns per montage page.")

    args = ap.parse_args()

    sweep_root = Path(args.sweep_root)
    out_dir = Path(args.out_dir) if args.out_dir else (sweep_root / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # resolve references (renders + gt)
    rgb_model_or_dir = Path(args.rgb_render)
    t_model_or_dir = Path(args.t_render)

    rgb_renders, rgb_gt, rgb_it = resolve_renders_and_gt(rgb_model_or_dir, force_iter=args.rgb_iter)
    t_renders, t_gt, t_it = resolve_renders_and_gt(t_model_or_dir, force_iter=args.t_iter)

    print("Resolved refs:")
    print("  RGB renders:", rgb_renders)
    print("  RGB gt     :", rgb_gt)
    print("  T   renders:", t_renders)
    print("  T   gt     :", t_gt)

    # discover methods
    methods = discover_methods(sweep_root)

    # auto render missing
    if args.auto_render:
        render_py = Path(args.render_py) if args.render_py else (Path(__file__).resolve().parent / "render.py")
        if not render_py.exists():
            raise FileNotFoundError(f"render.py not found: {render_py}")

        # infer render params from RGB cfg_args by default
        cfg = parse_cfg_args(rgb_model_or_dir if rgb_model_or_dir.is_dir() else rgb_model_or_dir.parent)
        source_path = Path(args.render_source) if args.render_source else Path(cfg.get("source_path", ""))
        images_subdir = args.render_images if args.render_images else str(cfg.get("images", "images"))
        resolution = int(args.render_resolution if args.render_resolution is not None else int(cfg.get("resolution", 1)))

        if not source_path.exists():
            raise FileNotFoundError(
                f"render_source not found. Pass --render_source explicitly. Got: {source_path}"
            )

        preferred_iter = args.render_iter
        if preferred_iter is None:
            # default: use the iteration of the provided T-model renders (e.g. ours_40000)
            preferred_iter = t_it
        if preferred_iter is None:
            # fallback: inspect T model dir for best ours_*
            try:
                best = find_best_ours_dir(t_model_or_dir, None)
                if best is not None:
                    preferred_iter = int(best.name.split('_')[-1])
            except Exception:
                preferred_iter = None
        if preferred_iter is None:
            preferred_iter = 0

        print(f"[AUTO_RENDER] using source={source_path} images={images_subdir} r={resolution} iter={preferred_iter}")

        # shared gt = RGB gt (avoid per-alpha duplication)
        shared_gt = rgb_gt if rgb_gt.exists() else None

        auto_render_methods(
            methods=methods,
            render_py=render_py,
            source_path=source_path,
            images_subdir=images_subdir,
            resolution=resolution,
            preferred_iter=int(preferred_iter),
            python_exe=args.python_exe,
            gt_mode=args.gt_mode,
            shared_gt=shared_gt,
            extra=args.render_extra,
            verbose=True,
        )

    # after possible rendering, resolve method renders/gt dirs
    for m in methods:
        if m.renders_dir is None:
            # try resolve best ours (might have existed already)
            try:
                r, g, it = resolve_renders_and_gt(m.model_dir, force_iter=args.render_iter)
                m.renders_dir = r if r.exists() else None
                m.gt_dir = g if g.exists() else None
                m.used_iter = it
            except Exception:
                pass

    # filter valid
    methods = [m for m in methods if m.renders_dir is not None and m.renders_dir.exists()]
    if not methods:
        raise RuntimeError(f"No methods with renders found under: {sweep_root}")

    # sample frames by names from RGB gt (stable ordering)
    sample_names = sample_frame_names(rgb_gt, args.sample_frames, args.seed)
    if not sample_names:
        # fallback to RGB renders
        sample_names = sample_frame_names(rgb_renders, args.sample_frames, args.seed)

    # evaluate
    rows_render_ref = []
    rows_gt_ref = []

    for m in tqdm(methods, desc="Evaluating", unit="method"):
        rr = eval_render_ref(
            method_renders=m.renders_dir,  # type: ignore
            rgb_renders=rgb_renders,
            t_renders=t_renders,
            thermal_mode=args.thermal_scalar,
            thermal_align=args.thermal_align,
            sample_names=sample_names,
        )
        gr = eval_vs_gt(
            method_renders=m.renders_dir,  # type: ignore
            gt_dir=rgb_gt,
            sample_names=sample_names,
        )

        rows_render_ref.append({
            "strategy": m.strategy,
            "alpha": m.alpha,
            "label": m.label,
            **{f"{k}_mean": v for k, v in rr.items()},
        })
        rows_gt_ref.append({
            "strategy": m.strategy,
            "alpha": m.alpha,
            "label": m.label,
            **{f"{k}_mean": v for k, v in gr.items()},
        })

    # write csv
    import pandas as pd  # type: ignore
    df_rr = pd.DataFrame(rows_render_ref).sort_values(["strategy", "alpha"])
    df_gt = pd.DataFrame(rows_gt_ref).sort_values(["strategy", "alpha"])
    df_rr.to_csv(out_dir / "summary_render_ref.csv", index=False)
    df_gt.to_csv(out_dir / "summary_gt_ref.csv", index=False)
    print("[OK] wrote:", out_dir / "summary_render_ref.csv")
    print("[OK] wrote:", out_dir / "summary_gt_ref.csv")

    # pareto (structure vs thermal)
    if "SSIM_Y_mean" in df_rr.columns and "Spearman_S_mean" in df_rr.columns:
        plot_lines(
            rows=rows_render_ref,
            out_path=out_dir / "pareto_lines.png",
            x_key="SSIM_Y_mean",
            y_key="Spearman_S_mean",
            title="Structure (SSIM_Y) vs Thermal (Spearman_S)",
        )
        print("[OK] wrote:", out_dir / "pareto_lines.png")

    # GT curves
    for metric in ["PSNR_mean", "SSIM_mean", "LPIPS_mean", "PSNR_Y_mean", "SSIM_Y_mean"]:
        if metric in df_gt.columns:
            plot_lines(
                rows=rows_gt_ref,
                out_path=out_dir / f"curve_{metric}.png",
                x_key="alpha",
                y_key=metric,
                title=f"GT metric vs alpha: {metric}",
            )
            print("[OK] wrote:", out_dir / f"curve_{metric}.png")

    # montage
    if not args.no_montage and sample_names:
        make_montage(
            out_dir=out_dir,
            methods=methods,
            rgb_renders=rgb_renders,
            t_renders=t_renders,
            sample_name=sample_names[0],
            cols_per_page=max(1, int(args.montage_cols)),
        )
        print("[OK] wrote montage_page*.png")

    print("Done.")


if __name__ == "__main__":
    main()
