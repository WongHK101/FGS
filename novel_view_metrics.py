#
# Novel-view metrics: render a small novel camera path and compute no-ref cleanliness proxies.
#
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.cameras import MiniCam
from utils.graphics_utils import getProjectionMatrix
from utils.general_utils import safe_state


def _rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _conv3x3(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = img.shape
    pad = np.pad(img, ((1, 1), (1, 1)), mode="edge")
    out = (
        kernel[0, 0] * pad[0:h, 0:w] +
        kernel[0, 1] * pad[0:h, 1:w + 1] +
        kernel[0, 2] * pad[0:h, 2:w + 2] +
        kernel[1, 0] * pad[1:h + 1, 0:w] +
        kernel[1, 1] * pad[1:h + 1, 1:w + 1] +
        kernel[1, 2] * pad[1:h + 1, 2:w + 2] +
        kernel[2, 0] * pad[2:h + 2, 0:w] +
        kernel[2, 1] * pad[2:h + 2, 1:w + 1] +
        kernel[2, 2] * pad[2:h + 2, 2:w + 2]
    )
    return out


def _sobel_mag(gray: np.ndarray) -> np.ndarray:
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    gx = _conv3x3(gray, kx)
    gy = _conv3x3(gray, ky)
    mag = np.hypot(gx, gy)
    mag /= (4.0 * math.sqrt(2.0))
    return np.clip(mag, 0.0, 1.0)


def _sobel_xy(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]], dtype=np.float32)
    ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.float32)
    gx = _conv3x3(gray, kx)
    gy = _conv3x3(gray, ky)
    return gx, gy


def _smooth_gray(gray: np.ndarray, iters: int = 2) -> np.ndarray:
    k = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=np.float32) / 16.0
    out = gray.astype(np.float32)
    for _ in range(max(1, int(iters))):
        out = _conv3x3(out, k)
    return out


def _local_contrast_norm(gray: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    mu = _smooth_gray(gray, iters=2)
    dev = np.abs(gray - mu)
    sigma = _smooth_gray(dev, iters=2)
    norm = (gray - mu) / (sigma + float(eps))
    return np.clip(norm, -4.0, 4.0).astype(np.float32)


def _texture_metrics(gray: np.ndarray) -> Tuple[float, float, float]:
    # Brightness-insensitive texture proxy: evaluate gradients on local-contrast-normalized gray.
    g = _local_contrast_norm(gray)
    gx, gy = _sobel_xy(g)
    gm = np.hypot(gx, gy)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    grad_p90 = float(np.percentile(gm, 90.0))
    thr = float(np.mean(gm) + 0.75 * np.std(gm))
    thr = max(thr, 1e-4)
    edge_density = float(np.mean(gm > thr))
    return tenengrad, grad_p90, edge_density


def _edge_anisotropy(gray: np.ndarray) -> float:
    # Directionality score in [0,1]: higher means clearer oriented structures, lower means isotropic clutter/smear.
    g = _local_contrast_norm(gray)
    gx, gy = _sobel_xy(g)
    mag = np.hypot(gx, gy)
    if mag.size <= 0:
        return 0.0
    thr = float(np.percentile(mag, 75.0))
    mask = mag > max(thr, 1e-6)
    if int(np.sum(mask)) < 16:
        return 0.0
    gxx = gx[mask] * gx[mask]
    gyy = gy[mask] * gy[mask]
    gxy = gx[mask] * gy[mask]
    jxx = float(np.mean(gxx))
    jyy = float(np.mean(gyy))
    jxy = float(np.mean(gxy))
    tr = max(jxx + jyy, 1e-12)
    det = (jxx * jyy) - (jxy * jxy)
    disc = max(0.0, 0.25 * tr * tr - det)
    root = math.sqrt(disc)
    l1 = 0.5 * tr + root
    l2 = max(0.0, 0.5 * tr - root)
    return float(np.clip((l1 - l2) / max(l1 + l2, 1e-12), 0.0, 1.0))


def _laplacian(gray: np.ndarray) -> np.ndarray:
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    return _conv3x3(gray, k)


def _air_mask_from_render(gray: np.ndarray, edge_mag: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    top_h = max(1, int(round(0.65 * h)))
    top = np.zeros((h, w), dtype=bool)
    top[:top_h, :] = True
    edge_thr = float(np.percentile(edge_mag, 55.0))
    air = np.logical_and(top, edge_mag <= edge_thr)
    # Fallback for very close-up scenes: still keep a stable "upper-view" proxy.
    if float(np.mean(air)) < 0.08:
        air = top
    return air


def _air_cleanliness_metrics(gray: np.ndarray, edge_mag: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    air = _air_mask_from_render(gray, edge_mag)
    cnt = int(air.sum())
    if cnt <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    lap = np.abs(_laplacian(gray))
    base = _smooth_gray(gray, iters=3)
    resid = gray - base
    resid_abs = np.abs(resid)

    air_edge = float(np.mean(edge_mag[air]))
    air_hf = float(np.mean(lap[air]))
    air_ratio = float(cnt) / float(gray.size)

    resid_thr = max(float(np.percentile(resid[air], 92.0)), 0.03)
    edge_low = float(np.percentile(edge_mag[air], 65.0))
    blob_mask = np.logical_and(air, np.logical_and(resid >= resid_thr, edge_mag <= edge_low))
    blob_ratio = float(blob_mask.sum()) / float(cnt)

    contrast_low = float(np.percentile(resid_abs[air], 35.0))
    gray_high = float(np.percentile(gray[air], 65.0))
    fog_mask = np.logical_and(air, np.logical_and(resid_abs <= contrast_low, gray >= gray_high))
    fog_ratio = float(fog_mask.sum()) / float(cnt)

    # Lower is cleaner; weighted to emphasize obvious floaters and haze.
    artifact_score = float(air_edge + air_hf + 2.0 * blob_ratio + 1.5 * fog_ratio)
    return air_ratio, air_edge, air_hf, blob_ratio, fog_ratio, artifact_score


def _bg_leak_ratio(render: np.ndarray, bg: int, thr: float = 3.0 / 255.0) -> float:
    if bg == 0:
        mask = np.max(render, axis=2) <= thr
    else:
        mask = np.min(render, axis=2) >= (1.0 - thr)
    return float(np.mean(mask))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def _parse_float_list(raw: str, default_vals: List[float], clip_min: float = None, clip_max: float = None) -> List[float]:
    if raw is None:
        vals = list(default_vals)
    else:
        text = str(raw).strip()
        if not text:
            vals = list(default_vals)
        else:
            vals = []
            for tok in text.split(","):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    vals.append(float(tok))
                except ValueError:
                    continue
    out: List[float] = []
    for v in vals:
        x = float(v)
        if clip_min is not None:
            x = max(float(clip_min), x)
        if clip_max is not None:
            x = min(float(clip_max), x)
        out.append(x)
    if not out:
        out = list(default_vals)
    return out


def _camera_mats(cam) -> Tuple[np.ndarray, np.ndarray]:
    w2c = cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)
    return w2c, c2w


def _camera_center(cam) -> np.ndarray:
    _, c2w = _camera_mats(cam)
    return c2w[:3, 3].copy()


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _build_c2w_lookat(cam_pos: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    z = _normalize(target - cam_pos)
    if float(np.linalg.norm(z)) < 1e-8:
        z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = _normalize(up_hint)
    if float(np.linalg.norm(up)) < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # Build a right-handed camera basis: x = up x forward, y = forward x x.
    x = np.cross(up, z)
    if float(np.linalg.norm(x)) < 1e-8:
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(alt, z))) > 0.95:
            alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = np.cross(alt, z)
    x = _normalize(x)
    y = _normalize(np.cross(z, x))
    R = np.stack([x, y, z], axis=1)
    R = _orthonormalize(R)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.astype(np.float32)
    c2w[:3, 3] = cam_pos.astype(np.float32)
    return c2w


def _build_c2w_from_forward_right(cam_pos: np.ndarray, forward: np.ndarray, right_hint: np.ndarray) -> np.ndarray:
    z = _normalize(forward.astype(np.float32))
    if float(np.linalg.norm(z)) < 1e-8:
        z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    x = right_hint.astype(np.float32) - float(np.dot(right_hint, z)) * z
    if float(np.linalg.norm(x)) < 1e-8:
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(alt, z))) > 0.95:
            alt = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        x = alt - float(np.dot(alt, z)) * z
    x = _normalize(x)
    y = _normalize(np.cross(z, x))
    R = np.stack([x, y, z], axis=1)
    R = _orthonormalize(R)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.astype(np.float32)
    c2w[:3, 3] = cam_pos.astype(np.float32)
    return c2w


def _nearest_neighbor_dist(centers: np.ndarray) -> np.ndarray:
    n = centers.shape[0]
    if n <= 1:
        return np.zeros((n,), dtype=np.float32)
    out = np.full((n,), np.inf, dtype=np.float32)
    chunk = 512
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        a = centers[i0:i1]
        d = np.linalg.norm(a[:, None, :] - centers[None, :, :], axis=2)
        rows = np.arange(i0, i1) - i0
        d[rows, np.arange(i0, i1)] = np.inf
        out[i0:i1] = d.min(axis=1)
    out[~np.isfinite(out)] = 0.0
    return out


def _camera_scene_reference(cams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    centers = np.stack([_camera_center(c) for c in cams], axis=0).astype(np.float32)
    center_med = np.median(centers, axis=0).astype(np.float32)
    centered = centers - center_med[None, :]

    if centered.shape[0] >= 3:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        up = vt[-1].astype(np.float32)
    else:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    up = _normalize(up)
    if float(np.linalg.norm(up)) < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # Resolve normal sign: prefer the side where most cameras lie, then +Z as fallback.
    side = float(np.median(np.dot(centered, up)))
    if side < 0.0 or (abs(side) <= 1e-6 and up[2] < 0.0):
        up = -up

    # Estimate a look-at center by least-squares intersection of camera forward rays.
    fwd = []
    for c in cams:
        _, c2w = _camera_mats(c)
        d = _normalize(c2w[:3, 2].astype(np.float32))
        if float(np.linalg.norm(d)) < 1e-8:
            d = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        fwd.append(d)
    fwd = np.stack(fwd, axis=0)
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros((3,), dtype=np.float64)
    for p, d in zip(centers.astype(np.float64), fwd.astype(np.float64)):
        m = np.eye(3, dtype=np.float64) - np.outer(d, d)
        A += m
        b += m @ p
    try:
        center = np.linalg.solve(A, b).astype(np.float32)
    except np.linalg.LinAlgError:
        center = center_med.copy()

    # Resolve forward sign so that most rays point toward the estimated center.
    t_med = float(np.median(np.einsum("ij,ij->i", (center[None, :] - centers), fwd)))
    if t_med < 0.0:
        fwd = -fwd

    cam_dist = np.linalg.norm(centers - center[None, :], axis=1)
    d50 = float(np.percentile(cam_dist, 50.0)) if cam_dist.size > 0 else 1.0
    d90 = float(np.percentile(cam_dist, 90.0)) if cam_dist.size > 0 else max(1.0, d50)
    d50 = max(0.05, d50)
    d90 = max(d50 + 1e-6, d90)
    return centers, center, up, d50, d90


def _scene_center_radius(scene: Scene, gaussians: GaussianModel) -> Tuple[np.ndarray, float]:
    xyz = gaussians.get_xyz.detach().cpu().numpy() if gaussians.get_xyz.numel() > 0 else np.zeros((0, 3), dtype=np.float32)
    if xyz.shape[0] > 0:
        center = np.mean(xyz, axis=0).astype(np.float32)
        dist = np.linalg.norm(xyz - center[None, :], axis=1)
        # Use a robust percentile to avoid a few outliers dominating far-view distance.
        radius = float(np.percentile(dist, 90.0))
        if not np.isfinite(radius) or radius <= 1e-6:
            radius = max(0.05, float(scene.cameras_extent) * 0.3)
        return center, radius
    cams = scene.getTestCameras() or scene.getTrainCameras()
    if cams:
        ctrs = np.stack([_camera_center(c) for c in cams], axis=0)
        center = np.mean(ctrs, axis=0).astype(np.float32)
        dist = np.linalg.norm(ctrs - center[None, :], axis=1)
        radius = float(np.percentile(dist, 80.0))
        if not np.isfinite(radius) or radius <= 1e-6:
            radius = max(0.05, float(scene.cameras_extent) * 0.3)
        return center, radius
    return np.zeros((3,), dtype=np.float32), max(0.05, float(scene.cameras_extent) * 0.3)


def _connected_components(mask: np.ndarray) -> List[Tuple[int, int, int]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[Tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            area = 0
            minx = maxx = x
            miny = maxy = y
            while stack:
                cy, cx = stack.pop()
                area += 1
                if cx < minx:
                    minx = cx
                if cx > maxx:
                    maxx = cx
                if cy < miny:
                    miny = cy
                if cy > maxy:
                    maxy = cy
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append((area, maxx - minx + 1, maxy - miny + 1))
    return comps


def _spike_score(edge_mag: np.ndarray, thr: float = 0.1, valid_mask: Optional[np.ndarray] = None) -> float:
    mask = edge_mag > thr
    if valid_mask is not None:
        if valid_mask.shape != mask.shape:
            return 0.0
        mask = np.logical_and(mask, valid_mask.astype(bool))
    ds = mask[::4, ::4]
    if valid_mask is not None:
        vm = valid_mask[::4, ::4].astype(bool)
        denom = int(vm.sum())
    else:
        denom = int(ds.size)
    if denom <= 0:
        return 0.0
    comps = _connected_components(ds)
    spike_pixels = 0
    for area, w, h in comps:
        if area < 2:
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar >= 4.0 and area <= 200:
            spike_pixels += area
    return float(spike_pixels) / float(denom)


def _collect_frame_tags(out_dir: Path, num_frames: int) -> List[Tuple[int, bool]]:
    tags: List[Tuple[int, bool]] = [(0, False) for _ in range(max(0, int(num_frames)))]
    files = sorted(out_dir.glob("*.png"))
    if len(files) < num_frames:
        return tags
    out: List[Tuple[int, bool]] = []
    for i in range(num_frames):
        name = files[i].name
        m = re.search(r"_dst(\d+)", name)
        dist_idx = int(m.group(1)) if m else 0
        is_topdown = "_topdown_" in name
        out.append((dist_idx, is_topdown))
    return out


def _write_bucket_means(results: Dict[str, Any], key_prefix: str, values: List[float], frame_tags: List[Tuple[int, bool]]) -> None:
    if not values or not frame_tags or len(values) != len(frame_tags):
        return
    bucket_vals: Dict[int, List[float]] = {}
    top_vals: List[float] = []
    for v, (dist_idx, is_topdown) in zip(values, frame_tags):
        if is_topdown:
            top_vals.append(float(v))
            continue
        bucket_vals.setdefault(int(dist_idx), []).append(float(v))
    for dist_idx in sorted(bucket_vals.keys()):
        arr = bucket_vals[dist_idx]
        if not arr:
            continue
        results[f"{key_prefix}_d{dist_idx:02d}_mean"] = float(np.mean(arr))
        results[f"{key_prefix}_d{dist_idx:02d}_count"] = int(len(arr))
    if top_vals:
        results[f"{key_prefix}_topdown_mean"] = float(np.mean(top_vals))
        results[f"{key_prefix}_topdown_count"] = int(len(top_vals))


def _bucket_pairs(results: Dict[str, Any], key_prefix: str) -> List[Tuple[int, float]]:
    pairs: List[Tuple[int, float]] = []
    pat = re.compile(rf"^{re.escape(key_prefix)}_d(\d+)_mean$")
    for k, v in results.items():
        m = pat.match(str(k))
        if not m:
            continue
        try:
            idx = int(m.group(1))
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            pairs.append((idx, fv))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _head_mean(pairs: List[Tuple[int, float]], n: int, fallback: float) -> float:
    if not pairs:
        return float(fallback)
    vals = [v for _, v in pairs[: max(1, int(n))]]
    return float(np.mean(vals)) if vals else float(fallback)


def _tail_mean(pairs: List[Tuple[int, float]], n: int, fallback: float) -> float:
    if not pairs:
        return float(fallback)
    vals = [v for _, v in pairs[-max(1, int(n)) :]]
    return float(np.mean(vals)) if vals else float(fallback)


def _prepare_png_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for old in path.glob("*.png"):
        try:
            old.unlink()
        except OSError:
            pass


def _qnorm_torch(x: torch.Tensor, qlo: float = 0.02, qhi: float = 0.98, eps: float = 1e-8) -> torch.Tensor:
    if x.numel() == 0:
        return x
    lo = torch.quantile(x, qlo)
    hi = torch.quantile(x, qhi)
    if (not torch.isfinite(lo)) or (not torch.isfinite(hi)):
        return torch.zeros_like(x)
    denom = hi - lo
    if torch.abs(denom) < eps:
        return torch.zeros_like(x)
    return torch.clamp((x - lo) / denom, 0.0, 1.0)


def _build_ellipsoid_proxy_override_color(gaussians: GaussianModel) -> Optional[torch.Tensor]:
    with torch.no_grad():
        dc = gaussians.get_features_dc
        if dc is None or dc.numel() == 0:
            return None
        # Use SH-DC color to keep appearance close to the model's native colors.
        # _features_dc is (N,1,3) in this repo.
        if dc.dim() == 3 and dc.shape[1] == 1:
            dc = dc[:, 0, :]
        dc = dc.float()
        rgb = torch.clamp(dc + 0.5, 0.0, 1.0)
        return rgb


def _render_save_pair(
    cam: MiniCam,
    gaussians: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    out_dir: Path,
    file_name: str,
    frames: List[np.ndarray],
    proxy_enabled: bool,
    proxy_dir: Optional[Path],
    proxy_override_color: Optional[torch.Tensor],
    bg_mode: int,
    bg_sens_means: Optional[List[float]] = None,
    bg_sens_ratios: Optional[List[float]] = None,
    bg_sens_thr: float = 8.0 / 255.0,
) -> None:
    render_out = render(cam, gaussians, pipe, bg_color, use_trained_exp=False)
    img = render_out["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    frames.append(img)
    img_u8 = (img * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_u8).save(out_dir / file_name)

    if bg_sens_means is not None and bg_sens_ratios is not None:
        if int(bg_mode) == 0:
            img_b = img
            bg_white = torch.tensor([1, 1, 1], dtype=torch.float32, device=bg_color.device)
            rw = render(cam, gaussians, pipe, bg_white, use_trained_exp=False)
            img_w = rw["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        else:
            img_w = img
            bg_black = torch.tensor([0, 0, 0], dtype=torch.float32, device=bg_color.device)
            rb = render(cam, gaussians, pipe, bg_black, use_trained_exp=False)
            img_b = rb["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        sens = np.mean(np.abs(img_b - img_w), axis=2)
        bg_sens_means.append(float(np.mean(sens)))
        bg_sens_ratios.append(float(np.mean(sens > float(bg_sens_thr))))

    if proxy_enabled and proxy_dir is not None and proxy_override_color is not None:
        # For ellipsoid diagnostics we force near-opaque alpha so distribution is easier to inspect.
        # Keep it local to proxy pass and restore immediately after render.
        op_backup = None
        aa_backup = None
        try:
            if hasattr(gaussians, "_opacity") and gaussians._opacity is not None:
                op_backup = gaussians._opacity.detach().clone()
                with torch.no_grad():
                    gaussians._opacity.data.fill_(12.0)  # sigmoid(12) ~= 0.999994
            if hasattr(pipe, "antialiasing"):
                aa_backup = bool(pipe.antialiasing)
                pipe.antialiasing = False
            render_proxy = render(
                cam,
                gaussians,
                pipe,
                bg_color,
                scaling_modifier=0.65,
                override_color=proxy_override_color,
                use_trained_exp=False,
            )
        finally:
            if aa_backup is not None:
                pipe.antialiasing = aa_backup
            if op_backup is not None:
                with torch.no_grad():
                    gaussians._opacity.data.copy_(op_backup)
        img_proxy = render_proxy["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        # Harder depth-edge outline for clearer ellipsoid boundaries.
        depth_t = render_proxy.get("depth", None)
        if depth_t is not None:
            try:
                depth = depth_t.detach().float().squeeze().cpu().numpy()
                if depth.ndim == 2 and depth.size > 0:
                    gx, gy = np.gradient(depth)
                    g = np.hypot(gx, gy)
                    p80 = float(np.percentile(g, 80.0))
                    p98 = float(np.percentile(g, 98.0))
                    if p98 > p80:
                        e = np.clip((g - p80) / (p98 - p80), 0.0, 1.0)
                        img_proxy = np.clip(img_proxy * (1.0 - 0.65 * e[..., None]), 0.0, 1.0)
            except Exception:
                pass
        img_proxy_u8 = (img_proxy * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        stem = Path(file_name).stem
        Image.fromarray(img_proxy_u8).save(proxy_dir / f"{stem}_ellip.png")


def _render_mode_orbit(
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    bg: int,
    n: int,
    out_dir: Path,
    proxy_enabled: bool = False,
    proxy_dir: Optional[Path] = None,
    proxy_override_color: Optional[torch.Tensor] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    cams = scene.getTestCameras() or scene.getTrainCameras()
    if not cams:
        raise RuntimeError("No cameras found to derive a novel path.")
    ref_cam = cams[0]

    w2c = ref_cam.world_view_transform.transpose(0, 1).detach().cpu().numpy()
    c2w = np.linalg.inv(w2c)
    right = _normalize(c2w[:3, 0])
    up = _normalize(c2w[:3, 1])
    forward = _normalize(c2w[:3, 2])

    center = c2w[:3, 3]
    radius = max(0.05, float(scene.cameras_extent) * 0.3)

    width = int(ref_cam.image_width)
    height = int(ref_cam.image_height)
    fovy = float(ref_cam.FoVy)
    fovx = float(ref_cam.FoVx)
    znear = float(ref_cam.znear)
    zfar = float(ref_cam.zfar)

    proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    bg_color = torch.tensor([1, 1, 1] if bg == 1 else [0, 0, 0], dtype=torch.float32, device="cuda")

    _prepare_png_dir(out_dir)
    if proxy_enabled and proxy_dir is not None:
        _prepare_png_dir(proxy_dir)
    frames: List[np.ndarray] = []
    bg_sens_means: List[float] = []
    bg_sens_ratios: List[float] = []
    bg_sens_means: List[float] = []
    bg_sens_ratios: List[float] = []

    for i in range(n):
        theta = 2.0 * math.pi * (i / float(n))
        delta = radius * (math.cos(theta) * right + math.sin(theta) * forward)
        delta += (0.1 * radius * math.sin(2.0 * theta)) * up
        c2w_new = c2w.copy()
        c2w_new[:3, 3] = center + delta
        w2c_new = np.linalg.inv(c2w_new)

        world_view = torch.tensor(w2c_new).transpose(0, 1).cuda()
        full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full_proj)

        _render_save_pair(
            cam, gaussians, pipe, bg_color, out_dir, f"{i:04d}.png", frames,
            proxy_enabled, proxy_dir, proxy_override_color, int(bg),
            bg_sens_means=bg_sens_means, bg_sens_ratios=bg_sens_ratios
        )

    return frames, {
        "mode": "orbit",
        "num_frames_target": int(n),
        "bg_sensitivity_thr": float(8.0 / 255.0),
        "bg_sensitivity_means": bg_sens_means,
        "bg_sensitivity_ratios": bg_sens_ratios,
    }


def _render_mode_test_offset(
    scene: Scene,
    gaussians: GaussianModel,
    pipe,
    bg: int,
    n: int,
    out_dir: Path,
    shift_lat: float,
    shift_up: float,
    lookat_blend: float,
    seed: int,
    proxy_enabled: bool = False,
    proxy_dir: Optional[Path] = None,
    proxy_override_color: Optional[torch.Tensor] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    cams = scene.getTestCameras() or scene.getTrainCameras()
    if not cams:
        raise RuntimeError("No cameras found to derive test-offset path.")
    n_total = len(cams)
    n_use = n_total if n <= 0 else min(int(n), n_total)
    if n_use <= 0:
        raise RuntimeError("No cameras selected for test-offset mode.")
    cams = cams[:n_use]

    centers, center_scene, _up_scene, _d50, _d90 = _camera_scene_reference(cams)
    nn_dist = _nearest_neighbor_dist(centers)
    fallback = max(0.05, float(scene.cameras_extent) * 0.05)
    nn_dist = np.maximum(nn_dist, fallback)

    rng = np.random.default_rng(int(seed))
    bg_color = torch.tensor([1, 1, 1] if bg == 1 else [0, 0, 0], dtype=torch.float32, device="cuda")
    _prepare_png_dir(out_dir)
    if proxy_enabled and proxy_dir is not None:
        _prepare_png_dir(proxy_dir)
    frames: List[np.ndarray] = []

    for i, src_cam in enumerate(cams):
        _, c2w = _camera_mats(src_cam)
        right = _normalize(c2w[:3, 0])
        up = _normalize(c2w[:3, 1])
        forward = _normalize(c2w[:3, 2])
        c = c2w[:3, 3].copy()
        local = float(nn_dist[i])
        theta = 2.0 * math.pi * (float(i) / float(max(1, n_use)))
        jitter = rng.uniform(-0.15, 0.15)
        delta = local * shift_lat * ((math.cos(theta + jitter) * right) + (math.sin(theta - jitter) * forward))
        delta += local * shift_up * (math.sin(2.0 * theta + jitter) * up)
        c_new = c + delta

        if lookat_blend > 0.0:
            c2w_look = _build_c2w_lookat(c_new, center_scene, up)
            R_old = c2w[:3, :3]
            R_new = (1.0 - lookat_blend) * R_old + lookat_blend * c2w_look[:3, :3]
            R_new = _orthonormalize(R_new.astype(np.float32))
            c2w_new = np.eye(4, dtype=np.float32)
            c2w_new[:3, :3] = R_new
            c2w_new[:3, 3] = c_new.astype(np.float32)
        else:
            c2w_new = c2w.astype(np.float32)
            c2w_new[:3, 3] = c_new.astype(np.float32)

        w2c_new = np.linalg.inv(c2w_new)
        width = int(src_cam.image_width)
        height = int(src_cam.image_height)
        fovy = float(src_cam.FoVy)
        fovx = float(src_cam.FoVx)
        znear = float(src_cam.znear)
        zfar = float(src_cam.zfar)
        proj = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
        world_view = torch.tensor(w2c_new, dtype=torch.float32, device="cuda").transpose(0, 1)
        full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view, full_proj)

        _render_save_pair(
            cam, gaussians, pipe, bg_color, out_dir, f"{i:04d}.png", frames,
            proxy_enabled, proxy_dir, proxy_override_color, int(bg),
            bg_sens_means=bg_sens_means, bg_sens_ratios=bg_sens_ratios
        )

    meta = {
        "mode": "test_offset",
        "num_frames_target": int(n),
        "num_frames_selected": int(n_use),
        "num_test_cameras": int(n_total),
        "shift_lat": float(shift_lat),
        "shift_up": float(shift_up),
        "lookat_blend": float(lookat_blend),
        "seed": int(seed),
        "bg_sensitivity_thr": float(8.0 / 255.0),
        "bg_sensitivity_means": bg_sens_means,
        "bg_sensitivity_ratios": bg_sens_ratios,
    }
    return frames, meta


def _render_mode_grid72(scene: Scene, gaussians: GaussianModel, pipe, bg: int, out_dir: Path,
                        near_scale: float, far_scale: float, far_blend: float,
                        far_cover_pad: float, far_cam_mult: float,
                        azimuth_count: int, pitch_list: List[float], dist_factors: List[float], include_topdown: bool,
                        jitter: bool, pos_jitter: float, ang_jitter_deg: float, seed: int,
                        proxy_enabled: bool = False, proxy_dir: Optional[Path] = None,
                        proxy_override_color: Optional[torch.Tensor] = None) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    cams = scene.getTestCameras() or scene.getTrainCameras()
    if not cams:
        raise RuntimeError("No cameras found to derive grid72 path.")
    ref_cam = cams[0]
    centers, center_scene, up_guess, cam_d50, cam_d90 = _camera_scene_reference(cams)
    # Camera image-up convention in this repo is typically opposite to geometric "up".
    # Use a camera-space up hint to avoid globally upside-down grid72 renders.
    up_cam = -up_guess
    ref_ups = []
    for c in cams:
        _, c2w_ref = _camera_mats(c)
        ref_ups.append(_normalize(c2w_ref[:3, 1].astype(np.float32)))
    ref_ups = np.stack(ref_ups, axis=0)
    # Use a global target up to keep all rendered frames consistently oriented.
    up_ref = up_cam.copy()
    if float(np.median(np.dot(ref_ups, up_ref))) < 0.0:
        up_ref = -up_ref
    azimuth_count = max(1, int(azimuth_count))
    pitches = sorted({float(np.clip(p, 5.0, 89.9)) for p in pitch_list})
    if not pitches:
        pitches = [30.0, 60.0]
    factors = sorted({float(max(0.0, f)) for f in dist_factors})
    if not factors:
        factors = [0.0, 0.5, 1.0]
    far_blend = float(np.clip(far_blend, 0.1, 1.0))
    far_cover_pad = max(1.0, float(far_cover_pad))
    far_cam_mult = max(1.0, float(far_cam_mult))

    near_d = max(0.05, near_scale * cam_d50)
    width = int(ref_cam.image_width)
    height = int(ref_cam.image_height)
    fovy = float(ref_cam.FoVy)
    fovx = float(ref_cam.FoVx)
    # Strict viewpoint locking: derive distances only from camera geometry.
    far_cover_raw = max(far_scale * cam_d50, cam_d90 * 1.05)
    far_cover_d = min(far_cover_raw * far_cover_pad, cam_d90 * far_cam_mult)
    far_cover_d = max(far_cover_d, near_d + 1e-3)
    far_anchor = max(near_d + far_blend * (far_cover_d - near_d), near_d + 1e-3)
    # If user requests factors beyond 1.0, switch to cover-span mapping so that
    # larger factors can genuinely extend farther (not stuck in compressed anchor span).
    use_cover_span = any(float(f) > 1.0 for f in factors)
    span = (far_cover_d - near_d) if use_cover_span else (far_anchor - near_d)
    dists = [near_d + f * span for f in factors]

    # Build horizontal basis from camera spread projected to the plane.
    centered = centers - center_scene[None, :]
    spread = centered - np.dot(centered, up_guess)[:, None] * up_guess[None, :]
    if spread.shape[0] >= 3:
        _, _, vt_h = np.linalg.svd(spread, full_matrices=False)
        h1 = _normalize(vt_h[0].astype(np.float32))
    else:
        h1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(h1, up_guess))) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(tmp, up_guess))) > 0.95:
            tmp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        h1 = _normalize(tmp - float(np.dot(tmp, up_guess)) * up_guess)
    h2 = _normalize(np.cross(up_guess, h1))

    # Use observed camera azimuths to avoid generating mostly out-of-coverage novel views.
    planar_x = np.dot(spread, h1)
    planar_y = np.dot(spread, h2)
    az_all = (np.degrees(np.arctan2(planar_y, planar_x)) + 360.0) % 360.0
    az_all = np.sort(az_all)
    if az_all.size >= azimuth_count:
        pick = np.linspace(0, az_all.size - 1, num=azimuth_count)
        pick = np.round(pick).astype(np.int32)
        azimuths = [float(az_all[int(i)]) for i in pick]
    else:
        azimuths = [k * (360.0 / float(azimuth_count)) for k in range(azimuth_count)]
    azimuths = sorted(azimuths)

    znear = float(ref_cam.znear)
    zfar = float(ref_cam.zfar)
    zfar_novel = max(zfar, far_anchor * 2.5, cam_d90 * 2.0)
    proj = getProjectionMatrix(znear=znear, zfar=zfar_novel, fovX=fovx, fovY=fovy).transpose(0, 1).cuda()
    bg_color = torch.tensor([1, 1, 1] if bg == 1 else [0, 0, 0], dtype=torch.float32, device="cuda")

    rng = np.random.default_rng(int(seed))
    _prepare_png_dir(out_dir)
    if proxy_enabled and proxy_dir is not None:
        _prepare_png_dir(proxy_dir)
    frames: List[np.ndarray] = []
    bg_sens_means: List[float] = []
    bg_sens_ratios: List[float] = []
    idx = 0
    roll_flip_count = 0
    for azi_i, azi in enumerate(azimuths):
        for pit_i, pitch in enumerate(pitches):
            for dist_i, dist in enumerate(dists):
                a = float(azi)
                p = float(pitch)
                d = float(dist)
                if jitter:
                    a += float(rng.uniform(-ang_jitter_deg, ang_jitter_deg))
                    p += float(rng.uniform(-ang_jitter_deg, ang_jitter_deg))
                    p = float(np.clip(p, 20.0, 89.9))
                    d *= (1.0 + float(rng.uniform(-pos_jitter, pos_jitter)))
                    d = max(0.05, d)
                ar = math.radians(a)
                pr = math.radians(p)
                horiz = (math.cos(ar) * h1) + (math.sin(ar) * h2)
                dir_center_to_cam = (math.cos(pr) * horiz) + (math.sin(pr) * up_guess)
                dir_center_to_cam = _normalize(dir_center_to_cam)
                cam_pos = center_scene + d * dir_center_to_cam
                c2w_new = _build_c2w_lookat(cam_pos, center_scene, up_cam)
                # Keep orientation globally consistent across all novel views.
                if float(np.dot(_normalize(c2w_new[:3, 1]), up_ref)) < 0.0:
                    c2w_new[:3, 0] *= -1.0
                    c2w_new[:3, 1] *= -1.0
                    roll_flip_count += 1
                w2c_new = np.linalg.inv(c2w_new)

                world_view = torch.tensor(w2c_new, dtype=torch.float32, device="cuda").transpose(0, 1)
                full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
                cam = MiniCam(width, height, fovy, fovx, znear, zfar_novel, world_view, full_proj)

                azi_tag = int(round((a % 360.0)))
                pit_tag = int(round(p))
                # Naming encodes deterministic inspection order:
                # direction index -> pitch index -> distance index.
                file_name = (
                    f"{idx:04d}_dir{azi_i:02d}_pit{pit_i:02d}_dst{dist_i:02d}"
                    f"_a{azi_tag:03d}_p{pit_tag:02d}_de{d:.2f}.png"
                )
                _render_save_pair(
                    cam, gaussians, pipe, bg_color, out_dir, file_name, frames,
                    proxy_enabled, proxy_dir, proxy_override_color, int(bg),
                    bg_sens_means=bg_sens_means, bg_sens_ratios=bg_sens_ratios
                )
                idx += 1

    if include_topdown:
        # Top-down at one azimuth, but for all configured distances.
        top_dir = up_guess
        top_dir = _normalize(top_dir)
        if float(np.linalg.norm(top_dir)) < 1e-8:
            top_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        # Fix top-down orientation using a single deterministic azimuth reference.
        top_az = float(azimuths[0]) if len(azimuths) > 0 else 0.0
        top_ar = math.radians(top_az)
        top_right_hint = _normalize((math.cos(top_ar) * h1) + (math.sin(top_ar) * h2))
        top_forward = _normalize(-top_dir)
        for dist_i, top_dist in enumerate(dists):
            top_cam_pos = center_scene + top_dist * top_dir
            c2w_new = _build_c2w_from_forward_right(top_cam_pos, top_forward, top_right_hint)
            w2c_new = np.linalg.inv(c2w_new)

            world_view = torch.tensor(w2c_new, dtype=torch.float32, device="cuda").transpose(0, 1)
            full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
            cam = MiniCam(width, height, fovy, fovx, znear, zfar_novel, world_view, full_proj)

            file_name = f"{idx:04d}_topdown_a{int(round(top_az))%360:03d}_dst{dist_i:02d}_de{top_dist:.2f}.png"
            _render_save_pair(
                cam, gaussians, pipe, bg_color, out_dir, file_name, frames,
                proxy_enabled, proxy_dir, proxy_override_color, int(bg),
                bg_sens_means=bg_sens_means, bg_sens_ratios=bg_sens_ratios
            )
            idx += 1

    meta = {
        "mode": "grid",
        "num_frames_target": int(len(azimuths) * len(pitches) * len(dists) + (len(dists) if include_topdown else 0)),
        "num_frames_selected": int(len(frames)),
        "scene_center": [float(center_scene[0]), float(center_scene[1]), float(center_scene[2])],
        "far_cover_distance": float(far_cover_d),
        "near_distance": float(near_d),
        "far_distance": float((near_d + 1.0 * span)),
        "distances": [float(d) for d in dists],
        "distance_factors": [float(f) for f in factors],
        "up_guess": [float(up_guess[0]), float(up_guess[1]), float(up_guess[2])],
        "up_ref": [float(up_ref[0]), float(up_ref[1]), float(up_ref[2])],
        "camera_distance_p50": float(cam_d50),
        "camera_distance_p90": float(cam_d90),
        "zfar_ref": float(zfar),
        "zfar_novel": float(zfar_novel),
        "azimuth_strategy": "observed_from_test_cameras" if az_all.size >= azimuth_count else "uniform_360_fallback",
        "azimuths_deg": [float(a) for a in azimuths],
        "azimuth_count": int(azimuth_count),
        "pitches_deg": [float(p) for p in pitches],
        "include_topdown": bool(include_topdown),
        "topdown_azimuth_deg": float(top_az) if include_topdown and len(azimuths) > 0 else 0.0,
        "ordering": "direction_then_pitch_then_distance_then_topdown",
        "distance_policy": "strict_locked: near=near_scale*p50; cover_far=camera_only; if any factor>1 use cover span else anchor span; d=near+factor*span",
        "distance_span_mode": "cover" if use_cover_span else "anchor",
        "roll_flip_count": int(roll_flip_count),
        "near_scale": float(near_scale),
        "far_scale": float(far_scale),
        "far_blend": float(far_blend),
        "far_cover_pad": float(far_cover_pad),
        "far_cam_mult": float(far_cam_mult),
        "jitter": bool(jitter),
        "pos_jitter": float(pos_jitter),
        "ang_jitter_deg": float(ang_jitter_deg),
        "seed": int(seed),
        "bg_sensitivity_thr": float(8.0 / 255.0),
        "bg_sensitivity_means": bg_sens_means,
        "bg_sensitivity_ratios": bg_sens_ratios,
    }
    return frames, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel-view metrics parameters")
    model = ModelParams(parser, sentinel=True)
    pipe = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--N", type=int, default=60, help="Number of novel views (default: 60; mode=test_offset and N<=0 uses all test cams)")
    parser.add_argument("--mode", type=str, default="orbit", choices=["orbit", "test_offset", "grid72"],
                        help="Novel-view generation mode (default: orbit)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for jittered modes")
    parser.add_argument("--bg", type=int, default=0, choices=[0, 1], help="Background color: 0=black, 1=white")
    parser.add_argument("--edge_thr", type=float, default=0.1, help="Edge threshold for spike score (default: 0.1)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: <model>/novel_views)")
    # test-offset mode
    parser.add_argument("--test_shift_lat", type=float, default=0.15, help="test_offset lateral shift factor wrt local camera spacing")
    parser.add_argument("--test_shift_up", type=float, default=0.03, help="test_offset vertical shift factor wrt local camera spacing")
    parser.add_argument("--test_lookat_blend", type=float, default=0.35, help="test_offset look-at center blend in [0,1]")
    # grid72 mode (configurable grid)
    parser.add_argument("--grid_near_scale", type=float, default=0.6, help="grid72 near distance scale * camera_distance_p50")
    parser.add_argument("--grid_far_scale", type=float, default=1.0, help="grid72 far distance scale * camera_distance_p50")
    parser.add_argument("--grid_far_blend", type=float, default=0.4, help="Blend factor from near->cover_far for practical far distance (default: 0.4)")
    parser.add_argument("--grid_far_cover_pad", type=float, default=1.08, help="Padding on FOV cover distance (default: 1.08)")
    parser.add_argument("--grid_far_cam_mult", type=float, default=2.0, help="Upper cap multiplier over camera_distance_p90 for far cover (default: 2.0)")
    parser.add_argument("--grid_azimuth_count", type=int, default=8, help="Number of azimuth directions (default: 8)")
    parser.add_argument("--grid_pitch_list", type=str, default="30,60", help="Comma-separated pitch degrees, e.g. 30,60 (default: 30,60)")
    parser.add_argument("--grid_distance_factors", type=str, default="0,0.5,1", help="Comma-separated factors >=0 between near and far, e.g. 0,0.5,1,1.5")
    parser.add_argument("--grid_no_topdown", action="store_true", default=False, help="Disable final top-down frame")
    parser.add_argument("--grid_jitter", action="store_true", default=False, help="Enable small random jitter in grid72")
    parser.add_argument("--grid_pos_jitter", type=float, default=0.05, help="grid72 distance jitter ratio")
    parser.add_argument("--grid_ang_jitter_deg", type=float, default=2.0, help="grid72 angle jitter in degrees")
    parser.add_argument("--dump_ellipsoid_proxy", action="store_true", default=False,
                        help="Dump per-view ellipsoid proxy images with exact same camera poses (default: off)")
    parser.add_argument("--ellipsoid_proxy_dir", type=str, default="",
                        help="Output directory for ellipsoid proxy images (default: <out_dir>_ellip)")
    args = get_combined_args(parser)

    safe_state(False)

    dataset = model.extract(args)
    pipeline = pipe.extract(args)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if args.mode == "orbit":
            out_dir = Path(args.model_path) / "novel_views"
        elif args.mode == "test_offset":
            out_dir = Path(args.model_path) / "novel_views_test_offset"
        else:
            out_dir = Path(args.model_path) / "novel_views_grid"

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    proxy_enabled = bool(args.dump_ellipsoid_proxy)
    if args.ellipsoid_proxy_dir:
        proxy_out_dir = Path(args.ellipsoid_proxy_dir)
    else:
        proxy_out_dir = out_dir.parent / f"{out_dir.name}_ellip"
    proxy_override_color = _build_ellipsoid_proxy_override_color(gaussians) if proxy_enabled else None

    with torch.no_grad():
        if args.mode == "orbit":
            frames, mode_meta = _render_mode_orbit(
                scene, gaussians, pipeline, args.bg, args.N, out_dir,
                proxy_enabled=proxy_enabled, proxy_dir=proxy_out_dir, proxy_override_color=proxy_override_color
            )
        elif args.mode == "test_offset":
            frames, mode_meta = _render_mode_test_offset(
                scene, gaussians, pipeline, args.bg, args.N, out_dir,
                shift_lat=float(args.test_shift_lat),
                shift_up=float(args.test_shift_up),
                lookat_blend=float(np.clip(args.test_lookat_blend, 0.0, 1.0)),
                seed=int(args.seed),
                proxy_enabled=proxy_enabled, proxy_dir=proxy_out_dir, proxy_override_color=proxy_override_color,
            )
        else:
            grid_pitches = _parse_float_list(args.grid_pitch_list, [30.0, 60.0], clip_min=5.0, clip_max=89.9)
            grid_dist_factors = _parse_float_list(args.grid_distance_factors, [0.0, 0.5, 1.0], clip_min=0.0, clip_max=None)
            frames, mode_meta = _render_mode_grid72(
                scene, gaussians, pipeline, args.bg, out_dir,
                near_scale=float(args.grid_near_scale),
                far_scale=float(args.grid_far_scale),
                far_blend=float(args.grid_far_blend),
                far_cover_pad=float(args.grid_far_cover_pad),
                far_cam_mult=float(args.grid_far_cam_mult),
                azimuth_count=int(args.grid_azimuth_count),
                pitch_list=grid_pitches,
                dist_factors=grid_dist_factors,
                include_topdown=not bool(args.grid_no_topdown),
                jitter=bool(args.grid_jitter),
                pos_jitter=float(max(0.0, args.grid_pos_jitter)),
                ang_jitter_deg=float(max(0.0, args.grid_ang_jitter_deg)),
                seed=int(args.seed),
                proxy_enabled=proxy_enabled, proxy_dir=proxy_out_dir, proxy_override_color=proxy_override_color,
            )

    frame_tags = _collect_frame_tags(out_dir, len(frames))
    bg_leaks: List[float] = []
    spike_scores_all: List[float] = []
    spike_scores_air: List[float] = []
    spike_scores_air_thr: List[float] = []
    tex_tenengrad: List[float] = []
    tex_grad_p90: List[float] = []
    tex_edge_density: List[float] = []
    edge_anisotropy: List[float] = []
    air_mask_ratios: List[float] = []
    air_edge_means: List[float] = []
    air_hf_means: List[float] = []
    air_blob_ratios: List[float] = []
    air_fog_ratios: List[float] = []
    air_artifact_scores: List[float] = []
    grays: List[np.ndarray] = []

    for img in frames:
        gray = _rgb_to_gray(img)
        grays.append(gray)
        edge_mag = _sobel_mag(gray)
        ten, gp90, edens = _texture_metrics(gray)
        ean = _edge_anisotropy(gray)
        air_ratio, air_edge, air_hf, air_blob, air_fog, air_score = _air_cleanliness_metrics(gray, edge_mag)
        air_mask = _air_mask_from_render(gray, edge_mag)
        bg_leaks.append(_bg_leak_ratio(img, args.bg))
        spike_scores_all.append(_spike_score(edge_mag, thr=args.edge_thr))
        if int(np.sum(air_mask)) > 0:
            air_thr = max(float(args.edge_thr) * 0.35, float(np.percentile(edge_mag[air_mask], 85.0)))
        else:
            air_thr = float(args.edge_thr)
        spike_scores_air_thr.append(float(air_thr))
        spike_scores_air.append(_spike_score(edge_mag, thr=air_thr, valid_mask=air_mask))
        tex_tenengrad.append(ten)
        tex_grad_p90.append(gp90)
        tex_edge_density.append(edens)
        edge_anisotropy.append(ean)
        air_mask_ratios.append(air_ratio)
        air_edge_means.append(air_edge)
        air_hf_means.append(air_hf)
        air_blob_ratios.append(air_blob)
        air_fog_ratios.append(air_fog)
        air_artifact_scores.append(air_score)

    flickers: List[float] = []
    for i in range(1, len(grays)):
        flickers.append(float(np.mean(np.abs(grays[i] - grays[i - 1]))))

    bg_sens_means = mode_meta.pop("bg_sensitivity_means", [])
    bg_sens_ratios = mode_meta.pop("bg_sensitivity_ratios", [])
    bg_sens_thr = float(mode_meta.get("bg_sensitivity_thr", 8.0 / 255.0))

    results = {
        "mode": str(args.mode),
        "num_frames": int(len(frames)),
        "bg": int(args.bg),
        "edge_thr": float(args.edge_thr),
        "BgLeakRatio_mean": float(np.mean(bg_leaks)) if bg_leaks else 0.0,
        "SpikeScore_mean": float(np.mean(spike_scores_all)) if spike_scores_all else 0.0,
        "SpikeScore_all_mean": float(np.mean(spike_scores_all)) if spike_scores_all else 0.0,
        "SpikeScore_air_mean": float(np.mean(spike_scores_air)) if spike_scores_air else 0.0,
        "SpikeScore_air_thr_mean": float(np.mean(spike_scores_air_thr)) if spike_scores_air_thr else float(args.edge_thr),
        "BgSensitivity_mean": float(np.mean(bg_sens_means)) if bg_sens_means else 0.0,
        "BgSensitivityRatio": float(np.mean(bg_sens_ratios)) if bg_sens_ratios else 0.0,
        "BgSensitivity_thr": float(bg_sens_thr),
        "TemporalFlicker_mean": float(np.mean(flickers)) if flickers else 0.0,
        "TextureTenengrad_mean": float(np.mean(tex_tenengrad)) if tex_tenengrad else 0.0,
        "TextureGradP90_mean": float(np.mean(tex_grad_p90)) if tex_grad_p90 else 0.0,
        "TextureEdgeDensityAdaptive_mean": float(np.mean(tex_edge_density)) if tex_edge_density else 0.0,
        "EdgeAnisotropy_mean": float(np.mean(edge_anisotropy)) if edge_anisotropy else 0.0,
        "AirMaskRatio_mean": float(np.mean(air_mask_ratios)) if air_mask_ratios else 0.0,
        "AirEdgeMean_mean": float(np.mean(air_edge_means)) if air_edge_means else 0.0,
        "AirHFMean_mean": float(np.mean(air_hf_means)) if air_hf_means else 0.0,
        "AirBlobRatio_mean": float(np.mean(air_blob_ratios)) if air_blob_ratios else 0.0,
        "AirFogRatio_mean": float(np.mean(air_fog_ratios)) if air_fog_ratios else 0.0,
        "AirArtifactScore_mean": float(np.mean(air_artifact_scores)) if air_artifact_scores else 0.0,
        "out_dir": str(out_dir),
        "ellipsoid_proxy_enabled": bool(proxy_enabled),
        "ellipsoid_proxy_out_dir": str(proxy_out_dir) if proxy_enabled else "",
        "ellipsoid_proxy_opacity_mode": "opaque" if proxy_enabled else "off",
    }
    results.update(mode_meta)
    if "distance_factors" in mode_meta and isinstance(mode_meta["distance_factors"], list):
        for i, v in enumerate(mode_meta["distance_factors"]):
            try:
                results[f"DistanceFactor_d{i:02d}"] = float(v)
            except (TypeError, ValueError):
                continue
    _write_bucket_means(results, "TextureTenengrad", tex_tenengrad, frame_tags)
    _write_bucket_means(results, "TextureGradP90", tex_grad_p90, frame_tags)
    _write_bucket_means(results, "TextureEdgeDensityAdaptive", tex_edge_density, frame_tags)
    _write_bucket_means(results, "EdgeAnisotropy", edge_anisotropy, frame_tags)
    _write_bucket_means(results, "BgLeakRatio", bg_leaks, frame_tags)
    _write_bucket_means(results, "SpikeScore", spike_scores_all, frame_tags)
    _write_bucket_means(results, "SpikeScore_all", spike_scores_all, frame_tags)
    _write_bucket_means(results, "SpikeScore_air", spike_scores_air, frame_tags)
    if len(bg_sens_means) == len(frame_tags):
        _write_bucket_means(results, "BgSensitivity", bg_sens_means, frame_tags)
    if len(bg_sens_ratios) == len(frame_tags):
        _write_bucket_means(results, "BgSensitivityRatio", bg_sens_ratios, frame_tags)
    _write_bucket_means(results, "AirEdgeMean", air_edge_means, frame_tags)
    _write_bucket_means(results, "AirHFMean", air_hf_means, frame_tags)
    _write_bucket_means(results, "AirBlobRatio", air_blob_ratios, frame_tags)
    _write_bucket_means(results, "AirFogRatio", air_fog_ratios, frame_tags)
    _write_bucket_means(results, "AirArtifactScore", air_artifact_scores, frame_tags)

    # SGF-oriented novel-view quality summary:
    # - StructureNearScore: emphasize near/mid clarity.
    # - CleanFarScore: emphasize far-field artifact suppression.
    # - NovelQualityScore: balanced summary (higher is better).
    tex_pairs = _bucket_pairs(results, "TextureTenengrad")
    ani_pairs = _bucket_pairs(results, "EdgeAnisotropy")
    air_pairs = _bucket_pairs(results, "AirArtifactScore")
    spike_pairs = _bucket_pairs(results, "SpikeScore_air")
    bg_pairs = _bucket_pairs(results, "BgSensitivityRatio")
    bg_leak_pairs = _bucket_pairs(results, "BgLeakRatio")

    tex_near2 = _head_mean(tex_pairs, 2, float(np.mean(tex_tenengrad)) if tex_tenengrad else 0.0)
    ani_near2 = _head_mean(ani_pairs, 2, float(np.mean(edge_anisotropy)) if edge_anisotropy else 0.0)
    air_far2 = _tail_mean(air_pairs, 2, float(np.mean(air_artifact_scores)) if air_artifact_scores else 0.0)
    spike_far2 = _tail_mean(spike_pairs, 2, float(np.mean(spike_scores_air)) if spike_scores_air else 0.0)
    bg_far2 = _tail_mean(bg_pairs, 2, float(np.mean(bg_sens_ratios)) if bg_sens_ratios else 0.0)
    bg_leak_far2 = _tail_mean(bg_leak_pairs, 2, float(np.mean(bg_leaks)) if bg_leaks else 0.0)
    flick_mean = float(np.mean(flickers)) if flickers else 0.0

    # Score shaping to [0,1] with fixed transforms (cross-experiment comparable on same dataset).
    s_tex = 1.0 - math.exp(-0.04 * max(0.0, tex_near2))
    s_ani = float(np.clip((ani_near2 - 0.10) / 0.45, 0.0, 1.0))
    structure_near_score = float(np.clip(0.75 * s_tex + 0.25 * s_ani, 0.0, 1.0))

    c_air = math.exp(-2.4 * max(0.0, air_far2))
    c_spike = math.exp(-120.0 * max(0.0, spike_far2))
    c_bg = math.exp(-10.0 * max(0.0, bg_far2))
    c_bg_leak = math.exp(-30.0 * max(0.0, bg_leak_far2))
    c_flick = math.exp(-3.0 * max(0.0, flick_mean))
    clean_far_score = float(np.clip(0.50 * c_air + 0.20 * c_spike + 0.15 * c_bg + 0.05 * c_bg_leak + 0.10 * c_flick, 0.0, 1.0))

    novel_quality_score = float(np.clip(0.60 * structure_near_score + 0.40 * clean_far_score, 0.0, 1.0))

    results["SGF_TextureNear2_mean"] = float(tex_near2)
    results["SGF_EdgeAnisotropyNear2_mean"] = float(ani_near2)
    results["SGF_AirArtifactFar2_mean"] = float(air_far2)
    results["SGF_SpikeFar2_mean"] = float(spike_far2)
    results["SGF_BgLeakFar2_mean"] = float(bg_far2)
    results["SGF_BgLeakLegacyFar2_mean"] = float(bg_leak_far2)
    results["SGF_StructureNearScore"] = structure_near_score
    results["SGF_CleanFarScore"] = clean_far_score
    results["SGF_NovelQualityScore"] = novel_quality_score

    if args.mode == "orbit":
        out_json = out_dir / "novel_view_metrics.json"
    elif args.mode == "grid72":
        out_json = out_dir / "novel_view_metrics_grid.json"
    else:
        out_json = out_dir / f"novel_view_metrics_{args.mode}.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
