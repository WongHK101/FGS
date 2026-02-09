#
# Novel-view metrics: render a small novel camera path and compute no-ref cleanliness proxies.
#
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

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


def _spike_score(edge_mag: np.ndarray, thr: float = 0.1) -> float:
    mask = edge_mag > thr
    ds = mask[::4, ::4]
    comps = _connected_components(ds)
    spike_pixels = 0
    for area, w, h in comps:
        if area < 2:
            continue
        ar = max(w, h) / max(1, min(w, h))
        if ar >= 4.0 and area <= 200:
            spike_pixels += area
    return float(spike_pixels) / float(ds.size)


def _render_novel_views(scene: Scene, gaussians: GaussianModel, pipe, bg: int, n: int, out_dir: Path) -> List[np.ndarray]:
    cams = scene.getTestCameras() or scene.getTrainCameras()
    if not cams:
        raise RuntimeError("No cameras found to derive a novel path.")
    ref_cam = cams[0]

    w2c = ref_cam.world_view_transform.transpose(0, 1).cpu().numpy()
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

    out_dir.mkdir(parents=True, exist_ok=True)
    frames: List[np.ndarray] = []

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

        render_out = render(cam, gaussians, pipe, bg_color, use_trained_exp=False)
        img = render_out["render"].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        frames.append(img)

        img_u8 = (img * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_u8).save(out_dir / f"{i:04d}.png")

    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Novel-view metrics parameters")
    model = ModelParams(parser, sentinel=True)
    pipe = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--N", type=int, default=60, help="Number of novel views (default: 60)")
    parser.add_argument("--bg", type=int, default=0, choices=[0, 1], help="Background color: 0=black, 1=white")
    parser.add_argument("--edge_thr", type=float, default=0.1, help="Edge threshold for spike score (default: 0.1)")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory (default: <model>/novel_views)")
    args = get_combined_args(parser)

    safe_state(False)

    dataset = model.extract(args)
    pipeline = pipe.extract(args)

    out_dir = Path(args.out_dir) if args.out_dir else (Path(args.model_path) / "novel_views")

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    with torch.no_grad():
        frames = _render_novel_views(scene, gaussians, pipeline, args.bg, args.N, out_dir)

    bg_leaks: List[float] = []
    spike_scores: List[float] = []
    grays: List[np.ndarray] = []

    for img in frames:
        gray = _rgb_to_gray(img)
        grays.append(gray)
        edge_mag = _sobel_mag(gray)
        bg_leaks.append(_bg_leak_ratio(img, args.bg))
        spike_scores.append(_spike_score(edge_mag, thr=args.edge_thr))

    flickers: List[float] = []
    for i in range(1, len(grays)):
        flickers.append(float(np.mean(np.abs(grays[i] - grays[i - 1]))))

    results = {
        "num_frames": int(len(frames)),
        "bg": int(args.bg),
        "edge_thr": float(args.edge_thr),
        "BgLeakRatio_mean": float(np.mean(bg_leaks)) if bg_leaks else 0.0,
        "SpikeScore_mean": float(np.mean(spike_scores)) if spike_scores else 0.0,
        "TemporalFlicker_mean": float(np.mean(flickers)) if flickers else 0.0,
        "out_dir": str(out_dir),
    }

    out_json = out_dir / "novel_view_metrics.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
