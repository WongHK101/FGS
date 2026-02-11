#
# Extra evaluation metrics for cleanliness / texture clarity / alignment robustness.
# Does NOT modify or depend on metrics.py behavior.
#
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


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


def _texture_metrics_lcn(gray: np.ndarray) -> Tuple[float, float]:
    g = _local_contrast_norm(gray)
    gx, gy = _sobel_xy(g)
    gm = np.hypot(gx, gy)
    tenengrad = float(np.mean(gx * gx + gy * gy))
    thr = float(np.mean(gm) + 0.75 * np.std(gm))
    thr = max(thr, 1e-4)
    edge_density = float(np.mean(gm > thr))
    return tenengrad, edge_density


def _laplacian(gray: np.ndarray) -> np.ndarray:
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype=np.float32)
    return _conv3x3(gray, k)


def _psnr_from_mse(mse: float, data_range: float = 1.0) -> float:
    if mse <= 0.0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _aligned_psnr_rgb(render: np.ndarray, gt: np.ndarray, k: int) -> float:
    h, w, _ = render.shape
    best = -float("inf")

    for dy in range(-k, k + 1):
        if dy >= 0:
            ry0, gy0, hh = dy, 0, h - dy
        else:
            ry0, gy0, hh = 0, -dy, h + dy
        if hh <= 0:
            continue
        for dx in range(-k, k + 1):
            if dx >= 0:
                rx0, gx0, ww = dx, 0, w - dx
            else:
                rx0, gx0, ww = 0, -dx, w + dx
            if ww <= 0:
                continue
            r = render[ry0:ry0 + hh, rx0:rx0 + ww, :]
            g = gt[gy0:gy0 + hh, gx0:gx0 + ww, :]
            mse = float(np.mean((r - g) ** 2))
            psnr = _psnr_from_mse(mse, 1.0)
            if psnr > best:
                best = psnr

    return best


def _aligned_psnr_gray(render: np.ndarray, gt: np.ndarray, k: int) -> float:
    h, w = render.shape
    best = -float("inf")

    for dy in range(-k, k + 1):
        if dy >= 0:
            ry0, gy0, hh = dy, 0, h - dy
        else:
            ry0, gy0, hh = 0, -dy, h + dy
        if hh <= 0:
            continue
        for dx in range(-k, k + 1):
            if dx >= 0:
                rx0, gx0, ww = dx, 0, w - dx
            else:
                rx0, gx0, ww = 0, -dx, w + dx
            if ww <= 0:
                continue
            r = render[ry0:ry0 + hh, rx0:rx0 + ww]
            g = gt[gy0:gy0 + hh, gx0:gx0 + ww]
            mse = float(np.mean((r - g) ** 2))
            psnr = _psnr_from_mse(mse, 1.0)
            if psnr > best:
                best = psnr

    return best


def _bg_leak_ratio(render: np.ndarray, bg: int, thr: float = 3.0 / 255.0) -> float:
    if bg == 0:
        mask = np.max(render, axis=2) <= thr
    else:
        mask = np.min(render, axis=2) >= (1.0 - thr)
    return float(np.mean(mask))


def _air_mask_from_gt(gray_g: np.ndarray, edge_g: np.ndarray) -> np.ndarray:
    # Empty-air proxy from GT: low-intensity + low-edge regions.
    q_gray = float(np.percentile(gray_g, 45.0))
    q_edge = float(np.percentile(edge_g, 40.0))
    mask = np.logical_and(gray_g <= q_gray, edge_g <= q_edge)
    # Keep metric stable when scene is mostly close-up.
    if float(np.mean(mask)) < 0.05:
        mask = gray_g <= float(np.percentile(gray_g, 30.0))
    return mask


def _masked_mean(arr: np.ndarray, mask: np.ndarray) -> float:
    if arr.shape != mask.shape:
        return 0.0
    cnt = int(mask.sum())
    if cnt <= 0:
        return 0.0
    return float(arr[mask].mean())


def _corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    if a.size == 0 or b.size == 0:
        return 0.0
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa < 1e-8 or sb < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _edge_f1(edge_r: np.ndarray, edge_g: np.ndarray, thr: float) -> float:
    pred = edge_r >= thr
    gt = edge_g >= thr
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    denom_p = tp + fp
    denom_r = tp + fn
    if denom_p == 0 or denom_r == 0:
        return 0.0
    precision = tp / denom_p
    recall = tp / denom_r
    denom_f1 = precision + recall
    if denom_f1 <= 0:
        return 0.0
    return 2.0 * precision * recall / denom_f1


def _format_float(v: float) -> str:
    if math.isfinite(v):
        return f"{v:.6f}"
    return str(v)


def _iter_pairs(renders_dir: Path, gt_dir: Path) -> Iterable[Tuple[str, np.ndarray, np.ndarray]]:
    names = sorted([f for f in os.listdir(renders_dir) if (renders_dir / f).is_file()])
    for name in names:
        gt_path = gt_dir / name
        if not gt_path.exists():
            continue
        yield name, _load_rgb(renders_dir / name), _load_rgb(gt_path)


def evaluate(model_paths: List[str], k: int, bg: int, edge_thr: float, save_json: bool) -> None:
    full_dict = {}

    for scene_dir in model_paths:
        print("")
        print("Scene:", scene_dir)
        test_dir = Path(scene_dir) / "test"
        if not test_dir.exists():
            print("  [WARN] test/ not found, skip.")
            continue

        full_dict[scene_dir] = {}

        for method in sorted(os.listdir(test_dir)):
            method_dir = test_dir / method
            renders_dir = method_dir / "renders"
            gt_dir = method_dir / "gt"
            if not renders_dir.exists() or not gt_dir.exists():
                continue

            edge_psnrs: List[float] = []
            edge_l1s: List[float] = []
            grad_corrs: List[float] = []
            edge_f1s: List[float] = []
            lap_var_r: List[float] = []
            lap_var_g: List[float] = []
            hf_r: List[float] = []
            hf_g: List[float] = []
            aligned_psnrs: List[float] = []
            aligned_edge_psnrs: List[float] = []
            bg_leaks: List[float] = []
            tex_lcn_ten_r: List[float] = []
            tex_lcn_ten_g: List[float] = []
            tex_lcn_edge_r: List[float] = []
            tex_lcn_edge_g: List[float] = []
            air_edge_excess: List[float] = []
            air_hf_mean: List[float] = []
            air_bright_excess: List[float] = []

            for _, render, gt in _iter_pairs(renders_dir, gt_dir):
                if render.shape != gt.shape:
                    raise ValueError(f"Render/GT size mismatch: {render.shape} vs {gt.shape}")
                gray_r = _rgb_to_gray(render)
                gray_g = _rgb_to_gray(gt)

                edge_r = _sobel_mag(gray_r)
                edge_g = _sobel_mag(gray_g)
                mse_edge = float(np.mean((edge_r - edge_g) ** 2))
                edge_psnrs.append(_psnr_from_mse(mse_edge, 1.0))
                edge_l1s.append(float(np.mean(np.abs(edge_r - edge_g))))
                grad_corrs.append(_corrcoef_safe(edge_r, edge_g))
                edge_f1s.append(_edge_f1(edge_r, edge_g, edge_thr))

                lap_r = _laplacian(gray_r)
                lap_g = _laplacian(gray_g)
                lap_var_r.append(float(np.var(lap_r)))
                lap_var_g.append(float(np.var(lap_g)))
                hf_r.append(float(np.mean(np.abs(lap_r))))
                hf_g.append(float(np.mean(np.abs(lap_g))))

                aligned_psnrs.append(_aligned_psnr_rgb(render, gt, k))
                aligned_edge_psnrs.append(_aligned_psnr_gray(edge_r, edge_g, k))
                bg_leaks.append(_bg_leak_ratio(render, bg))

                # Brightness-insensitive texture clarity.
                ten_r, ed_r = _texture_metrics_lcn(gray_r)
                ten_g, ed_g = _texture_metrics_lcn(gray_g)
                tex_lcn_ten_r.append(ten_r)
                tex_lcn_ten_g.append(ten_g)
                tex_lcn_edge_r.append(ed_r)
                tex_lcn_edge_g.append(ed_g)

                # Air cleanliness: artifact/fog proxies measured on GT-derived empty-air mask.
                air_mask = _air_mask_from_gt(gray_g, edge_g)
                air_edge_excess.append(_masked_mean(np.maximum(edge_r - edge_g, 0.0), air_mask))
                air_hf_mean.append(_masked_mean(np.abs(lap_r), air_mask))
                air_bright_excess.append(_masked_mean(np.maximum(gray_r - gray_g, 0.0), air_mask))

            if not edge_psnrs:
                continue

            mean_edge_psnr = float(np.mean(edge_psnrs))
            mean_edge_l1 = float(np.mean(edge_l1s))
            mean_grad_corr = float(np.mean(grad_corrs))
            mean_edge_f1 = float(np.mean(edge_f1s))
            mean_lap_r = float(np.mean(lap_var_r))
            mean_lap_g = float(np.mean(lap_var_g))
            lap_ratio = mean_lap_r / mean_lap_g if mean_lap_g > 0.0 else float("inf")
            mean_hf_r = float(np.mean(hf_r))
            mean_hf_g = float(np.mean(hf_g))
            mean_hf_diff = float(abs(mean_hf_r - mean_hf_g))
            mean_aligned_psnr = float(np.mean(aligned_psnrs))
            mean_aligned_edge_psnr = float(np.mean(aligned_edge_psnrs))
            mean_bg_leak = float(np.mean(bg_leaks))
            mean_tex_ten_r = float(np.mean(tex_lcn_ten_r))
            mean_tex_ten_g = float(np.mean(tex_lcn_ten_g))
            tex_ten_ratio = mean_tex_ten_r / mean_tex_ten_g if mean_tex_ten_g > 0.0 else float("inf")
            mean_tex_ed_r = float(np.mean(tex_lcn_edge_r))
            mean_tex_ed_g = float(np.mean(tex_lcn_edge_g))
            tex_ed_ratio = mean_tex_ed_r / mean_tex_ed_g if mean_tex_ed_g > 0.0 else float("inf")
            mean_air_edge_excess = float(np.mean(air_edge_excess))
            mean_air_hf = float(np.mean(air_hf_mean))
            mean_air_bright_excess = float(np.mean(air_bright_excess))

            print("Method:", method)
            print(f"  EdgePSNR(mean): {_format_float(mean_edge_psnr)}")
            print(f"  EdgeL1(mean): {_format_float(mean_edge_l1)}")
            print(f"  GradientCorr(mean): {_format_float(mean_grad_corr)}")
            print(f"  EdgeF1@{edge_thr:.3f}(mean): {_format_float(mean_edge_f1)}")
            print(f"  LapVar(render)(mean): {_format_float(mean_lap_r)}")
            print(f"  LapVar(gt)(mean): {_format_float(mean_lap_g)}")
            print(f"  LapVarRatio(mean): {_format_float(lap_ratio)}")
            print(f"  HFAbsMean(render)(mean): {_format_float(mean_hf_r)}")
            print(f"  HFAbsMean(gt)(mean): {_format_float(mean_hf_g)}")
            print(f"  HFAbsMeanDiff(mean): {_format_float(mean_hf_diff)}")
            print(f"  AlignedPSNR@{k}(mean): {_format_float(mean_aligned_psnr)}")
            print(f"  AlignedEdgePSNR@{k}(mean): {_format_float(mean_aligned_edge_psnr)}")
            print(f"  BgLeakRatio(mean): {_format_float(mean_bg_leak)}")
            print(f"  TextureLCN_Tenengrad(render)(mean): {_format_float(mean_tex_ten_r)}")
            print(f"  TextureLCN_Tenengrad(gt)(mean): {_format_float(mean_tex_ten_g)}")
            print(f"  TextureLCN_TenengradRatio(mean): {_format_float(tex_ten_ratio)}")
            print(f"  TextureLCN_EdgeDensity(render)(mean): {_format_float(mean_tex_ed_r)}")
            print(f"  TextureLCN_EdgeDensity(gt)(mean): {_format_float(mean_tex_ed_g)}")
            print(f"  TextureLCN_EdgeDensityRatio(mean): {_format_float(tex_ed_ratio)}")
            print(f"  AirArtifactEdgeExcess(mean): {_format_float(mean_air_edge_excess)}")
            print(f"  AirArtifactHFMean(mean): {_format_float(mean_air_hf)}")
            print(f"  AirArtifactBrightExcess(mean): {_format_float(mean_air_bright_excess)}")

            full_dict[scene_dir][method] = {
                "EdgePSNR": mean_edge_psnr,
                "EdgeL1": mean_edge_l1,
                "GradientCorr": mean_grad_corr,
                "EdgeF1": mean_edge_f1,
                "LapVar_render": mean_lap_r,
                "LapVar_gt": mean_lap_g,
                "LapVarRatio": lap_ratio,
                "HFAbsMean_render": mean_hf_r,
                "HFAbsMean_gt": mean_hf_g,
                "HFAbsMeanDiff": mean_hf_diff,
                "AlignedPSNR": mean_aligned_psnr,
                "AlignedEdgePSNR": mean_aligned_edge_psnr,
                "BgLeakRatio": mean_bg_leak,
                "TextureLCN_Tenengrad_render": mean_tex_ten_r,
                "TextureLCN_Tenengrad_gt": mean_tex_ten_g,
                "TextureLCN_TenengradRatio": tex_ten_ratio,
                "TextureLCN_EdgeDensity_render": mean_tex_ed_r,
                "TextureLCN_EdgeDensity_gt": mean_tex_ed_g,
                "TextureLCN_EdgeDensityRatio": tex_ed_ratio,
                "AirArtifactEdgeExcess": mean_air_edge_excess,
                "AirArtifactHFMean": mean_air_hf,
                "AirArtifactBrightExcess": mean_air_bright_excess,
            }

        if save_json:
            out_path = Path(scene_dir) / "results_plus.json"
            out_path.write_text(json.dumps(full_dict[scene_dir], indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extended metrics: edges, sharpness, alignment, background leak")
    parser.add_argument("--model_paths", "-m", required=True, nargs="+", type=str, default=[])
    parser.add_argument("--K", type=int, default=8, help="Search window for AlignedPSNR (default: 8)")
    parser.add_argument("--bg", type=int, default=0, choices=[0, 1], help="Background color: 0=black, 1=white")
    parser.add_argument("--edge_thr", type=float, default=0.1, help="Sobel edge threshold for EdgeF1 (default: 0.1)")
    parser.add_argument("--save_json", action="store_true", default=False, help="Write results_plus.json (default: off)")
    args = parser.parse_args()
    evaluate(args.model_paths, args.K, args.bg, args.edge_thr, args.save_json)


if __name__ == "__main__":
    main()
