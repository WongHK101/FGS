#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def _load_matcher(backend: str, repo_root: Path, weight_path: Path, match_thr: float, fine_thr: float):
    repo_root = repo_root.resolve()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)
    if backend == "xoftr":
        from test_relative_pose import load_xoftr  # type: ignore

        args = SimpleNamespace(
            match_threshold=match_thr,
            fine_threshold=fine_thr,
            ckpt=str(weight_path),
        )
        return load_xoftr(args)

    if backend == "minima_xoftr":
        from load_model import load_xoftr  # type: ignore

        args = SimpleNamespace(
            match_threshold=match_thr,
            fine_threshold=fine_thr,
            ckpt=str(weight_path),
        )
        return load_xoftr(args).from_paths

    raise ValueError(f"Unsupported backend: {backend}")


def _list_images(path: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for p in path.glob(ext):
            out[p.stem.lower()] = p
    return out


def _compute_homography(
    mkpts0: np.ndarray,
    mkpts1: np.ndarray,
    reproj: float,
    confidence: float,
    max_iters: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if len(mkpts0) < 4 or len(mkpts1) < 4:
        return None, None, "too_few_matches_for_h"
    if hasattr(cv2, "USAC_MAGSAC"):
        try:
            H, inliers = cv2.findHomography(
                mkpts0,
                mkpts1,
                method=cv2.USAC_MAGSAC,
                ransacReprojThreshold=reproj,
                maxIters=max_iters,
                confidence=confidence,
            )
            if H is not None and inliers is not None:
                return H, inliers, "USAC_MAGSAC"
        except Exception:
            pass
    try:
        H, inliers = cv2.findHomography(
            mkpts0,
            mkpts1,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj,
            maxIters=max_iters,
            confidence=confidence,
        )
        if H is not None and inliers is not None:
            return H, inliers, "RANSAC"
    except Exception:
        pass
    return None, None, "findHomography_failed"


def _valid_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _checkerboard(img_a: np.ndarray, img_b: np.ndarray, cell: int = 32) -> np.ndarray:
    h, w = img_a.shape[:2]
    yy, xx = np.indices((h, w))
    sel = ((yy // cell) + (xx // cell)) % 2 == 0
    out = img_a.copy()
    out[~sel] = img_b[~sel]
    return out


def _overlay(img_a: np.ndarray, img_b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(img_a, alpha, img_b, 1.0 - alpha, 0.0)


def _meta_dict(args: argparse.Namespace, total_pairs: int) -> Dict[str, object]:
    return {
        "backend": args.backend,
        "repo_root": str(Path(args.repo_root).resolve()),
        "repo_commit": _git_commit(Path(args.repo_root)),
        "weight_path": str(Path(args.weight_path).resolve()),
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "dataset_name": args.dataset_name,
        "input_mode": args.input_mode,
        "match_threshold": args.match_threshold,
        "fine_threshold": args.fine_threshold,
        "homography_method_preferred": "USAC_MAGSAC",
        "homography_fallback": "RANSAC",
        "reproj_threshold": args.reproj_threshold,
        "confidence": args.confidence,
        "max_iters": args.max_iters,
        "failure_rule": "matches<12 or inliers<8 or inlier_ratio<0.15",
        "opencv_version": cv2.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "total_pairs": total_pairs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run unified 2D registration wrapper for one backend and one dataset.")
    ap.add_argument("--backend", required=True, choices=["xoftr", "minima_xoftr"])
    ap.add_argument("--repo_root", required=True)
    ap.add_argument("--weight_path", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--input_mode", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--match_threshold", type=float, default=0.3)
    ap.add_argument("--fine_threshold", type=float, default=0.1)
    ap.add_argument("--reproj_threshold", type=float, default=6.0)
    ap.add_argument("--confidence", type=float, default=0.999)
    ap.add_argument("--max_iters", type=int, default=10000)
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    rgb_dir = dataset_dir / "rgb"
    th_dir = dataset_dir / "thermal"
    out_root = Path(args.out_root)
    reg_rgb_dir = out_root / "registered_rgb"
    reg_th_dir = out_root / "registered_thermal"
    vis_dir = out_root / "qualitative"
    _ensure_dir(out_root)
    _ensure_dir(reg_rgb_dir)
    _ensure_dir(reg_th_dir)
    _ensure_dir(vis_dir)

    rgb_map = _list_images(rgb_dir)
    th_map = _list_images(th_dir)
    stems = sorted(set(rgb_map.keys()) & set(th_map.keys()))
    meta = _meta_dict(args, len(stems))
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    matcher = _load_matcher(
        args.backend,
        Path(args.repo_root),
        Path(args.weight_path),
        args.match_threshold,
        args.fine_threshold,
    )

    rows: List[Dict[str, object]] = []
    success_examples: List[Tuple[str, float]] = []
    failure_examples: List[Tuple[str, float]] = []
    t0_all = time.time()

    for stem in stems:
        rgb_path = rgb_map[stem]
        th_path = th_map[stem]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        th = cv2.imread(str(th_path), cv2.IMREAD_COLOR)
        if rgb is None or th is None:
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "stem": stem,
                    "success": 0,
                    "n_matches": 0,
                    "n_inliers": 0,
                    "inlier_ratio": 0.0,
                    "runtime_ms": 0.0,
                    "fail_reason": "image_read_failed",
                    "solver": "",
                }
            )
            continue

        t0 = time.time()
        try:
            match_res = matcher(str(rgb_path), str(th_path))
        except Exception as exc:
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "stem": stem,
                    "success": 0,
                    "n_matches": 0,
                    "n_inliers": 0,
                    "inlier_ratio": 0.0,
                    "runtime_ms": (time.time() - t0) * 1000.0,
                    "fail_reason": f"matcher_error:{type(exc).__name__}",
                    "solver": "",
                }
            )
            failure_examples.append((stem, 0.0))
            continue

        mkpts0 = match_res.get("mkpts0")
        mkpts1 = match_res.get("mkpts1")
        n_matches = int(len(mkpts0)) if mkpts0 is not None else 0
        if mkpts0 is None or mkpts1 is None or n_matches < 12:
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "stem": stem,
                    "success": 0,
                    "n_matches": n_matches,
                    "n_inliers": 0,
                    "inlier_ratio": 0.0,
                    "runtime_ms": (time.time() - t0) * 1000.0,
                    "fail_reason": "matches_lt_12",
                    "solver": "",
                }
            )
            failure_examples.append((stem, 0.0))
            continue

        H, inliers, solver = _compute_homography(
            np.asarray(mkpts0, dtype=np.float32),
            np.asarray(mkpts1, dtype=np.float32),
            args.reproj_threshold,
            args.confidence,
            args.max_iters,
        )
        if H is None or inliers is None:
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "stem": stem,
                    "success": 0,
                    "n_matches": n_matches,
                    "n_inliers": 0,
                    "inlier_ratio": 0.0,
                    "runtime_ms": (time.time() - t0) * 1000.0,
                    "fail_reason": solver,
                    "solver": solver,
                }
            )
            failure_examples.append((stem, 0.0))
            continue

        inliers = inliers.reshape(-1).astype(bool)
        n_inliers = int(inliers.sum())
        inlier_ratio = float(n_inliers) / float(max(n_matches, 1))
        if n_inliers < 8 or inlier_ratio < 0.15:
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "stem": stem,
                    "success": 0,
                    "n_matches": n_matches,
                    "n_inliers": n_inliers,
                    "inlier_ratio": inlier_ratio,
                    "runtime_ms": (time.time() - t0) * 1000.0,
                    "fail_reason": "inliers_lt_rule",
                    "solver": solver,
                }
            )
            failure_examples.append((stem, inlier_ratio))
            continue

        h_t, w_t = th.shape[:2]
        warped = cv2.warpPerspective(rgb, H, (w_t, h_t))
        mask = cv2.warpPerspective(
            np.full((rgb.shape[0], rgb.shape[1]), 255, dtype=np.uint8),
            H,
            (w_t, h_t),
        )
        bbox = _valid_bbox(mask)
        if bbox is None:
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "stem": stem,
                    "success": 0,
                    "n_matches": n_matches,
                    "n_inliers": n_inliers,
                    "inlier_ratio": inlier_ratio,
                    "runtime_ms": (time.time() - t0) * 1000.0,
                    "fail_reason": "empty_valid_bbox",
                    "solver": solver,
                }
            )
            failure_examples.append((stem, inlier_ratio))
            continue

        x0, y0, x1, y1 = bbox
        warped_crop = warped[y0:y1, x0:x1]
        thermal_crop = th[y0:y1, x0:x1]
        if warped_crop.size == 0 or thermal_crop.size == 0:
            rows.append(
                {
                    "dataset": args.dataset_name,
                    "stem": stem,
                    "success": 0,
                    "n_matches": n_matches,
                    "n_inliers": n_inliers,
                    "inlier_ratio": inlier_ratio,
                    "runtime_ms": (time.time() - t0) * 1000.0,
                    "fail_reason": "empty_crop",
                    "solver": solver,
                }
            )
            failure_examples.append((stem, inlier_ratio))
            continue

        cv2.imwrite(str(reg_rgb_dir / f"{rgb_path.stem}.png"), warped_crop)
        cv2.imwrite(str(reg_th_dir / f"{th_path.stem}.png"), thermal_crop)

        overlay = _overlay(warped_crop, thermal_crop, alpha=0.5)
        checker = _checkerboard(warped_crop, thermal_crop, cell=32)
        cv2.imwrite(str(vis_dir / f"{stem}_overlay.png"), overlay)
        cv2.imwrite(str(vis_dir / f"{stem}_checker.png"), checker)

        rows.append(
            {
                "dataset": args.dataset_name,
                "stem": stem,
                "success": 1,
                "n_matches": n_matches,
                "n_inliers": n_inliers,
                "inlier_ratio": inlier_ratio,
                "runtime_ms": (time.time() - t0) * 1000.0,
                "fail_reason": "",
                "solver": solver,
                "crop_x0": x0,
                "crop_y0": y0,
                "crop_x1": x1,
                "crop_y1": y1,
            }
        )
        success_examples.append((stem, inlier_ratio))

    csv_path = out_root / "pair_stats.csv"
    fieldnames = [
        "dataset",
        "stem",
        "success",
        "n_matches",
        "n_inliers",
        "inlier_ratio",
        "runtime_ms",
        "fail_reason",
        "solver",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        **meta,
        "duration_s": time.time() - t0_all,
        "available_pairs": int(sum(int(r["success"]) for r in rows)),
        "available_rate": float(sum(int(r["success"]) for r in rows)) / float(max(len(rows), 1)),
        "success_examples_top2": [s for s, _ in sorted(success_examples, key=lambda x: x[1], reverse=True)[:2]],
        "failure_examples_top2": [s for s, _ in sorted(failure_examples, key=lambda x: x[1])[:2]],
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
