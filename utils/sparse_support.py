"""Sparse COLMAP support utilities.

This module is designed for the "Sparse-Support" family of constraints:
- Parse COLMAP sparse reconstruction outputs.
- Provide lightweight spatial queries over COLMAP points.

Notes
-----
* Dependency-light: requires numpy. Torch is optional.
* Safe to import even if COLMAP outputs are missing; functions raise
  FileNotFoundError with clear messages when called.
* API is kept stable to allow backward-compatible wiring into training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


# -----------------------------
# COLMAP I/O helpers
# -----------------------------


def _first_existing(paths: Sequence[Union[str, Path]]) -> Optional[Path]:
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None


def _looks_like_colmap_model_dir(d: Path) -> bool:
    """Whether a directory contains COLMAP model files."""
    if not d.exists() or not d.is_dir():
        return False
    for stem in ("cameras", "images", "points3D"):
        if (d / f"{stem}.bin").exists() or (d / f"{stem}.txt").exists():
            return True
    return False


def resolve_colmap_model_dir(sparse_root: Union[str, Path]) -> Path:
    """Resolve a COLMAP *model directory* that contains cameras/images/points3D.

    Accepts either:
      - a COLMAP model directory directly (contains points3D.bin/txt), or
      - a COLMAP sparse root directory (contains subdirs like sparse/0), or
      - a dataset root directory that contains a 'sparse' directory.

    Returns
    -------
    Path
        Directory containing COLMAP model files.

    Raises
    ------
    FileNotFoundError
        If no COLMAP model dir is found.
    """
    root = Path(sparse_root)
    if not root.exists():
        raise FileNotFoundError(f"COLMAP path does not exist: {root}")

    # Case 1: already a model dir.
    if _looks_like_colmap_model_dir(root):
        return root

    # Case 2: common structure: <root>/sparse/0
    candidates: List[Path] = []
    # If root itself is 'sparse', try numeric subfolders.
    if root.name.lower() == "sparse":
        candidates.extend([p for p in root.iterdir() if p.is_dir()])
    # If root contains 'sparse', try there.
    if (root / "sparse").exists():
        candidates.extend([p for p in (root / "sparse").iterdir() if p.is_dir()])

    # Also accept <root>/colmap/sparse/0
    if (root / "colmap" / "sparse").exists():
        candidates.extend([p for p in (root / "colmap" / "sparse").iterdir() if p.is_dir()])

    # Sort to prefer '0' then smallest index.
    def _key(p: Path) -> Tuple[int, str]:
        try:
            return (int(p.name), p.name)
        except Exception:
            return (10**9, p.name)

    for c in sorted(candidates, key=_key):
        if _looks_like_colmap_model_dir(c):
            return c

    raise FileNotFoundError(
        "Could not locate a COLMAP model directory under: "
        f"{root}. Expected files like points3D.bin or points3D.txt."
    )


@dataclass
class SparsePointCloud:
    xyz: np.ndarray  # (N,3) float32
    error: Optional[np.ndarray] = None  # (N,) float32
    track_len: Optional[np.ndarray] = None  # (N,) int32

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=np.float32)
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N,3), got {self.xyz.shape}")
        if self.error is not None:
            self.error = np.asarray(self.error, dtype=np.float32).reshape(-1)
        if self.track_len is not None:
            self.track_len = np.asarray(self.track_len, dtype=np.int32).reshape(-1)


def load_colmap_points3D(model_dir: Union[str, Path]) -> SparsePointCloud:
    """Load COLMAP points3D from a model directory."""
    model_dir = Path(model_dir)

    from utils.read_write_model import (  # local import to keep module import-safe
        read_points3D_binary,
        read_points3D_text,
    )

    p_bin = model_dir / "points3D.bin"
    p_txt = model_dir / "points3D.txt"
    if p_bin.exists():
        pts = read_points3D_binary(str(p_bin))
    elif p_txt.exists():
        pts = read_points3D_text(str(p_txt))
    else:
        raise FileNotFoundError(f"points3D.bin/txt not found in: {model_dir}")

    if len(pts) == 0:
        return SparsePointCloud(xyz=np.zeros((0, 3), dtype=np.float32))

    xyz = np.stack([p.xyz for p in pts.values()], axis=0).astype(np.float32)
    err = np.array([p.error for p in pts.values()], dtype=np.float32)
    tlen = np.array([len(p.image_ids) for p in pts.values()], dtype=np.int32)
    return SparsePointCloud(xyz=xyz, error=err, track_len=tlen)


def load_colmap_camera_centers(model_dir: Union[str, Path]) -> np.ndarray:
    """Load camera centers (world) from COLMAP images.bin/txt.

    Returns
    -------
    np.ndarray
        (M,3) float32 camera centers.
    """
    model_dir = Path(model_dir)

    from utils.read_write_model import (  # local import
        qvec2rotmat,
        read_images_binary,
        read_images_text,
    )

    i_bin = model_dir / "images.bin"
    i_txt = model_dir / "images.txt"
    if i_bin.exists():
        imgs = read_images_binary(str(i_bin))
    elif i_txt.exists():
        imgs = read_images_text(str(i_txt))
    else:
        raise FileNotFoundError(f"images.bin/txt not found in: {model_dir}")

    centers: List[np.ndarray] = []
    for im in imgs.values():
        R = qvec2rotmat(im.qvec)
        t = np.asarray(im.tvec, dtype=np.float64).reshape(3)
        c = (-R.T @ t).astype(np.float32)
        centers.append(c)

    if not centers:
        return np.zeros((0, 3), dtype=np.float32)

    return np.stack(centers, axis=0).astype(np.float32)


def robust_aabb(xyz: np.ndarray, quantile: float = 0.005, margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a robust AABB by trimming outliers by quantile.

    Parameters
    ----------
    xyz : (N,3)
    quantile : float
        Trim both tails per axis. 0.005 means keep [0.5%, 99.5%].
    margin : float
        Expand bounds by this amount (in world units).
    """
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.size == 0:
        mn = np.zeros((3,), dtype=np.float32)
        mx = np.zeros((3,), dtype=np.float32)
        return mn, mx

    q0 = float(np.clip(quantile, 0.0, 0.49))
    lo = np.quantile(xyz, q0, axis=0).astype(np.float32)
    hi = np.quantile(xyz, 1.0 - q0, axis=0).astype(np.float32)
    if margin != 0.0:
        m = float(margin)
        lo -= m
        hi += m
    return lo, hi


# -----------------------------
# Voxel-hash nearest-neighbor (small-batch)
# -----------------------------


def _voxel_index(xyz: np.ndarray, voxel: float) -> np.ndarray:
    return np.floor(xyz / float(voxel)).astype(np.int64)


class VoxelHashNN:
    """Very lightweight NN approximation via voxel hashing.

    Intended for querying a *small* batch of points (e.g. new Gaussians or
    densification candidates) against COLMAP sparse points.

    This intentionally avoids heavy dependencies (scipy/sklearn).
    """

    def __init__(
        self,
        points_xyz: np.ndarray,
        voxel_size: float,
        max_points_per_voxel: int = 256,
    ) -> None:
        self.voxel_size = float(voxel_size)
        self.max_points_per_voxel = int(max_points_per_voxel)

        pts = np.asarray(points_xyz, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points_xyz must be (N,3), got {pts.shape}")
        self.points_xyz = pts

        self._grid: Dict[Tuple[int, int, int], np.ndarray] = {}
        if len(pts) == 0:
            return

        vox = _voxel_index(pts, self.voxel_size)
        # group indices by voxel
        # Use python dict of lists for memory friendliness
        tmp: Dict[Tuple[int, int, int], List[int]] = {}
        for i, (x, y, z) in enumerate(vox.tolist()):
            k = (int(x), int(y), int(z))
            lst = tmp.get(k)
            if lst is None:
                tmp[k] = [i]
            else:
                if len(lst) < self.max_points_per_voxel:
                    lst.append(i)
        # finalize to numpy arrays
        for k, v in tmp.items():
            self._grid[k] = np.asarray(v, dtype=np.int32)

    def _candidate_indices_for_voxel(self, k: Tuple[int, int, int]) -> Optional[np.ndarray]:
        return self._grid.get(k, None)

    def query(
        self,
        query_xyz: np.ndarray,
        max_voxel_radius: int = 2,
        return_index: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Query nearest distance for each query point.

        Parameters
        ----------
        query_xyz : (M,3) float
        max_voxel_radius : int
            Search in a cube of +/-R voxels.
        return_index : bool
            Whether to also return nearest point index.

        Returns
        -------
        dists : (M,) float32
        idx : (M,) int32  (if return_index)
        """
        q = np.asarray(query_xyz, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, 3)
        if q.ndim != 2 or q.shape[1] != 3:
            raise ValueError(f"query_xyz must be (M,3), got {q.shape}")

        M = q.shape[0]
        d2_best = np.full((M,), np.inf, dtype=np.float32)
        i_best = np.full((M,), -1, dtype=np.int32)

        if len(self.points_xyz) == 0 or not self._grid:
            if return_index:
                return np.sqrt(d2_best), i_best
            return np.sqrt(d2_best)

        qvox = _voxel_index(q, self.voxel_size)
        R = int(max_voxel_radius)

        for m in range(M):
            vx, vy, vz = qvox[m].tolist()
            # search neighbor voxels
            cand: List[int] = []
            for dx in range(-R, R + 1):
                for dy in range(-R, R + 1):
                    for dz in range(-R, R + 1):
                        kk = (int(vx + dx), int(vy + dy), int(vz + dz))
                        inds = self._candidate_indices_for_voxel(kk)
                        if inds is not None:
                            cand.extend(inds.tolist())

            if not cand:
                continue

            cand_idx = np.asarray(cand, dtype=np.int32)
            pts = self.points_xyz[cand_idx]
            diff = pts - q[m : m + 1]
            d2 = np.sum(diff * diff, axis=1)
            j = int(np.argmin(d2))
            d2_best[m] = float(d2[j])
            i_best[m] = int(cand_idx[j])

        if return_index:
            return np.sqrt(d2_best), i_best
        return np.sqrt(d2_best)

    def query_torch(
        self,
        query_xyz: "torch.Tensor",
        max_voxel_radius: int = 2,
    ) -> "torch.Tensor":
        """Torch-friendly wrapper (runs on CPU; returns tensor on original device)."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is not available, cannot call query_torch")
        dev = query_xyz.device
        q_np = query_xyz.detach().float().cpu().numpy()
        d_np = self.query(q_np, max_voxel_radius=max_voxel_radius, return_index=False)
        return torch.from_numpy(d_np).to(dev)


# -----------------------------
# Convenience entrypoints for the pipeline
# -----------------------------


def build_sparse_support(
    colmap_path: Union[str, Path],
    *,
    voxel_size: float = 1.0,
    max_voxel_radius: int = 2,
    max_points: Optional[int] = None,
    error_prune_quantile: Optional[float] = 0.99,
    tracklen_min: int = 3,
) -> Tuple[SparsePointCloud, VoxelHashNN, Tuple[np.ndarray, np.ndarray]]:
    """High-level helper: load sparse points, filter, build NN index, compute AABB.

    Returns
    -------
    spc : SparsePointCloud
    index : VoxelHashNN
    aabb : (lo, hi)
    """
    model_dir = resolve_colmap_model_dir(colmap_path)
    spc = load_colmap_points3D(model_dir)

    xyz = spc.xyz
    if xyz.size == 0:
        idx = VoxelHashNN(xyz, voxel_size=voxel_size)
        aabb = robust_aabb(xyz)
        return spc, idx, aabb

    keep = np.ones((xyz.shape[0],), dtype=bool)

    if spc.track_len is not None and tracklen_min > 1:
        keep &= (spc.track_len >= int(tracklen_min))

    if spc.error is not None and error_prune_quantile is not None:
        q = float(np.clip(error_prune_quantile, 0.0, 1.0))
        thr = float(np.quantile(spc.error, q))
        keep &= (spc.error <= thr)

    xyz_f = xyz[keep]
    err_f = spc.error[keep] if spc.error is not None else None
    tlen_f = spc.track_len[keep] if spc.track_len is not None else None

    if max_points is not None and xyz_f.shape[0] > int(max_points):
        # Prefer longer tracks; if track_len absent, random sample.
        if tlen_f is not None:
            order = np.argsort(-tlen_f)  # desc
            order = order[: int(max_points)]
        else:
            rng = np.random.default_rng(0)
            order = rng.choice(xyz_f.shape[0], size=int(max_points), replace=False)
        xyz_f = xyz_f[order]
        if err_f is not None:
            err_f = err_f[order]
        if tlen_f is not None:
            tlen_f = tlen_f[order]

    spc_f = SparsePointCloud(xyz=xyz_f, error=err_f, track_len=tlen_f)
    aabb = robust_aabb(xyz_f, quantile=0.005, margin=0.0)

    index = VoxelHashNN(xyz_f, voxel_size=float(voxel_size))

    # Store query defaults as attributes (purely convenience; not relied on).
    index.default_max_voxel_radius = int(max_voxel_radius)  # type: ignore[attr-defined]

    return spc_f, index, aabb


def format_aabb(aabb: Tuple[np.ndarray, np.ndarray]) -> str:
    lo, hi = aabb
    lo = np.asarray(lo).reshape(3)
    hi = np.asarray(hi).reshape(3)
    return f"lo={lo.tolist()} hi={hi.tolist()}"

