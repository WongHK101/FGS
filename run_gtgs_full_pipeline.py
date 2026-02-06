# -*- coding: utf-8 -*-
"""
run_gtgs_full_pipeline_resumable.py

A resumable, one-command CLI runner for the end-to-end GeoTGS/FGS pipeline:

1) CFR crop/align (cfr.py)
2) Crop+EXIF evaluation (eval_crop_metrics.py) and (optionally) auto-pick best candidate
3) Prepare COLMAP input/ directory from chosen candidate
4) COLMAP pipeline with GPS priors + alignment (convert-gtgs.py)
5) Stage-1 3DGS train (RGB), render, metrics
6) Undistort thermal images using aligned sparse model (colmap image_undistorter)
7) Normalize thermal_UD/sparse layout (move files into sparse/0 if needed)
8) Stage-2 3DGS train (Thermal), render, metrics
9) Blend RGB+Thermal models (blend_model_strict_endpoints.py)
10) Evaluate sweep (eval_blend_sweep.py, with optional auto_render)

Key features vs the previous version:
- **Resumable by default**: If a step's expected outputs already exist, it will be skipped on rerun.
- Writes per-step markers under: <data_root>/_pipeline_state/*.json
- More robust COLMAP executable handling on Windows:
  - resolves "colmap" via PATH
  - supports colmap.bat / colmap.cmd (wraps with cmd.exe /c)
  - optionally tries PowerShell Get-Command to resolve "colmap" when PATH lookup fails

Assumptions:
- Put this file in the *root* of graphdeco-inria/gaussian-splatting repo,
  next to: train.py, render.py, metrics.py, and your helper scripts:
  cfr.py / eval_crop_metrics.py / convert-gtgs.py / blend_model_strict_endpoints.py / eval_blend_sweep.py
- Run this script using the same Python environment you use for 3DGS.
"""

from __future__ import annotations

import argparse
import math
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Small utilities
# ----------------------------

def eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def exists_nonempty_dir(p: Path) -> bool:
    return p.exists() and p.is_dir() and any(p.iterdir())

def _dir_size_bytes(p: Path) -> int:
    total = 0
    if not p.exists():
        return total
    if p.is_file():
        try:
            return p.stat().st_size
        except Exception:
            return 0
    for root, _, files in os.walk(p):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except Exception:
                continue
    return total

def _count_images_recursive(p: Path) -> int:
    if not p.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    count = 0
    for fp in p.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in exts:
            count += 1
    return count

def _ply_vertex_count(p: Path) -> Optional[int]:
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("element vertex"):
                    parts = line.strip().split()
                    return int(parts[-1])
                if line.strip() == "end_header":
                    break
    except Exception:
        return None
    return None


def _validate_finite_float(
    ap: argparse.ArgumentParser,
    name: str,
    v: Optional[float],
    *,
    min_value: Optional[float] = None,
    strict_positive: bool = False,
) -> None:
    """Argparse-time validation for optional floats.

    Notes:
      - Only called when the corresponding feature is enabled, so defaults remain unchanged.
      - When strict_positive=True, requires v > 0.
      - When min_value is set, requires v >= min_value.
    """
    if v is None:
        return
    try:
        fv = float(v)
    except Exception:
        ap.error(f"{name} must be a float, got {v!r}")
        return
    if not math.isfinite(fv):
        ap.error(f"{name} must be a finite float, got {v!r}")
    if strict_positive and fv <= 0.0:
        ap.error(f"{name} must be > 0, got {v!r}")
    if (min_value is not None) and fv < float(min_value):
        ap.error(f"{name} must be >= {min_value}, got {v!r}")


def _str2bool(v: str) -> bool:
    """Argparse-friendly bool parser."""
    if isinstance(v, bool):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v!r}")


def _build_tstruct_train_args(args: argparse.Namespace) -> List[str]:
    """Build extra args forwarded to *thermal* train.py for pseudo-color structure loss."""
    if float(getattr(args, "t_struct_grad_w", 0.0)) <= 0.0:
        return []
    norm_raw = getattr(args, "t_struct_grad_norm", True)
    if isinstance(norm_raw, str):
        norm = norm_raw.strip().lower() in {"1","true","yes","y","t","on"}
    else:
        norm = bool(norm_raw)
    return [
        "--t_struct_grad_w",
        str(float(getattr(args, "t_struct_grad_w", 0.0))),
        "--t_struct_grad_norm",
        "true" if norm else "false",
    ]


def _build_ss_train_args(*pos, **kw) -> List[str]:
    """Build train.py args for Sparse Support.

    Accepts either:
      - _build_ss_train_args(ap, args)
      - _build_ss_train_args(args, ap=ap)  (preferred)
      - _build_ss_train_args(args)         (errors are raised as RuntimeError)
    """
    ap: Optional[argparse.ArgumentParser] = kw.get("ap", None)
    args: Optional[argparse.Namespace] = None

    if len(pos) == 1 and isinstance(pos[0], argparse.Namespace):
        args = pos[0]
    elif len(pos) == 2 and isinstance(pos[0], argparse.ArgumentParser) and isinstance(pos[1], argparse.Namespace):
        ap, args = pos[0], pos[1]
    elif len(pos) == 2 and isinstance(pos[1], argparse.ArgumentParser) and isinstance(pos[0], argparse.Namespace):
        args, ap = pos[0], pos[1]
    else:
        raise TypeError("_build_ss_train_args expects (args) or (ap, args)")

    if args is None:
        raise RuntimeError("internal: args is None in _build_ss_train_args")

    # Validation (only when SS is enabled)
    if getattr(args, "ss_enable", False):
        # If ap is missing, fall back to RuntimeError with clear messages.
        def _err(msg: str) -> None:
            if ap is not None:
                ap.error(msg)
            raise RuntimeError(msg)

        # source is already validated by argparse choices
        try:
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_aabb_margin", args.ss_aabb_margin, min_value=0.0)
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_voxel_size", args.ss_voxel_size, strict_positive=True)
            _validate_finite_float(ap or argparse.ArgumentParser(add_help=False), "--ss_nn_dist_thr", args.ss_nn_dist_thr, min_value=0.0)
        except SystemExit:
            # argparse.error triggers SystemExit; just re-raise
            raise
        except Exception as e:
            _err(str(e))

    if not getattr(args, "ss_enable", False):
        return []

    out: List[str] = ["--ss_enable"]

    # Always forward explicit SS params once enabled.
    out += ["--ss_source", str(args.ss_source)]
    out += ["--ss_aabb_margin", str(args.ss_aabb_margin)]

    if args.ss_voxel_size is not None:
        out += ["--ss_voxel_size", str(args.ss_voxel_size)]
    if args.ss_nn_dist_thr is not None:
        out += ["--ss_nn_dist_thr", str(args.ss_nn_dist_thr)]
    return out


def list_images(dir_path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    if not dir_path.exists():
        return []
    out: List[Path] = []
    for fp in dir_path.iterdir():
        if fp.is_file() and fp.suffix.lower() in exts:
            out.append(fp)
    out.sort()
    return out

def contains_any_file(root: Path, names: Tuple[str, ...], max_depth: int = 2) -> bool:
    """
    Returns True if any file with basename in `names` exists within `root` up to `max_depth`.
    """
    if not root.exists():
        return False
    root = root.resolve()
    # BFS with depth
    queue: List[Tuple[Path, int]] = [(root, 0)]
    while queue:
        cur, d = queue.pop(0)
        try:
            for p in cur.iterdir():
                if p.is_file() and p.name in names:
                    return True
                if p.is_dir() and d < max_depth:
                    queue.append((p, d + 1))
        except Exception:
            continue
    return False

def hardlink_or_copy(src: Path, dst: Path, mode: str) -> None:
    """
    mode:
      - copy: shutil.copy2
      - hardlink: os.link when possible (same filesystem); fallback to copy
      - symlink: os.symlink when possible; fallback to copy
    """
    ensure_dir(dst.parent)
    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        try:
            if dst.exists():
                dst.unlink()
            os.symlink(src, dst)
            return
        except Exception:
            shutil.copy2(src, dst)
            return

    raise ValueError(f"Unknown link mode: {mode}")

def prepare_input_dir(src_images_dir: Path, dataset_root: Path, clean: bool, link_mode: str) -> Path:
    """
    Copy/link chosen aligned RGB images into <dataset_root>/input for COLMAP.
    """
    input_dir = dataset_root / "input"
    if clean and input_dir.exists():
        eprint(f"[INFO] Cleaning existing input dir: {input_dir}")
        shutil.rmtree(input_dir)
    ensure_dir(input_dir)

    src_imgs = list_images(src_images_dir)
    if not src_imgs:
        raise FileNotFoundError(f"No images found under: {src_images_dir}")

    # If input already has images and we're not cleaning, we still (re)sync only when needed.
    eprint(f"[INFO] Preparing COLMAP input/ from: {src_images_dir}  (count={len(src_imgs)}, mode={link_mode})")
    for fp in src_imgs:
        hardlink_or_copy(fp, input_dir / fp.name, mode=link_mode)
    return input_dir


# ----------------------------
# Windows executable resolution
# ----------------------------

def _ps_resolve_command(name: str) -> Optional[str]:
    """
    Try to resolve an executable/script name via PowerShell Get-Command.
    Helpful when the user can run "colmap" in PowerShell but Python cannot find it.
    """
    if os.name != "nt":
        return None
    ps = shutil.which("pwsh") or shutil.which("powershell")
    if not ps:
        return None
    try:
        # Not using -NoProfile: allow user profiles where aliases/functions might exist.
        out = subprocess.check_output(
            [ps, "-Command", f"(Get-Command {name} -ErrorAction SilentlyContinue).Source"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        out = out.strip()
        if not out:
            return None
        # first line is enough
        return out.splitlines()[0].strip() or None
    except Exception:
        return None

def _normalize_cmd_for_windows(cmd: List[str]) -> List[str]:
    if os.name != "nt" or not cmd:
        return cmd

    exe0 = cmd[0]
    resolved: Optional[str] = None

    p0 = Path(exe0)
    if p0.exists():
        resolved = str(p0.resolve())
    else:
        resolved = shutil.which(exe0)
        if resolved is None and (("\\" not in exe0) and ("/" not in exe0)):
            resolved = _ps_resolve_command(exe0)

    if resolved:
        cmd = [resolved] + cmd[1:]

    suffix = Path(cmd[0]).suffix.lower()
    # Batch scripts cannot be launched directly by CreateProcess; wrap with cmd.exe /c
    if suffix in (".bat", ".cmd"):
        cmd = ["cmd.exe", "/c"] + cmd

    return cmd

def run_cmd(
    cmd: List[str],
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
    stdout=None,
    stderr=None,
) -> None:
    cmd2 = _normalize_cmd_for_windows(cmd)
    cwd_str = str(cwd) if cwd else None
    pretty = " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])
    eprint(f"\n[RUN] {pretty}")
    subprocess.run(cmd2, cwd=cwd_str, env=env, check=True, stdout=stdout, stderr=stderr)


# ----------------------------
# Resumable step markers
# ----------------------------

def marker_path(state_dir: Path, step_name: str) -> Path:
    return state_dir / f"{step_name}.json"

def write_marker(marker: Path, step_name: str, cmd: List[str], cwd: Optional[Path], note: str = "") -> None:
    ensure_dir(marker.parent)
    payload = {
        "step": step_name,
        "time": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(cwd) if cwd else "",
        "cmd": cmd,
        "note": note,
    }
    marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def marker_matches(marker: Path, cmd: List[str]) -> bool:
    if not marker.exists():
        return False
    try:
        obj = json.loads(marker.read_text(encoding="utf-8"))
        return obj.get("cmd", None) == cmd
    except Exception:
        return False

def should_skip_step(state_dir: Path, step_name: str, cmd: List[str], outputs_ok: bool, force: bool) -> bool:
    """
    Skip when:
      - not forced, and
      - outputs look OK, and either:
          (a) marker exists and matches cmd, or
          (b) marker missing but outputs exist (auto-mark)
    """
    if force:
        return False
    m = marker_path(state_dir, step_name)
    if outputs_ok and marker_matches(m, cmd):
        eprint(f"[SKIP] {step_name}  (marker matched + outputs exist)")
        return True
    if outputs_ok and not m.exists():
        eprint(f"[SKIP] {step_name}  (outputs exist; auto-marking)")
        write_marker(m, step_name, cmd, cwd=None, note="auto-marked from existing outputs")
        return True
    return False


# ----------------------------
# Domain helpers
# ----------------------------

def ensure_sparse_0(sparse_dir: Path) -> Path:
    """
    Some tools expect sparse/0/ with cameras/images/points3D.
    If sparse_dir contains model files directly, move them into sparse/0/.
    """
    if not sparse_dir.exists():
        raise FileNotFoundError(f"sparse dir not found: {sparse_dir}")

    subdirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if any(d.name == "0" for d in subdirs):
        model0 = sparse_dir / "0"
    elif len(subdirs) == 1 and subdirs[0].name.isdigit():
        model0 = subdirs[0]
    else:
        model0 = sparse_dir / "0"
        ensure_dir(model0)

    moved = 0
    for fp in sparse_dir.iterdir():
        if fp.is_file():
            dst = model0 / fp.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(fp), str(dst))
            moved += 1

    if moved:
        eprint(f"[INFO] Moved {moved} sparse model files into: {model0}")

    return model0

@dataclass
class CropCandidate:
    tag: str
    rgb_dir: Path
    count: int
    mean: Dict[str, Optional[float]]

def pick_best_candidate(summary_all_json: Path, prefer: Tuple[str, ...] = ("edge_f1", "grad_ncc", "nmi", "edge_dice", "mi")) -> CropCandidate:
    """
    Auto-pick best crop candidate from eval_crop_metrics summary_all.json.

    Heuristic:
      sort by preferred metrics (descending) in order; skip missing values.
    """
    obj = json.loads(summary_all_json.read_text(encoding="utf-8"))
    candidates = obj.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates found in: {summary_all_json}")

    parsed: List[CropCandidate] = []
    for c in candidates:
        parsed.append(
            CropCandidate(
                tag=str(c.get("tag", "")),
                rgb_dir=Path(str(c.get("rgb_dir", ""))),
                count=int(c.get("count", 0)),
                mean=dict(c.get("mean", {})),
            )
        )

    parsed = [c for c in parsed if c.count > 0 and c.rgb_dir.exists()]
    if not parsed:
        raise RuntimeError("All candidates have count==0 or rgb_dir missing; cannot auto pick.")

    def key_fn(c: CropCandidate):
        ks = []
        for k in prefer:
            v = c.mean.get(k, None)
            ks.append(-1e30 if v is None else float(v))
        return tuple(ks)

    parsed.sort(key=key_fn, reverse=True)
    return parsed[0]


# ----------------------------
# Pipeline
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the full CFR->COLMAP->3DGS(RGB)->ThermalUD->3DGS(T)->Blend->Eval pipeline with one command (resumable)."
    )
    ap.add_argument("--data_root", required=True, help="Dataset root, e.g. F:\\databackup\\GeoTGS\\input\\PV-r4")
    ap.add_argument("--out_root", required=True, help="Output root, e.g. F:\\databackup\\GeoTGS\\output\\PV-r4")

    ap.add_argument("--rgb_dir", default="", help="RGB directory. Default: <data_root>/RGB")
    ap.add_argument("--th_dir", default="", help="Thermal directory. Default: <data_root>/thermal")

    ap.add_argument("--colmap", default="colmap", help="COLMAP executable (default: colmap). Can be colmap.exe / colmap.bat / full path.")
    ap.add_argument("--exiftool", default="exiftool", help="ExifTool executable (default: exiftool)")

    ap.add_argument("--align", default="auto", choices=["auto", "fit", "exif"], help="Which aligned RGB to use for COLMAP (default: auto)")
    ap.add_argument("--comparison", action="store_true", help="Enable cfr.py --comparison (writes side-by-side visuals; slower)")
    ap.add_argument("--link_mode", default="copy", choices=["copy", "hardlink", "symlink"],
                    help="How to put aligned images into data_root/input (default: copy). hardlink is fastest if same disk.")
    ap.add_argument("--clean_input", action="store_true", help="Clean data_root/input before preparing it")
    ap.add_argument("--clean_fit", action="store_true", help="Clean data_root/fit before running cfr.py")
    ap.add_argument("--clean_thermal_ud", action="store_true", help="Clean data_root/thermal_UD before thermal undistort")
    ap.add_argument("--clean_blend_out", action="store_true", help="Clean out_root/Model_F before blending (otherwise resumable)")

    ap.add_argument("--force", action="store_true", help="Force rerun steps even if outputs exist (disables resume skipping)")
    ap.add_argument("--dry_run", action="store_true", help="Print commands but do not execute")
    ap.add_argument("--skip_train", action="store_true", help="Skip BOTH training stages + their render/metrics (for debugging earlier steps)")
    ap.add_argument("--skip_blend", action="store_true", help="Skip blending + sweep eval")
    ap.add_argument("--save_cmds", action="store_true", default=False,
                    help="Save stage commands to out_root/cmd_*.txt (default: off)")
    ap.add_argument("--run_metrics_plus", action="store_true", default=False,
                    help="Run metrics_plus.py after metrics.py (default: off)")
    ap.add_argument("--metrics_plus_K", type=int, default=8,
                    help="AlignedPSNR search window for metrics_plus.py (default: 8)")
    ap.add_argument("--metrics_plus_bg", type=int, default=0, choices=[0, 1],
                    help="Background color for metrics_plus.py (default: 0)")
    ap.add_argument("--run_novel_view_metrics", action="store_true", default=False,
                    help="Run novel_view_metrics.py after thermal metrics (default: off)")
    ap.add_argument("--novel_view_N", type=int, default=60,
                    help="Number of novel views for novel_view_metrics.py (default: 60)")
    ap.add_argument("--novel_bg", type=int, default=0, choices=[0, 1],
                    help="Background color for novel_view_metrics.py (default: 0)")
    ap.add_argument("--debug_dump", action="store_true", default=False,
                    help="Write pipeline debug JSON (default: off)")
    ap.add_argument("--debug_dump_path", type=str, default=None,
                    help="Debug JSON path (default: out_root/pipeline_debug.json)")
    ap.add_argument("--profile_pipeline", action="store_true", default=False,
                    help="Enable pipeline profiling (durations/metadata; default: off)")
    ap.add_argument("--profile_out", type=str, default=None,
                    help="Profiling JSON path (default: out_root/pipeline_profile.json)")
    ap.add_argument("--profile_collect_sizes", action="store_true", default=False,
                    help="Collect file/dir sizes in profile (default: off)")
    ap.add_argument("--profile_collect_counts", action="store_true", default=False,
                    help="Collect counts (PLY vertices / image counts) in profile (default: off)")
    ap.add_argument("--profile_save_logs", action="store_true", default=False,
                    help="Redirect each run step stdout+stderr to out_root/log_<step>.txt (default: off)")

    # Step range control (for partial runs / resume)
    ap.add_argument("--from_step", type=int, default=1, help="Execute steps starting from this number (1-14).")
    ap.add_argument("--to_step", type=int, default=14, help="Execute steps up to this number (1-14).")

    # COLMAP defaults copied from your example
    ap.add_argument("--camera", default="SIMPLE_RADIAL")
    ap.add_argument("--matching", default="spatial", choices=["spatial", "exhaustive", "sequential", "vocab_tree"])
    ap.add_argument("--matcher_args", default="--SpatialMatching.max_num_neighbors=80 --SpatialMatching.max_distance=500")
    ap.add_argument("--mapper_multiple_models", type=int, default=1)
    ap.add_argument("--min_model_size", type=int, default=5)
    ap.add_argument("--init_min_num_inliers", type=int, default=50)
    ap.add_argument("--abs_pose_min_num_inliers", type=int, default=20)
    ap.add_argument("--use_model_aligner", action="store_true", default=True)
    ap.add_argument("--model_aligner_args", default="--ref_is_gps=1 --alignment_type=enu --alignment_max_error=30.0")
    ap.add_argument("--prior_position_std_m", type=float, default=1.0)
    ap.add_argument("--wgs84_code", type=int, default=0)

    # Stage 1 training defaults (RGB)
    ap.add_argument("--rgb_iter", type=int, default=30000)
    ap.add_argument("--rgb_res", type=int, default=1)
    ap.add_argument("--rgb_densify_from", type=int, default=1500)
    ap.add_argument("--rgb_densify_until", type=int, default=10000)
    ap.add_argument("--rgb_densify_interval", type=int, default=300)
    ap.add_argument("--rgb_densify_grad", type=float, default=0.001)
    ap.add_argument("--rgb_lambda_dssim", type=float, default=0.3)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    # Sparse Support (Improvement 1) - forwarded to train.py only when enabled
    ap.add_argument("--ss_enable", action="store_true", help="Enable sparse support gating (default: off)")
    ap.add_argument("--ss_enable_rgb", action="store_true", default=False,
                    help="Enable sparse support for RGB stage only (overrides --ss_enable when set)")
    ap.add_argument("--ss_enable_t", action="store_true", default=False,
                    help="Enable sparse support for Thermal stage only (overrides --ss_enable when set)")
    ap.add_argument(
        "--ss_source",
        default="colmap_sparse",
        choices=["colmap_sparse", "init_pcd"],
        help="Sparse support source (default: colmap_sparse)",
    )
    ap.add_argument(
        "--ss_aabb_margin",
        type=float,
        default=0.0,
        help="AABB margin for sparse support (default: 0.0)",
    )
    ap.add_argument(
        "--ss_voxel_size",
        type=float,
        default=None,
        help="Voxel size for VoxelHashNN index (default: None -> AABB-only)",
    )
    ap.add_argument(
        "--ss_nn_dist_thr",
        type=float,
        default=None,
        help="Reserved NN distance threshold for gating (default: None)",
    )

    # Stage 2 training defaults (Thermal)
    ap.add_argument("--t_iter", type=int, default=40000)
    ap.add_argument("--t_res", type=int, default=1)
    ap.add_argument("--t_feature_lr", type=float, default=0.001)
    ap.add_argument("--t_lambda_dssim", type=float, default=0.05)
    ap.add_argument("--ss_prune_before_thermal", action="store_true", default=False,
                    help="Thermal-only: prune outside sparse support after restore (default: off)")
    ap.add_argument("--clamp_scale_max", type=float, default=None,
                    help="Thermal-only: clamp max gaussian scale after restore/prune (default: None)")
    ap.add_argument("--thermal_reset_features", action="store_true", default=False,
                    help="Thermal-only: reset SH features after restore/prune/clamp (default: off)")
    ap.add_argument("--debug_gaussian_stats", action="store_true", default=False,
                    help="Thermal-only: log gaussian stats after restore/before save (default: off)")
    ap.add_argument("--sgf_disable", action="store_true", default=False,
                    help="Thermal-only: disable SGF (reapply LRs after restore) (default: off)")

    # Improvement 4: thermal pseudo-color structure gradient loss (default: disabled)
    ap.add_argument("--t_struct_grad_w", type=float, default=0.0,
                    help="Thermal pseudo-color structure gradient loss weight (0 disables; forwarded to thermal train only).")
    ap.add_argument("--t_struct_grad_norm", type=_str2bool, nargs="?", const=True, default=True,
                    help="Whether to normalize structure grad loss (default: True). Use --t_struct_grad_norm false to disable.")

    # Blend defaults
    ap.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    ap.add_argument("--methods", nargs="+", default=[
        "sh_only", "sh_opacity", "sh_opacity_scale", "sh_opacity_geom",
        "dc_ycc_only", "sh_opacity_dc_ycc"
    ])
    ap.add_argument("--verify_endpoints", action="store_true", default=True)

    # Eval sweep
    ap.add_argument("--auto_render", action="store_true", default=True)

    args = ap.parse_args()

    # Validate step range
    if args.from_step < 1 or args.to_step > 14 or args.from_step > args.to_step:
        ap.error("--from_step/--to_step must satisfy 1 <= from_step <= to_step <= 14")

    # Validate improvement-4 params (always validated; only forwarded when enabled)
    if not math.isfinite(float(getattr(args, "t_struct_grad_w", 0.0))) or float(getattr(args, "t_struct_grad_w", 0.0)) < 0.0:
        ap.error("--t_struct_grad_w must be a finite float >= 0")
    # Sparse Support argument sanity (only when enabled for any stage)
    ss_any_enabled = bool(args.ss_enable or args.ss_enable_rgb or args.ss_enable_t)
    if ss_any_enabled:
        _validate_finite_float(ap, "--ss_aabb_margin", args.ss_aabb_margin, min_value=0.0)
        _validate_finite_float(ap, "--ss_voxel_size", args.ss_voxel_size, strict_positive=True)
        _validate_finite_float(ap, "--ss_nn_dist_thr", args.ss_nn_dist_thr, min_value=0.0)

    ss_stage_override = bool(args.ss_enable_rgb) or bool(args.ss_enable_t)
    if ss_stage_override:
        ss_enable_rgb = bool(args.ss_enable_rgb)
        ss_enable_t = bool(args.ss_enable_t)
    else:
        ss_enable_rgb = bool(args.ss_enable)
        ss_enable_t = bool(args.ss_enable)

    def _build_ss_args_for_stage(enable_flag: bool) -> List[str]:
        if not enable_flag:
            return []
        tmp = argparse.Namespace(**vars(args))
        tmp.ss_enable = True
        try:
            return _build_ss_train_args(tmp, ap=ap)
        except ValueError as e:
            ap.error(str(e))
        return []

    ss_train_extra: List[str] = _build_ss_args_for_stage(ss_enable_rgb)
    ss_train2_extra: List[str] = _build_ss_args_for_stage(ss_enable_t)
    # Improvement-4 forwarding args (only forwarded when enabled; safe no-op otherwise)
    tstruct_train_extra: List[str] = _build_tstruct_train_args(args)
    def _maybe_raise_file_not_found(msg: str) -> None:
        if args.dry_run:
            eprint(f"[WARN] {msg} (dry_run)")
            return
        raise FileNotFoundError(msg)



    gs_root = Path(__file__).resolve().parent  # gaussian-splatting repo root
    py = sys.executable
    metrics_plus_path = gs_root / "metrics_plus.py"
    if args.run_metrics_plus and not metrics_plus_path.exists():
        raise FileNotFoundError(f"metrics_plus.py not found: {metrics_plus_path}")
    novel_view_metrics_path = gs_root / "novel_view_metrics.py"
    if args.run_novel_view_metrics and not novel_view_metrics_path.exists():
        raise FileNotFoundError(f"novel_view_metrics.py not found: {novel_view_metrics_path}")

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()

    rgb_dir = Path(args.rgb_dir).resolve() if args.rgb_dir else (data_root / "RGB")
    th_dir = Path(args.th_dir).resolve() if args.th_dir else (data_root / "thermal")

    fit_dir = data_root / "fit"
    metrics_out = fit_dir / "metrics"
    input_dir = data_root / "input"
    thermal_ud = data_root / "thermal_UD"

    model_rgb = out_root / "Model_RGB"
    model_t = out_root / "Model_T"
    model_f = out_root / "Model_F"
    eval_out = out_root / "eval"

    ensure_dir(out_root)
    train1_cmd = None
    train2_cmd = None
    render1_cmd = None
    render2_cmd = None
    metrics1_cmd = None
    metrics2_cmd = None
    blend_cmd = None
    sweep_cmd = None
    ckpt_rgb = None
    ckpt_t = None

    # State dir for resumable markers
    state_dir = data_root / "_pipeline_state"
    ensure_dir(state_dir)

    def _in_step_range(n: int) -> bool:
        return args.from_step <= n <= args.to_step

    debug_enabled = bool(args.debug_dump)
    debug_dump_written = False
    step_order = [
        "01_cfr",
        "02_eval_crop",
        "03_prepare_input",
        "04_convert_gtgs",
        "05_train_rgb",
        "06_render_rgb",
        "07_metrics_rgb",
        "08_undistort_thermal",
        "09_normalize_sparse_ud",
        "10_train_thermal",
        "11_render_thermal",
        "12_metrics_thermal",
        "13_blend",
        "14_eval_sweep",
    ]
    debug_steps: Dict[str, Dict[str, object]] = {}
    if debug_enabled:
        for name in step_order:
            debug_steps[name] = {
                "decision": "not_reached",
                "marker": str(marker_path(state_dir, name)),
            }

    profile_enabled = bool(args.profile_pipeline)
    profile_dump_written = False
    profile_start_iso = datetime.now().isoformat(timespec="seconds")
    profile_start_perf = time.perf_counter()
    profile_steps: Dict[str, Dict[str, object]] = {}
    profile_run_meta: Dict[str, Dict[str, object]] = {}
    if profile_enabled:
        for name in step_order:
            profile_steps[name] = {"decision": "not_reached"}

    def _format_cmd_for_debug(cmd: List[str]) -> str:
        cmd2 = _normalize_cmd_for_windows(cmd)
        return " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])

    def _now_iso() -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _json_safe(v):
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, dict):
            return {str(k): _json_safe(vv) for k, vv in v.items()}
        if isinstance(v, (list, tuple)):
            return [_json_safe(x) for x in v]
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        return str(v)

    def _collect_artifacts(step_name: str) -> Dict[str, object]:
        rgb_ply = model_rgb / "point_cloud" / f"iteration_{args.rgb_iter}" / "point_cloud.ply"
        t_ply = model_t / "point_cloud" / f"iteration_{args.t_iter}" / "point_cloud.ply"
        paths: Dict[str, Optional[str]] = {
            "out_root": str(out_root),
            "model_rgb": str(model_rgb),
            "model_t": str(model_t),
            "model_f": str(model_f),
            "ckpt_rgb": str(ckpt_rgb) if ckpt_rgb is not None else None,
            "ckpt_t": str(ckpt_t) if ckpt_t is not None else None,
            "rgb_ply": str(rgb_ply),
            "t_ply": str(t_ply),
            "thermal_ud": str(thermal_ud),
            "render_rgb_dir": str(model_rgb / "test"),
            "render_t_dir": str(model_t / "test"),
            "novel_views_dir": str(model_t / "novel_views"),
            "metrics_rgb": str(model_rgb / "results.json"),
            "metrics_t": str(model_t / "results.json"),
        }

        exists = {}
        for k, v in paths.items():
            if v is None:
                exists[k] = False
            else:
                exists[k] = Path(v).exists()

        artifacts: Dict[str, object] = {"paths": paths, "exists": exists}

        if args.profile_collect_sizes:
            sizes: Dict[str, Optional[int]] = {}
            for k in ("ckpt_rgb", "ckpt_t", "rgb_ply", "t_ply", "out_root"):
                v = paths.get(k, None)
                if v is None:
                    sizes[k] = None
                else:
                    sizes[k] = _dir_size_bytes(Path(v))
            artifacts["sizes_bytes"] = sizes

        if args.profile_collect_counts:
            counts: Dict[str, Optional[int]] = {}
            counts["rgb_ply_vertices"] = _ply_vertex_count(rgb_ply)
            counts["t_ply_vertices"] = _ply_vertex_count(t_ply)
            counts["render_rgb_images"] = _count_images_recursive(model_rgb / "test")
            counts["render_t_images"] = _count_images_recursive(model_t / "test")
            counts["novel_views_images"] = _count_images_recursive(model_t / "novel_views")
            artifacts["counts"] = counts

        return artifacts

    def _profile_update_step(
        step_name: str,
        decision: Optional[str] = None,
        cmd: Optional[List[str]] = None,
        outputs_ok: Optional[bool] = None,
        note: str = "",
    ) -> None:
        if not profile_enabled:
            return
        entry = profile_steps.get(step_name, {})
        if decision is not None:
            entry["decision"] = decision
        if cmd is not None:
            entry["cmd"] = _format_cmd_for_debug(cmd)
        if outputs_ok is not None:
            entry["outputs_ok"] = bool(outputs_ok)
        if note:
            entry["note"] = note
        if step_name in profile_run_meta:
            entry.update(profile_run_meta.get(step_name, {}))
        if "start" not in entry:
            now = _now_iso()
            entry["start"] = now
            entry["end"] = now
            entry["duration_s"] = 0.0
            entry["returncode"] = None
        entry["artifacts"] = _collect_artifacts(step_name)
        profile_steps[step_name] = entry

    def _profile_step_start(step_name: Optional[str], cmd: Optional[List[str]] = None) -> Optional[float]:
        if not profile_enabled or not step_name:
            return None
        entry = profile_run_meta.get(step_name, {})
        if "start" not in entry:
            entry["start"] = _now_iso()
            entry["duration_s"] = 0.0
        if cmd is not None:
            entry["cmd"] = _format_cmd_for_debug(cmd)
        profile_run_meta[step_name] = entry
        return time.perf_counter()

    def _profile_step_end(step_name: Optional[str], t0: Optional[float], returncode: Optional[int] = None, log_path: Optional[Path] = None) -> None:
        if not profile_enabled or not step_name or t0 is None:
            return
        entry = profile_run_meta.get(step_name, {})
        entry["end"] = _now_iso()
        entry["duration_s"] = float(entry.get("duration_s", 0.0)) + float(time.perf_counter() - t0)
        if returncode is not None:
            entry["returncode"] = returncode
        else:
            entry.setdefault("returncode", None)
        if log_path is not None:
            entry["log_path"] = str(log_path)
        profile_run_meta[step_name] = entry
        _profile_update_step(step_name, decision="run")

    def _record_step(step_name: str, decision: str, cmd: Optional[List[str]] = None,
                     outputs_ok: Optional[bool] = None, note: str = "") -> None:
        if not debug_enabled:
            pass
        else:
            entry = debug_steps.get(step_name, {})
            entry["decision"] = decision
            if cmd is not None:
                entry["cmd"] = _format_cmd_for_debug(cmd)
            if outputs_ok is not None:
                entry["outputs_ok"] = bool(outputs_ok)
            if note:
                entry["note"] = note
            if "marker" not in entry:
                entry["marker"] = str(marker_path(state_dir, step_name))
            debug_steps[step_name] = entry
        _profile_update_step(step_name, decision=decision, cmd=cmd, outputs_ok=outputs_ok, note=note)

    def _write_debug_dump(reason: str = "") -> None:
        nonlocal debug_dump_written
        if (not debug_enabled) or debug_dump_written:
            return
        debug_dump_written = True
        dump_path = Path(args.debug_dump_path).expanduser() if args.debug_dump_path else (out_root / "pipeline_debug.json")
        ensure_dir(dump_path.parent)

        render_cmd = None
        if render2_cmd is not None and _in_step_range(11):
            render_cmd = render2_cmd
        elif render1_cmd is not None and _in_step_range(6):
            render_cmd = render1_cmd

        metrics_cmd = None
        if metrics2_cmd is not None and _in_step_range(12):
            metrics_cmd = metrics2_cmd
        elif metrics1_cmd is not None and _in_step_range(7):
            metrics_cmd = metrics1_cmd

        payload = {
            "reason": reason,
            "args": _json_safe(vars(args)),
            "cmds": {
                "train1_cmd": _format_cmd_for_debug(train1_cmd) if train1_cmd else None,
                "train2_cmd": _format_cmd_for_debug(train2_cmd) if train2_cmd else None,
                "render_cmd": _format_cmd_for_debug(render_cmd) if render_cmd else None,
                "metrics_cmd": _format_cmd_for_debug(metrics_cmd) if metrics_cmd else None,
                "render1_cmd": _format_cmd_for_debug(render1_cmd) if render1_cmd else None,
                "render2_cmd": _format_cmd_for_debug(render2_cmd) if render2_cmd else None,
                "metrics1_cmd": _format_cmd_for_debug(metrics1_cmd) if metrics1_cmd else None,
                "metrics2_cmd": _format_cmd_for_debug(metrics2_cmd) if metrics2_cmd else None,
            },
            "flags": {
                "ss_enable": bool(getattr(args, "ss_enable", False)),
                "ss_enable_rgb": bool(getattr(args, "ss_enable_rgb", False)),
                "ss_enable_t": bool(getattr(args, "ss_enable_t", False)),
                "ss_source": getattr(args, "ss_source", None),
                "ss_aabb_margin": getattr(args, "ss_aabb_margin", None),
                "ss_voxel_size": getattr(args, "ss_voxel_size", None),
                "ss_nn_dist_thr": getattr(args, "ss_nn_dist_thr", None),
                "ss_prune_before_thermal": bool(getattr(args, "ss_prune_before_thermal", False)),
                "clamp_scale_max": getattr(args, "clamp_scale_max", None),
                "thermal_reset_features": bool(getattr(args, "thermal_reset_features", False)),
                "sgf_disable": bool(getattr(args, "sgf_disable", False)),
                "debug_gaussian_stats": bool(getattr(args, "debug_gaussian_stats", False)),
                "save_cmds": bool(getattr(args, "save_cmds", False)),
                "run_metrics_plus": bool(getattr(args, "run_metrics_plus", False)),
                "metrics_plus_K": getattr(args, "metrics_plus_K", None),
                "metrics_plus_bg": getattr(args, "metrics_plus_bg", None),
                "run_novel_view_metrics": bool(getattr(args, "run_novel_view_metrics", False)),
                "novel_view_N": getattr(args, "novel_view_N", None),
                "novel_bg": getattr(args, "novel_bg", None),
                "dry_run": bool(getattr(args, "dry_run", False)),
            },
            "paths": {
                "data_root": str(data_root),
                "out_root": str(out_root),
                "model_rgb": str(model_rgb),
                "model_t": str(model_t),
                "model_f": str(model_f),
                "ckpt_rgb": str(ckpt_rgb) if ckpt_rgb is not None else None,
                "ckpt_t": str(ckpt_t) if ckpt_t is not None else None,
                "thermal_ud": str(thermal_ud),
                "state_dir": str(state_dir),
            },
            "steps": [_json_safe(debug_steps.get(name, {})) for name in step_order],
        }
        dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        eprint(f"[INFO] DebugDump: {dump_path}")

    def _write_profile_dump(reason: str = "") -> None:
        nonlocal profile_dump_written
        if (not profile_enabled) or profile_dump_written:
            return
        profile_dump_written = True
        dump_path = Path(args.profile_out).expanduser() if args.profile_out else (out_root / "pipeline_profile.json")
        ensure_dir(dump_path.parent)

        time_end = _now_iso()
        total_s = time.perf_counter() - profile_start_perf

        payload = {
            "meta": {
                "time_start": profile_start_iso,
                "time_end": time_end,
                "total_s": float(total_s),
                "cwd": str(Path.cwd()),
                "python": sys.executable,
                "reason": reason or "done",
                "args": _json_safe(vars(args)),
                "flags": {
                    "ss_enable": bool(getattr(args, "ss_enable", False)),
                    "ss_enable_rgb": bool(getattr(args, "ss_enable_rgb", False)),
                    "ss_enable_t": bool(getattr(args, "ss_enable_t", False)),
                },
            },
            "steps": _json_safe(profile_steps),
        }
        dump_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        eprint(f"[INFO] ProfileDump: {dump_path}")

    profile_reason = {"value": ""}
    if profile_enabled:
        import atexit

        def _profile_excepthook(exctype, value, tb):
            profile_reason["value"] = f"error:{getattr(exctype, '__name__', 'Exception')}"
            _write_profile_dump(profile_reason["value"])
            sys.__excepthook__(exctype, value, tb)

        sys.excepthook = _profile_excepthook
        atexit.register(lambda: _write_profile_dump(profile_reason["value"]))


    def maybe_run(cmd: List[str], cwd: Optional[Path] = None, step_name: Optional[str] = None):
        t0 = _profile_step_start(step_name, cmd)
        log_path: Optional[Path] = None
        if args.profile_save_logs and step_name:
            log_path = out_root / f"log_{step_name}.txt"
        if args.dry_run:
            cmd2 = _normalize_cmd_for_windows(cmd)
            pretty = " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])
            eprint(f"\n[DRY] {pretty}")
            _profile_step_end(step_name, t0, returncode=None, log_path=log_path)
            return
        rc = None
        try:
            if log_path is not None:
                ensure_dir(log_path.parent)
                with log_path.open("a", encoding="utf-8") as f:
                    run_cmd(cmd, cwd=cwd, stdout=f, stderr=f)
            else:
                run_cmd(cmd, cwd=cwd)
            rc = 0
        except subprocess.CalledProcessError as e:
            rc = e.returncode
            raise
        finally:
            _profile_step_end(step_name, t0, returncode=rc, log_path=log_path)

    # Expected CFR outputs
    cand_fit = fit_dir / "image" / "image-fit"
    cand_exif = fit_dir / "image" / "image-exif"

    # -------- 1) CFR
    if args.clean_fit and fit_dir.exists():
        eprint(f"[INFO] Cleaning fit dir: {fit_dir}")
        shutil.rmtree(fit_dir)

    cfr_cmd = [py, "cfr.py", "--rgb_dir", str(rgb_dir), "--th_dir", str(th_dir), "--out_dir", str(fit_dir), "--align", "both", "--stage", "both" ,"--comparison"]
    if args.comparison:
        cfr_cmd.append("--comparison")

    cfr_outputs_ok = cand_fit.exists() and cand_exif.exists() and (len(list_images(cand_fit)) > 0) and (len(list_images(cand_exif)) > 0)
    if not _in_step_range(1):
        eprint("[SKIP] 01_cfr (outside selected step range)")
        _record_step("01_cfr", "skip_range", cfr_cmd, outputs_ok=cfr_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "01_cfr", cfr_cmd, outputs_ok=cfr_outputs_ok, force=args.force)
        if skip:
            _record_step("01_cfr", "skip", cfr_cmd, outputs_ok=cfr_outputs_ok)
        else:
            ensure_dir(fit_dir)
            maybe_run(cfr_cmd, cwd=gs_root, step_name="01_cfr")
            # re-evaluate
            cfr_outputs_ok = cand_fit.exists() and cand_exif.exists() and (len(list_images(cand_fit)) > 0) and (len(list_images(cand_exif)) > 0)
            if not cfr_outputs_ok:
                _maybe_raise_file_not_found(f"Expected cfr outputs missing. Need both:\n  {cand_fit}\n  {cand_exif}")
            _record_step("01_cfr", "run", cfr_cmd, outputs_ok=cfr_outputs_ok)
            write_marker(marker_path(state_dir, "01_cfr"), "01_cfr", cfr_cmd, cwd=gs_root)

    # -------- 2) Evaluate crop candidates (fit + exif)
    ensure_dir(metrics_out)
    summary_all = metrics_out / "summary_all.json"
    eval_cmd_base = [py, "eval_crop_metrics.py", "--th_dir", str(th_dir),
                     "--rgb_dir", str(cand_fit), "--rgb_dir", str(cand_exif),
                     "--tag", "fit", "--tag", "exif",
                     "--out_dir", str(metrics_out)]
    eval_outputs_ok = summary_all.exists() and summary_all.stat().st_size > 50

    if not _in_step_range(2):

        eprint("[SKIP] 02_eval_crop (outside selected step range)")

        _record_step("02_eval_crop", "skip_range", eval_cmd_base, outputs_ok=eval_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "02_eval_crop", eval_cmd_base, outputs_ok=eval_outputs_ok, force=args.force)
        if skip:
            _record_step("02_eval_crop", "skip", eval_cmd_base, outputs_ok=eval_outputs_ok)
        else:
            # Try with --ssim first (if available), then fallback without it.
            try:
                maybe_run(eval_cmd_base + ["--ssim"], cwd=gs_root, step_name="02_eval_crop")
            except subprocess.CalledProcessError:
                eprint("[WARN] eval_crop_metrics.py failed with --ssim. Retrying without --ssim ...")
                maybe_run(eval_cmd_base, cwd=gs_root, step_name="02_eval_crop")
            eval_outputs_ok = summary_all.exists() and summary_all.stat().st_size > 50
            if not eval_outputs_ok:
                _maybe_raise_file_not_found(f"summary_all.json not found or empty: {summary_all}")
            _record_step("02_eval_crop", "run", eval_cmd_base, outputs_ok=eval_outputs_ok)
            write_marker(marker_path(state_dir, "02_eval_crop"), "02_eval_crop", eval_cmd_base, cwd=gs_root)

    # -------- 3) Decide which candidate to use, then prepare input/
    if args.align == "fit":
        chosen_tag = "fit"
        chosen_dir = cand_fit
    elif args.align == "exif":
        chosen_tag = "exif"
        chosen_dir = cand_exif
    else:
        best = pick_best_candidate(summary_all)
        chosen_tag = best.tag
        chosen_dir = best.rgb_dir
        eprint(f"[INFO] Auto-picked candidate: {chosen_tag}  (from {summary_all})")

    # Prepare input dir
    if args.clean_input and input_dir.exists():
        eprint(f"[INFO] Cleaning input dir: {input_dir}")
        shutil.rmtree(input_dir)

    prep_cmd = [py, "-c", f"print('prepare_input: {chosen_tag} -> {input_dir}')"]  # marker cmd (informational)
    prep_outputs_ok = input_dir.exists() and (len(list_images(input_dir)) > 0)

    if not _in_step_range(3):

        eprint("[SKIP] 03_prepare_input (outside selected step range)")

        _record_step("03_prepare_input", "skip_range", prep_cmd, outputs_ok=prep_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "03_prepare_input", prep_cmd, outputs_ok=prep_outputs_ok, force=args.force)
        if skip:
            _record_step("03_prepare_input", "skip", prep_cmd, outputs_ok=prep_outputs_ok)
        else:
            t0 = _profile_step_start("03_prepare_input", prep_cmd)
            rc = 0
            try:
                prepare_input_dir(chosen_dir, data_root, clean=False, link_mode=args.link_mode)
            except FileNotFoundError as e:
                rc = 1
                _maybe_raise_file_not_found(str(e))
            finally:
                _profile_step_end("03_prepare_input", t0, returncode=rc)
            prep_outputs_ok = input_dir.exists() and (len(list_images(input_dir)) > 0)
            if not prep_outputs_ok:
                _maybe_raise_file_not_found(f"input dir has no images: {input_dir}")
            _record_step("03_prepare_input", "run", prep_cmd, outputs_ok=prep_outputs_ok)
            write_marker(marker_path(state_dir, "03_prepare_input"), "03_prepare_input", prep_cmd, cwd=gs_root, note=f"chosen={chosen_tag}; src={chosen_dir}")

    # -------- 4) COLMAP (convert-gtgs.py)
    sparse_aligned = data_root / "distorted" / "sparse_aligned"
    convert_cmd = [
        py, "convert-gtgs.py",
        "-s", str(data_root),
        "--colmap_executable", str(args.colmap),
        "--exiftool_executable", str(args.exiftool),
        "--wgs84_code", str(args.wgs84_code),
        "--prior_position_std_m", str(args.prior_position_std_m),
        "--camera", str(args.camera),
        "--matching", str(args.matching),
        "--matcher_args", str(args.matcher_args),
        "--mapper_multiple_models", str(args.mapper_multiple_models),
        "--min_model_size", str(args.min_model_size),
        "--init_min_num_inliers", str(args.init_min_num_inliers),
        "--abs_pose_min_num_inliers", str(args.abs_pose_min_num_inliers),
        "--use_model_aligner",
        "--model_aligner_args", str(args.model_aligner_args),
    ]

    convert_outputs_ok = sparse_aligned.exists() and contains_any_file(sparse_aligned, ("cameras.bin", "cameras.txt"), max_depth=3)

    if not _in_step_range(4):

        eprint("[SKIP] 04_convert_gtgs (outside selected step range)")

        _record_step("04_convert_gtgs", "skip_range", convert_cmd, outputs_ok=convert_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "04_convert_gtgs", convert_cmd, outputs_ok=convert_outputs_ok, force=args.force)
        if skip:
            _record_step("04_convert_gtgs", "skip", convert_cmd, outputs_ok=convert_outputs_ok)
        else:
            maybe_run(convert_cmd, cwd=gs_root, step_name="04_convert_gtgs")
            convert_outputs_ok = sparse_aligned.exists() and contains_any_file(sparse_aligned, ("cameras.bin", "cameras.txt"), max_depth=3)
            if not convert_outputs_ok:
                _maybe_raise_file_not_found(f"Aligned sparse model not found/invalid: {sparse_aligned}")
            _record_step("04_convert_gtgs", "run", convert_cmd, outputs_ok=convert_outputs_ok)
            write_marker(marker_path(state_dir, "04_convert_gtgs"), "04_convert_gtgs", convert_cmd, cwd=gs_root)

    # Optional early stop after COLMAP
    if args.to_step <= 4:
        eprint("[INFO] Step range ends at 04_convert_gtgs. Done.")
        profile_reason["value"] = "end_at_step4"
        _write_debug_dump("end_at_step4")
        return

    if args.skip_train:
        if args.from_step > 4:
            eprint("[INFO] --skip_train ignored because --from_step > 4 (you selected later steps).")
        else:
            eprint("[INFO] --skip_train set. Stopping after COLMAP.")
            profile_reason["value"] = "skip_train"
            _write_debug_dump("skip_train")
            return

    # -------- 5) Stage-1 training (RGB)
    ensure_dir(model_rgb)
    ckpt_rgb = model_rgb / f"chkpnt{args.rgb_iter}.pth"

    train1_cmd = [
        py, "train.py",
        "-s", str(data_root),
        "--images", "images",
        "-m", str(model_rgb),
        "-r", str(args.rgb_res),
        "--iterations", str(args.rgb_iter),
        "--checkpoint_iterations", str(args.rgb_iter),
        "--data_device", str(args.device),
        "--eval",
        "--densify_from_iter", str(args.rgb_densify_from),
        "--densify_until_iter", str(args.rgb_densify_until),
        "--densification_interval", str(args.rgb_densify_interval),
        "--densify_grad_threshold", str(args.rgb_densify_grad),
        "--lambda_dssim", str(args.rgb_lambda_dssim),
    ]

    # Forward sparse support opts only when enabled for RGB stage.
    if ss_train_extra:
        train1_cmd.extend(ss_train_extra)

    train1_outputs_ok = ckpt_rgb.exists()
    if not _in_step_range(5):
        eprint("[SKIP] 05_train_rgb (outside selected step range)")
        _record_step("05_train_rgb", "skip_range", train1_cmd, outputs_ok=train1_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "05_train_rgb", train1_cmd, outputs_ok=train1_outputs_ok, force=args.force)
        if skip:
            _record_step("05_train_rgb", "skip", train1_cmd, outputs_ok=train1_outputs_ok)
        else:
            maybe_run(train1_cmd, cwd=gs_root, step_name="05_train_rgb")
            train1_outputs_ok = ckpt_rgb.exists()
            if not train1_outputs_ok:
                _maybe_raise_file_not_found(f"RGB checkpoint not found after training: {ckpt_rgb}")
            _record_step("05_train_rgb", "run", train1_cmd, outputs_ok=train1_outputs_ok)
            write_marker(marker_path(state_dir, "05_train_rgb"), "05_train_rgb", train1_cmd, cwd=gs_root)

    # Render RGB (keep -r consistent with training to avoid mismatched intrinsics/resolution)
    render1_cmd = [py, "render.py", "-m", str(model_rgb), "-s", str(data_root), "-r", str(args.rgb_res)]
    render1_outputs_ok = (model_rgb / "test").exists() and contains_any_file(model_rgb / "test", ("00000.png",), max_depth=5)  # weak check
    # Better: any image in test dir
    if (model_rgb / "test").exists():
        try:
            render1_outputs_ok = any(p.suffix.lower() in (".png", ".jpg", ".jpeg") for p in (model_rgb / "test").rglob("*"))
        except Exception:
            pass

    if not _in_step_range(6):

        eprint("[SKIP] 06_render_rgb (outside selected step range)")

        _record_step("06_render_rgb", "skip_range", render1_cmd, outputs_ok=render1_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "06_render_rgb", render1_cmd, outputs_ok=render1_outputs_ok, force=args.force)
        if skip:
            _record_step("06_render_rgb", "skip", render1_cmd, outputs_ok=render1_outputs_ok)
        else:
            maybe_run(render1_cmd, cwd=gs_root, step_name="06_render_rgb")
            _record_step("06_render_rgb", "run", render1_cmd, outputs_ok=render1_outputs_ok)
            write_marker(marker_path(state_dir, "06_render_rgb"), "06_render_rgb", render1_cmd, cwd=gs_root)

    metrics1_cmd = [py, "metrics.py", "-m", str(model_rgb)]
    metrics1_outputs_ok = (model_rgb / "results.json").exists() or (model_rgb / "results.txt").exists()
    if not _in_step_range(7):
        eprint("[SKIP] 07_metrics_rgb (outside selected step range)")
        _record_step("07_metrics_rgb", "skip_range", metrics1_cmd, outputs_ok=metrics1_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "07_metrics_rgb", metrics1_cmd, outputs_ok=metrics1_outputs_ok, force=args.force)
        if skip:
            _record_step("07_metrics_rgb", "skip", metrics1_cmd, outputs_ok=metrics1_outputs_ok)
        else:
            maybe_run(metrics1_cmd, cwd=gs_root, step_name="07_metrics_rgb")
            if args.run_metrics_plus:
                metrics_plus_cmd = [py, "metrics_plus.py", "-m", str(model_rgb),
                                    "--K", str(args.metrics_plus_K), "--bg", str(args.metrics_plus_bg)]
                maybe_run(metrics_plus_cmd, cwd=gs_root, step_name="07_metrics_rgb")
            _record_step("07_metrics_rgb", "run", metrics1_cmd, outputs_ok=metrics1_outputs_ok)
            # even if we can't detect output, write marker so reruns can skip
            write_marker(marker_path(state_dir, "07_metrics_rgb"), "07_metrics_rgb", metrics1_cmd, cwd=gs_root)

    # -------- 6) Undistort thermal using aligned sparse model
    if args.clean_thermal_ud and thermal_ud.exists():
        eprint(f"[INFO] Cleaning existing thermal_UD: {thermal_ud}")
        shutil.rmtree(thermal_ud)

    undistort_cmd = [
        str(args.colmap), "image_undistorter",
        "--image_path", str(th_dir),
        "--input_path", str(sparse_aligned),
        "--output_path", str(thermal_ud),
        "--output_type", "COLMAP",
    ]
    undistort_outputs_ok = thermal_ud.exists() and (thermal_ud / "images").exists() and (len(list_images(thermal_ud / "images")) > 0) and (thermal_ud / "sparse").exists()
    # Preflight: step 08 requires sparse_aligned from step 04
    if _in_step_range(8):
        if not sparse_aligned.exists() or not contains_any_file(sparse_aligned, ("cameras.bin", "cameras.txt"), max_depth=3):
            _maybe_raise_file_not_found(
                "Missing COLMAP sparse_aligned model required for thermal undistort.\n"
                f"Expected: {sparse_aligned} (with cameras.bin/txt).\n"
                "Run with --from_step 4 (or earlier), or fix your COLMAP outputs."
            )
        if not th_dir.exists():
            _maybe_raise_file_not_found(f"Thermal directory not found: {th_dir}")

    if not _in_step_range(8):
        eprint("[SKIP] 08_undistort_thermal (outside selected step range)")
        _record_step("08_undistort_thermal", "skip_range", undistort_cmd, outputs_ok=undistort_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "08_undistort_thermal", undistort_cmd, outputs_ok=undistort_outputs_ok, force=args.force)
        if skip:
            _record_step("08_undistort_thermal", "skip", undistort_cmd, outputs_ok=undistort_outputs_ok)
        else:
            maybe_run(undistort_cmd, cwd=gs_root, step_name="08_undistort_thermal")
            undistort_outputs_ok = thermal_ud.exists() and (thermal_ud / "images").exists() and (len(list_images(thermal_ud / "images")) > 0) and (thermal_ud / "sparse").exists()
            if not undistort_outputs_ok:
                _maybe_raise_file_not_found(f"thermal_UD seems incomplete: {thermal_ud}")
            _record_step("08_undistort_thermal", "run", undistort_cmd, outputs_ok=undistort_outputs_ok)
            write_marker(marker_path(state_dir, "08_undistort_thermal"), "08_undistort_thermal", undistort_cmd, cwd=gs_root)

    # -------- 7) Normalize sparse layout for thermal_UD
    sparse_dir_ud = thermal_ud / "sparse"
    norm_cmd = [py, "-c", f"print('ensure_sparse_0: {sparse_dir_ud}')"]  # marker cmd
    norm_outputs_ok = (sparse_dir_ud / "0").exists() and contains_any_file(sparse_dir_ud / "0", ("cameras.bin", "cameras.txt"), max_depth=1)

    if not _in_step_range(9):

        eprint("[SKIP] 09_normalize_sparse_ud (outside selected step range)")

        _record_step("09_normalize_sparse_ud", "skip_range", norm_cmd, outputs_ok=norm_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "09_normalize_sparse_ud", norm_cmd, outputs_ok=norm_outputs_ok, force=args.force)
        if skip:
            _record_step("09_normalize_sparse_ud", "skip", norm_cmd, outputs_ok=norm_outputs_ok)
        else:
            t0 = _profile_step_start("09_normalize_sparse_ud", norm_cmd)
            rc = 0
            try:
                ensure_sparse_0(sparse_dir_ud)
            except FileNotFoundError as e:
                rc = 1
                _maybe_raise_file_not_found(str(e))
            finally:
                _profile_step_end("09_normalize_sparse_ud", t0, returncode=rc)
        norm_outputs_ok = (sparse_dir_ud / "0").exists() and contains_any_file(sparse_dir_ud / "0", ("cameras.bin", "cameras.txt"), max_depth=1)
        if not norm_outputs_ok:
            _maybe_raise_file_not_found(f"thermal_UD sparse/0 not found or missing cameras.*: {sparse_dir_ud / '0'}")
        _record_step("09_normalize_sparse_ud", "run", norm_cmd, outputs_ok=norm_outputs_ok)
        write_marker(marker_path(state_dir, "09_normalize_sparse_ud"), "09_normalize_sparse_ud", norm_cmd, cwd=gs_root)

    # -------- 8) Stage-2 training (Thermal)
    ensure_dir(model_t)
    ckpt_t = model_t / f"chkpnt{args.t_iter}.pth"
    if not ckpt_rgb.exists():
        _maybe_raise_file_not_found(f"Start checkpoint not found: {ckpt_rgb}")

    train2_cmd = [
        py, "train.py",
        "-s", str(thermal_ud),
        "--images", "images",
        "-m", str(model_t),
        "--start_checkpoint", str(ckpt_rgb),
        "-r", str(args.t_res),
        "--iterations", str(args.t_iter),
        "--checkpoint_iterations", str(args.t_iter),

        # Freeze geometry-related params
        "--position_lr_init", "0", "--position_lr_final", "0",
        "--scaling_lr", "0", "--rotation_lr", "0",
        "--opacity_lr", "0",

        "--feature_lr", str(args.t_feature_lr),

        # Disable densification & opacity resets
        "--densify_from_iter", "999999",
        "--densify_until_iter", "0",
        "--densification_interval", "999999",
        "--opacity_reset_interval", "999999",

        "--lambda_dssim", str(args.t_lambda_dssim),
        "--eval",
    ]

    if ss_train2_extra:
        train2_cmd.extend(ss_train2_extra)
    if args.ss_prune_before_thermal:
        train2_cmd.append("--ss_prune_before_thermal")
    if args.clamp_scale_max is not None:
        train2_cmd.extend(["--clamp_scale_max", str(args.clamp_scale_max)])
    if args.thermal_reset_features:
        train2_cmd.append("--thermal_reset_features")
    if args.debug_gaussian_stats:
        train2_cmd.append("--debug_gaussian_stats")
    if args.sgf_disable:
        train2_cmd.append("--sgf_disable")

    train2_cmd.extend(tstruct_train_extra)
    train2_outputs_ok = ckpt_t.exists()
    # Preflight: step 10 requires stage-1 checkpoint and thermal_UD dataset
    if _in_step_range(10):
        if not ckpt_rgb.exists():
            _maybe_raise_file_not_found(
                f"Stage-1 RGB checkpoint missing: {ckpt_rgb}\n"
                "Run with --from_step 5 (or earlier) to train RGB first, or set --rgb_iter to match an existing checkpoint."
            )
        if not thermal_ud.exists() or not (thermal_ud / "images").exists():
            _maybe_raise_file_not_found(f"thermal_UD dataset missing: {thermal_ud} (need images/). Run step 08 first.")

    if not _in_step_range(10):
        eprint("[SKIP] 10_train_thermal (outside selected step range)")
        _record_step("10_train_thermal", "skip_range", train2_cmd, outputs_ok=train2_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "10_train_thermal", train2_cmd, outputs_ok=train2_outputs_ok, force=args.force)
        if skip:
            _record_step("10_train_thermal", "skip", train2_cmd, outputs_ok=train2_outputs_ok)
        else:
            maybe_run(train2_cmd, cwd=gs_root, step_name="10_train_thermal")
            train2_outputs_ok = ckpt_t.exists()
            if not train2_outputs_ok:
                _maybe_raise_file_not_found(f"Thermal checkpoint not found after training: {ckpt_t}")
            _record_step("10_train_thermal", "run", train2_cmd, outputs_ok=train2_outputs_ok)
            write_marker(marker_path(state_dir, "10_train_thermal"), "10_train_thermal", train2_cmd, cwd=gs_root)

    render2_cmd = [py, "render.py", "-m", str(model_t), "-s", str(thermal_ud), "-r", str(args.t_res)]
    render2_outputs_ok = (model_t / "test").exists()
    if (model_t / "test").exists():
        try:
            render2_outputs_ok = any(p.suffix.lower() in (".png", ".jpg", ".jpeg") for p in (model_t / "test").rglob("*"))
        except Exception:
            pass

    if not _in_step_range(11):

        eprint("[SKIP] 11_render_thermal (outside selected step range)")

        _record_step("11_render_thermal", "skip_range", render2_cmd, outputs_ok=render2_outputs_ok)

    else:
        skip = should_skip_step(state_dir, "11_render_thermal", render2_cmd, outputs_ok=render2_outputs_ok, force=args.force)
        if skip:
            _record_step("11_render_thermal", "skip", render2_cmd, outputs_ok=render2_outputs_ok)
        else:
            maybe_run(render2_cmd, cwd=gs_root, step_name="11_render_thermal")
            _record_step("11_render_thermal", "run", render2_cmd, outputs_ok=render2_outputs_ok)
            write_marker(marker_path(state_dir, "11_render_thermal"), "11_render_thermal", render2_cmd, cwd=gs_root)

    metrics2_cmd = [py, "metrics.py", "-m", str(model_t)]
    metrics2_outputs_ok = (model_t / "results.json").exists() or (model_t / "results.txt").exists()
    if args.save_cmds:
        def _format_cmd_for_save(cmd: List[str]) -> str:
            cmd2 = _normalize_cmd_for_windows(cmd)
            return " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])

        def _save_cmd(path: Path, cmd: List[str]) -> None:
            path.write_text(_format_cmd_for_save(cmd) + "\n", encoding="utf-8")

        render_cmd = render2_cmd if _in_step_range(11) else render1_cmd
        metrics_cmd = metrics2_cmd if _in_step_range(12) else metrics1_cmd
        _save_cmd(out_root / "cmd_train1.txt", train1_cmd)
        _save_cmd(out_root / "cmd_train2.txt", train2_cmd)
        _save_cmd(out_root / "cmd_render.txt", render_cmd)
        _save_cmd(out_root / "cmd_metrics.txt", metrics_cmd)
        eprint(f"[INFO] SavedCmds: {out_root}")
    if not _in_step_range(12):
        eprint("[SKIP] 12_metrics_thermal (outside selected step range)")
        _record_step("12_metrics_thermal", "skip_range", metrics2_cmd, outputs_ok=metrics2_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "12_metrics_thermal", metrics2_cmd, outputs_ok=metrics2_outputs_ok, force=args.force)
        if skip:
            _record_step("12_metrics_thermal", "skip", metrics2_cmd, outputs_ok=metrics2_outputs_ok)
        else:
            maybe_run(metrics2_cmd, cwd=gs_root, step_name="12_metrics_thermal")
            if args.run_metrics_plus:
                metrics_plus_cmd = [py, "metrics_plus.py", "-m", str(model_t),
                                    "--K", str(args.metrics_plus_K), "--bg", str(args.metrics_plus_bg)]
                maybe_run(metrics_plus_cmd, cwd=gs_root, step_name="12_metrics_thermal")
            if args.run_novel_view_metrics:
                novel_cmd = [py, "novel_view_metrics.py", "-m", str(model_t),
                             "--N", str(args.novel_view_N), "--bg", str(args.novel_bg)]
                maybe_run(novel_cmd, cwd=gs_root, step_name="12_metrics_thermal")
            _record_step("12_metrics_thermal", "run", metrics2_cmd, outputs_ok=metrics2_outputs_ok)
            write_marker(marker_path(state_dir, "12_metrics_thermal"), "12_metrics_thermal", metrics2_cmd, cwd=gs_root)

    # Optional early stop after stage-2
    if args.to_step <= 12:
        eprint("[INFO] Step range ends at 12_metrics_thermal. Done.")
        profile_reason["value"] = "end_at_step12"
        _write_debug_dump("end_at_step12")
        return

    if args.skip_blend:
        if args.from_step > 12:
            eprint("[INFO] --skip_blend ignored because --from_step > 12 (you selected later steps).")
        else:
            eprint("[INFO] --skip_blend set. Stopping after stage-2 training.")
            profile_reason["value"] = "skip_blend"
            _write_debug_dump("skip_blend")
            return

    # -------- 9) Blend models
    if args.clean_blend_out and model_f.exists():
        eprint(f"[INFO] Cleaning existing blend output: {model_f}")
        shutil.rmtree(model_f)

    blend_cmd = [
        py, "blend_model_strict_endpoints.py",
        "--rgb_model_dir", str(model_rgb), "--rgb_iter", str(args.rgb_iter),
        "--t_model_dir", str(model_t), "--t_iter", str(args.t_iter),
        "--alphas", str(args.alphas),
        "--out_root", str(model_f),
        "--out_iter", str(args.t_iter),
        "--methods",
    ] + list(args.methods)

    if args.verify_endpoints:
        blend_cmd.append("--verify_endpoints")
    # only add clean_out when explicitly requested (so resume works)
    if args.clean_blend_out:
        blend_cmd.append("--clean_out")

    blend_outputs_ok = model_f.exists() and any(p.is_dir() for p in model_f.iterdir())
    # Preflight: step 13 requires RGB/T trained point clouds at requested iterations
    if _in_step_range(13):
        rgb_ply = model_rgb / "point_cloud" / f"iteration_{args.rgb_iter}" / "point_cloud.ply"
        t_ply = model_t / "point_cloud" / f"iteration_{args.t_iter}" / "point_cloud.ply"
        if not rgb_ply.exists():
            _maybe_raise_file_not_found(
                f"RGB point cloud missing for blend: {rgb_ply}\n"
                "Run step 05 (RGB train) first, or set --rgb_iter to an existing iteration."
            )
        if not t_ply.exists():
            _maybe_raise_file_not_found(
                f"Thermal point cloud missing for blend: {t_ply}\n"
                "Run step 10 (thermal train) first, or set --t_iter to an existing iteration."
            )

    if not _in_step_range(13):
        eprint("[SKIP] 13_blend (outside selected step range)")
        _record_step("13_blend", "skip_range", blend_cmd, outputs_ok=blend_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "13_blend", blend_cmd, outputs_ok=blend_outputs_ok, force=args.force)
        if skip:
            _record_step("13_blend", "skip", blend_cmd, outputs_ok=blend_outputs_ok)
        else:
            maybe_run(blend_cmd, cwd=gs_root, step_name="13_blend")
            blend_outputs_ok = model_f.exists() and any(p.is_dir() for p in model_f.iterdir())
            if not blend_outputs_ok:
                _maybe_raise_file_not_found(f"Blend output looks empty: {model_f}")
            _record_step("13_blend", "run", blend_cmd, outputs_ok=blend_outputs_ok)
            write_marker(marker_path(state_dir, "13_blend"), "13_blend", blend_cmd, cwd=gs_root)

    # -------- 10) Evaluate sweep
    ensure_dir(eval_out)
    sweep_cmd = [
        py, "eval_blend_sweep.py",
        "--sweep_root", str(model_f),
        "--rgb_render", str(model_rgb),
        "--t_render", str(model_t),
        "--out_dir", str(eval_out),
    ]
    if args.auto_render:
        sweep_cmd.append("--auto_render")

    sweep_outputs_ok = (eval_out / "summary.csv").exists() and (eval_out / "summary.csv").stat().st_size > 50
    if not _in_step_range(14):
        eprint("[SKIP] 14_eval_sweep (outside selected step range)")
        _record_step("14_eval_sweep", "skip_range", sweep_cmd, outputs_ok=sweep_outputs_ok)
    else:
        skip = should_skip_step(state_dir, "14_eval_sweep", sweep_cmd, outputs_ok=sweep_outputs_ok, force=args.force)
        if skip:
            _record_step("14_eval_sweep", "skip", sweep_cmd, outputs_ok=sweep_outputs_ok)
        else:
            maybe_run(sweep_cmd, cwd=gs_root, step_name="14_eval_sweep")
            sweep_outputs_ok = (eval_out / "summary.csv").exists() and (eval_out / "summary.csv").stat().st_size > 50
            if not sweep_outputs_ok:
                eprint("[WARN] eval_blend_sweep finished but summary.csv not found; please check logs.")
            _record_step("14_eval_sweep", "run", sweep_cmd, outputs_ok=sweep_outputs_ok)
            write_marker(marker_path(state_dir, "14_eval_sweep"), "14_eval_sweep", sweep_cmd, cwd=gs_root)

    profile_reason["value"] = "done"
    _write_debug_dump("done")
    eprint("\n[DONE] Full pipeline finished.")


if __name__ == "__main__":
    main()
