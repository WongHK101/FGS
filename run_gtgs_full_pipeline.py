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

def run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> None:
    cmd2 = _normalize_cmd_for_windows(cmd)
    cwd_str = str(cwd) if cwd else None
    pretty = " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])
    eprint(f"\n[RUN] {pretty}")
    subprocess.run(cmd2, cwd=cwd_str, env=env, check=True)


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
    # Sparse Support argument sanity (only when enabled)
    ss_train_extra: List[str] = []
    if args.ss_enable:
        _validate_finite_float(ap, "--ss_aabb_margin", args.ss_aabb_margin, min_value=0.0)
        _validate_finite_float(ap, "--ss_voxel_size", args.ss_voxel_size, strict_positive=True)
        _validate_finite_float(ap, "--ss_nn_dist_thr", args.ss_nn_dist_thr, min_value=0.0)
        try:
            ss_train_extra = _build_ss_train_args(args, ap=ap)
        except ValueError as e:
            ap.error(str(e))
    # Improvement-4 forwarding args (only forwarded when enabled; safe no-op otherwise)
    tstruct_train_extra: List[str] = _build_tstruct_train_args(args)



    gs_root = Path(__file__).resolve().parent  # gaussian-splatting repo root
    py = sys.executable

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

    # State dir for resumable markers
    state_dir = data_root / "_pipeline_state"
    ensure_dir(state_dir)

    def _in_step_range(n: int) -> bool:
        return args.from_step <= n <= args.to_step


    def maybe_run(cmd: List[str], cwd: Optional[Path] = None):
        if args.dry_run:
            cmd2 = _normalize_cmd_for_windows(cmd)
            pretty = " ".join([f'"{c}"' if (" " in c or "\t" in c) else c for c in cmd2])
            eprint(f"\n[DRY] {pretty}")
            return
        run_cmd(cmd, cwd=cwd)

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
    elif not should_skip_step(state_dir, "01_cfr", cfr_cmd, outputs_ok=cfr_outputs_ok, force=args.force):
        ensure_dir(fit_dir)
        maybe_run(cfr_cmd, cwd=gs_root)
        # re-evaluate
        cfr_outputs_ok = cand_fit.exists() and cand_exif.exists() and (len(list_images(cand_fit)) > 0) and (len(list_images(cand_exif)) > 0)
        if not cfr_outputs_ok:
            raise FileNotFoundError(f"Expected cfr outputs missing. Need both:\n  {cand_fit}\n  {cand_exif}")
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

    elif not should_skip_step(state_dir, "02_eval_crop", eval_cmd_base, outputs_ok=eval_outputs_ok, force=args.force):
        # Try with --ssim first (if available), then fallback without it.
        try:
            maybe_run(eval_cmd_base + ["--ssim"], cwd=gs_root)
        except subprocess.CalledProcessError:
            eprint("[WARN] eval_crop_metrics.py failed with --ssim. Retrying without --ssim ...")
            maybe_run(eval_cmd_base, cwd=gs_root)
        eval_outputs_ok = summary_all.exists() and summary_all.stat().st_size > 50
        if not eval_outputs_ok:
            raise FileNotFoundError(f"summary_all.json not found or empty: {summary_all}")
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

    elif not should_skip_step(state_dir, "03_prepare_input", prep_cmd, outputs_ok=prep_outputs_ok, force=args.force):
        prepare_input_dir(chosen_dir, data_root, clean=False, link_mode=args.link_mode)
        prep_outputs_ok = input_dir.exists() and (len(list_images(input_dir)) > 0)
        if not prep_outputs_ok:
            raise FileNotFoundError(f"input dir has no images: {input_dir}")
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

    elif not should_skip_step(state_dir, "04_convert_gtgs", convert_cmd, outputs_ok=convert_outputs_ok, force=args.force):
        maybe_run(convert_cmd, cwd=gs_root)
        convert_outputs_ok = sparse_aligned.exists() and contains_any_file(sparse_aligned, ("cameras.bin", "cameras.txt"), max_depth=3)
        if not convert_outputs_ok:
            raise FileNotFoundError(f"Aligned sparse model not found/invalid: {sparse_aligned}")
        write_marker(marker_path(state_dir, "04_convert_gtgs"), "04_convert_gtgs", convert_cmd, cwd=gs_root)

    # Optional early stop after COLMAP
    if args.to_step <= 4:
        eprint("[INFO] Step range ends at 04_convert_gtgs. Done.")
        return

    if args.skip_train:
        if args.from_step > 4:
            eprint("[INFO] --skip_train ignored because --from_step > 4 (you selected later steps).")
        else:
            eprint("[INFO] --skip_train set. Stopping after COLMAP.")
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

    # Forward sparse support opts only when explicitly enabled.
    if args.ss_enable:
        train1_cmd.extend(ss_train_extra)

    train1_outputs_ok = ckpt_rgb.exists()
    if not _in_step_range(5):
        eprint("[SKIP] 05_train_rgb (outside selected step range)")
    elif not should_skip_step(state_dir, "05_train_rgb", train1_cmd, outputs_ok=train1_outputs_ok, force=args.force):
        maybe_run(train1_cmd, cwd=gs_root)
        train1_outputs_ok = ckpt_rgb.exists()
        if not train1_outputs_ok:
            raise FileNotFoundError(f"RGB checkpoint not found after training: {ckpt_rgb}")
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

    elif not should_skip_step(state_dir, "06_render_rgb", render1_cmd, outputs_ok=render1_outputs_ok, force=args.force):
        maybe_run(render1_cmd, cwd=gs_root)
        write_marker(marker_path(state_dir, "06_render_rgb"), "06_render_rgb", render1_cmd, cwd=gs_root)

    metrics1_cmd = [py, "metrics.py", "-m", str(model_rgb)]
    metrics1_outputs_ok = (model_rgb / "results.json").exists() or (model_rgb / "results.txt").exists()
    if not _in_step_range(7):
        eprint("[SKIP] 07_metrics_rgb (outside selected step range)")
    elif not should_skip_step(state_dir, "07_metrics_rgb", metrics1_cmd, outputs_ok=metrics1_outputs_ok, force=args.force):
        maybe_run(metrics1_cmd, cwd=gs_root)
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
            raise FileNotFoundError(
                "Missing COLMAP sparse_aligned model required for thermal undistort.\n"
                f"Expected: {sparse_aligned} (with cameras.bin/txt).\n"
                "Run with --from_step 4 (or earlier), or fix your COLMAP outputs."
            )
        if not th_dir.exists():
            raise FileNotFoundError(f"Thermal directory not found: {th_dir}")

    if not _in_step_range(8):
        eprint("[SKIP] 08_undistort_thermal (outside selected step range)")
    elif not should_skip_step(state_dir, "08_undistort_thermal", undistort_cmd, outputs_ok=undistort_outputs_ok, force=args.force):
        maybe_run(undistort_cmd, cwd=gs_root)
        undistort_outputs_ok = thermal_ud.exists() and (thermal_ud / "images").exists() and (len(list_images(thermal_ud / "images")) > 0) and (thermal_ud / "sparse").exists()
        if not undistort_outputs_ok:
            raise FileNotFoundError(f"thermal_UD seems incomplete: {thermal_ud}")
        write_marker(marker_path(state_dir, "08_undistort_thermal"), "08_undistort_thermal", undistort_cmd, cwd=gs_root)

    # -------- 7) Normalize sparse layout for thermal_UD
    sparse_dir_ud = thermal_ud / "sparse"
    norm_cmd = [py, "-c", f"print('ensure_sparse_0: {sparse_dir_ud}')"]  # marker cmd
    norm_outputs_ok = (sparse_dir_ud / "0").exists() and contains_any_file(sparse_dir_ud / "0", ("cameras.bin", "cameras.txt"), max_depth=1)

    if not _in_step_range(9):

        eprint("[SKIP] 09_normalize_sparse_ud (outside selected step range)")

    elif not should_skip_step(state_dir, "09_normalize_sparse_ud", norm_cmd, outputs_ok=norm_outputs_ok, force=args.force):
        ensure_sparse_0(sparse_dir_ud)
        norm_outputs_ok = (sparse_dir_ud / "0").exists() and contains_any_file(sparse_dir_ud / "0", ("cameras.bin", "cameras.txt"), max_depth=1)
        if not norm_outputs_ok:
            raise FileNotFoundError(f"thermal_UD sparse/0 not found or missing cameras.*: {sparse_dir_ud / '0'}")
        write_marker(marker_path(state_dir, "09_normalize_sparse_ud"), "09_normalize_sparse_ud", norm_cmd, cwd=gs_root)

    # -------- 8) Stage-2 training (Thermal)
    ensure_dir(model_t)
    ckpt_t = model_t / f"chkpnt{args.t_iter}.pth"
    if not ckpt_rgb.exists():
        raise FileNotFoundError(f"Start checkpoint not found: {ckpt_rgb}")

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

    train2_cmd.extend(tstruct_train_extra)
    train2_outputs_ok = ckpt_t.exists()
    # Preflight: step 10 requires stage-1 checkpoint and thermal_UD dataset
    if _in_step_range(10):
        if not ckpt_rgb.exists():
            raise FileNotFoundError(
                f"Stage-1 RGB checkpoint missing: {ckpt_rgb}\n"
                "Run with --from_step 5 (or earlier) to train RGB first, or set --rgb_iter to match an existing checkpoint."
            )
        if not thermal_ud.exists() or not (thermal_ud / "images").exists():
            raise FileNotFoundError(f"thermal_UD dataset missing: {thermal_ud} (need images/). Run step 08 first.")

    if not _in_step_range(10):
        eprint("[SKIP] 10_train_thermal (outside selected step range)")
    elif not should_skip_step(state_dir, "10_train_thermal", train2_cmd, outputs_ok=train2_outputs_ok, force=args.force):
        maybe_run(train2_cmd, cwd=gs_root)
        train2_outputs_ok = ckpt_t.exists()
        if not train2_outputs_ok:
            raise FileNotFoundError(f"Thermal checkpoint not found after training: {ckpt_t}")
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

    elif not should_skip_step(state_dir, "11_render_thermal", render2_cmd, outputs_ok=render2_outputs_ok, force=args.force):
        maybe_run(render2_cmd, cwd=gs_root)
        write_marker(marker_path(state_dir, "11_render_thermal"), "11_render_thermal", render2_cmd, cwd=gs_root)

    metrics2_cmd = [py, "metrics.py", "-m", str(model_t)]
    metrics2_outputs_ok = (model_t / "results.json").exists() or (model_t / "results.txt").exists()
    if not _in_step_range(12):
        eprint("[SKIP] 12_metrics_thermal (outside selected step range)")
    elif not should_skip_step(state_dir, "12_metrics_thermal", metrics2_cmd, outputs_ok=metrics2_outputs_ok, force=args.force):
        maybe_run(metrics2_cmd, cwd=gs_root)
        write_marker(marker_path(state_dir, "12_metrics_thermal"), "12_metrics_thermal", metrics2_cmd, cwd=gs_root)

    # Optional early stop after stage-2
    if args.to_step <= 12:
        eprint("[INFO] Step range ends at 12_metrics_thermal. Done.")
        return

    if args.skip_blend:
        if args.from_step > 12:
            eprint("[INFO] --skip_blend ignored because --from_step > 12 (you selected later steps).")
        else:
            eprint("[INFO] --skip_blend set. Stopping after stage-2 training.")
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
            raise FileNotFoundError(
                f"RGB point cloud missing for blend: {rgb_ply}\n"
                "Run step 05 (RGB train) first, or set --rgb_iter to an existing iteration."
            )
        if not t_ply.exists():
            raise FileNotFoundError(
                f"Thermal point cloud missing for blend: {t_ply}\n"
                "Run step 10 (thermal train) first, or set --t_iter to an existing iteration."
            )

    if not _in_step_range(13):
        eprint("[SKIP] 13_blend (outside selected step range)")
    elif not should_skip_step(state_dir, "13_blend", blend_cmd, outputs_ok=blend_outputs_ok, force=args.force):
        maybe_run(blend_cmd, cwd=gs_root)
        blend_outputs_ok = model_f.exists() and any(p.is_dir() for p in model_f.iterdir())
        if not blend_outputs_ok:
            raise FileNotFoundError(f"Blend output looks empty: {model_f}")
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
    elif not should_skip_step(state_dir, "14_eval_sweep", sweep_cmd, outputs_ok=sweep_outputs_ok, force=args.force):
        maybe_run(sweep_cmd, cwd=gs_root)
        sweep_outputs_ok = (eval_out / "summary.csv").exists() and (eval_out / "summary.csv").stat().st_size > 50
        if not sweep_outputs_ok:
            eprint("[WARN] eval_blend_sweep finished but summary.csv not found; please check logs.")
        write_marker(marker_path(state_dir, "14_eval_sweep"), "14_eval_sweep", sweep_cmd, cwd=gs_root)

    eprint("\n[DONE] Full pipeline finished.")


if __name__ == "__main__":
    main()
