import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


METHODS = ["Ours", "ThermalGaussian_OMMG", "Thermal3D_GS", "ThermoNeRF"]
DATASETS = ["PVpanel", "Orchard", "Building", "Road", "TransmissionTower"]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _sorted_files(path: Path, pattern: str = "*") -> List[Path]:
    return sorted([p for p in path.glob(pattern) if p.is_file()])


def _ours_pairs(root: Path, dataset: str) -> List[Tuple[Path, Path]]:
    src_root = Path((root / "Ours" / dataset / "SOURCE_PATH.txt").read_text(encoding="utf-8").strip())
    model_t = src_root / "Model_T" / "test" / "ours_60000"
    renders = _sorted_files(model_t / "renders")
    gts = _sorted_files(model_t / "gt")
    return list(zip(renders, gts))


def _ommg_pairs(root: Path, dataset: str) -> List[Tuple[Path, Path]]:
    base = root / "ThermalGaussian_OMMG" / dataset / "test" / "ours_30000"
    renders = _sorted_files(base / "renders_thermal")
    gts = _sorted_files(base / "gt_thermal")
    return list(zip(renders, gts))


def _thermal3d_pairs(root: Path, dataset: str) -> List[Tuple[Path, Path]]:
    base = root / "Thermal3D_GS" / dataset / "test" / "ours_30000"
    renders = _sorted_files(base / "renders")
    gts = _sorted_files(base / "gt")
    return list(zip(renders, gts))


def _thermonerf_pairs(root: Path, thermonerf_data_root: Path, dataset: str) -> List[Tuple[Path, Path]]:
    eval_dir = root / "ThermoNeRF" / dataset / "eval"
    render_files = sorted(
        p for p in eval_dir.glob("thermal_*.jpg") if not p.name.startswith("thermal_combined_")
    )
    tf_path = thermonerf_data_root / dataset / "transforms.json"
    obj = json.loads(tf_path.read_text(encoding="utf-8"))
    frame_map: Dict[str, str] = {}
    for frame in obj["frames"]:
        frame_map[frame["file_path"].replace("\\", "/")] = frame["thermal_file_path"].replace("\\", "/")
    gt_files = []
    for rel in obj.get("test_filenames", []):
        thermal_rel = frame_map[rel.replace("\\", "/")]
        gt_files.append((thermonerf_data_root / dataset / thermal_rel).resolve())
    if len(render_files) != len(gt_files):
        raise RuntimeError(f"ThermoNeRF pair count mismatch for {dataset}: renders={len(render_files)} gt={len(gt_files)}")
    return list(zip(render_files, gt_files))


def _pairs_for_method(root: Path, thermonerf_data_root: Path, method: str, dataset: str) -> List[Tuple[Path, Path]]:
    if method == "Ours":
        return _ours_pairs(root, dataset)
    if method == "ThermalGaussian_OMMG":
        return _ommg_pairs(root, dataset)
    if method == "Thermal3D_GS":
        return _thermal3d_pairs(root, dataset)
    if method == "ThermoNeRF":
        return _thermonerf_pairs(root, thermonerf_data_root, dataset)
    raise KeyError(method)


def _prepare_scene(root: Path, unified_root: Path, thermonerf_data_root: Path, method: str, dataset: str, force: bool) -> Path:
    scene_dir = unified_root / method / dataset
    renders_dir = scene_dir / "test" / method / "renders"
    gt_dir = scene_dir / "test" / method / "gt"
    if force and scene_dir.exists():
        shutil.rmtree(scene_dir)
    renders_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    pairs = _pairs_for_method(root, thermonerf_data_root, method, dataset)
    if not pairs:
        raise RuntimeError(f"No pairs found for {method}/{dataset}")
    for idx, (render_src, gt_src) in enumerate(pairs):
        render_dst = renders_dir / f"{idx:05d}{render_src.suffix.lower()}"
        gt_dst = gt_dir / f"{idx:05d}{gt_src.suffix.lower()}"
        _link_or_copy(render_src, render_dst)
        _link_or_copy(gt_src, gt_dst)
    return scene_dir


def _run_cmd(cmd: List[str], workdir: Path) -> None:
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(workdir), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--thermonerf_data_root", default=r"E:\3DGS\ThermoNeRF\dataset\input")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--force", action="store_true", default=False)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--bg", type=int, default=0)
    ap.add_argument("--edge_thr", type=float, default=0.1)
    ap.add_argument("--extra_iqa", type=str, default="flip,dists,fsim,vif,ms-ssim,gmsd,haarpsi,niqe,brisque,piqe,hdrvdp3")
    ap.add_argument("--extra_iqa_space", type=str, default="y")
    ap.add_argument("--extra_iqa_device", type=str, default="cuda")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    root = Path(args.root)
    unified_root = root / "UnifiedEval"
    thermonerf_data_root = Path(args.thermonerf_data_root)

    for method in args.methods:
        for dataset in args.datasets:
            print(f"\n=== {method} / {dataset} ===")
            scene_dir = _prepare_scene(root, unified_root, thermonerf_data_root, method, dataset, args.force)
            if args.force or not (scene_dir / "results.json").exists():
                _run_cmd([sys.executable, "metrics.py", "-m", str(scene_dir)], repo_root)
            if args.force or not (scene_dir / "results_plus.json").exists():
                _run_cmd(
                    [
                        sys.executable,
                        "metrics_plus.py",
                        "-m",
                        str(scene_dir),
                        "--save_json",
                        "--K",
                        str(args.k),
                        "--bg",
                        str(args.bg),
                        "--edge_thr",
                        str(args.edge_thr),
                        "--extra_iqa",
                        args.extra_iqa,
                        "--extra_iqa_space",
                        args.extra_iqa_space,
                        "--extra_iqa_device",
                        args.extra_iqa_device,
                    ],
                    repo_root,
                )


if __name__ == "__main__":
    main()
