#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


DATASETS = ["PVpanel", "Orchard", "Building", "Road", "TransmissionTower"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, obj: Dict) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_logged(cmd: List[str], cwd: Path, log_path: Path) -> Tuple[int, float]:
    _ensure_dir(log_path.parent)
    t0 = time.time()
    with log_path.open("a", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=f).returncode
    return int(rc), float(time.time() - t0)


def _base_pipeline_cmd(
    py: str,
    data_root: Path,
    out_root: Path,
    align: str,
) -> List[str]:
    return [
        py,
        "run_gtgs_full_pipeline.py",
        "--data_root",
        str(data_root),
        "--out_root",
        str(out_root),
        "--align",
        align,
        "--from_step",
        "1",
        "--to_step",
        "10",
        "--rgb_iter",
        "1000",
        "--t_iter",
        "2000",
        "--rgb_res",
        "8",
        "--t_res",
        "8",
        "--skip_blend",
        "--force",
        "--clean_input",
        "--clean_fit",
        "--clean_thermal_ud",
        "--no_comparison",
        "--save_cmds",
    ]


def _run_one(
    *,
    py: str,
    repo_root: Path,
    input_root: Path,
    out_root: Path,
    exp_name: str,
    dataset: str,
    align: str,
    cfr_extra: List[str],
    status_path: Path,
) -> None:
    exp_dir = out_root / exp_name / dataset
    data_root = input_root / dataset
    log_path = out_root / exp_name / "logs" / f"{dataset}.log"
    cmd = _base_pipeline_cmd(py, data_root, exp_dir, align) + cfr_extra
    key = f"{exp_name}/{dataset}"

    status = _load_json(status_path)
    prev = status.get(key, {})
    if isinstance(prev, dict) and prev.get("status") == "done":
        return

    status[key] = {
        "status": "running",
        "dataset": dataset,
        "exp_name": exp_name,
        "data_root": str(data_root),
        "out_root": str(exp_dir),
        "log_path": str(log_path),
        "command": cmd,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(status_path, status)

    rc, duration_s = _run_logged(cmd, repo_root, log_path)
    status = _load_json(status_path)
    status[key] = {
        **status.get(key, {}),
        "status": "done" if rc == 0 else "failed",
        "returncode": int(rc),
        "duration_s": float(duration_s),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(status_path, status)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run CFR CVPR supplement (E4/E1/E5) with resume.")
    ap.add_argument("--input_root", default=r"F:\databackup\xr6\input")
    ap.add_argument("--out_root", default=r"F:\databackup\xr6\output\CFR_CVPR_Supplement")
    ap.add_argument("--python_exe", default=sys.executable)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    _ensure_dir(out_root)
    status_path = out_root / "campaign_status.json"
    if not status_path.exists():
        _write_json(status_path, {"meta": {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "root": str(out_root)}})

    # E4: COLMAP-friendly ablation (fit-full / fit-noKupdate / fit-naiveK)
    e4_variants = [
        ("E4_COLMAP_Ablation/fit_full", "fit", ["--cfr_fit_k_mode", "full"]),
        ("E4_COLMAP_Ablation/fit_noKupdate", "fit", ["--cfr_fit_k_mode", "no_kupdate"]),
        ("E4_COLMAP_Ablation/fit_naiveK", "fit", ["--cfr_fit_k_mode", "naive_k"]),
    ]
    for exp_name, align, extra in e4_variants:
        for ds in DATASETS:
            _run_one(
                py=args.python_exe,
                repo_root=repo_root,
                input_root=input_root,
                out_root=out_root,
                exp_name=exp_name,
                dataset=ds,
                align=align,
                cfr_extra=extra,
                status_path=status_path,
            )

    # E1: EXIF noise robustness (EXIF-only vs Ours-CFR-fit)
    e1_levels = [
        ("n00", 0.0, False),
        ("n05", 5.0, False),
        ("n10", 10.0, False),
        ("n20", 20.0, False),
        ("missing", 0.0, True),
    ]
    for tag, noise_pct, missing in e1_levels:
        for method_name, align in (("exif_only", "exif"), ("ours_fit", "fit")):
            extra = []
            if noise_pct > 0:
                extra += ["--cfr_exif_noise_pct", str(noise_pct)]
            if missing:
                extra += ["--cfr_exif_missing"]
            exp_name = f"E1_EXIF_Noise/{method_name}_{tag}"
            for ds in DATASETS:
                _run_one(
                    py=args.python_exe,
                    repo_root=repo_root,
                    input_root=input_root,
                    out_root=out_root,
                    exp_name=exp_name,
                    dataset=ds,
                    align=align,
                    cfr_extra=extra,
                    status_path=status_path,
                )

    # E5: fit aggregation ablation (median / mean / per_pair)
    e5_variants = [
        ("E5_Fit_Aggregation/median", ["--cfr_fit_agg_mode", "median"]),
        ("E5_Fit_Aggregation/mean", ["--cfr_fit_agg_mode", "mean"]),
        ("E5_Fit_Aggregation/per_pair", ["--cfr_fit_agg_mode", "per_pair"]),
    ]
    for exp_name, extra in e5_variants:
        for ds in DATASETS:
            _run_one(
                py=args.python_exe,
                repo_root=repo_root,
                input_root=input_root,
                out_root=out_root,
                exp_name=exp_name,
                dataset=ds,
                align="fit",
                cfr_extra=extra,
                status_path=status_path,
            )

    # Mark pending groups so downstream summary scripts can track TODOs explicitly.
    status = _load_json(status_path)
    for pending in ("E2_Overlap_Stress", "E3_Res_Stress", "E6_External", "E7_Qualitative_Failure_Board"):
        status.setdefault(pending, {"status": "pending"})
    status["meta"] = {
        **status.get("meta", {}),
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_blocks": ["E4_COLMAP_Ablation", "E1_EXIF_Noise", "E5_Fit_Aggregation"],
        "pending_blocks": ["E2_Overlap_Stress", "E3_Res_Stress", "E6_External", "E7_Qualitative_Failure_Board"],
    }
    _write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
