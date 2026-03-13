#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


DATASETS = ["PVpanel", "Orchard", "Building", "Road", "TransmissionTower"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_reason(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    lines = [ln.strip() for ln in log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-80:] if ln.strip()]
    if not lines:
        return ""
    return " | ".join(lines[-3:])


def _run_logged(cmd: List[str], cwd: Path, log_path: Path) -> int:
    _ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=f)
        return int(proc.returncode)


def _run_raw_direct_tt(py: str, repo_root: Path, input_root: Path, necessity_root: Path) -> None:
    dataset = "TransmissionTower"
    setting = "raw_direct"
    workdata_root = necessity_root / "workdata" / setting / dataset
    out_root = necessity_root / "runs" / setting / dataset
    log_path = necessity_root / "logs" / f"{setting}__{dataset}.log"
    status_path = necessity_root / "campaign_status.json"
    campaign_log = necessity_root / "run_cfr_necessity.log"
    cmd = [
        py, "run_gtgs_full_pipeline.py",
        "--data_root", str(workdata_root),
        "--out_root", str(out_root),
        "--align", "raw",
        "--from_step", "1",
        "--to_step", "10",
        "--rgb_iter", "1000",
        "--t_iter", "2000",
        "--rgb_res", "8",
        "--t_res", "8",
        "--skip_blend",
        "--force",
        "--clean_input",
        "--clean_fit",
        "--clean_thermal_ud",
        "--no_comparison",
        "--save_cmds",
    ]
    run_meta = {
        "dataset": dataset,
        "setting": setting,
        "data_root": str(workdata_root),
        "out_root": str(out_root),
        "status": "running",
        "command": cmd,
        "log_path": str(log_path),
    }
    _write_json(out_root / "run_meta.json", run_meta)
    status = _load_json(status_path)
    status[f"{setting}/{dataset}"] = {"status": "running", "log_path": str(log_path), "out_root": str(out_root)}
    _write_json(status_path, status)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with campaign_log.open("a", encoding="utf-8") as clog:
        clog.write(f"[{stamp}] RESTART {setting}/{dataset}\n")
    t0 = time.time()
    rc = _run_logged(cmd, repo_root, log_path)
    duration_s = time.time() - t0
    final_status = "done" if rc == 0 else "failed"
    reason = "" if rc == 0 else _tail_reason(log_path)
    run_meta.update({"status": final_status, "returncode": rc, "duration_s": duration_s, "failure_reason": reason})
    _write_json(out_root / "run_meta.json", run_meta)
    status = _load_json(status_path)
    status[f"{setting}/{dataset}"] = {
        "status": final_status,
        "returncode": rc,
        "duration_s": duration_s,
        "log_path": str(log_path),
        "out_root": str(out_root),
    }
    _write_json(status_path, status)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with campaign_log.open("a", encoding="utf-8") as clog:
        clog.write(f"[{stamp}] END {setting}/{dataset} status={final_status} duration_s={duration_s:.3f}\n")


def _fit_reuse_stub(fit_root: Path, out_root: Path, dataset: str) -> None:
    dst = out_root / "fit_reuse" / dataset
    _ensure_dir(dst)
    (dst / "SOURCE_PATH.txt").write_text(str(fit_root / dataset), encoding="utf-8")


def _run_exif_pass(py: str, repo_root: Path, exif_work_root: Path, out_root: Path, dataset: str, pass_name: str, from_step: int, to_step: int, skip_blend: bool) -> Dict[str, object]:
    run_root = out_root / "runs" / dataset
    log_path = out_root / "logs" / f"{dataset}__{pass_name}.log"
    cmd = [
        py, "run_gtgs_full_pipeline.py",
        "--data_root", str(exif_work_root / dataset),
        "--out_root", str(run_root),
        "--align", "exif",
        "--from_step", str(from_step),
        "--to_step", str(to_step),
        "--rgb_iter", "30000",
        "--t_iter", "60000",
        "--rgb_res", "4",
        "--t_res", "4",
        "--force",
        "--save_cmds",
    ]
    if skip_blend:
        cmd.append("--skip_blend")
    t0 = time.time()
    rc = _run_logged(cmd, repo_root, log_path)
    return {
        "status": "done" if rc == 0 else "failed",
        "returncode": rc,
        "duration_s": time.time() - t0,
        "log_path": str(log_path),
        "command": cmd,
        "failure_reason": "" if rc == 0 else _tail_reason(log_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Finish CFR necessity and launch EXIF-vs-FIT full-quality runs.")
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--fit_root", required=True)
    ap.add_argument("--necessity_root", required=True)
    ap.add_argument("--exif_full_root", required=True)
    ap.add_argument("--python_exe", default=sys.executable)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    input_root = Path(args.input_root)
    fit_root = Path(args.fit_root)
    necessity_root = Path(args.necessity_root)
    exif_full_root = Path(args.exif_full_root)
    _ensure_dir(exif_full_root / "runs")
    _ensure_dir(exif_full_root / "logs")
    _ensure_dir(exif_full_root / "fit_reuse")
    _ensure_dir(exif_full_root / "Summaries")

    # 1) Finish raw_direct/TransmissionTower and CFR necessity summaries.
    _run_raw_direct_tt(args.python_exe, repo_root, input_root, necessity_root)
    subprocess.run(
        [
            args.python_exe,
            "summarize_cfr_necessity.py",
            "--input_root", str(input_root),
            "--fit_root", str(fit_root),
            "--workdata_root", str(necessity_root / "workdata"),
            "--runs_root", str(necessity_root / "runs"),
            "--out_dir", str(necessity_root / "Summaries"),
        ],
        cwd=str(repo_root),
        check=True,
    )

    # 2) Run exif full quality campaign.
    status_path = exif_full_root / "campaign_status.json"
    campaign_log = exif_full_root / "run_exif_full.log"
    status = {}
    for dataset in DATASETS:
        _fit_reuse_stub(fit_root, exif_full_root, dataset)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with campaign_log.open("a", encoding="utf-8") as clog:
            clog.write(f"[{stamp}] START exif_full/{dataset} pass_rgb\n")
        rgb = _run_exif_pass(args.python_exe, repo_root, necessity_root / "workdata" / "exif_only", exif_full_root, dataset, "pass_rgb", 5, 7, True)
        status[f"exif_full/{dataset}/pass_rgb"] = rgb
        _write_json(status_path, status)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with campaign_log.open("a", encoding="utf-8") as clog:
            clog.write(f"[{stamp}] END exif_full/{dataset} pass_rgb status={rgb['status']} duration_s={rgb['duration_s']:.3f}\n")
        if rgb["status"] != "done":
            continue

        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with campaign_log.open("a", encoding="utf-8") as clog:
            clog.write(f"[{stamp}] START exif_full/{dataset} pass_thermal_fusion\n")
        tf = _run_exif_pass(args.python_exe, repo_root, necessity_root / "workdata" / "exif_only", exif_full_root, dataset, "pass_thermal_fusion", 10, 14, False)
        status[f"exif_full/{dataset}/pass_thermal_fusion"] = tf
        _write_json(status_path, status)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with campaign_log.open("a", encoding="utf-8") as clog:
            clog.write(f"[{stamp}] END exif_full/{dataset} pass_thermal_fusion status={tf['status']} duration_s={tf['duration_s']:.3f}\n")

    subprocess.run(
        [
            args.python_exe,
            "summarize_cfr_final_quality.py",
            "--fit_root", str(fit_root),
            "--fit_source_csv", r"F:\databackup\xr6\output\Summaries\Source_xr6_main.csv",
            "--exif_full_root", str(exif_full_root / "runs"),
            "--status_json", str(exif_full_root / "campaign_status.json"),
            "--fit_input_root", str(input_root),
            "--exif_work_root", str(necessity_root / "workdata" / "exif_only"),
            "--out_dir", str(exif_full_root / "Summaries"),
        ],
        cwd=str(repo_root),
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
