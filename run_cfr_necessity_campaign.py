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


def _junction(link: Path, target: Path) -> None:
    if link.exists():
        return
    _ensure_dir(link.parent)
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True)


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


def _prepare_workdata(input_root: Path, workdata_root: Path, setting: str, dataset: str) -> Path:
    src = input_root / dataset
    dst = workdata_root / setting / dataset
    _ensure_dir(dst)
    _junction(dst / "RGB", src / "RGB")
    _junction(dst / "thermal", src / "thermal")
    (dst / "SOURCE_ROOT.txt").write_text(str(src), encoding="utf-8")
    return dst


def _fit_reuse_stub(fit_root: Path, out_root: Path, dataset: str) -> None:
    src = fit_root / dataset
    dst = out_root / "fit_reuse" / dataset
    _ensure_dir(dst)
    (dst / "SOURCE_PATH.txt").write_text(str(src), encoding="utf-8")
    meta = {
        "dataset": dataset,
        "setting": "fit",
        "status": "reused",
        "source_path": str(src),
    }
    _write_json(dst / "run_meta.json", meta)


def _run_one(py: str, repo_root: Path, input_root: Path, workdata_root: Path, runs_root: Path, logs_root: Path,
             fit_root: Path, campaign_log: Path, status_path: Path, setting: str, dataset: str) -> None:
    if setting == "fit":
        _fit_reuse_stub(fit_root, runs_root.parent, dataset)
        return

    data_root = _prepare_workdata(input_root, workdata_root, setting, dataset)
    out_root = runs_root / setting / dataset
    _ensure_dir(out_root)
    log_path = logs_root / f"{setting}__{dataset}.log"
    align = "exif" if setting == "exif_only" else "raw"
    cmd: List[str] = [
        py, "run_gtgs_full_pipeline.py",
        "--data_root", str(data_root),
        "--out_root", str(out_root),
        "--align", align,
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
    meta = {
        "dataset": dataset,
        "setting": setting,
        "data_root": str(data_root),
        "out_root": str(out_root),
        "status": "running",
        "command": cmd,
        "log_path": str(log_path),
    }
    _write_json(out_root / "run_meta.json", meta)
    status_obj = _load_json(status_path)
    status_obj[f"{setting}/{dataset}"] = {"status": "running", "log_path": str(log_path), "out_root": str(out_root)}
    _write_json(status_path, status_obj)

    t0 = time.time()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with campaign_log.open("a", encoding="utf-8") as clog:
        clog.write(f"[{stamp}] START {setting}/{dataset}\n")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        f.flush()
        rc = None
        try:
            subprocess.run(cmd, cwd=str(repo_root), check=True, stdout=f, stderr=f)
            rc = 0
            status = "done"
            failure_reason = ""
        except subprocess.CalledProcessError as exc:
            rc = exc.returncode
            status = "failed"
            failure_reason = _tail_reason(log_path)
        duration_s = time.time() - t0
    meta.update({
        "status": status,
        "returncode": rc,
        "duration_s": duration_s,
        "failure_reason": failure_reason,
    })
    _write_json(out_root / "run_meta.json", meta)
    status_obj = _load_json(status_path)
    status_obj[f"{setting}/{dataset}"] = {
        "status": status,
        "returncode": rc,
        "duration_s": duration_s,
        "log_path": str(log_path),
        "out_root": str(out_root),
    }
    _write_json(status_path, status_obj)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with campaign_log.open("a", encoding="utf-8") as clog:
        clog.write(f"[{stamp}] END {setting}/{dataset} status={status} duration_s={duration_s:.3f}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run CFR necessity experiments (fit reuse + exif/raw smoke runs).")
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--fit_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--python_exe", default=sys.executable)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    input_root = Path(args.input_root)
    fit_root = Path(args.fit_root)
    out_root = Path(args.out_root)
    workdata_root = out_root / "workdata"
    runs_root = out_root / "runs"
    logs_root = out_root / "logs"
    summaries_root = out_root / "Summaries"
    _ensure_dir(workdata_root)
    _ensure_dir(runs_root)
    _ensure_dir(logs_root)
    _ensure_dir(summaries_root)
    status_path = out_root / "campaign_status.json"
    _write_json(status_path, {})

    # Fit rows are reused, not rerun.
    for dataset in DATASETS:
        _fit_reuse_stub(fit_root, out_root, dataset)

    campaign_log = out_root / "run_cfr_necessity.log"
    for setting in ("exif_only", "raw_direct"):
        for dataset in DATASETS:
            _run_one(args.python_exe, repo_root, input_root, workdata_root, runs_root, logs_root, fit_root, campaign_log, status_path, setting, dataset)

    subprocess.run(
        [
            args.python_exe,
            "summarize_cfr_necessity.py",
            "--input_root", str(input_root),
            "--fit_root", str(fit_root),
            "--workdata_root", str(workdata_root),
            "--runs_root", str(runs_root),
            "--out_dir", str(summaries_root),
        ],
        cwd=str(repo_root),
        check=True,
    )
    status_obj = _load_json(status_path)
    status_obj["summary"] = {"status": "done", "out_dir": str(summaries_root)}
    _write_json(status_path, status_obj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
