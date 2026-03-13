#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DATASETS = ["PVpanel", "Orchard", "Building", "Road", "TransmissionTower"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str], cwd: Path, log_path: Path) -> int:
    _ensure_dir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("CMD: " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=f)
        return proc.returncode


def _official_sample_pairs() -> Dict[str, Dict[str, str]]:
    return {
        "xoftr": {
            "fig1": r"E:\3DGS\RGBT2D\XoFTR\assets\METU_VisTIR_samples\cloudy\scene_7\visible\images\IM_04525.jpg",
            "fig2": r"E:\3DGS\RGBT2D\XoFTR\assets\METU_VisTIR_samples\cloudy\scene_7\thermal\images\IM_01139.jpg",
        },
        "minima_xoftr": {
            "fig1": r"E:\3DGS\RGBT2D\MINIMA\third_party\XoFTR\assets\METU_VisTIR_samples\cloudy\scene_7\visible\images\IM_04525.jpg",
            "fig2": r"E:\3DGS\RGBT2D\MINIMA\third_party\XoFTR\assets\METU_VisTIR_samples\cloudy\scene_7\thermal\images\IM_01139.jpg",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run CFR 2D paper campaign.")
    ap.add_argument("--campaign_root", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--repo_root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--xoftr_python", default=r"D:\anaconda\envs\xoftr\python.exe")
    ap.add_argument("--minima_python", default=r"D:\anaconda\envs\minima\python.exe")
    ap.add_argument("--xoftr_repo", default=r"E:\3DGS\RGBT2D\XoFTR")
    ap.add_argument("--minima_repo", default=r"E:\3DGS\RGBT2D\MINIMA")
    ap.add_argument("--xoftr_weight", default=r"E:\3DGS\RGBT2D\XoFTR\weights\weights_xoftr_640.ckpt")
    ap.add_argument("--minima_weight", default=r"E:\3DGS\RGBT2D\MINIMA\weights\minima_xoftr.ckpt")
    ap.add_argument("--cfr_necessity_root", required=True)
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    campaign_root = Path(args.campaign_root)
    dataset_root = Path(args.dataset_root)
    s0_root = campaign_root / "S0_OfficialSanity"
    s2_root = campaign_root / "S2_Main_2DRawReplacement"
    s3_root = campaign_root / "S3_Appendix_AssistedInputAnalysis"
    summaries_root = campaign_root / "Summaries"
    logs_root = campaign_root / "logs"
    _ensure_dir(campaign_root)
    _ensure_dir(s0_root)
    _ensure_dir(s2_root)
    _ensure_dir(s3_root)
    _ensure_dir(summaries_root)
    _ensure_dir(logs_root)

    manifest: Dict[str, object] = {
        "campaign_root": str(campaign_root),
        "dataset_root": str(dataset_root),
        "reused_s1_root": str(Path(args.cfr_necessity_root).resolve()),
        "steps": {},
    }

    s1_root = campaign_root / "S1_Main_CFRNecessity"
    _ensure_dir(s1_root)
    (s1_root / "SOURCE_PATH.txt").write_text(str(Path(args.cfr_necessity_root).resolve()), encoding="utf-8")
    manifest["steps"]["S1_Main_CFRNecessity"] = {"status": "reused", "source": str(Path(args.cfr_necessity_root).resolve())}

    sample_pairs = _official_sample_pairs()
    for backend in ("xoftr", "minima_xoftr"):
        out_dir = s0_root / backend
        _ensure_dir(out_dir)
        py = args.xoftr_python if backend == "xoftr" else args.minima_python
        repo = args.xoftr_repo if backend == "xoftr" else args.minima_repo
        weight = args.xoftr_weight if backend == "xoftr" else args.minima_weight
        tmp_root = out_dir / "_tmp_dataset"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        _ensure_dir(tmp_root / "rgb")
        _ensure_dir(tmp_root / "thermal")
        # Use the same stem on both modalities so pair matching is not empty.
        sample_name = "metu_sample_pair.jpg"
        shutil.copy2(sample_pairs[backend]["fig1"], tmp_root / "rgb" / sample_name)
        shutil.copy2(sample_pairs[backend]["fig2"], tmp_root / "thermal" / sample_name)
        cmd = [
            py,
            str(repo_root / "run_2d_pair_registration.py"),
            "--backend", backend,
            "--repo_root", repo,
            "--weight_path", weight,
            "--dataset_dir", str(tmp_root),
            "--dataset_name", "METU_sample",
            "--input_mode", "official_sample",
            "--out_root", str(out_dir),
        ]
        rc = _run(cmd, repo_root, logs_root / f"S0__{backend}.log")
        manifest["steps"][f"S0_OfficialSanity/{backend}"] = {"returncode": rc, "out_dir": str(out_dir)}

    combos = [
        ("S2_Main_2DRawReplacement", s2_root, "raw"),
        ("S3_Appendix_AssistedInputAnalysis", s3_root, "cfr_fit"),
    ]
    for step_name, step_root, input_mode in combos:
        for backend in ("xoftr", "minima_xoftr"):
            py = args.xoftr_python if backend == "xoftr" else args.minima_python
            repo = args.xoftr_repo if backend == "xoftr" else args.minima_repo
            weight = args.xoftr_weight if backend == "xoftr" else args.minima_weight
            for dataset in DATASETS:
                out_dir = step_root / backend / dataset
                _ensure_dir(out_dir)
                dataset_dir = dataset_root / input_mode / dataset
                cmd = [
                    py,
                    str(repo_root / "run_2d_pair_registration.py"),
                    "--backend", backend,
                    "--repo_root", repo,
                    "--weight_path", weight,
                    "--dataset_dir", str(dataset_dir),
                    "--dataset_name", dataset,
                    "--input_mode", input_mode,
                    "--out_root", str(out_dir),
                ]
                rc = _run(cmd, repo_root, logs_root / f"{step_name}__{backend}__{dataset}.log")
                manifest["steps"][f"{step_name}/{backend}/{dataset}"] = {"returncode": rc, "out_dir": str(out_dir)}
                if rc == 0:
                    cmd_eval = [
                        sys.executable,
                        str(repo_root / "eval_crop_metrics.py"),
                        "--th_dir", str(out_dir / "registered_thermal"),
                        "--rgb_dir", str(out_dir / "registered_rgb"),
                        "--tag", backend,
                        "--out_dir", str(out_dir / "metrics"),
                        "--ssim",
                    ]
                    rc_eval = _run(cmd_eval, repo_root, logs_root / f"{step_name}__{backend}__{dataset}__metrics.log")
                    manifest["steps"][f"{step_name}/{backend}/{dataset}"]["eval_returncode"] = rc_eval

    manifest_path = campaign_root / "campaign_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    cmd_sum = [
        sys.executable,
        str(repo_root / "summarize_cfr_2d_campaign.py"),
        "--campaign_root", str(campaign_root),
        "--cfr_necessity_root", str(Path(args.cfr_necessity_root).resolve()),
        "--out_dir", str(summaries_root),
    ]
    rc_sum = _run(cmd_sum, repo_root, logs_root / "summarize.log")
    manifest["steps"]["Summaries"] = {"returncode": rc_sum, "out_dir": str(summaries_root)}
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if rc_sum == 0 else rc_sum


if __name__ == "__main__":
    raise SystemExit(main())
