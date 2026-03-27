#!/usr/bin/env python
from __future__ import annotations

import json
import math
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch


SUMMARY_ROOT = Path(r"F:\databackup\xr6\output\Summaries")
BASE = Path(r"F:\databackup\xr6\output\Ch4_2_MainComparison")
OLD_M00M01 = SUMMARY_ROOT / "M00_M01_AssistantPack_lerp_evalfull_20260320_022415"
LATEST_GROUP = "M01_OursFull_Default_rawRebuild_step1to14_20260325"
LATEST_ROOT = BASE / LATEST_GROUP
M01_FULL_RUN = BASE / "_m01_full_step1to14_20260325_1"
M01_RESUME_RUN = BASE / "_m01_full_step1to14_resume_20260326_1"
SCENES = ["Building", "Orchard", "PVpanel", "Road", "TransmissionTower"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_metrics(obj: dict) -> dict:
    if not isinstance(obj, dict) or not obj:
        return {}
    if all(isinstance(v, (int, float, bool)) for v in obj.values()):
        return obj
    for k in sorted(obj.keys()):
        if "ours" in str(k).lower() and isinstance(obj[k], dict):
            return obj[k]
    for k in sorted(obj.keys()):
        if isinstance(obj[k], dict):
            return obj[k]
    return {}


def parse_results(path: Path) -> dict:
    return {str(k): v for k, v in select_metrics(load_json(path)).items()} if path.exists() else {}


def safe_float(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        fv = float(v)
        if math.isfinite(fv):
            return fv
    return None


def ckpt_count(path: Path):
    if not path.exists():
        return None
    obj = torch.load(str(path), map_location="cpu", weights_only=False)
    return int(obj[0][1].shape[0])


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def zip_dir(root: Path) -> Path:
    z = root.with_suffix(".zip")
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for cur, _, files in os.walk(root):
            for name in files:
                fp = Path(cur) / name
                zf.write(fp, fp.relative_to(root.parent))
    return z


def build_run_manifest() -> pd.DataFrame:
    rows = []
    for queue in [M01_FULL_RUN, M01_RESUME_RUN]:
        if not queue.exists():
            continue
        df = pd.read_csv(queue / "manifest.csv")
        if not df.empty:
            df = df.copy()
            df["queue_root"] = str(queue)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SUMMARY_ROOT / f"SOTA_OursRefresh_latestM01_step1to14_20260326_{stamp}"
    for sub in ["tables", "logs", "source_files"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    run_manifest = build_run_manifest()
    old_df = pd.read_csv(OLD_M00M01 / "tables" / "M00_M01_Source.csv")
    old_df = old_df[old_df["exp_group"] == "M01_OursFull_Default"].copy()

    rows = []
    for scene in SCENES:
        model_rgb = LATEST_ROOT / scene / "Model_RGB"
        model_t = LATEST_ROOT / scene / "Model_T"
        rgb = parse_results(model_rgb / "results.json")
        rgb_plus = parse_results(model_rgb / "results_plus.json")
        t = parse_results(model_t / "results.json")
        t_plus = parse_results(model_t / "results_plus.json")
        nv = load_json(model_t / "novel_views_grid" / "novel_view_metrics_grid.json")
        ckpt = ckpt_count(model_rgb / "chkpnt30000.pth")

        rows.append(
            {
                "exp_group": LATEST_GROUP,
                "dataset": scene,
                "RGB_PSNR": safe_float(rgb.get("PSNR")),
                "RGB_SSIM": safe_float(rgb.get("SSIM")),
                "RGB_LPIPS": safe_float(rgb.get("LPIPS")),
                "RGB_AlignedGradientCorr": safe_float(rgb_plus.get("AlignedGradientCorr")),
                "RGB_AlignedEdgeF1": safe_float(rgb_plus.get("AlignedEdgeF1")),
                "RGB_BgLeakRatio_band": safe_float(rgb_plus.get("BgLeakRatio_band")),
                "RGB_EdgeHaloScore": safe_float(rgb_plus.get("EdgeHaloScore")),
                "RGB_gaussians": ckpt,
                "T_PSNR": safe_float(t.get("PSNR")),
                "T_SSIM": safe_float(t.get("SSIM")),
                "T_LPIPS": safe_float(t.get("LPIPS")),
                "T_AlignedGradientCorr": safe_float(t_plus.get("AlignedGradientCorr")),
                "T_Flicker": safe_float(nv.get("TemporalFlicker_local_mean")),
                "T_AirArtifact": safe_float(nv.get("AirArtifactScore_mean")),
                "T_BgLeakRatio_band": safe_float(t_plus.get("BgLeakRatio_band")),
                "T_EdgeHaloScore": safe_float(t_plus.get("EdgeHaloScore")),
                "T_gaussians": ckpt,
                "source_dir_rgb": str(model_rgb),
                "source_dir_t": str(model_t),
            }
        )

    source_df = pd.DataFrame(rows)
    per_scene = source_df[
        [
            "dataset",
            "RGB_PSNR",
            "RGB_SSIM",
            "RGB_LPIPS",
            "RGB_gaussians",
            "T_PSNR",
            "T_SSIM",
            "T_LPIPS",
            "T_gaussians",
        ]
    ].copy()
    macro = pd.DataFrame(
        [
            {
                "setting": "Ours (latest M01 step1-14)",
                "RGB_PSNR": float(per_scene["RGB_PSNR"].mean()),
                "RGB_SSIM": float(per_scene["RGB_SSIM"].mean()),
                "RGB_LPIPS": float(per_scene["RGB_LPIPS"].mean()),
                "RGB_gaussians": int(round(per_scene["RGB_gaussians"].mean())),
                "T_PSNR": float(per_scene["T_PSNR"].mean()),
                "T_SSIM": float(per_scene["T_SSIM"].mean()),
                "T_LPIPS": float(per_scene["T_LPIPS"].mean()),
                "T_gaussians": int(round(per_scene["T_gaussians"].mean())),
            }
        ]
    )

    diff_rows = []
    for scene in SCENES:
        old_row = old_df[old_df["dataset"] == scene].iloc[0]
        new_row = per_scene[per_scene["dataset"] == scene].iloc[0]
        diff_rows.append(
            {
                "dataset": scene,
                "old_RGB_PSNR": old_row["RGB_PSNR"],
                "new_RGB_PSNR": new_row["RGB_PSNR"],
                "delta_RGB_PSNR": new_row["RGB_PSNR"] - old_row["RGB_PSNR"],
                "old_RGB_SSIM": old_row["RGB_SSIM"],
                "new_RGB_SSIM": new_row["RGB_SSIM"],
                "delta_RGB_SSIM": new_row["RGB_SSIM"] - old_row["RGB_SSIM"],
                "old_RGB_LPIPS": old_row["RGB_LPIPS"],
                "new_RGB_LPIPS": new_row["RGB_LPIPS"],
                "delta_RGB_LPIPS": new_row["RGB_LPIPS"] - old_row["RGB_LPIPS"],
                "old_RGB_gaussians": int(old_row["RGB_gaussians"]),
                "new_RGB_gaussians": int(new_row["RGB_gaussians"]),
                "delta_RGB_gaussians": int(new_row["RGB_gaussians"]) - int(old_row["RGB_gaussians"]),
                "old_T_PSNR": old_row["T_PSNR"],
                "new_T_PSNR": new_row["T_PSNR"],
                "delta_T_PSNR": new_row["T_PSNR"] - old_row["T_PSNR"],
                "old_T_SSIM": old_row["T_SSIM"],
                "new_T_SSIM": new_row["T_SSIM"],
                "delta_T_SSIM": new_row["T_SSIM"] - old_row["T_SSIM"],
                "old_T_LPIPS": old_row["T_LPIPS"],
                "new_T_LPIPS": new_row["T_LPIPS"],
                "delta_T_LPIPS": new_row["T_LPIPS"] - old_row["T_LPIPS"],
                "old_T_gaussians": int(old_row["T_gaussians"]),
                "new_T_gaussians": int(new_row["T_gaussians"]),
                "delta_T_gaussians": int(new_row["T_gaussians"]) - int(old_row["T_gaussians"]),
            }
        )
    diff_df = pd.DataFrame(diff_rows)
    diff_macro = {"dataset": "Mean"}
    for col in diff_df.columns:
        if col != "dataset":
            diff_macro[col] = float(diff_df[col].mean())
    diff_df = pd.concat([diff_df, pd.DataFrame([diff_macro])], ignore_index=True)

    macro_display = macro.copy()
    for col in ["RGB_PSNR", "RGB_SSIM", "RGB_LPIPS", "T_PSNR", "T_SSIM", "T_LPIPS"]:
        macro_display[col] = macro_display[col].round(6)
    macro_display["RGB_gaussians"] = macro_display["RGB_gaussians"].astype(int)
    macro_display["T_gaussians"] = macro_display["T_gaussians"].astype(int)

    tables = out_dir / "tables"
    source_df.to_csv(tables / "Ours_LatestM1_Source.csv", index=False, encoding="utf-8-sig")
    per_scene.to_csv(tables / "Ours_LatestM1_RGBT_PerScene.csv", index=False, encoding="utf-8-sig")
    macro.to_csv(tables / "Ours_LatestM1_RGBT_Macro.csv", index=False, encoding="utf-8-sig")
    diff_df.to_csv(tables / "Ours_LatestM1_vs_OldArchived_Diff.csv", index=False, encoding="utf-8-sig")
    run_manifest.to_csv(tables / "Ours_LatestM1_RunManifest.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(tables / "Ours_LatestM1_Comparison.xlsx", engine="openpyxl") as writer:
        source_df.to_excel(writer, sheet_name="Source", index=False)
        per_scene.to_excel(writer, sheet_name="PerScene", index=False)
        macro.to_excel(writer, sheet_name="Macro", index=False)
        diff_df.to_excel(writer, sheet_name="DiffVsOld", index=False)
        run_manifest.to_excel(writer, sheet_name="RunManifest", index=False)

    logs = out_dir / "logs"
    for src in [
        M01_FULL_RUN / "master.log",
        M01_FULL_RUN / "manifest.csv",
        M01_RESUME_RUN / "master.log",
        M01_RESUME_RUN / "manifest.csv",
    ]:
        copy_if_exists(src, logs / src.parent.name / src.name)

    for scene in SCENES:
        dst = out_dir / "source_files" / LATEST_GROUP / scene
        dst.mkdir(parents=True, exist_ok=True)
        model_rgb = LATEST_ROOT / scene / "Model_RGB"
        model_t = LATEST_ROOT / scene / "Model_T"
        for src, name in [
            (model_rgb / "cfg_args", "Model_RGB_cfg_args"),
            (model_rgb / "results.json", "Model_RGB_results.json"),
            (model_rgb / "results_plus.json", "Model_RGB_results_plus.json"),
            (model_t / "cfg_args", "Model_T_cfg_args"),
            (model_t / "results.json", "Model_T_results.json"),
            (model_t / "results_plus.json", "Model_T_results_plus.json"),
            (model_t / "novel_views_grid" / "novel_view_metrics_grid.json", "Model_T_novel_view_metrics_grid.json"),
        ]:
            copy_if_exists(src, dst / name)

    readme = f"""# Ours SOTA Refresh (Latest M01 Step1-14)

This package refreshes the `Ours` row for RGB/T SOTA tables using:
- `{LATEST_GROUP}`

It should replace the older `Ours` row that previously came from:
- `{OLD_M00M01}`

Why this refresh exists:
- latest `M01` has now completed the full `step1-14` pipeline,
- the grouped-ablation main package has already been updated to the latest 4 groups,
- this package keeps the SOTA `Ours` row synchronized with the latest `M01`.

Macro means:
{md_table(macro_display)}
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "QA.json").write_text(
        json.dumps(
            {
                "head_commit": "715e0a6",
                "exp_group": LATEST_GROUP,
                "old_reference_pack": str(OLD_M00M01),
                "scenes": SCENES,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "package_type": "SOTA_OursRefresh_latestM01_step1to14",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "exp_group": LATEST_GROUP,
                "scenes": SCENES,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "_build_info.txt").write_text(
        "Built from latest completed M01 step1-14 outputs on 2026-03-26.",
        encoding="utf-8",
    )
    zip_path = zip_dir(out_dir)

    print("OUT_DIR", out_dir)
    print("ZIP", zip_path)
    print("MACRO")
    print(macro_display.to_string(index=False))


if __name__ == "__main__":
    main()
