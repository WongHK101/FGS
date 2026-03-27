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
RUN_ROOT = BASE / "_clean_ablation_rawRebuild_20260323_1"
OLD_GROUP = SUMMARY_ROOT / "GroupedAblation_AssistantPack_rerunM00M01_20260323_030908"
OLD_M00M01 = SUMMARY_ROOT / "M00_M01_AssistantPack_lerp_evalfull_20260320_022415"
SCENES = ["Building", "Orchard", "PVpanel", "Road", "TransmissionTower"]
GROUPS = [
    ("M00_Baseline_ModulesOff_rawRebuild_20260323", "Baseline Transfer (M00)", True, "none", 0.025),
    ("G01_M00_plus_SSP_rawRebuild_20260323", "M00 + SSP", True, "m00_plus_ssp", 0.025),
    ("G02_M00_plus_STT_rawRebuild_20260323", "M00 + STT", True, "m00_plus_stt", 0.0002),
    ("M01_OursFull_Default_rawRebuild_20260323", "Stabilized Transfer (M01)", False, "none", 0.0002),
]


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


def ply_count(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        for line in f:
            s = line.decode("ascii", "ignore").strip()
            if s.startswith("element vertex "):
                return int(s.split()[-1])
            if s == "end_header":
                break
    return None


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


def main() -> None:
    run_manifest = pd.read_csv(RUN_ROOT / "manifest.csv")
    old_group_source = pd.read_csv(OLD_GROUP / "tables" / "GroupedAblation_Source.csv")
    old_m00m01_source = pd.read_csv(OLD_M00M01 / "tables" / "M00_M01_Source.csv")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    group_pkg = SUMMARY_ROOT / f"GroupedAblation_AssistantPack_rawRebuild_20260323_{stamp}"
    ours_pkg = SUMMARY_ROOT / f"SOTA_OursRefresh_rawRebuild_20260323_{stamp}"
    for pkg in [group_pkg, ours_pkg]:
        for sub in ["tables", "logs", "source_files"]:
            (pkg / sub).mkdir(parents=True, exist_ok=True)

    rows, audits = [], []
    for exp_group, setting, baseline_off, ablation_mode, t_opacity_lr in GROUPS:
        for scene in SCENES:
            exp_dir = BASE / exp_group / scene
            rgb_dir = exp_dir / "Model_RGB"
            t_dir = exp_dir / "Model_T"
            rgb = parse_results(rgb_dir / "results.json")
            rgb_plus = parse_results(rgb_dir / "results_plus.json")
            t = parse_results(t_dir / "results.json")
            t_plus = parse_results(t_dir / "results_plus.json")
            nv = load_json(t_dir / "novel_views_grid" / "novel_view_metrics_grid.json")
            rgb_ckpt = rgb_dir / "chkpnt30000.pth"
            rgb_ply = rgb_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            t_ckpt = t_dir / "chkpnt60000.pth"
            t_ply = t_dir / "point_cloud" / "iteration_60000" / "point_cloud.ply"
            run_row = run_manifest[(run_manifest["scene"] == scene) & (run_manifest["group"] == exp_group)].iloc[0]
            rows.append(
                {
                    "exp_group": exp_group, "setting": setting, "dataset": scene,
                    "RGB_PSNR": safe_float(rgb.get("PSNR")), "RGB_SSIM": safe_float(rgb.get("SSIM")), "RGB_LPIPS": safe_float(rgb.get("LPIPS")),
                    "RGB_AlignedGradientCorr": safe_float(rgb_plus.get("AlignedGradientCorr")), "RGB_AlignedEdgeF1": safe_float(rgb_plus.get("AlignedEdgeF1")),
                    "RGB_BgLeakRatio_band": safe_float(rgb_plus.get("BgLeakRatio_band")), "RGB_EdgeHaloScore": safe_float(rgb_plus.get("EdgeHaloScore")),
                    "RGB_gaussians": ckpt_count(rgb_ckpt),
                    "T_PSNR": safe_float(t.get("PSNR")), "T_SSIM": safe_float(t.get("SSIM")), "T_LPIPS": safe_float(t.get("LPIPS")),
                    "T_AlignedGradientCorr": safe_float(t_plus.get("AlignedGradientCorr")),
                    "T_Flicker": safe_float(nv.get("TemporalFlicker_local_mean")), "T_AirArtifact": safe_float(nv.get("AirArtifactScore_mean")),
                    "BgLeakRatio_band": safe_float(t_plus.get("BgLeakRatio_band")), "EdgeHaloScore": safe_float(t_plus.get("EdgeHaloScore")),
                    "SpikeScore_air_mean": safe_float(nv.get("SpikeScore_air_mean")), "BgSensitivity_mean": safe_float(nv.get("BgSensitivity_mean")),
                    "T_gaussians": ckpt_count(t_ckpt), "run_status": run_row["status"], "run_returncode": run_row["returncode"],
                    "run_elapsed_sec": run_row["elapsed_sec"], "run_log_path": run_row["log_path"], "baseline_modules_off": baseline_off,
                    "grouped_ablation_mode": ablation_mode, "t_opacity_lr": t_opacity_lr, "source_dir_rgb": str(rgb_dir), "source_dir_t": str(t_dir),
                }
            )
            audits.append(
                {
                    "exp_group": exp_group, "setting": setting, "dataset": scene,
                    "RGB_ckpt_gaussians": ckpt_count(rgb_ckpt), "RGB_ply_vertices": ply_count(rgb_ply),
                    "T_ckpt_gaussians": ckpt_count(t_ckpt), "T_ply_vertices": ply_count(t_ply),
                }
            )

    source_df = pd.DataFrame(rows).sort_values(["setting", "dataset"]).reset_index(drop=True)
    gauss_df = pd.DataFrame(audits).sort_values(["setting", "dataset"]).reset_index(drop=True)
    gauss_df["rgb_consistent"] = gauss_df["RGB_ckpt_gaussians"] == gauss_df["RGB_ply_vertices"]
    gauss_df["t_consistent"] = gauss_df["T_ckpt_gaussians"] == gauss_df["T_ply_vertices"]
    gauss_df["rgb_equals_t"] = gauss_df["RGB_ckpt_gaussians"] == gauss_df["T_ckpt_gaussians"]

    main_df = source_df.groupby("setting", sort=False, observed=False)[
        ["T_PSNR", "T_SSIM", "T_LPIPS", "T_AlignedGradientCorr", "T_Flicker", "T_AirArtifact", "T_gaussians",
         "BgLeakRatio_band", "EdgeHaloScore", "SpikeScore_air_mean", "BgSensitivity_mean"]
    ].mean().reset_index()
    main_display = main_df.copy()
    for col in [c for c in main_display.columns if c not in ("setting", "T_gaussians")]:
        main_display[col] = main_display[col].round(6)
    main_display["T_gaussians"] = main_display["T_gaussians"].round().astype(int)

    diff_rows = []
    for setting in ["Baseline Transfer (M00)", "Stabilized Transfer (M01)"]:
        for scene in SCENES:
            old_r = old_group_source[(old_group_source["setting"] == setting) & (old_group_source["dataset"] == scene)].iloc[0]
            new_r = source_df[(source_df["setting"] == setting) & (source_df["dataset"] == scene)].iloc[0]
            diff_rows.append({"setting": setting, "dataset": scene, "old_T_PSNR": old_r["T_PSNR"], "new_T_PSNR": new_r["T_PSNR"],
                              "delta_T_PSNR": new_r["T_PSNR"] - old_r["T_PSNR"], "old_T_gaussians": int(old_r["T_gaussians"]),
                              "new_T_gaussians": int(new_r["T_gaussians"]), "delta_T_gaussians": int(new_r["T_gaussians"]) - int(old_r["T_gaussians"])})
    diff_df = pd.DataFrame(diff_rows)

    ours_old = old_m00m01_source[old_m00m01_source["exp_group"] == "M01_OursFull_Default"]
    ours_new = source_df[source_df["exp_group"] == "M01_OursFull_Default_rawRebuild_20260323"]
    ours_diff = []
    for scene in SCENES:
        old_r = ours_old[ours_old["dataset"] == scene].iloc[0]
        new_r = ours_new[ours_new["dataset"] == scene].iloc[0]
        ours_diff.append({"dataset": scene, "old_RGB_PSNR": old_r["RGB_PSNR"], "new_RGB_PSNR": new_r["RGB_PSNR"],
                          "delta_RGB_PSNR": new_r["RGB_PSNR"] - old_r["RGB_PSNR"], "old_T_PSNR": old_r["T_PSNR"],
                          "new_T_PSNR": new_r["T_PSNR"], "delta_T_PSNR": new_r["T_PSNR"] - old_r["T_PSNR"],
                          "old_T_gaussians": int(old_r["T_gaussians"]), "new_T_gaussians": int(new_r["T_gaussians"])})
    ours_diff_df = pd.DataFrame(ours_diff)

    # grouped package
    source_df.to_csv(group_pkg / "tables" / "GroupedAblation_Source.csv", index=False, encoding="utf-8-sig")
    main_df.to_csv(group_pkg / "tables" / "GroupedAblation_MainTable.csv", index=False, encoding="utf-8-sig")
    main_display.to_csv(group_pkg / "tables" / "GroupedAblation_MainTable_display.csv", index=False, encoding="utf-8-sig")
    gauss_df.to_csv(group_pkg / "tables" / "GroupedAblation_GaussianAudit.csv", index=False, encoding="utf-8-sig")
    diff_df.to_csv(group_pkg / "tables" / "GroupedAblation_OldVsRawRebuild_Diff.csv", index=False, encoding="utf-8-sig")
    run_manifest.to_csv(group_pkg / "tables" / "GroupedAblation_RunManifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(GROUPS, columns=["exp_group", "setting", "baseline_modules_off", "grouped_ablation_mode", "t_opacity_lr"]).to_csv(
        group_pkg / "tables" / "GroupedAblation_CommandAudit.csv", index=False, encoding="utf-8-sig"
    )
    with pd.ExcelWriter(group_pkg / "tables" / "GroupedAblation_Comparison.xlsx", engine="openpyxl") as writer:
        main_df.to_excel(writer, sheet_name="MainTable", index=False)
        source_df.to_excel(writer, sheet_name="Source", index=False)
        gauss_df.to_excel(writer, sheet_name="GaussianAudit", index=False)
        diff_df.to_excel(writer, sheet_name="OldVsRawRebuild", index=False)

    for p in [RUN_ROOT / "master.log", RUN_ROOT / "manifest.csv", *RUN_ROOT.glob("*.log")]:
        copy_if_exists(p, group_pkg / "logs" / p.name)
    for exp_group, _, _, _, _ in GROUPS:
        for scene in SCENES:
            dst = group_pkg / "source_files" / exp_group / scene
            dst.mkdir(parents=True, exist_ok=True)
            model_t = BASE / exp_group / scene / "Model_T"
            model_rgb = BASE / exp_group / scene / "Model_RGB"
            for src, name in [
                (model_t / "cfg_args", "Model_T_cfg_args"), (model_t / "results.json", "Model_T_results.json"),
                (model_t / "results_plus.json", "Model_T_results_plus.json"),
                (model_t / "novel_views_grid" / "novel_view_metrics_grid.json", "Model_T_novel_view_metrics_grid.json"),
                (model_rgb / "cfg_args", "Model_RGB_cfg_args"), (model_rgb / "results.json", "Model_RGB_results.json"),
                (model_rgb / "results_plus.json", "Model_RGB_results_plus.json"),
            ]:
                copy_if_exists(src, dst / name)

    group_readme = f"""# Grouped Ablation Assistant Pack (Raw-Rebuild Clean Batch)

This package is the new grouped-ablation source of truth for the clean unified batch.
It supersedes:
- `{OLD_GROUP}`

Why this rebuild was necessary:
- the historical `cfr_fit` backup matched a raw rerun,
- the historical `sparse` backup did **not** match the sparse regenerated from raw,
- current `HEAD=715e0a6` recovers archived `Building/M01` RGB quality once sparse is rebuilt from raw.

Final 4 rows:
1. `Baseline Transfer (M00)` = `M00_Baseline_ModulesOff_rawRebuild_20260323`
2. `M00 + SSP` = `G01_M00_plus_SSP_rawRebuild_20260323`
3. `M00 + STT` = `G02_M00_plus_STT_rawRebuild_20260323`
4. `Stabilized Transfer (M01)` = `M01_OursFull_Default_rawRebuild_20260323`

Shared protocol:
- shared prep per scene: `step1-4`
- per-group run: `step5-12`
- `rgb_res=4`, `t_res=4`, `rgb_iter=30000`, `t_iter=60000`

Important note for the paper assistant:
- baseline / `M00`: `t_opacity_lr = 0.025`
- `STT` / `G02`: `t_opacity_lr = 0.0002`
- full `M01`: `t_opacity_lr = 0.0002`

Macro table:
{md_table(main_display)}
"""
    (group_pkg / "README.md").write_text(group_readme, encoding="utf-8")
    (group_pkg / "SUMMARY.md").write_text(
        "# Grouped Ablation Summary (Raw-Rebuild Clean Batch)\n\n"
        "- SSP mainly improves cleanliness / compactness.\n"
        "- STT mainly improves thermal-stage stability / artifact suppression.\n"
        "- This package supersedes earlier grouped-ablation packs that mixed provenance.\n",
        encoding="utf-8",
    )
    (group_pkg / "QA.json").write_text(json.dumps({"head_commit": "715e0a6", "single_provenance_clean_batch": True}, indent=2), encoding="utf-8")
    (group_pkg / "manifest.json").write_text(json.dumps({"groups": [g[0] for g in GROUPS], "scenes": SCENES}, indent=2), encoding="utf-8")
    (group_pkg / "_build_info.txt").write_text("Built from clean rawRebuild outputs on 2026-03-23.", encoding="utf-8")
    group_zip = zip_dir(group_pkg)

    # ours package
    ours_macro = pd.DataFrame(
        [
            {
                "setting": "Ours (rawRebuild M01)",
                "RGB_PSNR": float(ours_new["RGB_PSNR"].mean()),
                "RGB_SSIM": float(ours_new["RGB_SSIM"].mean()),
                "RGB_LPIPS": float(ours_new["RGB_LPIPS"].mean()),
                "RGB_gaussians": int(round(ours_new["RGB_gaussians"].mean())),
                "T_PSNR": float(ours_new["T_PSNR"].mean()),
                "T_SSIM": float(ours_new["T_SSIM"].mean()),
                "T_LPIPS": float(ours_new["T_LPIPS"].mean()),
                "T_gaussians": int(round(ours_new["T_gaussians"].mean())),
            }
        ]
    )
    ours_new.to_csv(ours_pkg / "tables" / "Ours_RawRebuild_Source.csv", index=False, encoding="utf-8-sig")
    ours_new[["dataset", "RGB_PSNR", "RGB_SSIM", "RGB_LPIPS", "RGB_gaussians", "T_PSNR", "T_SSIM", "T_LPIPS", "T_gaussians"]].to_csv(
        ours_pkg / "tables" / "Ours_RawRebuild_RGBT_PerScene.csv", index=False, encoding="utf-8-sig"
    )
    ours_macro.to_csv(ours_pkg / "tables" / "Ours_RawRebuild_RGBT_Macro.csv", index=False, encoding="utf-8-sig")
    ours_diff_df.to_csv(ours_pkg / "tables" / "Ours_RawRebuild_vs_OldArchived_Diff.csv", index=False, encoding="utf-8-sig")
    gauss_df[gauss_df["exp_group"] == "M01_OursFull_Default_rawRebuild_20260323"].to_csv(
        ours_pkg / "tables" / "Ours_RawRebuild_GaussianAudit.csv", index=False, encoding="utf-8-sig"
    )
    with pd.ExcelWriter(ours_pkg / "tables" / "Ours_RawRebuild_Comparison.xlsx", engine="openpyxl") as writer:
        ours_new.to_excel(writer, sheet_name="Source", index=False)
        ours_macro.to_excel(writer, sheet_name="Macro", index=False)
        ours_diff_df.to_excel(writer, sheet_name="DiffVsOld", index=False)
    for p in [RUN_ROOT / "master.log", RUN_ROOT / "manifest.csv", *RUN_ROOT.glob("*M01_OursFull_Default_rawRebuild_20260323*.log")]:
        copy_if_exists(p, ours_pkg / "logs" / p.name)
    for scene in SCENES:
        dst = ours_pkg / "source_files" / "M01_OursFull_Default_rawRebuild_20260323" / scene
        dst.mkdir(parents=True, exist_ok=True)
        model_t = BASE / "M01_OursFull_Default_rawRebuild_20260323" / scene / "Model_T"
        model_rgb = BASE / "M01_OursFull_Default_rawRebuild_20260323" / scene / "Model_RGB"
        for src, name in [
            (model_t / "cfg_args", "Model_T_cfg_args"), (model_t / "results.json", "Model_T_results.json"),
            (model_t / "results_plus.json", "Model_T_results_plus.json"),
            (model_t / "novel_views_grid" / "novel_view_metrics_grid.json", "Model_T_novel_view_metrics_grid.json"),
            (model_rgb / "cfg_args", "Model_RGB_cfg_args"), (model_rgb / "results.json", "Model_RGB_results.json"),
            (model_rgb / "results_plus.json", "Model_RGB_results_plus.json"),
        ]:
            copy_if_exists(src, dst / name)
    ours_macro_display = ours_macro.copy()
    for col in ["RGB_PSNR", "RGB_SSIM", "RGB_LPIPS", "T_PSNR", "T_SSIM", "T_LPIPS"]:
        ours_macro_display[col] = ours_macro_display[col].round(6)
    ours_macro_display["RGB_gaussians"] = ours_macro_display["RGB_gaussians"].astype(int)
    ours_macro_display["T_gaussians"] = ours_macro_display["T_gaussians"].astype(int)
    ours_readme = f"""# Ours SOTA Refresh (Raw-Rebuild M01)

This package refreshes the `Ours` row for RGB/T SOTA tables using:
- `M01_OursFull_Default_rawRebuild_20260323`

It should be treated as a replacement candidate for the old archived `Ours` row in:
- `{OLD_M00M01}`

Why this refresh exists:
- the historical `cfr_fit` backup matches a raw rerun,
- the historical `sparse` backup does not match the sparse regenerated from raw,
- current `HEAD=715e0a6` reproduces archived `Building/M01` RGB quality closely once sparse is regenerated from raw.

Macro means:
{md_table(ours_macro_display)}
"""
    (ours_pkg / "README.md").write_text(ours_readme, encoding="utf-8")
    (ours_pkg / "QA.json").write_text(json.dumps({"head_commit": "715e0a6", "exp_group": "M01_OursFull_Default_rawRebuild_20260323"}, indent=2), encoding="utf-8")
    (ours_pkg / "manifest.json").write_text(json.dumps({"exp_group": "M01_OursFull_Default_rawRebuild_20260323", "scenes": SCENES}, indent=2), encoding="utf-8")
    (ours_pkg / "_build_info.txt").write_text("Built from clean rawRebuild M01 outputs on 2026-03-23.", encoding="utf-8")
    ours_zip = zip_dir(ours_pkg)

    print("GROUP_PKG", group_pkg)
    print("GROUP_ZIP", group_zip)
    print("OURS_PKG", ours_pkg)
    print("OURS_ZIP", ours_zip)
    print("GROUP_MAIN")
    print(main_display.to_string(index=False))
    print("OURS_MACRO")
    print(ours_macro_display.to_string(index=False))


if __name__ == "__main__":
    main()
