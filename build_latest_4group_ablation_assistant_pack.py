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
SCENES = ["Building", "Orchard", "PVpanel", "Road", "TransmissionTower"]
ALPHAS = {0.25, 0.5, 0.75}

RAW_REBUILD_RUN = BASE / "_clean_ablation_rawRebuild_20260323_1"
M00_STEP13_14_RUN = BASE / "_m00_step13to14_retry2_20260324_012445"
G01_G02_STEP13_14_RUN = BASE / "_g01_g02_step13to14_20260324_204500"
M01_FULL_RUN = BASE / "_m01_full_step1to14_20260325_1"
M01_RESUME_RUN = BASE / "_m01_full_step1to14_resume_20260326_1"
OLD_M00M01 = SUMMARY_ROOT / "M00_M01_AssistantPack_lerp_evalfull_20260320_022415"

GROUPS = [
    {
        "exp_group": "M00_Baseline_ModulesOff_rawRebuild_20260323",
        "setting": "Baseline Transfer (M00)",
        "baseline_modules_off": True,
        "grouped_ablation_mode": "none",
        "t_opacity_lr": 0.025,
        "notes": "Latest clean raw-rebuild M00.",
    },
    {
        "exp_group": "G01_M00_plus_SSP_rawRebuild_20260323",
        "setting": "M00 + SSP",
        "baseline_modules_off": True,
        "grouped_ablation_mode": "m00_plus_ssp",
        "t_opacity_lr": 0.025,
        "notes": "Latest clean raw-rebuild G01.",
    },
    {
        "exp_group": "G02_M00_plus_STT_rawRebuild_20260323",
        "setting": "M00 + STT",
        "baseline_modules_off": True,
        "grouped_ablation_mode": "m00_plus_stt",
        "t_opacity_lr": 0.0002,
        "notes": "Latest clean raw-rebuild G02.",
    },
    {
        "exp_group": "M01_OursFull_Default_rawRebuild_step1to14_20260325",
        "setting": "Stabilized Transfer (M01)",
        "baseline_modules_off": False,
        "grouped_ablation_mode": "none",
        "t_opacity_lr": 0.0002,
        "notes": "Latest clean raw-rebuild M01 with completed step1-14.",
    },
]

F_COLUMN_MAP = {
    "fusion_MI_total_mean": "F_Fusion_MI",
    "fusion_QABF_mean": "F_Fusion_QABF",
    "t_ref_Spearman_S_mean": "F_Thermal_Spearman",
    "rgb_ref_SSIM_Y_mean": "F_RGB_ref_SSIM_Y",
    "PSNR_Y_mean": "F_PSNR_Y",
    "LPIPS_mean": "F_LPIPS",
}


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


def aggregate_f_summary(summary_csv: Path) -> tuple[dict[str, float], dict[str, object]]:
    df = pd.read_csv(summary_csv)
    sub = df[df["alpha"].isin(ALPHAS)].copy()
    if sub.empty:
        raise RuntimeError(f"No alpha subset rows in {summary_csv}")
    strategies = sorted(sub["strategy"].dropna().unique().tolist())
    values = {out_col: float(sub[src_col].mean()) for src_col, out_col in F_COLUMN_MAP.items()}
    meta = {
        "f_source_summary_csv": str(summary_csv),
        "f_rows_used": int(len(sub)),
        "f_strategy_count": int(len(strategies)),
        "f_strategies": ",".join(strategies),
        "f_alpha_subset": "0.25,0.5,0.75",
    }
    return values, meta


def load_rawrebuild_manifest() -> pd.DataFrame:
    df = pd.read_csv(RAW_REBUILD_RUN / "manifest.csv")
    return df.rename(columns={"group": "exp_group", "scene": "dataset"})


def load_m01_combined_manifest() -> pd.DataFrame:
    rows = []
    first = pd.read_csv(M01_FULL_RUN / "manifest.csv")
    if not first.empty:
        first = first.copy()
        first["queue"] = "initial"
        rows.append(first)
    second = pd.read_csv(M01_RESUME_RUN / "manifest.csv")
    if not second.empty:
        second = second.copy()
        second["queue"] = "resume"
        rows.append(second)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def manifest_phase_col(df: pd.DataFrame) -> str:
    if "phase" in df.columns:
        return "phase"
    if "stage" in df.columns:
        return "stage"
    raise KeyError("Manifest has neither 'phase' nor 'stage'")


def manifest_elapsed(row: pd.Series) -> float | None:
    for key in ["elapsed_sec", "duration_sec"]:
        if key in row.index and pd.notna(row[key]):
            return float(row[key])
    return None


def manifest_log_path(row: pd.Series) -> str | None:
    for key in ["log_path", "scene_log", "log"]:
        if key in row.index and pd.notna(row[key]):
            return str(row[key])
    return None


def infer_main_run_row(exp_group: str, scene: str, raw_df: pd.DataFrame, m01_df: pd.DataFrame) -> dict[str, object]:
    if exp_group != "M01_OursFull_Default_rawRebuild_step1to14_20260325":
        sub = raw_df[(raw_df["exp_group"] == exp_group) & (raw_df["dataset"] == scene)]
        if len(sub) != 1:
            raise RuntimeError(f"Expected one raw manifest row for {exp_group}/{scene}, got {len(sub)}")
        row = sub.iloc[0]
        return {
            "main_run_status": row["status"],
            "main_run_returncode": row["returncode"],
            "main_run_elapsed_sec": row["elapsed_sec"],
            "main_run_log_path": row["log_path"],
            "main_run_queue": str(RAW_REBUILD_RUN),
        }

    # For latest M01, Building finished in the initial queue; later scenes may be resumed.
    candidates = m01_df[m01_df["scene"] == scene].copy()
    phase_col = manifest_phase_col(candidates)
    phase_text = candidates[phase_col].astype(str)
    if scene in {"Building", "PVpanel", "Road", "TransmissionTower"}:
        sub = candidates[phase_text.str.contains(r"step5[-_]?14|step5to14", case=False, regex=True, na=False)]
        if len(sub) == 0:
            raise RuntimeError(f"No step5-14 row for latest M01/{scene}")
        row = sub.iloc[-1]
    else:
        # Orchard spans two queues; use the successful resume row if available.
        sub = candidates[
            (phase_text.str.contains(r"step13[-_]?14|step13to14", case=False, regex=True, na=False))
            & (candidates["status"] == "done")
        ]
        if len(sub) == 0:
            sub = candidates[candidates["status"] == "done"]
        row = sub.iloc[-1]
    return {
        "main_run_status": row["status"],
        "main_run_returncode": row["returncode"],
        "main_run_elapsed_sec": manifest_elapsed(row),
        "main_run_log_path": manifest_log_path(row),
        "main_run_queue": " + ".join(
            str(p)
            for p in [M01_FULL_RUN, M01_RESUME_RUN]
            if p.exists()
        ),
    }


def load_step13_14_manifest(exp_group: str) -> pd.DataFrame:
    if exp_group == "M00_Baseline_ModulesOff_rawRebuild_20260323":
        return pd.read_csv(M00_STEP13_14_RUN / "manifest.csv")
    if exp_group in {"G01_M00_plus_SSP_rawRebuild_20260323", "G02_M00_plus_STT_rawRebuild_20260323"}:
        return pd.read_csv(G01_G02_STEP13_14_RUN / "manifest.csv")
    # latest M01 gets F-stage from the fully completed step1-14 rerun itself
    return pd.DataFrame()


def infer_f_run_row(exp_group: str, scene: str, m01_df: pd.DataFrame) -> dict[str, object]:
    if exp_group == "M00_Baseline_ModulesOff_rawRebuild_20260323":
        df = pd.read_csv(M00_STEP13_14_RUN / "manifest.csv")
        sub = df[(df["scene"] == scene) & (df["group"] == exp_group)]
        row = sub.iloc[0]
        queue = str(M00_STEP13_14_RUN)
    elif exp_group in {"G01_M00_plus_SSP_rawRebuild_20260323", "G02_M00_plus_STT_rawRebuild_20260323"}:
        df = pd.read_csv(G01_G02_STEP13_14_RUN / "manifest.csv")
        sub = df[(df["scene"] == scene) & (df["group"] == exp_group)]
        row = sub.iloc[0]
        queue = str(G01_G02_STEP13_14_RUN)
    else:
        candidates = m01_df[m01_df["scene"] == scene].copy()
        phase_col = manifest_phase_col(candidates)
        phase_text = candidates[phase_col].astype(str)
        sub = candidates[phase_text.str.contains(r"step13[-_]?14|step13to14", case=False, regex=True, na=False)]
        if len(sub) == 0:
            sub = candidates[phase_text.str.contains(r"step5[-_]?14|step5to14", case=False, regex=True, na=False)]
        row = sub.iloc[-1]
        queue = " + ".join(str(p) for p in [M01_FULL_RUN, M01_RESUME_RUN] if p.exists())
    return {
        "f_run_status": row["status"],
        "f_run_returncode": row["returncode"],
        "f_run_elapsed_sec": manifest_elapsed(row),
        "f_run_log_path": manifest_log_path(row),
        "f_run_queue": queue,
    }


def format_display(df: pd.DataFrame, int_cols: set[str]) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col == "setting":
            continue
        if col in int_cols:
            out[col] = out[col].map(lambda v: f"{int(round(float(v)))}")
        else:
            out[col] = out[col].map(lambda v: f"{float(v):.6f}")
    return out


def build_old_m01_diff(latest_source_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    old_rgbt = pd.read_csv(OLD_M00M01 / "tables" / "M00_M01_Source.csv")
    old_rgbt = old_rgbt[old_rgbt["exp_group"] == "M01_OursFull_Default"].copy()
    latest = latest_source_df[
        latest_source_df["exp_group"] == "M01_OursFull_Default_rawRebuild_step1to14_20260325"
    ].copy()

    old_f_rows = []
    for scene in SCENES:
        f_vals, _ = aggregate_f_summary(OLD_M00M01 / "eval" / "M01_OursFull_Default" / scene / "eval" / "summary.csv")
        old_f_rows.append({"dataset": scene, **f_vals})
    old_f = pd.DataFrame(old_f_rows)

    rows = []
    for scene in SCENES:
        old_row = old_rgbt[old_rgbt["dataset"] == scene].iloc[0]
        old_f_row = old_f[old_f["dataset"] == scene].iloc[0]
        new_row = latest[latest["dataset"] == scene].iloc[0]
        rows.append(
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
                "old_F_Fusion_MI": old_f_row["F_Fusion_MI"],
                "new_F_Fusion_MI": new_row["F_Fusion_MI"],
                "delta_F_Fusion_MI": new_row["F_Fusion_MI"] - old_f_row["F_Fusion_MI"],
                "old_F_Fusion_QABF": old_f_row["F_Fusion_QABF"],
                "new_F_Fusion_QABF": new_row["F_Fusion_QABF"],
                "delta_F_Fusion_QABF": new_row["F_Fusion_QABF"] - old_f_row["F_Fusion_QABF"],
                "old_F_Thermal_Spearman": old_f_row["F_Fusion_MI"] if False else old_f_row["F_Thermal_Spearman"],
                "new_F_Thermal_Spearman": new_row["F_Thermal_Spearman"],
                "delta_F_Thermal_Spearman": new_row["F_Thermal_Spearman"] - old_f_row["F_Thermal_Spearman"],
                "old_F_RGB_ref_SSIM_Y": old_f_row["F_RGB_ref_SSIM_Y"],
                "new_F_RGB_ref_SSIM_Y": new_row["F_RGB_ref_SSIM_Y"],
                "delta_F_RGB_ref_SSIM_Y": new_row["F_RGB_ref_SSIM_Y"] - old_f_row["F_RGB_ref_SSIM_Y"],
                "old_F_PSNR_Y": old_f_row["F_PSNR_Y"],
                "new_F_PSNR_Y": new_row["F_PSNR_Y"],
                "delta_F_PSNR_Y": new_row["F_PSNR_Y"] - old_f_row["F_PSNR_Y"],
                "old_F_LPIPS": old_f_row["F_LPIPS"],
                "new_F_LPIPS": new_row["F_LPIPS"],
                "delta_F_LPIPS": new_row["F_LPIPS"] - old_f_row["F_LPIPS"],
            }
        )

    diff_df = pd.DataFrame(rows)
    macro = {
        "dataset": "Mean",
    }
    for col in diff_df.columns:
        if col == "dataset":
            continue
        macro[col] = float(diff_df[col].mean()) if col.startswith(("old_", "new_", "delta_")) else diff_df[col].mean()
    diff_df = pd.concat([diff_df, pd.DataFrame([macro])], ignore_index=True)
    return diff_df, old_f


def main() -> None:
    raw_manifest = load_rawrebuild_manifest()
    m01_manifest = load_m01_combined_manifest()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SUMMARY_ROOT / f"GroupedAblation_AssistantPack_latest4groups_20260326_{stamp}"
    for sub in ["tables", "logs", "source_files"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    source_rows = []
    gauss_rows = []
    manifest_rows = []
    for cfg in GROUPS:
        exp_group = cfg["exp_group"]
        setting = cfg["setting"]
        for scene in SCENES:
            exp_dir = BASE / exp_group / scene
            rgb_dir = exp_dir / "Model_RGB"
            t_dir = exp_dir / "Model_T"
            rgb = parse_results(rgb_dir / "results.json")
            rgb_plus = parse_results(rgb_dir / "results_plus.json")
            t = parse_results(t_dir / "results.json")
            t_plus = parse_results(t_dir / "results_plus.json")
            nv = load_json(t_dir / "novel_views_grid" / "novel_view_metrics_grid.json")
            f_vals, f_meta = aggregate_f_summary(exp_dir / "eval" / "summary.csv")
            rgb_ckpt = rgb_dir / "chkpnt30000.pth"
            rgb_ply = rgb_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
            t_ckpt = t_dir / "chkpnt60000.pth"
            t_ply = t_dir / "point_cloud" / "iteration_60000" / "point_cloud.ply"

            source_rows.append(
                {
                    "exp_group": exp_group,
                    "setting": setting,
                    "dataset": scene,
                    "RGB_PSNR": safe_float(rgb.get("PSNR")),
                    "RGB_SSIM": safe_float(rgb.get("SSIM")),
                    "RGB_LPIPS": safe_float(rgb.get("LPIPS")),
                    "RGB_AlignedGradientCorr": safe_float(rgb_plus.get("AlignedGradientCorr")),
                    "RGB_AlignedEdgeF1": safe_float(rgb_plus.get("AlignedEdgeF1")),
                    "RGB_BgLeakRatio_band": safe_float(rgb_plus.get("BgLeakRatio_band")),
                    "RGB_EdgeHaloScore": safe_float(rgb_plus.get("EdgeHaloScore")),
                    "RGB_gaussians": ckpt_count(rgb_ckpt),
                    "T_PSNR": safe_float(t.get("PSNR")),
                    "T_SSIM": safe_float(t.get("SSIM")),
                    "T_LPIPS": safe_float(t.get("LPIPS")),
                    "T_AlignedGradientCorr": safe_float(t_plus.get("AlignedGradientCorr")),
                    "T_Flicker": safe_float(nv.get("TemporalFlicker_local_mean")),
                    "T_AirArtifact": safe_float(nv.get("AirArtifactScore_mean")),
                    "BgLeakRatio_band": safe_float(t_plus.get("BgLeakRatio_band")),
                    "EdgeHaloScore": safe_float(t_plus.get("EdgeHaloScore")),
                    "SpikeScore_air_mean": safe_float(nv.get("SpikeScore_air_mean")),
                    "BgSensitivity_mean": safe_float(nv.get("BgSensitivity_mean")),
                    "T_gaussians": ckpt_count(t_ckpt),
                    **f_vals,
                    **f_meta,
                    "baseline_modules_off": cfg["baseline_modules_off"],
                    "grouped_ablation_mode": cfg["grouped_ablation_mode"],
                    "t_opacity_lr": cfg["t_opacity_lr"],
                    "source_dir_rgb": str(rgb_dir),
                    "source_dir_t": str(t_dir),
                    "source_dir_f": str(exp_dir / "Model_F"),
                    "notes": cfg["notes"],
                }
            )
            gauss_rows.append(
                {
                    "exp_group": exp_group,
                    "setting": setting,
                    "dataset": scene,
                    "RGB_ckpt_gaussians": ckpt_count(rgb_ckpt),
                    "RGB_ply_vertices": ply_count(rgb_ply),
                    "T_ckpt_gaussians": ckpt_count(t_ckpt),
                    "T_ply_vertices": ply_count(t_ply),
                }
            )
            manifest_rows.append(
                {
                    "exp_group": exp_group,
                    "setting": setting,
                    "dataset": scene,
                    **infer_main_run_row(exp_group, scene, raw_manifest, m01_manifest),
                    **infer_f_run_row(exp_group, scene, m01_manifest),
                    "rgb_dir": str(rgb_dir),
                    "t_dir": str(t_dir),
                    "f_eval_summary_csv": str(exp_dir / "eval" / "summary.csv"),
                }
            )

    source_df = pd.DataFrame(source_rows).sort_values(["setting", "dataset"]).reset_index(drop=True)
    gauss_df = pd.DataFrame(gauss_rows).sort_values(["setting", "dataset"]).reset_index(drop=True)
    manifest_df = pd.DataFrame(manifest_rows).sort_values(["setting", "dataset"]).reset_index(drop=True)
    gauss_df["rgb_consistent"] = gauss_df["RGB_ckpt_gaussians"] == gauss_df["RGB_ply_vertices"]
    gauss_df["t_consistent"] = gauss_df["T_ckpt_gaussians"] == gauss_df["T_ply_vertices"]
    gauss_df["rgb_equals_t"] = gauss_df["RGB_ckpt_gaussians"] == gauss_df["T_ckpt_gaussians"]

    t_macro_cols = [
        "T_PSNR", "T_SSIM", "T_LPIPS", "T_AlignedGradientCorr", "T_Flicker", "T_AirArtifact",
        "T_gaussians", "BgLeakRatio_band", "EdgeHaloScore", "SpikeScore_air_mean", "BgSensitivity_mean",
    ]
    rgbt_macro_cols = ["RGB_PSNR", "RGB_SSIM", "RGB_LPIPS", "RGB_gaussians", "T_PSNR", "T_SSIM", "T_LPIPS", "T_gaussians"]
    f_macro_cols = [
        "T_PSNR", "T_SSIM", "T_LPIPS", "T_gaussians", "BgLeakRatio_band", "EdgeHaloScore",
        "F_Fusion_MI", "F_Fusion_QABF", "F_Thermal_Spearman", "F_RGB_ref_SSIM_Y", "F_PSNR_Y", "F_LPIPS",
    ]

    t_macro = source_df.groupby("setting", sort=False, observed=False)[t_macro_cols].mean().reset_index()
    rgbt_macro = source_df.groupby("setting", sort=False, observed=False)[rgbt_macro_cols].mean().reset_index()
    f_macro = source_df.groupby("setting", sort=False, observed=False)[f_macro_cols].mean().reset_index()

    t_display = format_display(t_macro, {"T_gaussians"})
    rgbt_display = format_display(rgbt_macro, {"RGB_gaussians", "T_gaussians"})
    f_display = format_display(f_macro, {"T_gaussians"})

    old_m01_diff_df, old_m01_f_df = build_old_m01_diff(source_df)

    tables = out_dir / "tables"
    source_df.to_csv(tables / "GroupedAblation_Source.csv", index=False, encoding="utf-8-sig")
    t_macro.to_csv(tables / "GroupedAblation_MainTable.csv", index=False, encoding="utf-8-sig")
    t_display.to_csv(tables / "GroupedAblation_MainTable_display.csv", index=False, encoding="utf-8-sig")
    rgbt_macro.to_csv(tables / "GroupedAblation_RGBT_Macro.csv", index=False, encoding="utf-8-sig")
    rgbt_display.to_csv(tables / "GroupedAblation_RGBT_Macro_display.csv", index=False, encoding="utf-8-sig")
    f_macro.to_csv(tables / "GroupedAblation_FStage_Macro.csv", index=False, encoding="utf-8-sig")
    f_display.to_csv(tables / "GroupedAblation_FStage_Display.csv", index=False, encoding="utf-8-sig")
    gauss_df.to_csv(tables / "GroupedAblation_GaussianAudit.csv", index=False, encoding="utf-8-sig")
    manifest_df.to_csv(tables / "GroupedAblation_RunManifest.csv", index=False, encoding="utf-8-sig")
    old_m01_diff_df.to_csv(tables / "LatestM01_vs_OldArchived_Diff.csv", index=False, encoding="utf-8-sig")
    old_m01_f_df.to_csv(tables / "OldArchived_M01_FStage_Aggregated.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(GROUPS).to_csv(tables / "GroupedAblation_CommandAudit.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(tables / "GroupedAblation_Comparison.xlsx", engine="openpyxl") as writer:
        t_display.to_excel(writer, sheet_name="TStage_Display", index=False)
        t_macro.to_excel(writer, sheet_name="TStage_Macro", index=False)
        rgbt_display.to_excel(writer, sheet_name="RGBT_Display", index=False)
        rgbt_macro.to_excel(writer, sheet_name="RGBT_Macro", index=False)
        f_display.to_excel(writer, sheet_name="FStage_Display", index=False)
        f_macro.to_excel(writer, sheet_name="FStage_Macro", index=False)
        source_df.to_excel(writer, sheet_name="Source", index=False)
        gauss_df.to_excel(writer, sheet_name="GaussianAudit", index=False)
        manifest_df.to_excel(writer, sheet_name="RunManifest", index=False)
        old_m01_diff_df.to_excel(writer, sheet_name="LatestM01_vs_Old", index=False)

    logs = out_dir / "logs"
    for src in [
        RAW_REBUILD_RUN / "master.log",
        RAW_REBUILD_RUN / "manifest.csv",
        M00_STEP13_14_RUN / "master.log",
        M00_STEP13_14_RUN / "manifest.csv",
        G01_G02_STEP13_14_RUN / "master.log",
        G01_G02_STEP13_14_RUN / "manifest.csv",
        M01_FULL_RUN / "master.log",
        M01_FULL_RUN / "manifest.csv",
        M01_RESUME_RUN / "master.log",
        M01_RESUME_RUN / "manifest.csv",
    ]:
        copy_if_exists(src, logs / src.parent.name / src.name)

    for exp_group in [cfg["exp_group"] for cfg in GROUPS]:
        for scene in SCENES:
            dst = out_dir / "source_files" / exp_group / scene
            dst.mkdir(parents=True, exist_ok=True)
            exp_dir = BASE / exp_group / scene
            for src, name in [
                (exp_dir / "Model_RGB" / "cfg_args", "Model_RGB_cfg_args"),
                (exp_dir / "Model_RGB" / "results.json", "Model_RGB_results.json"),
                (exp_dir / "Model_RGB" / "results_plus.json", "Model_RGB_results_plus.json"),
                (exp_dir / "Model_T" / "cfg_args", "Model_T_cfg_args"),
                (exp_dir / "Model_T" / "results.json", "Model_T_results.json"),
                (exp_dir / "Model_T" / "results_plus.json", "Model_T_results_plus.json"),
                (exp_dir / "Model_T" / "novel_views_grid" / "novel_view_metrics_grid.json", "Model_T_novel_view_metrics_grid.json"),
                (exp_dir / "eval" / "summary.csv", "F_eval_summary.csv"),
                (exp_dir / "eval" / "summary_gt_ref.csv", "F_eval_summary_gt_ref.csv"),
                (exp_dir / "eval" / "summary_render_ref.csv", "F_eval_summary_render_ref.csv"),
                (exp_dir / "Model_F" / "sh_opacity_geom" / "0.5" / "cfg_args", "Model_F_sh_opacity_geom_a0p5_cfg_args"),
            ]:
                copy_if_exists(src, dst / name)

    manifest_json = {
        "package_type": "GroupedAblation_AssistantPack_latest4groups",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "groups": [cfg["exp_group"] for cfg in GROUPS],
        "scenes": SCENES,
        "alpha_subset_for_f_stage": [0.25, 0.5, 0.75],
        "sources": {
            "base": str(BASE),
            "raw_rebuild_run": str(RAW_REBUILD_RUN),
            "m00_step13_14_run": str(M00_STEP13_14_RUN),
            "g01_g02_step13_14_run": str(G01_G02_STEP13_14_RUN),
            "m01_full_run": str(M01_FULL_RUN),
            "m01_resume_run": str(M01_RESUME_RUN),
            "old_m01_pack": str(OLD_M00M01),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    qa = {
        "head_commit": "715e0a6",
        "latest_m01_group": "M01_OursFull_Default_rawRebuild_step1to14_20260325",
        "checks": {
            "gaussian_rows": len(gauss_df),
            "source_rows": len(source_df),
            "run_manifest_rows": len(manifest_df),
            "f_rows_per_scene_expected": 18,
            "f_strategy_count_expected": 6,
        },
    }
    (out_dir / "QA.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""# Grouped Ablation Assistant Pack (Latest 4 Groups)

This package refreshes grouped-ablation writing material using the latest available 4 groups:

1. `Baseline Transfer (M00)` = `M00_Baseline_ModulesOff_rawRebuild_20260323`
2. `M00 + SSP` = `G01_M00_plus_SSP_rawRebuild_20260323`
3. `M00 + STT` = `G02_M00_plus_STT_rawRebuild_20260323`
4. `Stabilized Transfer (M01)` = `M01_OursFull_Default_rawRebuild_step1to14_20260325`

Compared with the earlier grouped-ablation clean batch, this package updates `M01`
to the latest fully completed `step1-14` rerun.

## Protocol

- `align = fit`
- `rgb_res = 4`
- `t_res = 4`
- `rgb_iter = 30000`
- `t_iter = 60000`
- `step13-14` uses `blend_model_strict_endpoints.py`
- `dc_y_from = lerp`
- default `endpoint_mode = blend`

## Tables For The Paper Assistant

- `tables/GroupedAblation_MainTable_display.csv`: T-stage grouped-ablation macro table
- `tables/GroupedAblation_RGBT_Macro_display.csv`: RGB/T macro table for the 4 groups
- `tables/GroupedAblation_FStage_Display.csv`: F-stage extension table using Table-1 aggregation
- `tables/LatestM01_vs_OldArchived_Diff.csv`: latest-vs-old M01 difference table
- `tables/GroupedAblation_Comparison.xlsx`: all of the above in one workbook

## T-Stage Macro Table

{md_table(t_display)}

## F-Stage Extension Table

{md_table(f_display)}
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    (out_dir / "_build_info.txt").write_text(
        "Built from latest M00/G01/G02 rawRebuild outputs plus latest completed M01 step1-14 rerun.",
        encoding="utf-8",
    )
    zip_path = zip_dir(out_dir)

    print("OUT_DIR", out_dir)
    print("ZIP", zip_path)
    print("T_DISPLAY")
    print(t_display.to_string(index=False))
    print("F_DISPLAY")
    print(f_display.to_string(index=False))


if __name__ == "__main__":
    main()
