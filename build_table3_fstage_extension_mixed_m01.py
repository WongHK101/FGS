from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd


SUMMARY_ROOT = Path(r"F:\databackup\xr6\output\Summaries")
BASE = Path(r"F:\databackup\xr6\output\Ch4_2_MainComparison")
SCENES = ["Building", "Orchard", "PVpanel", "Road", "TransmissionTower"]
ALPHAS = {0.25, 0.5, 0.75}

NEW_T_SOURCE = (
    SUMMARY_ROOT
    / "GroupedAblation_AssistantPack_rawRebuild_20260323_20260323_175411"
    / "tables"
    / "GroupedAblation_Source.csv"
)
OLD_T_SOURCE = (
    SUMMARY_ROOT
    / "GroupedAblation_AssistantPack_20260323_002921"
    / "tables"
    / "GroupedAblation_Source.csv"
)
OLD_M01_F_PACK = (
    SUMMARY_ROOT / "M00_M01_AssistantPack_lerp_evalfull_20260320_022415" / "eval"
)
M00_STEP13_14_RUN = BASE / "_m00_step13to14_retry2_20260324_012445"
G01_G02_STEP13_14_RUN = BASE / "_g01_g02_step13to14_20260324_204500"

SETTINGS = [
    {
        "setting": "Baseline Transfer (M00)",
        "exp_group": "M00_Baseline_ModulesOff_rawRebuild_20260323",
        "t_source_csv": NEW_T_SOURCE,
        "f_summary_kind": "live",
        "f_eval_root": BASE / "M00_Baseline_ModulesOff_rawRebuild_20260323",
        "notes": "New rawRebuild M00; step13-14 completed on 2026-03-24.",
    },
    {
        "setting": "M00 + SSP",
        "exp_group": "G01_M00_plus_SSP_rawRebuild_20260323",
        "t_source_csv": NEW_T_SOURCE,
        "f_summary_kind": "live",
        "f_eval_root": BASE / "G01_M00_plus_SSP_rawRebuild_20260323",
        "notes": "New rawRebuild G01; step13-14 completed on 2026-03-24.",
    },
    {
        "setting": "M00 + STT",
        "exp_group": "G02_M00_plus_STT_rawRebuild_20260323",
        "t_source_csv": NEW_T_SOURCE,
        "f_summary_kind": "live",
        "f_eval_root": BASE / "G02_M00_plus_STT_rawRebuild_20260323",
        "notes": "New rawRebuild G02; step13-14 completed on 2026-03-24.",
    },
    {
        "setting": "Stabilized Transfer (M01)",
        "exp_group": "M01_OursFull_Default",
        "t_source_csv": OLD_T_SOURCE,
        "f_summary_kind": "pack",
        "f_eval_root": OLD_M01_F_PACK / "M01_OursFull_Default",
        "notes": "Old 2026-03-19 lerp rerun M01 retained to stay source-consistent with Table 1 and Table 2 Ours.",
    },
]

T_COLUMNS = [
    "T_PSNR",
    "T_SSIM",
    "T_LPIPS",
    "T_AlignedGradientCorr",
    "T_Flicker",
    "T_AirArtifact",
    "T_gaussians",
    "BgLeakRatio_band",
    "EdgeHaloScore",
    "SpikeScore_air_mean",
    "BgSensitivity_mean",
]

F_COLUMN_MAP = {
    "fusion_MI_total_mean": "F_Fusion_MI",
    "fusion_QABF_mean": "F_Fusion_QABF",
    "t_ref_Spearman_S_mean": "F_Thermal_Spearman",
    "rgb_ref_SSIM_Y_mean": "F_RGB_ref_SSIM_Y",
    "PSNR_Y_mean": "F_PSNR_Y",
    "LPIPS_mean": "F_LPIPS",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _t_row(t_df: pd.DataFrame, exp_group: str, setting: str, scene: str) -> pd.Series:
    sub = t_df[
        (t_df["exp_group"] == exp_group)
        & (t_df["setting"] == setting)
        & (t_df["dataset"] == scene)
    ]
    if len(sub) != 1:
        raise RuntimeError(f"Expected exactly one T row for {exp_group}/{scene}, got {len(sub)}")
    return sub.iloc[0]


def _f_summary_path(setting_cfg: dict[str, object], scene: str) -> Path:
    root = Path(setting_cfg["f_eval_root"])
    if setting_cfg["f_summary_kind"] == "pack":
        return root / scene / "eval" / "summary.csv"
    return root / scene / "eval" / "summary.csv"


def _f_scene_aggregate(summary_csv: Path) -> tuple[dict[str, float], dict[str, object]]:
    df = _read_csv(summary_csv)
    sub = df[df["alpha"].isin(ALPHAS)].copy()
    if sub.empty:
        raise RuntimeError(f"No alpha subset rows in {summary_csv}")
    strategies = sorted(sub["strategy"].dropna().unique().tolist())
    values = {
        out_col: float(sub[src_col].mean())
        for src_col, out_col in F_COLUMN_MAP.items()
    }
    meta = {
        "f_source_summary_csv": str(summary_csv),
        "f_rows_used": int(len(sub)),
        "f_strategy_count": int(len(strategies)),
        "f_strategies": ",".join(strategies),
        "f_alpha_subset": "0.25,0.5,0.75",
    }
    return values, meta


def _format_main(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    int_cols = ["T_gaussians"]
    float6_cols = [c for c in out.columns if c not in {"setting", *int_cols}]
    for col in float6_cols:
        out[col] = out[col].map(lambda v: f"{float(v):.6f}")
    for col in int_cols:
        out[col] = out[col].map(lambda v: f"{int(round(float(v)))}")
    return out


def _zip_dir(root: Path) -> Path:
    zip_path = root.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for cur, _, files in os.walk(root):
            for name in files:
                fp = Path(cur) / name
                zf.write(fp, fp.relative_to(root.parent))
    return zip_path


def main() -> None:
    new_t_df = _read_csv(NEW_T_SOURCE)
    old_t_df = _read_csv(OLD_T_SOURCE)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SUMMARY_ROOT / f"GroupedAblation_FStage_Extended_mixedM01_20260324_{stamp}"
    tables_dir = out_dir / "tables"
    logs_dir = out_dir / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, object]] = []
    source_manifest_rows: list[dict[str, object]] = []

    for cfg in SETTINGS:
        t_df = new_t_df if Path(cfg["t_source_csv"]) == NEW_T_SOURCE else old_t_df
        for scene in SCENES:
            t_row = _t_row(t_df, str(cfg["exp_group"]), str(cfg["setting"]), scene)
            f_summary = _f_summary_path(cfg, scene)
            f_vals, f_meta = _f_scene_aggregate(f_summary)
            row = {
                "setting": cfg["setting"],
                "exp_group": cfg["exp_group"],
                "dataset": scene,
                **{col: t_row[col] for col in T_COLUMNS},
                **f_vals,
                **f_meta,
                "t_source_csv": str(cfg["t_source_csv"]),
                "notes": cfg["notes"],
            }
            source_rows.append(row)
            source_manifest_rows.append(
                {
                    "setting": cfg["setting"],
                    "dataset": scene,
                    "t_source_csv": str(cfg["t_source_csv"]),
                    "f_source_summary_csv": str(f_summary),
                    "notes": cfg["notes"],
                }
            )

    source_df = pd.DataFrame(source_rows)
    order = [cfg["setting"] for cfg in SETTINGS]
    source_df["setting"] = pd.Categorical(source_df["setting"], categories=order, ordered=True)
    source_df = source_df.sort_values(["setting", "dataset"], kind="stable").reset_index(drop=True)

    macro_num_cols = [
        "T_PSNR",
        "T_SSIM",
        "T_LPIPS",
        "T_AlignedGradientCorr",
        "T_Flicker",
        "T_AirArtifact",
        "T_gaussians",
        "BgLeakRatio_band",
        "EdgeHaloScore",
        "SpikeScore_air_mean",
        "BgSensitivity_mean",
        "F_Fusion_MI",
        "F_Fusion_QABF",
        "F_Thermal_Spearman",
        "F_RGB_ref_SSIM_Y",
        "F_PSNR_Y",
        "F_LPIPS",
    ]
    macro_df = (
        source_df.groupby("setting", as_index=False)[macro_num_cols]
        .mean(numeric_only=True)
        .reset_index(drop=True)
    )
    macro_df["setting"] = pd.Categorical(macro_df["setting"], categories=order, ordered=True)
    macro_df = macro_df.sort_values("setting", kind="stable").reset_index(drop=True)

    display_cols = [
        "setting",
        "T_PSNR",
        "T_SSIM",
        "T_LPIPS",
        "T_gaussians",
        "BgLeakRatio_band",
        "EdgeHaloScore",
        "F_Fusion_MI",
        "F_Fusion_QABF",
        "F_Thermal_Spearman",
        "F_RGB_ref_SSIM_Y",
        "F_PSNR_Y",
        "F_LPIPS",
    ]
    display_df = _format_main(macro_df[display_cols])

    source_csv = tables_dir / "GroupedAblation_FStage_Extended_Source.csv"
    macro_csv = tables_dir / "GroupedAblation_FStage_Extended_Macro.csv"
    display_csv = tables_dir / "GroupedAblation_FStage_Extended_Display.csv"
    source_manifest_csv = tables_dir / "GroupedAblation_FStage_Extended_SourceManifest.csv"

    source_df.to_csv(source_csv, index=False, encoding="utf-8-sig")
    macro_df.to_csv(macro_csv, index=False, encoding="utf-8-sig")
    display_df.to_csv(display_csv, index=False, encoding="utf-8-sig")
    pd.DataFrame(source_manifest_rows).to_csv(source_manifest_csv, index=False, encoding="utf-8-sig")

    xlsx = tables_dir / "GroupedAblation_FStage_Extended_Comparison.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        display_df.to_excel(writer, sheet_name="Display", index=False)
        macro_df.to_excel(writer, sheet_name="Macro", index=False)
        source_df.to_excel(writer, sheet_name="Source", index=False)
        pd.DataFrame(source_manifest_rows).to_excel(writer, sheet_name="SourceManifest", index=False)

    copy_map = {
        M00_STEP13_14_RUN / "manifest.csv": logs_dir / "M00_step13to14_manifest.csv",
        G01_G02_STEP13_14_RUN / "manifest.csv": logs_dir / "G01_G02_step13to14_manifest.csv",
        G01_G02_STEP13_14_RUN / "master.log": logs_dir / "G01_G02_step13to14_master.log",
    }
    for src, dst in copy_map.items():
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    qa = {
        "alpha_subset": [0.25, 0.5, 0.75],
        "strategy_count_expected": 6,
        "setting_order": order,
        "mixed_provenance": True,
        "provenance_note": (
            "M00/G01/G02 use rawRebuild_20260323 step13-14 outputs; "
            "M01 intentionally reuses the 2026-03-19 lerp rerun to stay source-consistent "
            "with Table 1 and Table 2 Ours."
        ),
        "checks": [],
    }
    for _, row in source_df.iterrows():
        qa["checks"].append(
            {
                "setting": row["setting"],
                "dataset": row["dataset"],
                "f_rows_used": int(row["f_rows_used"]),
                "f_strategy_count": int(row["f_strategy_count"]),
                "f_source_summary_csv": row["f_source_summary_csv"],
            }
        )

    (out_dir / "QA.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "package_type": "GroupedAblation_FStage_Extended_mixedM01",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tables": {
            "source_csv": str(source_csv),
            "macro_csv": str(macro_csv),
            "display_csv": str(display_csv),
            "xlsx": str(xlsx),
            "source_manifest_csv": str(source_manifest_csv),
        },
        "sources": {
            "new_t_source_csv": str(NEW_T_SOURCE),
            "old_t_source_csv": str(OLD_T_SOURCE),
            "old_m01_f_pack_root": str(OLD_M01_F_PACK),
            "new_m00_group": str(BASE / "M00_Baseline_ModulesOff_rawRebuild_20260323"),
            "new_g01_group": str(BASE / "G01_M00_plus_SSP_rawRebuild_20260323"),
            "new_g02_group": str(BASE / "G02_M00_plus_STT_rawRebuild_20260323"),
            "old_m01_group": str(BASE / "M01_OursFull_Default"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    readme = f"""# Grouped Ablation F-Stage Extension (Mixed M01)

This package extends grouped ablation Table 3 to the fusion stage (F) using the same
per-scene aggregation rule as Table 1:

- fusion strategies: all 6 strategies in `summary.csv`
- alpha subset: `{{0.25, 0.5, 0.75}}`
- per-scene value: mean across the 18 selected rows
- macro value: mean across the 5 scenes

## Provenance

- `Baseline Transfer (M00)`: new `M00_Baseline_ModulesOff_rawRebuild_20260323`
- `M00 + SSP`: new `G01_M00_plus_SSP_rawRebuild_20260323`
- `M00 + STT`: new `G02_M00_plus_STT_rawRebuild_20260323`
- `Stabilized Transfer (M01)`: old `M01_OursFull_Default` from the 2026-03-19 lerp rerun

The package is intentionally mixed-provenance by design:

- `M01` is kept old so that it stays source-consistent with Table 1 and Table 2 Ours.
- `M00/G01/G02` use the new rawRebuild batch and the newly completed step13-14 runs.

## Step13-14 Protocol

- blend script: `blend_model_strict_endpoints.py`
- `dc_y_from = lerp`
- default `endpoint_mode = blend`
- eval script: `eval_blend_sweep.py`

## Main Tables

- `tables/GroupedAblation_FStage_Extended_Display.csv`
- `tables/GroupedAblation_FStage_Extended_Macro.csv`
- `tables/GroupedAblation_FStage_Extended_Source.csv`
- `tables/GroupedAblation_FStage_Extended_Comparison.xlsx`

## Notes

- `T_*` metrics come from grouped ablation source tables.
- `F_*` metrics come from `eval/summary.csv` using the same aggregation rule as Table 1.
- `M01` F-stage values come from the frozen lerp-evalfull package / old 2026-03-19 outputs.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    _zip_dir(out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
