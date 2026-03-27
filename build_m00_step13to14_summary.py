from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd


SCENES = ["Building", "Orchard", "PVpanel", "Road", "TransmissionTower"]
GROUP_NAME = "M00_Baseline_ModulesOff_rawRebuild_20260323"
GROUP_ROOT = Path(r"F:\databackup\xr6\output\Ch4_2_MainComparison") / GROUP_NAME
RUN_DIR = Path(
    r"F:\databackup\xr6\output\Ch4_2_MainComparison\_m00_step13to14_retry2_20260324_012445"
)
SUMMARIES_ROOT = Path(r"F:\databackup\xr6\output\Summaries")


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _best_row(df: pd.DataFrame, metric: str, higher_is_better: bool) -> dict[str, object]:
    if metric not in df.columns:
        return {
            "metric": metric,
            "direction": "max" if higher_is_better else "min",
            "strategy": None,
            "alpha": None,
            "value": None,
        }
    sub = df[["strategy", "alpha", metric]].dropna()
    if sub.empty:
        return {
            "metric": metric,
            "direction": "max" if higher_is_better else "min",
            "strategy": None,
            "alpha": None,
            "value": None,
        }
    row = sub.sort_values(metric, ascending=not higher_is_better).iloc[0]
    return {
        "metric": metric,
        "direction": "max" if higher_is_better else "min",
        "strategy": row["strategy"],
        "alpha": float(row["alpha"]),
        "value": float(row[metric]),
    }


def _scene_eval_dir(scene: str) -> Path:
    return GROUP_ROOT / scene / "eval"


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = SUMMARIES_ROOT / f"M00_BlendSweep_AssistantPack_rawRebuild_20260324_{ts}"
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[pd.DataFrame] = []
    scene_index_rows: list[dict[str, object]] = []
    for scene in SCENES:
        eval_dir = _scene_eval_dir(scene)
        summary_csv = eval_dir / "summary.csv"
        summary_gt_csv = eval_dir / "summary_gt_ref.csv"
        summary_rr_csv = eval_dir / "summary_render_ref.csv"
        if not summary_csv.exists():
            raise FileNotFoundError(summary_csv)
        df = pd.read_csv(summary_csv)
        df.insert(0, "scene", scene)
        all_rows.append(df)
        scene_index_rows.append(
            {
                "scene": scene,
                "eval_dir": str(eval_dir),
                "summary_csv": str(summary_csv),
                "summary_gt_ref_csv": str(summary_gt_csv),
                "summary_render_ref_csv": str(summary_rr_csv),
                "model_f_dir": str(GROUP_ROOT / scene / "Model_F"),
                "summary_rows": int(len(df)),
            }
        )

    all_df = pd.concat(all_rows, ignore_index=True)
    num_cols = _numeric_columns(all_df.drop(columns=["scene", "strategy", "label"], errors="ignore"))
    macro_df = (
        all_df.groupby(["strategy", "alpha"], as_index=False)[num_cols]
        .mean(numeric_only=True)
        .sort_values(["strategy", "alpha"], kind="stable")
        .reset_index(drop=True)
    )

    best_metrics = [
        ("PSNR_mean", True),
        ("SSIM_mean", True),
        ("LPIPS_mean", False),
        ("fusion_MI_total_mean", True),
        ("fusion_QABF_mean", True),
        ("t_ref_Spearman_S_mean", True),
        ("t_ref_SSIM_S_mean", True),
    ]
    best_df = pd.DataFrame([_best_row(macro_df, m, hib) for m, hib in best_metrics])

    scene_index_df = pd.DataFrame(scene_index_rows)
    manifest_df = pd.read_csv(RUN_DIR / "manifest.csv")

    all_csv = tables_dir / "M00_BlendSweep_AllScenes.csv"
    macro_csv = tables_dir / "M00_BlendSweep_Macro.csv"
    best_csv = tables_dir / "M00_BlendSweep_BestByMetric.csv"
    scene_index_csv = tables_dir / "M00_BlendSweep_SceneIndex.csv"
    manifest_csv = tables_dir / "M00_BlendSweep_RunManifest.csv"

    all_df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    macro_df.to_csv(macro_csv, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_csv, index=False, encoding="utf-8-sig")
    scene_index_df.to_csv(scene_index_csv, index=False, encoding="utf-8-sig")
    manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    xlsx_path = tables_dir / "M00_BlendSweep_Comparison.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        macro_df.to_excel(writer, sheet_name="Macro", index=False)
        best_df.to_excel(writer, sheet_name="BestByMetric", index=False)
        all_df.to_excel(writer, sheet_name="AllScenes", index=False)
        scene_index_df.to_excel(writer, sheet_name="SceneIndex", index=False)
        manifest_df.to_excel(writer, sheet_name="RunManifest", index=False)

    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# M00 Step13-14 Summary",
                "",
                f"- Source group: `{GROUP_NAME}`",
                f"- Source root: `{GROUP_ROOT}`",
                f"- Step13-14 run dir: `{RUN_DIR}`",
                "- Scope: latest clean-batch `M00` blend (`Model_F`) and `eval_blend_sweep` outputs for all five scenes.",
                "- Alpha semantics: `alpha` is the thermal-model weight for blended fields; `alpha=0` is RGB-side, `alpha=1` is thermal-side for the fields blended by that strategy.",
                "- Current blend default is continuous `endpoint_mode=blend`, not hard `copy` endpoints.",
                "",
                "## Files",
                f"- `tables/{all_csv.name}`: per-scene combined summary rows.",
                f"- `tables/{macro_csv.name}`: five-scene macro means by `(strategy, alpha)`.",
                f"- `tables/{best_csv.name}`: best macro row for selected metrics.",
                f"- `tables/{scene_index_csv.name}`: scene-to-source path index.",
                f"- `tables/{manifest_csv.name}`: step13-14 queue manifest.",
                f"- `tables/{xlsx_path.name}`: Excel workbook with the same content.",
                "",
                "## Notes",
                "- `summary.csv` is the merged blend-sweep table produced by `eval_blend_sweep.py`.",
                "- Per-scene raw files remain in each scene's `eval/` directory (`summary.csv`, `summary_gt_ref.csv`, `summary_render_ref.csv`, plots, montages).",
                "- This package only summarizes `M00` step13-14. It does not modify grouped-ablation or SOTA tables.",
            ]
        ),
        encoding="utf-8",
    )

    manifest_json = out_dir / "manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "group": GROUP_NAME,
                "source_root": str(GROUP_ROOT),
                "run_dir": str(RUN_DIR),
                "scenes": SCENES,
                "tables": {
                    "all_scenes": str(all_csv),
                    "macro": str(macro_csv),
                    "best_by_metric": str(best_csv),
                    "scene_index": str(scene_index_csv),
                    "run_manifest": str(manifest_csv),
                    "xlsx": str(xlsx_path),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    zip_path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            zf.write(p, p.relative_to(out_dir.parent))

    print(f"OUT_DIR={out_dir}")
    print(f"ZIP={zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
