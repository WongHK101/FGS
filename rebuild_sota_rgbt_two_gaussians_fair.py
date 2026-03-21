from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as tf
from PIL import Image

from lpipsPyTorch import lpips
from utils.image_utils import psnr
from utils.loss_utils import ssim


SCENES = ["Building", "Orchard", "PVpanel", "Road", "TransmissionTower"]
METHOD_ORDER = [
    "ThermoNeRF",
    "Thermal3D_GS",
    "ThermalGaussian_MFTG",
    "ThermalGaussian_MSMG",
    "ThermalGaussian_OMMG",
    "Ours",
]


def _round_r4_size(size: Tuple[int, int]) -> Tuple[int, int]:
    w, h = size
    return round(w / 4.0), round(h / 4.0)


def _load_rgb_tensor(path: Path, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return tf.to_tensor(img).unsqueeze(0).to(device)


def _load_rgb_tensor_from_image(img: Image.Image, size: Tuple[int, int], device: torch.device) -> torch.Tensor:
    img = img.convert("RGB")
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
    return tf.to_tensor(img).unsqueeze(0).to(device)


def _compute_triplet(render: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    return {
        "PSNR": float(psnr(render[:, :3], gt[:, :3]).item()),
        "SSIM": float(ssim(render[:, :3], gt[:, :3]).item()),
        "LPIPS": float(lpips(render[:, :3], gt[:, :3], net_type="vgg").item()),
    }


def _evaluate_thermonerf_scene(
    scene: str,
    thermonerf_root: Path,
    thermonerf_data_root: Path,
    device: torch.device,
) -> Dict[str, object]:
    eval_dir = thermonerf_root / scene / "eval"
    data_dir = thermonerf_data_root / scene
    transforms = json.loads((data_dir / "transforms.json").read_text(encoding="utf-8"))

    test_files: List[str] = transforms.get("test_filenames", [])
    if not test_files:
        raise RuntimeError(f"No test_filenames found in {data_dir / 'transforms.json'}")

    frame_map = {
        frame["file_path"].replace("\\", "/"): frame["thermal_file_path"].replace("\\", "/")
        for frame in transforms["frames"]
    }

    rgb_scores: List[Dict[str, float]] = []
    t_scores: List[Dict[str, float]] = []
    audit_rows: List[Dict[str, object]] = []

    for idx, rel_rgb in enumerate(test_files):
        rel_rgb = rel_rgb.replace("\\", "/")
        gt_rgb_path = (data_dir / rel_rgb).resolve()
        rgb_combined_path = eval_dir / f"img_{idx:05d}.jpg"
        combined_t_path = eval_dir / f"thermal_combined_{idx:05d}.jpg"

        if not gt_rgb_path.exists():
            raise FileNotFoundError(f"Missing ThermoNeRF RGB GT: {gt_rgb_path}")
        if not rgb_combined_path.exists():
            raise FileNotFoundError(f"Missing ThermoNeRF RGB combined image: {rgb_combined_path}")
        if not combined_t_path.exists():
            raise FileNotFoundError(f"Missing ThermoNeRF thermal combined render: {combined_t_path}")

        gt_rgb_img = Image.open(gt_rgb_path).convert("RGB")
        target_size = _round_r4_size(gt_rgb_img.size)

        rgb_combined = Image.open(rgb_combined_path).convert("RGB")
        rgb_w, rgb_h = rgb_combined.size
        rgb_gt_img = rgb_combined.crop((0, 0, rgb_w // 2, rgb_h))
        rgb_pred_img = rgb_combined.crop((rgb_w // 2, 0, rgb_w, rgb_h))
        render_rgb = _load_rgb_tensor_from_image(rgb_pred_img, target_size, device)
        gt_rgb = _load_rgb_tensor_from_image(rgb_gt_img, target_size, device)
        rgb_metric = _compute_triplet(render_rgb, gt_rgb)
        rgb_scores.append(rgb_metric)

        combined = Image.open(combined_t_path).convert("RGB")
        full_w, full_h = combined.size
        half_w = full_w // 2
        gt_t_img = combined.crop((0, 0, half_w, full_h))
        pred_t_img = combined.crop((half_w, 0, full_w, full_h))
        gt_t = _load_rgb_tensor_from_image(gt_t_img, target_size, device)
        pred_t = _load_rgb_tensor_from_image(pred_t_img, target_size, device)
        t_metric = _compute_triplet(pred_t, gt_t)
        t_scores.append(t_metric)

        audit_rows.append(
            {
                "scene": scene,
                "view_idx": idx,
                "rgb_combined": str(rgb_combined_path),
                "rgb_gt_external": str(gt_rgb_path),
                "thermal_combined": str(combined_t_path),
                "target_w": target_size[0],
                "target_h": target_size[1],
                "thermal_rel": frame_map[rel_rgb],
            }
        )

    def _mean(items: List[Dict[str, float]], key: str) -> float:
        return float(np.mean([row[key] for row in items]))

    return {
        "dataset": scene,
        "RGB_PSNR": _mean(rgb_scores, "PSNR"),
        "RGB_SSIM": _mean(rgb_scores, "SSIM"),
        "RGB_LPIPS": _mean(rgb_scores, "LPIPS"),
        "T_PSNR": _mean(t_scores, "PSNR"),
        "T_SSIM": _mean(t_scores, "SSIM"),
        "T_LPIPS": _mean(t_scores, "LPIPS"),
        "view_count": len(test_files),
        "render_size_r4": f"{audit_rows[0]['target_w']}x{audit_rows[0]['target_h']}",
        "audit_rows": audit_rows,
    }


def _load_base_rows(
    ours_csv: Path,
    tg_csv: Path,
    paper_final_csv: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    ours = pd.read_csv(ours_csv)
    ours = ours[ours["exp_group"] == "M01_OursFull_Default"]
    for scene in SCENES:
        r = ours[ours["dataset"] == scene].iloc[0]
        rows.append(
            {
                "Scene": scene,
                "Method": "Ours",
                "RGB PSNR": float(r["RGB_PSNR"]),
                "RGB SSIM": float(r["RGB_SSIM"]),
                "RGB LPIPS": float(r["RGB_LPIPS"]),
                "RGB Gaussian": float(r["RGB_gaussians"]),
                "T PSNR": float(r["T_PSNR"]),
                "T SSIM": float(r["T_SSIM"]),
                "T LPIPS": float(r["T_LPIPS"]),
                "T Gaussian": float(r["T_gaussians"]),
            }
        )

    tg = pd.read_csv(tg_csv)
    for method in ["ThermalGaussian_MFTG", "ThermalGaussian_MSMG", "ThermalGaussian_OMMG"]:
        sub = tg[tg["method"] == method]
        for scene in SCENES:
            r = sub[sub["dataset"] == scene].iloc[0]
            if method == "ThermalGaussian_OMMG":
                rgb_g = float(r["gaussians_total"])
                t_g = float(r["gaussians_total"])
            else:
                rgb_g = float(r["color_gaussians"])
                t_g = float(r["thermal_gaussians"])
            rows.append(
                {
                    "Scene": scene,
                    "Method": method,
                    "RGB PSNR": float(r["RGB_PSNR"]),
                    "RGB SSIM": float(r["RGB_SSIM"]),
                    "RGB LPIPS": float(r["RGB_LPIPS"]),
                    "RGB Gaussian": rgb_g,
                    "T PSNR": float(r["T_PSNR"]),
                    "T SSIM": float(r["T_SSIM"]),
                    "T LPIPS": float(r["T_LPIPS"]),
                    "T Gaussian": t_g,
                }
            )

    paper_final = pd.read_csv(paper_final_csv)
    t3d = paper_final[paper_final["method"] == "Thermal3D_GS"][["dataset", "gaussian_count"]]
    for scene in SCENES:
        unified = json.loads(
            (
                Path(r"F:\databackup\xr6\output\SOTA_Comparison\UnifiedEval\Thermal3D_GS")
                / scene
                / "results.json"
            ).read_text(encoding="utf-8")
        )
        vals = next(iter(unified.values()))
        g = float(t3d[t3d["dataset"] == scene]["gaussian_count"].iloc[0])
        rows.append(
            {
                "Scene": scene,
                "Method": "Thermal3D_GS",
                "RGB PSNR": np.nan,
                "RGB SSIM": np.nan,
                "RGB LPIPS": np.nan,
                "RGB Gaussian": np.nan,
                "T PSNR": float(vals["PSNR"]),
                "T SSIM": float(vals["SSIM"]),
                "T LPIPS": float(vals["LPIPS"]),
                "T Gaussian": g,
            }
        )

    return pd.DataFrame(rows)


def _build_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["RGB PSNR", "T PSNR"]:
        out[col] = out[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    for col in ["RGB SSIM", "RGB LPIPS", "T SSIM", "T LPIPS"]:
        out[col] = out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    for col in ["RGB Gaussian", "T Gaussian"]:
        out[col] = out[col].map(lambda x: f"{int(round(x)):,}" if pd.notna(x) else "N/A")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=r"F:\databackup\xr6\output")
    ap.add_argument("--thermonerf_root", default=r"F:\databackup\xr6\output\SOTA_Comparison\ThermoNeRF")
    ap.add_argument("--thermonerf_data_root", default=r"E:\3DGS\ThermoNeRF\dataset\input")
    ap.add_argument(
        "--ours_csv",
        default=r"F:\databackup\xr6\output\Summaries\M00_M01_AssistantPack_lerp_evalfull_20260320_022415\tables\M00_M01_Source.csv",
    )
    ap.add_argument(
        "--tg_csv",
        default=r"F:\databackup\xr6\output\Summaries\ThermalGaussian_Variants_AssistantPack_20260320_004104\tables\ThermalGaussian_Variants_Source.csv",
    )
    ap.add_argument(
        "--paper_final_csv",
        default=r"F:\databackup\xr6\output\Summaries\Paper_Final_Source.csv",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_root) / f"SOTA_RGBT_TwoGaussians_FairR4_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_df = _load_base_rows(Path(args.ours_csv), Path(args.tg_csv), Path(args.paper_final_csv))

    thermonerf_rows = []
    thermonerf_audit = []
    for scene in SCENES:
        result = _evaluate_thermonerf_scene(scene, Path(args.thermonerf_root), Path(args.thermonerf_data_root), device)
        thermonerf_rows.append(
            {
                "Scene": scene,
                "Method": "ThermoNeRF",
                "RGB PSNR": result["RGB_PSNR"],
                "RGB SSIM": result["RGB_SSIM"],
                "RGB LPIPS": result["RGB_LPIPS"],
                "RGB Gaussian": np.nan,
                "T PSNR": result["T_PSNR"],
                "T SSIM": result["T_SSIM"],
                "T LPIPS": result["T_LPIPS"],
                "T Gaussian": np.nan,
            }
        )
        for row in result["audit_rows"]:
            row["normalization_rule"] = "RGB: native render vs GT image, both resized to r=4; T: split thermal_combined into GT/pred gray halves, both resized to r=4."
            thermonerf_audit.append(row)

    thermonerf_df = pd.DataFrame(thermonerf_rows)
    merged = pd.concat([base_df[base_df["Method"] != "ThermoNeRF"], thermonerf_df], ignore_index=True)
    merged["Method"] = pd.Categorical(merged["Method"], METHOD_ORDER, ordered=True)
    merged["Scene"] = pd.Categorical(merged["Scene"], SCENES, ordered=True)
    merged = merged.sort_values(["Scene", "Method"]).reset_index(drop=True)

    mean_df = (
        merged.groupby("Method", observed=True)
        .mean(numeric_only=True)
        .reset_index()
    )
    mean_df.insert(0, "Scene", "Mean")
    mean_df["Method"] = pd.Categorical(mean_df["Method"], METHOD_ORDER, ordered=True)
    mean_df = mean_df.sort_values("Method").reset_index(drop=True)

    full_raw = pd.concat([merged, mean_df], ignore_index=True)
    display = _build_display(full_raw)

    source_audit_rows = [
        {
            "Method": "ThermoNeRF",
            "Scene": scene,
            "field_group": "RGB metrics",
            "source": str(Path(args.thermonerf_root) / scene / "eval" / "img_*.jpg"),
            "note": "Fair-r4 rebuild from ThermoNeRF native RGB combined images: split img_*.jpg into GT/pred halves, then resize both to r=4 and score with official 3DGS metrics.py functions.",
        }
        for scene in SCENES
    ] + [
        {
            "Method": "ThermoNeRF",
            "Scene": scene,
            "field_group": "T metrics",
            "source": str(Path(args.thermonerf_root) / scene / "eval" / "thermal_combined_*.jpg"),
            "note": "Fair-r4 rebuild by splitting thermal_combined into GT(gray) and pred(gray), then resizing both to r=4 and scoring with official 3DGS metrics.py functions.",
        }
        for scene in SCENES
    ] + [
        {
            "Method": "ThermoNeRF",
            "Scene": scene,
            "field_group": "RGB Gaussian / T Gaussian",
            "source": "",
            "note": "NeRF representation; no Gaussian count exposed.",
        }
        for scene in SCENES
    ]

    old_audit = pd.read_csv(Path(args.out_root) / "SOTA_RGBT_TwoGaussians_20260320_060948" / "SOTA_RGBT_TwoGaussians_source_audit.csv")
    old_audit = old_audit[old_audit["Method"] != "ThermoNeRF"].copy()
    source_audit = pd.concat([old_audit, pd.DataFrame(source_audit_rows)], ignore_index=True)

    missing_notes = pd.read_csv(Path(args.out_root) / "SOTA_RGBT_TwoGaussians_20260320_060948" / "SOTA_RGBT_TwoGaussians_MissingNotes.csv")
    # Keep previous notes but update ThermoNeRF note wording.
    missing_notes = missing_notes[missing_notes["method"] != "ThermoNeRF"].copy()
    missing_notes = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "method": "ThermoNeRF",
                        "field": "RGB Gaussian / T Gaussian",
                        "status": "N/A",
                        "reason": "NeRF representation; no Gaussian count exists by definition.",
                        "how_to_fill": "Do not fill; keep N/A.",
                    }
                ]
            ),
            missing_notes,
        ],
        ignore_index=True,
    )

    readme = f"""# SOTA RGB/T Table (Fair r=4 ThermoNeRF Repair)

This folder contains a repaired version of the RGB/T comparison table.

## What was repaired
- ThermoNeRF RGB metrics were recomputed at `r=4` from native `img_*.jpg` renders and GT RGB test images.
- ThermoNeRF RGB metrics were recomputed at `r=4` by splitting native `img_*.jpg` combined images into GT/pred halves.
- ThermoNeRF thermal metrics were recomputed at `r=4` by splitting `thermal_combined_*.jpg` into GT(gray) and prediction(gray) halves.
- Metric functions are the same as the official GraphDECO 3DGS `metrics.py` implementation in this repo: `PSNR`, `SSIM`, and `LPIPS`.

## Important caveat
- ThermoNeRF thermal outputs currently available in archived artifacts are grayscale visualizations from the official evaluator, not the pseudo-color thermal JPGs used by the 3DGS-family methods.
- Therefore this repair is much fairer than the previous table (which incorrectly compared gray predictions to pseudo-color GT), but it still reflects ThermoNeRF's native gray thermal visualization space.
- No retraining was performed. The repair was extracted from existing evaluation artifacts only.

## Files
- `SOTA_RGBT_TwoGaussians_FairR4.xlsx`: main workbook
- `SOTA_RGBT_TwoGaussians_FairR4_raw.csv`: raw numeric table
- `SOTA_RGBT_TwoGaussians_FairR4_display.csv`: display-ready table
- `SOTA_RGBT_TwoGaussians_FairR4_source_audit.csv`: source-by-source provenance
- `SOTA_RGBT_TwoGaussians_FairR4_ThermoNeRF_audit.csv`: per-view ThermoNeRF repair audit
- `SOTA_RGBT_TwoGaussians_FairR4_MissingNotes.csv`: justified N/A items
"""

    raw_csv = out_dir / "SOTA_RGBT_TwoGaussians_FairR4_raw.csv"
    display_csv = out_dir / "SOTA_RGBT_TwoGaussians_FairR4_display.csv"
    audit_csv = out_dir / "SOTA_RGBT_TwoGaussians_FairR4_source_audit.csv"
    tn_audit_csv = out_dir / "SOTA_RGBT_TwoGaussians_FairR4_ThermoNeRF_audit.csv"
    missing_csv = out_dir / "SOTA_RGBT_TwoGaussians_FairR4_MissingNotes.csv"
    xlsx_path = out_dir / "SOTA_RGBT_TwoGaussians_FairR4.xlsx"
    readme_path = out_dir / "README.md"

    full_raw.to_csv(raw_csv, index=False)
    display.to_csv(display_csv, index=False)
    source_audit.to_csv(audit_csv, index=False)
    pd.DataFrame(thermonerf_audit).to_csv(tn_audit_csv, index=False)
    missing_notes.to_csv(missing_csv, index=False)
    readme_path.write_text(readme, encoding="utf-8")

    with pd.ExcelWriter(xlsx_path) as writer:
        display.to_excel(writer, index=False, sheet_name="display")
        full_raw.to_excel(writer, index=False, sheet_name="raw")
        source_audit.to_excel(writer, index=False, sheet_name="source_audit")
        pd.DataFrame(thermonerf_audit).to_excel(writer, index=False, sheet_name="thermonerf_audit")
        missing_notes.to_excel(writer, index=False, sheet_name="missing_notes")

    print(f"Wrote repaired table to: {out_dir}")


if __name__ == "__main__":
    main()
