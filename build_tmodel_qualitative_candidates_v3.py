import argparse
import csv
import json
import shutil
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image


SCENES = ("Building", "PVpanel")
METHOD_ORDER = (
    "ThermoNeRF",
    "Thermal3D_GS",
    "ThermalGaussian_MSMG",
    "ThermalGaussian_MFTG",
    "ThermalGaussian_OMMG",
    "Ours",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild T-model qualitative candidates using Ours/T_ref pseudo-color "
            "style as the display anchor instead of grayscale-unifying every method."
        )
    )
    parser.add_argument(
        "--old_root",
        type=Path,
        default=Path(r"F:\databackup\xr6\output\TModel_Qualitative_Candidates_20260320_074500"),
        help="Existing candidate package used to source scene/view selections.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(r"F:\databackup\xr6\output"),
        help="Parent output directory for the rebuilt v3 package.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional custom output tag. Defaults to timestamped v3 name.",
    )
    return parser.parse_args()


def read_rows(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def open_rgb(path, target_size=None):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if target_size is not None and img.size != target_size:
            img = img.resize(target_size, Image.Resampling.BILINEAR)
        return img


def load_thermonerf_pred(eval_dir, view_id, target_size):
    combined_path = eval_dir / f"thermal_combined_{view_id}.jpg"
    with Image.open(combined_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        pred = img.crop((w // 2, 0, w, h))
        if pred.size != target_size:
            pred = pred.resize(target_size, Image.Resampling.BILINEAR)
        return pred


def to_gray_array(img):
    return np.asarray(img.convert("L"), dtype=np.float32)


def build_tref_rgb_lut(tref_rgb):
    tref_arr = np.asarray(tref_rgb.convert("RGB"), dtype=np.float32)
    tref_gray = np.asarray(tref_rgb.convert("L"), dtype=np.float32)
    lo = float(np.percentile(tref_gray, 1.0))
    hi = float(np.percentile(tref_gray, 99.0))
    if hi <= lo:
        lo = float(np.min(tref_gray))
        hi = float(np.max(tref_gray))
    if hi <= lo:
        hi = lo + 1.0

    bins = np.clip(np.round((tref_gray - lo) / (hi - lo) * 255.0), 0, 255).astype(np.int32)
    sums = np.zeros((256, 3), dtype=np.float64)
    counts = np.zeros(256, dtype=np.int64)

    flat_bins = bins.reshape(-1)
    flat_rgb = tref_arr.reshape(-1, 3)
    for idx in range(flat_bins.shape[0]):
        b = int(flat_bins[idx])
        sums[b] += flat_rgb[idx]
        counts[b] += 1

    lut = np.zeros((256, 3), dtype=np.float64)
    present = counts > 0
    lut[present] = sums[present] / counts[present, None]

    present_idx = np.flatnonzero(present)
    if present_idx.size == 0:
        lut[:] = 128.0
    elif present_idx.size == 1:
        lut[:] = lut[present_idx[0]]
    else:
        for c in range(3):
            lut[:, c] = np.interp(
                np.arange(256),
                present_idx.astype(np.float64),
                lut[present_idx, c],
            )

    lut = np.clip(np.round(lut), 0, 255).astype(np.uint8)
    return lut, lo, hi


def apply_tref_style_to_gray(gray_rgb_img, tref_lut, lo, hi):
    gray = np.asarray(gray_rgb_img.convert("L"), dtype=np.float32)
    bins = np.clip(np.round((gray - lo) / (hi - lo) * 255.0), 0, 255).astype(np.int32)
    styled = tref_lut[bins]
    return Image.fromarray(styled.astype(np.uint8), mode="RGB")


def copy_png(src_img, out_path):
    src_img.save(out_path)


def build_scene(scene, old_root, out_dir):
    rows = read_rows(old_root / scene / "scene_candidates.csv")
    summary_rows = []

    for row in rows:
        view_id = row["view_id"]
        old_manifest_path = old_root / scene / f"view_{view_id}" / "manifest.json"
        old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))

        view_dir = out_dir / scene / f"view_{view_id}"
        ensure_dir(view_dir)

        tref_dir = Path(old_manifest["source_dir_tref"])
        ours_dir = Path(old_manifest["source_dir_ours"])
        ommg_dir = Path(old_manifest["source_dir_thermogaussian_ommg"])
        t3d_dir = Path(old_manifest["source_dir_thermal3dgs"])
        msmg_dir = Path(old_manifest["source_dir_thermogaussian_msmg"])
        mftg_dir = Path(old_manifest["source_dir_thermogaussian_mftg"])
        thermonerf_dir = Path(old_manifest["source_dir_thermonerf"])

        tref_path = tref_dir / f"{view_id}.png"
        tref_img = open_rgb(tref_path)
        target_size = tref_img.size
        tref_lut, lo, hi = build_tref_rgb_lut(tref_img)

        images = OrderedDict()
        images["T_ref"] = tref_img
        images["Ours"] = open_rgb(ours_dir / f"{view_id}.png", target_size)
        images["ThermalGaussian_OMMG"] = open_rgb(ommg_dir / f"{view_id}.png", target_size)
        images["Thermal3D_GS"] = open_rgb(t3d_dir / f"{view_id}.png", target_size)
        images["ThermalGaussian_MSMG"] = open_rgb(msmg_dir / f"{view_id}.png", target_size)
        images["ThermalGaussian_MFTG"] = open_rgb(mftg_dir / f"{view_id}.png", target_size)
        thermonerf_pred = load_thermonerf_pred(thermonerf_dir, view_id, target_size)
        images["ThermoNeRF"] = apply_tref_style_to_gray(thermonerf_pred, tref_lut, lo, hi)

        for name, img in images.items():
            copy_png(img, view_dir / f"{name}.png")

        new_manifest = {
            "scene": scene,
            "view_id": view_id,
            "candidate_rank": old_manifest.get("candidate_rank"),
            "why_selected": old_manifest.get("why_selected", ""),
            "suggested_crop_xywh": old_manifest.get("suggested_crop_xywh"),
            "all_methods_present": "yes",
            "column_order": list(METHOD_ORDER),
            "colormap_name": "ours/T_ref pseudo-color baseline; ThermoNeRF recolored to match the same family",
            "normalization_rule": (
                "Archived pseudo-color outputs from Ours, Thermal3D-GS, and ThermalGaussian "
                "variants are preserved. ThermoNeRF uses the prediction half from "
                "thermal_combined_<view>.jpg, resized to the shared canvas, then recolored "
                "using a per-view RGB lookup table derived from T_ref after applying the "
                "same T_ref luminance [1%, 99%] percentile range."
            ),
            "source_dir_ours": old_manifest["source_dir_ours"],
            "source_dir_thermogaussian_ommg": old_manifest["source_dir_thermogaussian_ommg"],
            "source_dir_thermal3dgs": old_manifest["source_dir_thermal3dgs"],
            "source_dir_thermogaussian_msmg": old_manifest["source_dir_thermogaussian_msmg"],
            "source_dir_thermogaussian_mftg": old_manifest["source_dir_thermogaussian_mftg"],
            "source_dir_thermonerf": old_manifest["source_dir_thermonerf"],
            "source_dir_tref": old_manifest["source_dir_tref"],
            "thermonerf_processing": {
                "source_file": f"thermal_combined_{view_id}.jpg",
                "source_region": "right half (prediction)",
                "target_canvas": list(target_size),
                "tref_luminance_p1": round(lo, 4),
                "tref_luminance_p99": round(hi, 4),
                "color_transfer": "per-view lookup table fitted from T_ref luminance -> RGB"
            },
            "v1_manifest_path": str(old_manifest_path),
            "v1_package_root": str(old_root),
            "display_protocol_version": "v3_ours_baseline_pseudocolor",
            "notes": (
                "This v3 package keeps Ours/T_ref pseudo-color style as the baseline display "
                "family. It avoids forcing every method to grayscale; only ThermoNeRF is "
                "adapted to that family because its archived thermal outputs are grayscale."
            ),
        }
        (view_dir / "manifest.json").write_text(
            json.dumps(new_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary_rows.append(
            {
                "scene": scene,
                "view_id": view_id,
                "candidate_rank": row.get("candidate_rank", ""),
                "why_selected": row.get("why_selected", ""),
                "suggested_crop_xywh": row.get("suggested_crop_xywh", ""),
                "all_methods_present": "yes",
                "colormap_name": new_manifest["colormap_name"],
                "normalization_rule": new_manifest["normalization_rule"],
                "tref_luminance_p1": f"{lo:.4f}",
                "tref_luminance_p99": f"{hi:.4f}",
                "source_dir_ours": new_manifest["source_dir_ours"],
                "source_dir_thermogaussian_ommg": new_manifest["source_dir_thermogaussian_ommg"],
                "source_dir_thermal3dgs": new_manifest["source_dir_thermal3dgs"],
                "source_dir_thermogaussian_msmg": new_manifest["source_dir_thermogaussian_msmg"],
                "source_dir_thermogaussian_mftg": new_manifest["source_dir_thermogaussian_mftg"],
                "source_dir_thermonerf": new_manifest["source_dir_thermonerf"],
                "source_dir_tref": new_manifest["source_dir_tref"],
            }
        )

    return summary_rows


def write_csv(rows, out_path):
    if not rows:
        return
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_readme(out_dir, old_root):
    text = f"""# TModel Qualitative Candidates v3

This package supersedes both:

- `{old_root}`
- the intermediate grayscale-only v2 package

It keeps the same held-out test-view selections for:

- `Building` (10 views)
- `PVpanel` (10 views)

## Purpose

This is a **candidate-image package**, not the final paper collage. It is meant for later
manual selection and layout.

## Design Choice

This v3 package uses **Ours / T_ref pseudo-color style as the display baseline**.

- Ours, Thermal3D-GS, and ThermalGaussian variants keep their archived pseudo-color thermal renders.
- ThermoNeRF does not have a matching pseudo-color thermal archive, so its thermal prediction is
  extracted from `thermal_combined_<view>.jpg` and recolored using a per-view lookup table derived
  from `T_ref`.

This avoids forcing every method into grayscale just because ThermoNeRF is stored that way.

## Fixed Column Order

1. `ThermoNeRF`
2. `Thermal3D-GS`
3. `ThermalGaussian-MSMG`
4. `ThermalGaussian-MFTG`
5. `ThermalGaussian-OMMG`
6. `Ours`

Each view folder also includes `T_ref.png`.

## Notes

- Held-out test views only.
- Same scene/view IDs as v1.
- No retraining and no rerendering.
- This is still a packaging/re-visualization step; it does not change any quantitative result.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def zip_dir(src_dir, zip_path):
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir))


def main():
    args = parse_args()
    tag = args.tag or f"TModel_Qualitative_Candidates_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out_root / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    all_rows = []
    for scene in SCENES:
        all_rows.extend(build_scene(scene, args.old_root, out_dir))

    write_csv(all_rows, out_dir / "scene_summary.csv")
    build_readme(out_dir, args.old_root)
    zip_dir(out_dir, out_dir.with_suffix(".zip"))
    print(out_dir)
    print(out_dir.with_suffix(".zip"))


if __name__ == "__main__":
    main()
