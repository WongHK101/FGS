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
THERMAL_METHOD_ORDER = (
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
            "Rebuild the T-model qualitative candidate pack with a unified "
            "grayscale display rule from archived held-out test-view renders."
        )
    )
    parser.add_argument(
        "--old_root",
        type=Path,
        default=Path(r"F:\databackup\xr6\output\TModel_Qualitative_Candidates_20260320_074500"),
        help="Existing candidate package used to source view lists and manifests.",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(r"F:\databackup\xr6\output"),
        help="Parent output directory for the rebuilt candidate package.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional custom directory tag. Defaults to timestamped v2 name.",
    )
    return parser.parse_args()


def read_rows(csv_path):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def to_gray_array(img):
    return np.asarray(img.convert("L"), dtype=np.float32)


def normalize_with_range(gray, lo, hi):
    if hi <= lo:
        return np.zeros_like(gray, dtype=np.uint8)
    arr = (gray - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.round(arr * 255.0).astype(np.uint8)
    return arr


def open_standard_image(path, target_size):
    with Image.open(path) as img:
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.BILINEAR)
        return img.convert("RGB")


def open_thermonerf_pred(eval_dir, view_id, target_size):
    combined_path = eval_dir / f"thermal_combined_{view_id}.jpg"
    with Image.open(combined_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        half_w = w // 2
        pred = img.crop((half_w, 0, w, h))
        if pred.size != target_size:
            pred = pred.resize(target_size, Image.Resampling.BILINEAR)
        return pred.convert("RGB")


def save_gray_png(gray_u8, out_path):
    Image.fromarray(gray_u8, mode="L").save(out_path)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def build_scene(scene, old_root, out_dir):
    scene_candidates = read_rows(old_root / scene / "scene_candidates.csv")
    scene_summary_rows = []
    for row in scene_candidates:
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
        with Image.open(tref_path) as tref_probe:
            target_size = tref_probe.size
        target_img = open_standard_image(tref_path, target_size)

        images = OrderedDict()
        images["T_ref"] = target_img
        images["Ours"] = open_standard_image(ours_dir / f"{view_id}.png", target_size)
        images["ThermalGaussian_OMMG"] = open_standard_image(ommg_dir / f"{view_id}.png", target_size)
        images["Thermal3D_GS"] = open_standard_image(t3d_dir / f"{view_id}.png", target_size)
        images["ThermalGaussian_MSMG"] = open_standard_image(msmg_dir / f"{view_id}.png", target_size)
        images["ThermalGaussian_MFTG"] = open_standard_image(mftg_dir / f"{view_id}.png", target_size)
        images["ThermoNeRF"] = open_thermonerf_pred(thermonerf_dir, view_id, target_size)

        tref_gray = to_gray_array(images["T_ref"])
        lo = float(np.percentile(tref_gray, 1.0))
        hi = float(np.percentile(tref_gray, 99.0))
        if hi <= lo:
            lo = float(np.min(tref_gray))
            hi = float(np.max(tref_gray))

        for label, img in images.items():
            gray = to_gray_array(img)
            gray_u8 = normalize_with_range(gray, lo, hi)
            save_gray_png(gray_u8, view_dir / f"{label}.png")

        new_manifest = {
            "scene": scene,
            "view_id": view_id,
            "candidate_rank": old_manifest.get("candidate_rank"),
            "why_selected": old_manifest.get("why_selected", ""),
            "suggested_crop_xywh": old_manifest.get("suggested_crop_xywh"),
            "all_methods_present": "yes",
            "column_order": list(THERMAL_METHOD_ORDER),
            "colormap_name": "shared grayscale display from archived thermal RGB outputs",
            "normalization_rule": (
                "Per-view shared grayscale normalization. Convert each archived output "
                "to luminance, then apply the same [1%, 99%] percentile range measured "
                "from T_ref luminance. No per-method auto-contrast. ThermoNeRF uses the "
                "prediction half from thermal_combined_<view>.jpg."
            ),
            "source_dir_ours": old_manifest["source_dir_ours"],
            "source_dir_thermogaussian_ommg": old_manifest["source_dir_thermogaussian_ommg"],
            "source_dir_thermal3dgs": old_manifest["source_dir_thermal3dgs"],
            "source_dir_thermogaussian_msmg": old_manifest["source_dir_thermogaussian_msmg"],
            "source_dir_thermogaussian_mftg": old_manifest["source_dir_thermogaussian_mftg"],
            "source_dir_thermonerf": old_manifest["source_dir_thermonerf"],
            "source_dir_tref": old_manifest["source_dir_tref"],
            "normalization_range_luminance_1_99": [round(lo, 4), round(hi, 4)],
            "v1_manifest_path": str(old_manifest_path),
            "v1_package_root": str(old_root),
            "display_protocol_version": "v2_shared_gray_r4",
            "notes": (
                "This v2 package is for qualitative comparison only. It reuses the same "
                "held-out test views as v1 and applies a unified grayscale display rule "
                "without retraining or rerendering any method."
            ),
        }
        (view_dir / "manifest.json").write_text(
            json.dumps(new_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        scene_summary_rows.append(
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
    return scene_summary_rows


def write_csv(rows, out_path):
    if not rows:
        return
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_readme(out_dir, old_root):
    text = f"""# TModel Qualitative Candidates v2

This package supersedes the earlier candidate pack at:

- `{old_root}`

It rebuilds the same held-out test-view candidates for:

- `Building` (10 views)
- `PVpanel` (10 views)

## Purpose

This is a **candidate-image package**, not the final paper collage. It is meant for later
manual selection and layout.

## What Changed From v1

- Uses the same scene/view selections as v1.
- Does **not** retrain or rerender any method.
- Re-extracts archived outputs and applies a **shared grayscale display rule** per view.
- Uses the same normalization range for every method within the same view.
- ThermoNeRF now uses the prediction half from `thermal_combined_<view>.jpg`, rather than
  mixing native gray output with other methods' pseudo-color outputs.

## Fixed Column Order

Left to right:

1. `ThermoNeRF`
2. `Thermal3D-GS`
3. `ThermalGaussian-MSMG`
4. `ThermalGaussian-MFTG`
5. `ThermalGaussian-OMMG`
6. `Ours`

Each view folder also includes `T_ref.png`.

## Display Rule

- Held-out test views only (with GT thermal reference).
- All methods within the same scene/view are resized to the same canvas.
- All images are converted to grayscale luminance.
- A shared `[1%, 99%]` luminance percentile range is measured from `T_ref` and then applied
  to every method in that view.
- No per-method auto-enhancement.

## Notes

- Because several archived baseline outputs are stored only as thermal RGB visualizations
  rather than raw scalar temperature maps, this v2 package standardizes **display** rather
  than reconstructing original thermal scalars.
- This makes the qualitative comparison much more visually consistent than v1, but it is
  still a packaging/re-visualization step, not a new experiment.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def zip_dir(src_dir, zip_path):
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir))


def main():
    args = parse_args()
    old_root = args.old_root
    tag = args.tag or f"TModel_Qualitative_Candidates_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out_root / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    all_rows = []
    for scene in SCENES:
        all_rows.extend(build_scene(scene, old_root, out_dir))

    write_csv(all_rows, out_dir / "scene_summary.csv")
    build_readme(out_dir, old_root)
    zip_dir(out_dir, out_dir.with_suffix(".zip"))
    print(out_dir)
    print(out_dir.with_suffix(".zip"))


if __name__ == "__main__":
    main()
