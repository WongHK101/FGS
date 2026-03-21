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


SCENES = ("Building", "PVpanel", "Orchard", "Road", "TransmissionTower")
TOP_K = 20
METHOD_ORDER = (
    "ThermoNeRF",
    "Thermal3D_GS",
    "ThermalGaussian_MSMG",
    "ThermalGaussian_MFTG",
    "ThermalGaussian_OMMG",
    "Ours",
)
SOURCE_TEMPLATES = {
    "source_dir_ours": r"F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default\{scene}\Model_T\test\ours_60000\renders",
    "source_dir_tref": r"F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default\{scene}\Model_T\test\ours_60000\gt",
    "source_dir_thermogaussian_ommg": r"F:\databackup\xr6\output\SOTA_Comparison\ThermalGaussian_OMMG\{scene}\test\ours_30000\renders_thermal",
    "source_dir_thermal3dgs": r"F:\databackup\xr6\output\SOTA_Comparison\Thermal3D_GS\{scene}\test\ours_30000\renders",
    "source_dir_thermogaussian_msmg": r"E:\3DGS\TGS\output\MSMG\{scene}\thermal_test\ours_30000\renders",
    "source_dir_thermogaussian_mftg": r"E:\3DGS\TGS\output\MFTG\{scene}\thermal_test\ours_30000\renders",
    "source_dir_thermonerf": r"F:\databackup\xr6\output\SOTA_Comparison\ThermoNeRF\{scene}\eval",
}
PER_VIEW_TEMPLATES = {
    "ours": r"F:\databackup\xr6\output\Ch4_2_MainComparison\M01_OursFull_Default\{scene}\Model_T\per_view.json",
    "t3d": r"F:\databackup\xr6\output\SOTA_Comparison\Thermal3D_GS\{scene}\per_view.json",
    "ommg": r"F:\databackup\xr6\output\SOTA_Comparison\ThermalGaussian_OMMG\{scene}\per_view.json",
    "msmg": r"E:\3DGS\TGS\output\MSMG\{scene}\per_view.json",
    "mftg": r"E:\3DGS\TGS\output\MFTG\{scene}\per_view.json",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a v4 T-model qualitative candidate package for all five scenes, "
            "using Ours/T_ref pseudo-color style as the baseline and keeping "
            "ThermoNeRF in native grayscale."
        )
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path(r"F:\databackup\xr6\output"),
        help="Parent output directory for the candidate package.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional custom output tag. Defaults to timestamped v4 name.",
    )
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def open_rgb(path, target_size=None):
    with Image.open(path) as img:
        img = img.convert("RGB")
        if target_size is not None and img.size != target_size:
            img = img.resize(target_size, Image.Resampling.BILINEAR)
        return img


def load_thermonerf_native(eval_dir, view_id, target_size):
    img = open_rgb(eval_dir / f"thermal_{view_id}.jpg", target_size)
    return img


def load_per_view_metric_map(path, metric_key):
    data = read_json(path)
    top_key = next(iter(data.keys()))
    metric_block = data[top_key][metric_key]
    out = {}
    for k, v in metric_block.items():
        stem = Path(k).stem
        out[stem] = float(v)
    return out


def list_common_views(scene):
    roots = {k: Path(v.format(scene=scene)) for k, v in SOURCE_TEMPLATES.items()}
    ours_views = {p.stem for p in roots["source_dir_ours"].glob("*.png")}
    tref_views = {p.stem for p in roots["source_dir_tref"].glob("*.png")}
    ommg_views = {p.stem for p in roots["source_dir_thermogaussian_ommg"].glob("*.png")}
    t3d_views = {p.stem for p in roots["source_dir_thermal3dgs"].glob("*.png")}
    msmg_views = {p.stem for p in roots["source_dir_thermogaussian_msmg"].glob("*.png")}
    mftg_views = {p.stem for p in roots["source_dir_thermogaussian_mftg"].glob("*.png")}
    thermonerf_views = {p.stem.split("_", 1)[1] for p in roots["source_dir_thermonerf"].glob("thermal_*.jpg") if not p.name.startswith("thermal_combined_")}
    common = ours_views & tref_views & ommg_views & t3d_views & msmg_views & mftg_views & thermonerf_views
    return sorted(common)


def gradient_energy(rgb_img):
    g = np.asarray(rgb_img.convert("L"), dtype=np.float32) / 255.0
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    return float(dx.mean() + dy.mean())


def compute_crop_box(ours_img, baseline_imgs):
    ours = np.asarray(ours_img.convert("L"), dtype=np.float32)
    baseline_stack = np.stack([np.asarray(x.convert("L"), dtype=np.float32) for x in baseline_imgs], axis=0)
    base = baseline_stack.mean(axis=0)
    diff = np.abs(ours - base)
    gy, gx = np.gradient(ours)
    score = diff * (0.25 + np.sqrt(gx * gx + gy * gy))
    y, x = np.unravel_index(np.argmax(score), score.shape)
    h, w = ours.shape
    crop_w = max(96, int(round(w * 0.38)))
    crop_h = max(96, int(round(h * 0.38)))
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    x0 = int(np.clip(x - crop_w // 2, 0, max(0, w - crop_w)))
    y0 = int(np.clip(y - crop_h // 2, 0, max(0, h - crop_h)))
    return [x0, y0, crop_w, crop_h]


def choose_views_for_scene(scene):
    common = list_common_views(scene)
    if len(common) < TOP_K:
        raise RuntimeError(f"{scene}: only {len(common)} common held-out views found, need {TOP_K}.")

    ours_psnr = load_per_view_metric_map(Path(PER_VIEW_TEMPLATES["ours"].format(scene=scene)), "PSNR")
    t3d_psnr = load_per_view_metric_map(Path(PER_VIEW_TEMPLATES["t3d"].format(scene=scene)), "PSNR")
    ommg_psnr = load_per_view_metric_map(Path(PER_VIEW_TEMPLATES["ommg"].format(scene=scene)), "thermal_PSNR")
    msmg_psnr = load_per_view_metric_map(Path(PER_VIEW_TEMPLATES["msmg"].format(scene=scene)), "thermal PSNR")
    mftg_psnr = load_per_view_metric_map(Path(PER_VIEW_TEMPLATES["mftg"].format(scene=scene)), "thermal PSNR")

    tref_dir = Path(SOURCE_TEMPLATES["source_dir_tref"].format(scene=scene))
    scored = []
    for view_id in common:
        baselines = np.array([
            t3d_psnr[f"{view_id}"],
            ommg_psnr[f"{view_id}"],
            msmg_psnr[f"{view_id}"],
            mftg_psnr[f"{view_id}"],
        ], dtype=np.float32)
        ours_val = float(ours_psnr[f"{view_id}"])
        with Image.open(tref_dir / f"{view_id}.png") as tref_img:
            edge = gradient_energy(tref_img.convert("RGB"))
        mean_base = float(baselines.mean())
        spread = float(baselines.max() - baselines.min())
        score = (ours_val - mean_base) + 0.35 * spread + 12.0 * edge
        scored.append(
            {
                "scene": scene,
                "view_id": view_id,
                "selection_score": score,
                "ours_psnr": ours_val,
                "t3d_psnr": float(baselines[0]),
                "ommg_psnr": float(baselines[1]),
                "msmg_psnr": float(baselines[2]),
                "mftg_psnr": float(baselines[3]),
                "baseline_mean_psnr": mean_base,
                "baseline_spread_psnr": spread,
                "gt_edge_energy": edge,
            }
        )
    scored.sort(key=lambda x: (-x["selection_score"], x["view_id"]))
    return scored[:TOP_K]


def why_selected(row):
    delta = row["ours_psnr"] - row["baseline_mean_psnr"]
    if delta >= 3.0:
        return "large Ours-vs-baseline thermal advantage; strong structure and hotspot contrast"
    if delta >= 1.5:
        return "clear Ours-vs-baseline thermal advantage; visually structured hotspot region"
    if row["baseline_spread_psnr"] >= 3.0:
        return "strong cross-baseline visual spread; useful for manual qualitative discrimination"
    return "representative held-out view with stable structure and visible cross-method differences"


def build_scene(scene, out_dir):
    selected = choose_views_for_scene(scene)
    scene_rows = []
    src = {k: Path(v.format(scene=scene)) for k, v in SOURCE_TEMPLATES.items()}

    for rank, row in enumerate(selected, start=1):
        view_id = row["view_id"]
        view_dir = out_dir / scene / f"view_{view_id}"
        ensure_dir(view_dir)

        tref = open_rgb(src["source_dir_tref"] / f"{view_id}.png")
        target_size = tref.size
        ours = open_rgb(src["source_dir_ours"] / f"{view_id}.png", target_size)
        ommg = open_rgb(src["source_dir_thermogaussian_ommg"] / f"{view_id}.png", target_size)
        t3d = open_rgb(src["source_dir_thermal3dgs"] / f"{view_id}.png", target_size)
        msmg = open_rgb(src["source_dir_thermogaussian_msmg"] / f"{view_id}.png", target_size)
        mftg = open_rgb(src["source_dir_thermogaussian_mftg"] / f"{view_id}.png", target_size)
        thermonerf = load_thermonerf_native(src["source_dir_thermonerf"], view_id, target_size)

        images = OrderedDict(
            [
                ("T_ref", tref),
                ("ThermoNeRF", thermonerf),
                ("Thermal3D_GS", t3d),
                ("ThermalGaussian_MSMG", msmg),
                ("ThermalGaussian_MFTG", mftg),
                ("ThermalGaussian_OMMG", ommg),
                ("Ours", ours),
            ]
        )
        for name, img in images.items():
            img.save(view_dir / f"{name}.png")

        crop = compute_crop_box(ours, [t3d, msmg, mftg, ommg])
        manifest = {
            "scene": scene,
            "view_id": view_id,
            "candidate_rank": rank,
            "why_selected": why_selected(row),
            "suggested_crop_xywh": crop,
            "all_methods_present": "yes",
            "column_order": list(METHOD_ORDER),
            "colormap_name": "native pseudo-color for Ours/T_ref/Gaussian baselines; native grayscale for ThermoNeRF",
            "normalization_rule": (
                "No additional per-method contrast stretch. Ours, T_ref, Thermal3D-GS, and "
                "ThermalGaussian variants keep their archived pseudo-color outputs. ThermoNeRF "
                "keeps its archived native grayscale thermal output and is resized only to the "
                "shared held-out test-view canvas."
            ),
            **{k: str(v) for k, v in src.items()},
            "selection_metrics": {
                "selection_score": round(row["selection_score"], 4),
                "ours_psnr": round(row["ours_psnr"], 4),
                "baseline_mean_psnr": round(row["baseline_mean_psnr"], 4),
                "baseline_spread_psnr": round(row["baseline_spread_psnr"], 4),
                "gt_edge_energy": round(row["gt_edge_energy"], 4),
                "selection_baselines_for_score": ["Thermal3D_GS", "ThermalGaussian_MSMG", "ThermalGaussian_MFTG", "ThermalGaussian_OMMG"],
                "thermonerf_used_for_score": False,
            },
            "display_protocol_version": "v4_ours_pseudocolor_with_native_thermonerf_gray",
            "notes": (
                "This v4 package uses Ours/T_ref pseudo-color style as the display baseline. "
                "ThermoNeRF is intentionally kept in its archived native grayscale thermal form "
                "rather than being recolored."
            ),
        }
        (view_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        scene_rows.append(
            {
                "scene": scene,
                "view_id": view_id,
                "candidate_rank": rank,
                "why_selected": manifest["why_selected"],
                "suggested_crop_xywh": json.dumps(crop),
                "all_methods_present": "yes",
                "colormap_name": manifest["colormap_name"],
                "normalization_rule": manifest["normalization_rule"],
                "selection_score": f"{row['selection_score']:.4f}",
                "ours_psnr": f"{row['ours_psnr']:.4f}",
                "baseline_mean_psnr": f"{row['baseline_mean_psnr']:.4f}",
                "baseline_spread_psnr": f"{row['baseline_spread_psnr']:.4f}",
                "gt_edge_energy": f"{row['gt_edge_energy']:.4f}",
                **{k: str(v) for k, v in src.items()},
            }
        )
    return scene_rows


def write_csv(rows, out_path):
    if not rows:
        return
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_readme(out_dir):
    text = """# TModel Qualitative Candidates v4

This package prepares held-out test-view thermal qualitative candidates for all five scenes:

- Building
- PVpanel
- Orchard
- Road
- TransmissionTower

Each scene includes 20 candidate views.

## Purpose

This is a candidate-image package for later manual selection and paper figure assembly.
It is not the final collage.

## Display Policy

- `Ours`, `T_ref`, `Thermal3D-GS`, and `ThermalGaussian` variants keep their archived native
  pseudo-color thermal outputs.
- `ThermoNeRF` keeps its archived native grayscale thermal output.
- No per-method extra contrast stretch is applied.
- All methods are resized only as needed to the shared held-out test-view canvas.

## Fixed Column Order

1. ThermoNeRF
2. Thermal3D-GS
3. ThermalGaussian-MSMG
4. ThermalGaussian-MFTG
5. ThermalGaussian-OMMG
6. Ours

## Selection Rule

Candidate ranking is based on:

- Ours thermal PSNR advantage over the mean of Gaussian-style baselines
- cross-baseline PSNR spread
- GT edge energy

ThermoNeRF is included visually in the package but is not used in the numeric ranking score,
because its archived thermal output remains in a different native display space.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def zip_dir(src_dir, zip_path):
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in src_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(src_dir))


def main():
    args = parse_args()
    tag = args.tag or f"TModel_Qualitative_Candidates_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = args.out_root / tag
    if out_dir.exists():
        shutil.rmtree(out_dir)
    ensure_dir(out_dir)

    all_rows = []
    for scene in SCENES:
        all_rows.extend(build_scene(scene, out_dir))

    write_csv(all_rows, out_dir / "scene_summary.csv")
    build_readme(out_dir)
    zip_dir(out_dir, out_dir.with_suffix(".zip"))
    print(out_dir)
    print(out_dir.with_suffix(".zip"))


if __name__ == "__main__":
    main()
