#!/usr/bin/env python
# Summarize ablation metrics into an Excel sheet with best values bolded.
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font
except Exception as exc:
    raise SystemExit("openpyxl is required for this script. Install it in your environment.") from exc


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _to_number(v) -> Optional[float]:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        fv = float(v)
        if math.isfinite(fv):
            return fv
        return None
    return None


def _select_method(d: Dict) -> Optional[Dict]:
    if not isinstance(d, dict) or not d:
        return None
    # If values are numeric, treat as direct metrics dict.
    if all(isinstance(v, (int, float, bool)) for v in d.values()):
        return d
    # Otherwise, choose a method key.
    keys = sorted(d.keys())
    if not keys:
        return None
    for k in keys:
        if "ours" in str(k).lower():
            if isinstance(d.get(k), dict):
                return d.get(k)
    # Fallback to first dict-like entry.
    for k in keys:
        if isinstance(d.get(k), dict):
            return d.get(k)
    return None


def _read_json(path: Path) -> Optional[Dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        _warn(f"Failed to parse JSON: {path} ({e})")
        return None


def _parse_results_json(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    obj = _read_json(path)
    if obj is None:
        return out
    metrics = _select_method(obj) if isinstance(obj, dict) else None
    if metrics is None:
        _warn(f"Unexpected JSON structure (no metrics found): {path}")
        return out
    for k, v in metrics.items():
        fv = _to_number(v)
        if fv is not None:
            out[str(k)] = fv
    return out


def _parse_results_plus_json(path: Path) -> Dict[str, float]:
    return _parse_results_json(path)


def _parse_novel_view_json(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    obj = _read_json(path)
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        if "OpacityHoles" in str(k):
            continue
        fv = _to_number(v)
        if fv is not None:
            out[str(k)] = fv
    return out


def _find_latest_iteration_ply(model_dir: Path) -> Optional[Path]:
    pc_dir = model_dir / "point_cloud"
    if not pc_dir.exists():
        return None
    best_iter = -1
    best_ply: Optional[Path] = None
    for child in pc_dir.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"iteration_(\d+)$", child.name)
        if not m:
            continue
        try:
            it = int(m.group(1))
        except Exception:
            continue
        ply = child / "point_cloud.ply"
        if ply.exists() and it > best_iter:
            best_iter = it
            best_ply = ply
    return best_ply


def _find_latest_ckpt(model_dir: Path) -> Optional[Path]:
    best_iter = -1
    best_ckpt: Optional[Path] = None
    for p in model_dir.glob("chkpnt*.pth"):
        m = re.match(r"chkpnt(\d+)\.pth$", p.name)
        if not m:
            continue
        try:
            it = int(m.group(1))
        except Exception:
            continue
        if it > best_iter:
            best_iter = it
            best_ckpt = p
    return best_ckpt


def _ply_vertex_count_header(ply_path: Path) -> Optional[int]:
    try:
        with ply_path.open("r", encoding="utf-8", errors="ignore") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("element vertex"):
                    parts = line.split()
                    if len(parts) >= 3:
                        return int(parts[2])
                if line == "end_header":
                    break
    except Exception:
        return None
    return None


def _parse_results_txt(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        _warn(f"Failed to read results.txt: {path} ({e})")
        return out
    pattern = re.compile(r"(SSIM|PSNR|LPIPS)\s*:\s*([0-9.+-eE]+)")
    for m in pattern.finditer(text):
        k = m.group(1).strip()
        try:
            out[k] = float(m.group(2))
        except Exception:
            continue
    return out


def _collect_model_metrics(model_dir: Path, label: str) -> Tuple[Dict[str, float], Dict[str, Set[str]]]:
    record: Dict[str, float] = {}
    sources: Dict[str, Set[str]] = {}

    if not model_dir.exists():
        _warn(f"Model dir missing: {model_dir} ({label})")
        return record, sources

    results_json = model_dir / "results.json"
    results_txt = model_dir / "results.txt"
    results_plus = model_dir / "results_plus.json"
    novel_json = model_dir / "novel_view_metrics.json"
    novel_json_alt = model_dir / "novel_views" / "novel_view_metrics.json"

    if results_json.exists():
        vals = _parse_results_json(results_json)
    elif results_txt.exists():
        vals = _parse_results_txt(results_txt)
    else:
        vals = {}
        _warn(f"Missing results.json/results.txt in {model_dir} ({label})")

    for k, v in vals.items():
        record[k] = v
        sources.setdefault(k, set()).add("metrics")

    if results_plus.exists():
        vals = _parse_results_plus_json(results_plus)
        for k, v in vals.items():
            record[k] = v
            sources.setdefault(k, set()).add("metrics_plus")
    else:
        _warn(f"Missing results_plus.json in {model_dir} ({label})")

    if novel_json.exists() or novel_json_alt.exists():
        target = novel_json if novel_json.exists() else novel_json_alt
        vals = _parse_novel_view_json(target)
        for k, v in vals.items():
            record[k] = v
            sources.setdefault(k, set()).add("novel")
    else:
        _warn(f"Missing novel_view_metrics.json in {model_dir} ({label})")

    return record, sources


def _load_profile(exp_dir: Path) -> Optional[Dict]:
    prof_path = exp_dir / "pipeline_profile.json"
    if not prof_path.exists():
        _warn(f"Missing pipeline_profile.json: {prof_path}")
        return None
    obj = _read_json(prof_path)
    if not isinstance(obj, dict):
        return None
    return obj


def _parse_profile(obj: Optional[Dict]) -> Dict[str, float]:
    record: Dict[str, float] = {}
    if not isinstance(obj, dict):
        return record

    meta = obj.get("meta", {})
    if isinstance(meta, dict):
        total_s = _to_number(meta.get("total_s"))
        if total_s is not None:
            record["time_total_s"] = total_s

    steps = obj.get("steps", {})
    items: List[Tuple[str, Dict]] = []
    if isinstance(steps, dict):
        for k, v in steps.items():
            if isinstance(v, dict):
                items.append((str(k), v))
    elif isinstance(steps, list):
        for entry in steps:
            if not isinstance(entry, dict):
                continue
            step_name = entry.get("step") or entry.get("name") or entry.get("id")
            if step_name:
                items.append((str(step_name), entry))

    size_map: Dict[str, float] = {}
    count_map: Dict[str, float] = {}

    for step_name, step in items:
        dur = _to_number(step.get("duration_s"))
        if dur is not None:
            record[f"time_{step_name}_s"] = dur
        art = step.get("artifacts", {})
        if isinstance(art, dict):
            sizes = art.get("sizes_bytes", {})
            if isinstance(sizes, dict):
                for k, v in sizes.items():
                    fv = _to_number(v)
                    if fv is not None:
                        size_map[str(k)] = fv
            counts = art.get("counts", {})
            if isinstance(counts, dict):
                for k, v in counts.items():
                    fv = _to_number(v)
                    if fv is not None:
                        count_map[str(k)] = fv

    for k, v in size_map.items():
        record[f"size_{k}_mb"] = v / (1024.0 * 1024.0)
    for k, v in count_map.items():
        record[f"count_{k}"] = v

    return record


def _parse_profile_args(obj: Optional[Dict]) -> Dict[str, object]:
    record: Dict[str, object] = {}
    if not isinstance(obj, dict):
        return record
    meta = obj.get("meta", {})
    args = meta.get("args", {}) if isinstance(meta, dict) else {}
    if isinstance(args, dict):
        keys = [
            "rgb_iter",
            "t_iter",
            "rgb_res",
            "t_res",
            "ss_enable",
            "ss_enable_rgb",
            "ss_enable_t",
            "sgf_disable",
            "clamp_scale_max",
            "thermal_reset_features",
        ]
        for k in keys:
            if k in args:
                record[k] = args.get(k)
        try:
            record["args_json"] = json.dumps(args, ensure_ascii=False)
        except Exception:
            record["args_json"] = str(args)
    return record


def _parse_profile_artifacts(obj: Optional[Dict]) -> Dict[str, object]:
    record: Dict[str, object] = {}
    if not isinstance(obj, dict):
        return record

    steps = obj.get("steps", {})
    items: List[Tuple[str, Dict]] = []
    if isinstance(steps, dict):
        for k, v in steps.items():
            if isinstance(v, dict):
                items.append((str(k), v))
    elif isinstance(steps, list):
        for entry in steps:
            if not isinstance(entry, dict):
                continue
            step_name = entry.get("step") or entry.get("name") or entry.get("id")
            if step_name:
                items.append((str(step_name), entry))

    exists_map: Dict[str, bool] = {}
    for _, step in items:
        art = step.get("artifacts", {})
        if not isinstance(art, dict):
            continue
        exists = art.get("exists", {})
        if not isinstance(exists, dict):
            continue
        for k, v in exists.items():
            exists_map[str(k)] = bool(v) or exists_map.get(str(k), False)

    for k, v in exists_map.items():
        record[f"exists_{k}"] = 1 if v else 0

    return record


def _read_cmd(path: Path) -> Optional[str]:
    if not path.exists():
        _warn(f"Missing command file: {path}")
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text.strip()
    except Exception as e:
        _warn(f"Failed to read command file: {path} ({e})")
        return None


def _prefix_metrics(record: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}{k}": v for k, v in record.items()}


def _prefix_sources(sources: Dict[str, Set[str]], prefix: str) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    for k, v in sources.items():
        out[f"{prefix}{k}"] = set(v)
    return out


def _merge_sources(dst: Dict[str, Set[str]], src: Dict[str, Set[str]]) -> None:
    for k, v in src.items():
        dst.setdefault(k, set()).update(v)


def _collect_columns(rows: List[Dict[str, object]], preferred: Optional[List[str]] = None) -> List[str]:
    keys: Set[str] = set()
    for r in rows:
        keys.update(r.keys())
    keys.discard("Experiment")
    cols: List[str] = ["Experiment"]
    if preferred:
        for k in preferred:
            if k in keys:
                cols.append(k)
                keys.remove(k)
    cols.extend(sorted(keys))
    return cols


def _write_sheet(
    wb: Workbook,
    title: str,
    rows: List[Dict[str, object]],
    key_sources: Optional[Dict[str, Set[str]]] = None,
    *,
    bold_best: bool = True,
    preferred_cols: Optional[List[str]] = None,
    max_width: int = 60,
) -> None:
    if wb.sheetnames and wb.active.title == "Sheet" and title == "Ablation_T":
        ws = wb.active
        ws.title = title
    else:
        ws = wb.create_sheet(title)

    cols = _collect_columns(rows, preferred=preferred_cols)
    ws.freeze_panes = "B2"
    header_font = Font(bold=True)

    for col_idx, key in enumerate(cols, start=1):
        cell = ws.cell(row=1, column=col_idx, value=key)
        cell.font = header_font

    for row_idx, r in enumerate(rows, start=2):
        for col_idx, key in enumerate(cols, start=1):
            ws.cell(row=row_idx, column=col_idx, value=r.get(key, None))

    if bold_best:
        bold_font = Font(bold=True)
        for col_idx, key in enumerate(cols, start=1):
            if key == "Experiment":
                continue
            pref = _column_preference(key, (key_sources or {}).get(key, set()))
            if pref is None:
                continue
            values: List[Tuple[int, float]] = []
            for row_idx in range(2, 2 + len(rows)):
                v = ws.cell(row=row_idx, column=col_idx).value
                fv = _to_number(v)
                if fv is not None:
                    values.append((row_idx, fv))
            if not values:
                continue
            best = max(v for _, v in values) if pref == "high" else min(v for _, v in values)
            for row_idx, v in values:
                if abs(v - best) <= 1e-9:
                    ws.cell(row=row_idx, column=col_idx).font = bold_font

    for col_idx, key in enumerate(cols, start=1):
        max_len = len(str(key))
        for row_idx in range(2, 2 + len(rows)):
            v = ws.cell(row=row_idx, column=col_idx).value
            if v is None:
                continue
            max_len = max(max_len, len(str(v)))
        width = max(8, min(max_width, max_len + 2))
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = width


def _column_preference(key: str, sources: Set[str]) -> Optional[str]:
    k = key.lower()
    if k.startswith("time_") or k.endswith("_s"):
        return "low"
    if k.startswith("size_") or ("_mb" in k) or ("_bytes" in k):
        return "low"
    if k.startswith("t_") or k.startswith("rgb_"):
        k = k.split("_", 1)[1]
    bigger = {
        "psnr",
        "ssim",
        "edgepsnr",
        "edgepsnr(mean)",
        "gradientcorr",
        "gradientcorr(mean)",
        "edgef1",
        "edgef1(mean)",
        "lapvar(render)(mean)",
        "lapvarratio(mean)",
        "lapvarratio",
        "lapvar_render",
        "alignedpsnr@k(mean)",
        "alignededgepsnr@k(mean)",
        "alignedpsnr",
        "alignededgepsnr",
    }
    smaller = {
        "lpips",
        "edgel1",
        "edgel1(mean)",
        "hfabsmeandiff",
        "bgleakratio",
        "bgleakratio(mean)",
    }
    if k in bigger:
        return "high"
    if k in smaller:
        return "low"
    if "novel" in sources:
        if "corr" in k:
            return "high"
        return "low"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation metrics into Excel")
    parser.add_argument("--root", required=True, type=str, help="Root directory containing experiment subdirs")
    parser.add_argument("--out", required=True, type=str, help="Output xlsx path")
    args = parser.parse_args()

    root = Path(args.root)
    out_path = Path(args.out)

    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    exp_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    rows_t: List[Dict[str, object]] = []
    rows_rgb: List[Dict[str, object]] = []
    rows_time: List[Dict[str, object]] = []
    rows_size: List[Dict[str, object]] = []
    rows_lite: List[Dict[str, object]] = []
    rows_cmds: List[Dict[str, object]] = []
    rows_args: List[Dict[str, object]] = []
    rows_artifacts: List[Dict[str, object]] = []

    key_sources_t: Dict[str, Set[str]] = {}
    key_sources_rgb: Dict[str, Set[str]] = {}

    for exp in exp_dirs:
        model_t = exp / "Model_T"
        model_rgb = exp / "Model_RGB"

        t_metrics, t_sources = _collect_model_metrics(model_t, "Model_T")
        rgb_metrics, rgb_sources = _collect_model_metrics(model_rgb, "Model_RGB")

        t_pref = _prefix_metrics(t_metrics, "T_")
        rgb_pref = _prefix_metrics(rgb_metrics, "RGB_")

        _merge_sources(key_sources_t, _prefix_sources(t_sources, "T_"))
        _merge_sources(key_sources_rgb, _prefix_sources(rgb_sources, "RGB_"))

        profile_obj = _load_profile(exp)
        profile = _parse_profile(profile_obj)
        profile_args = _parse_profile_args(profile_obj)
        profile_artifacts = _parse_profile_artifacts(profile_obj)

        rows_t.append({"Experiment": exp.name, **t_pref, **profile})
        rows_rgb.append({"Experiment": exp.name, **rgb_pref})

        time_cols = {k: v for k, v in profile.items() if k.startswith("time_") or k.endswith("_s")}
        size_cols = {k: v for k, v in profile.items() if k.startswith("size_") or k.startswith("count_") or "_mb" in k or "_bytes" in k}
        rows_time.append({"Experiment": exp.name, **time_cols})
        rows_size.append({"Experiment": exp.name, **size_cols})

        # Offline lightweight stats (no profile_collect_* needed)
        rgb_ply = _find_latest_iteration_ply(model_rgb)
        t_ply = _find_latest_iteration_ply(model_t)
        rgb_vertices = _ply_vertex_count_header(rgb_ply) if rgb_ply else None
        t_vertices = _ply_vertex_count_header(t_ply) if t_ply else None
        rgb_ply_bytes = rgb_ply.stat().st_size if rgb_ply and rgb_ply.exists() else None
        t_ply_bytes = t_ply.stat().st_size if t_ply and t_ply.exists() else None
        rgb_ckpt = _find_latest_ckpt(model_rgb)
        t_ckpt = _find_latest_ckpt(model_t)
        rgb_ckpt_bytes = rgb_ckpt.stat().st_size if rgb_ckpt and rgb_ckpt.exists() else None
        t_ckpt_bytes = t_ckpt.stat().st_size if t_ckpt and t_ckpt.exists() else None
        rows_lite.append(
            {
                "Experiment": exp.name,
                "rgb_ply_vertices_offline": rgb_vertices,
                "t_ply_vertices_offline": t_vertices,
                "rgb_ply_bytes_offline": rgb_ply_bytes,
                "t_ply_bytes_offline": t_ply_bytes,
                "rgb_ckpt_bytes_offline": rgb_ckpt_bytes,
                "t_ckpt_bytes_offline": t_ckpt_bytes,
            }
        )

        cmd_train1 = _read_cmd(exp / "cmd_train1.txt")
        cmd_train2 = _read_cmd(exp / "cmd_train2.txt")
        cmd_render = _read_cmd(exp / "cmd_render.txt")
        cmd_metrics = _read_cmd(exp / "cmd_metrics.txt")
        rows_cmds.append(
            {
                "Experiment": exp.name,
                "cmd_train1": cmd_train1,
                "cmd_train2": cmd_train2,
                "cmd_render": cmd_render,
                "cmd_metrics": cmd_metrics,
            }
        )
        rows_args.append({"Experiment": exp.name, **profile_args})
        rows_artifacts.append({"Experiment": exp.name, **profile_artifacts})

    wb = Workbook()
    _write_sheet(wb, "Ablation_T", rows_t, key_sources_t, bold_best=True, max_width=60)
    _write_sheet(wb, "Ablation_RGB", rows_rgb, key_sources_rgb, bold_best=True, max_width=60)
    _write_sheet(wb, "StageTime", rows_time, {}, bold_best=True, max_width=60)
    _write_sheet(wb, "SizeCount", rows_size, {}, bold_best=True, max_width=60)
    _write_sheet(wb, "LiteStats", rows_lite, {}, bold_best=False, max_width=60)
    _write_sheet(wb, "RunArgs", rows_args, {}, bold_best=False, max_width=60)
    _write_sheet(wb, "Artifacts", rows_artifacts, {}, bold_best=False, max_width=60)
    _write_sheet(
        wb,
        "Commands",
        rows_cmds,
        {},
        bold_best=False,
        preferred_cols=["cmd_train1", "cmd_train2", "cmd_render", "cmd_metrics"],
        max_width=60,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))


if __name__ == "__main__":
    main()
