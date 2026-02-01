#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总多组实验(out_root)的评测结果，便于直观对比。

读取位置（每个 out_root 下尽量自动匹配）：
  - Model_RGB/results.json 或 results.txt：SSIM/PSNR/LPIPS
  - Model_T/results.json 或 results.txt：SSIM/PSNR/LPIPS
  - Eval/summary_render_ref.csv：融合结果(渲染参考)多 alpha/策略的汇总
  - Eval/summary_gt_ref.csv：融合结果(GT参考)多 alpha/策略的汇总

用法（PowerShell）：
  python summarize_ablation_eval.py `
    "F:\\...\\PVpanel_r8_baseline3dgs" `
    "F:\\...\\PVpanel_r8_adpp_default" `
    "F:\\...\\PVpanel_r8_adpp_noedge" `
    "F:\\...\\PVpanel_r8_adpp_nofog" `
    --output "F:\\...\\ablation_eval_summary.xlsx"
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

_FLOAT_RE = re.compile(r"([-+]?\d*\.\d+|[-+]?\d+)")


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = _FLOAT_RE.search(x)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def pick_metrics_block(obj: Any) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """从 results.json 里挑出一个最可能的 metrics block。"""
    if not isinstance(obj, dict):
        return None, None

    # 常见嵌套
    for k in ("results", "metrics"):
        if k in obj and isinstance(obj[k], dict):
            obj = obj[k]

    if "ours_30000" in obj and isinstance(obj["ours_30000"], dict):
        return "ours_30000", obj["ours_30000"]

    # ours_XXXXX 取迭代数最大的
    ours = [k for k in obj.keys() if isinstance(k, str) and k.startswith("ours_")]
    best_k, best_it = None, -1
    for k in ours:
        m = re.search(r"(\d+)$", k)
        if m:
            it = int(m.group(1))
            if it > best_it:
                best_it, best_k = it, k
    if best_k and isinstance(obj.get(best_k), dict):
        return best_k, obj[best_k]

    # 兜底：任何带 PSNR/SSIM/LPIPS 的 dict
    for k, v in obj.items():
        if isinstance(v, dict) and any(t in v for t in ("PSNR", "SSIM", "LPIPS")):
            return str(k), v

    return None, None


def parse_model_metrics(model_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"method": None, "PSNR": None, "SSIM": None, "LPIPS": None, "src": None}
    if not model_dir.exists():
        return out

    json_cands = [
        model_dir / "results.json",
        model_dir / "metrics.json",
        model_dir / "results" / "results.json",
    ]
    txt_cands = [
        model_dir / "results.txt",
        model_dir / "metrics.txt",
    ]

    p_json = next((p for p in json_cands if p.exists()), None)
    if p_json:
        try:
            obj = json.loads(p_json.read_text(encoding="utf-8"))
            method, blk = pick_metrics_block(obj)
            if blk:
                out["method"] = method
                out["PSNR"] = safe_float(blk.get("PSNR"))
                out["SSIM"] = safe_float(blk.get("SSIM"))
                out["LPIPS"] = safe_float(blk.get("LPIPS"))
                out["src"] = str(p_json)
                return out
        except Exception:
            pass

    p_txt = next((p for p in txt_cands if p.exists()), None)
    if p_txt:
        try:
            text = p_txt.read_text(encoding="utf-8", errors="ignore")
            def find(label: str) -> Optional[float]:
                m = re.search(rf"{label}\s*[:=]\s*([-+]?\d*\.\d+|[-+]?\d+)", text)
                return float(m.group(1)) if m else None
            out["PSNR"] = find("PSNR")
            out["SSIM"] = find("SSIM")
            out["LPIPS"] = find("LPIPS")
            out["src"] = str(p_txt)
            return out
        except Exception:
            pass

    # 进一步兜底：目录下任意 results*.json/metrics*.json
    for p in sorted(model_dir.glob("results*.json")) + sorted(model_dir.glob("metrics*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            method, blk = pick_metrics_block(obj)
            if blk:
                out["method"] = method
                out["PSNR"] = safe_float(blk.get("PSNR"))
                out["SSIM"] = safe_float(blk.get("SSIM"))
                out["LPIPS"] = safe_float(blk.get("LPIPS"))
                out["src"] = str(p)
                return out
        except Exception:
            continue

    return out


def read_csv_dicts(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def best_row(rows: List[Dict[str, Any]], psnr_keys: List[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not rows:
        return None, None
    cols = set(rows[0].keys())
    key = next((k for k in psnr_keys if k in cols), None)
    if key is None:
        cands = [c for c in cols if ("PSNR" in c and c.endswith("_mean"))]
        key = cands[0] if cands else None
    if key is None:
        return None, None

    b, bv = None, float("-inf")
    for r in rows:
        v = safe_float(r.get(key))
        if v is None:
            continue
        if v > bv:
            b, bv = r, v
    return b, key


def maybe(best: Optional[Dict[str, Any]], *keys: str) -> Optional[float]:
    if not best:
        return None
    for k in keys:
        v = safe_float(best.get(k))
        if v is not None:
            return v
    return None


def parse_eval(out_root: Path) -> Dict[str, Any]:
    eval_dir = out_root / "Eval"
    out: Dict[str, Any] = {"rr": None, "gr": None}
    if not eval_dir.exists():
        return out

    rr = read_csv_dicts(eval_dir / "summary_render_ref.csv")
    gr = read_csv_dicts(eval_dir / "summary_gt_ref.csv")

    rr_best, _ = best_row(rr, ["F_vs_T_PSNR_mean", "F_vs_R_PSNR_mean", "T_vs_R_PSNR_mean", "PSNR_mean"])
    gr_best, _ = best_row(gr, ["PSNR_Y_mean", "PSNR_mean"])

    out["rr"] = rr_best
    out["gr"] = gr_best
    return out


def summarize(out_root: Path) -> Dict[str, Any]:
    rgb = parse_model_metrics(out_root / "Model_RGB")
    th = parse_model_metrics(out_root / "Model_T")
    ev = parse_eval(out_root)
    rr = ev.get("rr")
    gr = ev.get("gr")

    return {
        "实验名": out_root.name,
        "out_root": str(out_root),

        "RGB_PSNR": rgb["PSNR"],
        "RGB_SSIM": rgb["SSIM"],
        "RGB_LPIPS": rgb["LPIPS"],

        "T_PSNR": th["PSNR"],
        "T_SSIM": th["SSIM"],
        "T_LPIPS": th["LPIPS"],

        "融合(RenderRef)_策略": (rr or {}).get("strategy") if rr else None,
        "融合(RenderRef)_alpha": safe_float((rr or {}).get("alpha")) if rr else None,
        "融合(RenderRef)_PSNR": maybe(rr, "F_vs_T_PSNR_mean", "F_vs_R_PSNR_mean", "T_vs_R_PSNR_mean", "PSNR_mean"),
        "融合(RenderRef)_SSIM": maybe(rr, "F_vs_T_SSIM_mean", "F_vs_R_SSIM_mean", "T_vs_R_SSIM_mean", "SSIM_mean"),
        "融合(RenderRef)_LPIPS": maybe(rr, "F_vs_T_LPIPS_mean", "F_vs_R_LPIPS_mean", "T_vs_R_LPIPS_mean", "LPIPS_mean"),

        "融合(GTRef)_策略": (gr or {}).get("strategy") if gr else None,
        "融合(GTRef)_alpha": safe_float((gr or {}).get("alpha")) if gr else None,
        "融合(GTRef)_PSNR": maybe(gr, "PSNR_Y_mean", "PSNR_mean"),
        "融合(GTRef)_SSIM": maybe(gr, "SSIM_Y_mean", "SSIM_mean"),
        "融合(GTRef)_LPIPS": maybe(gr, "LPIPS_mean"),

        "RGB_metrics_src": rgb.get("src"),
        "T_metrics_src": th.get("src"),
    }


def print_table(rows: List[Dict[str, Any]]) -> None:
    cols = [
        "实验名",
        "RGB_PSNR", "RGB_SSIM", "RGB_LPIPS",
        "T_PSNR", "T_SSIM", "T_LPIPS",
        "融合(RenderRef)_策略", "融合(RenderRef)_alpha", "融合(RenderRef)_PSNR",
        "融合(GTRef)_策略", "融合(GTRef)_alpha", "融合(GTRef)_PSNR",
    ]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.4f}"
        return "" if v is None else str(v)

    widths = {c: max(len(c), *(len(fmt(r.get(c))) for r in rows)) for c in cols}
    sep = " | "
    head = sep.join(c.ljust(widths[c]) for c in cols)
    line = "-+-".join("-" * widths[c] for c in cols)
    print(head)
    print(line)
    for r in rows:
        print(sep.join(fmt(r.get(c)).ljust(widths[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_roots", nargs="+", help="四组(或多组)实验 out_root 路径")
    ap.add_argument("--output", default="ablation_eval_summary.xlsx", help="输出 xlsx 路径（同时写同名 csv）")
    args = ap.parse_args()

    outs = [Path(p) for p in args.out_roots]
    rows = [summarize(p) for p in outs]

    print_table(rows)

    out_xlsx = Path(args.output)
    out_csv = out_xlsx.with_suffix(".csv")

    if pd is not None:
        df = pd.DataFrame(rows)
        df.to_excel(out_xlsx, index=False)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 已写入：{out_xlsx}")
        print(f"[OK] 已写入：{out_csv}")
    else:
        with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[OK] pandas 不可用，仅写入：{out_csv}")


if __name__ == "__main__":
    main()
