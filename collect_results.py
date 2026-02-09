# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_txt_metrics(path: Path) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {"PSNR": None, "SSIM": None, "LPIPS": None}
    if not path.exists():
        return out
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return out
    pat = re.compile(r"(PSNR|SSIM|LPIPS)[:\\s]+([0-9.]+)")
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        k = m.group(1)
        try:
            out[k] = float(m.group(2))
        except Exception:
            continue
    return out


def _pick_method(results: Dict, preferred: str) -> Tuple[Optional[str], Dict]:
    if preferred in results:
        return preferred, results[preferred]
    if results:
        first = next(iter(results.keys()))
        return first, results[first]
    return None, {}


def _read_model_metrics(model_dir: Path, method: str) -> Dict[str, Optional[float]]:
    json_path = model_dir / "results.json"
    txt_path = model_dir / "results.txt"
    obj = _read_json(json_path)
    if isinstance(obj, dict) and obj:
        _, metrics = _pick_method(obj, method)
        return {
            "PSNR": float(metrics.get("PSNR")) if metrics.get("PSNR") is not None else None,
            "SSIM": float(metrics.get("SSIM")) if metrics.get("SSIM") is not None else None,
            "LPIPS": float(metrics.get("LPIPS")) if metrics.get("LPIPS") is not None else None,
        }
    return _read_txt_metrics(txt_path)


def _format_float(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect PSNR/SSIM/LPIPS from multiple out_root folders.")
    ap.add_argument("--roots", nargs="+", required=True, help="List of out_root directories (one per candidate).")
    ap.add_argument("--tags", nargs="*", default=None, help="Optional tags for each root (default: basename of root).")
    ap.add_argument("--method", default="ours", help="Method key inside results.json (default: ours).")
    ap.add_argument("--out_csv", default="collect_results.csv", help="Output CSV path.")
    ap.add_argument("--out_json", default="collect_results.json", help="Output JSON path.")
    args = ap.parse_args()

    roots = [Path(p) for p in args.roots]
    tags: List[str] = []
    if args.tags:
        if len(args.tags) != len(roots):
            raise ValueError("--tags length must match --roots length")
        tags = list(args.tags)
    else:
        tags = [p.name for p in roots]

    rows = []
    for tag, root in zip(tags, roots):
        model_rgb = root / "Model_RGB"
        model_t = root / "Model_T"
        rgb = _read_model_metrics(model_rgb, method=str(args.method))
        th = _read_model_metrics(model_t, method=str(args.method))
        row = {
            "tag": tag,
            "root": str(root),
            "rgb_psnr": rgb.get("PSNR"),
            "rgb_ssim": rgb.get("SSIM"),
            "rgb_lpips": rgb.get("LPIPS"),
            "t_psnr": th.get("PSNR"),
            "t_ssim": th.get("SSIM"),
            "t_lpips": th.get("LPIPS"),
        }
        rows.append(row)

    # CSV
    csv_lines = ["tag,root,rgb_psnr,rgb_ssim,rgb_lpips,t_psnr,t_ssim,t_lpips"]
    for r in rows:
        csv_lines.append(
            ",".join([
                str(r.get("tag", "")),
                str(r.get("root", "")),
                _format_float(r.get("rgb_psnr")),
                _format_float(r.get("rgb_ssim")),
                _format_float(r.get("rgb_lpips")),
                _format_float(r.get("t_psnr")),
                _format_float(r.get("t_ssim")),
                _format_float(r.get("t_lpips")),
            ])
        )
    Path(args.out_csv).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote: {args.out_csv} and {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
