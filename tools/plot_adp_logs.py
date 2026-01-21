# tools/plot_adp_logs.py
# Plot ADP logs (adp_cycle.csv / adp_iter.csv) into PNGs for paper figures.
#
# Usage:
#   python tools/plot_adp_logs.py --log_dir <model_path>
#   python tools/plot_adp_logs.py --cycle_csv <path/to/adp_cycle.csv>
#   python tools/plot_adp_logs.py --iter_csv <path/to/adp_iter.csv>
#
# Notes:
# - Uses matplotlib only (no seaborn).
# - Does not set colors; matplotlib defaults are used.

import argparse
import os
import csv
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt


def _read_csv(path: str) -> Dict[str, List[Optional[float]]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols: Dict[str, List[Optional[float]]] = {k: [] for k in (reader.fieldnames or [])}
        for row in reader:
            for k in cols.keys():
                v = row.get(k, "")
                if v is None or v == "":
                    cols[k].append(None)
                else:
                    try:
                        cols[k].append(float(v))
                    except Exception:
                        cols[k].append(None)
    return cols


def _plot_series(x: List[Optional[float]], y: List[Optional[float]], title: str, xlabel: str, ylabel: str, out_path: str):
    # Filter None pairs
    xs, ys = [], []
    for xi, yi in zip(x, y):
        if xi is None or yi is None:
            continue
        xs.append(xi)
        ys.append(yi)
    if len(xs) == 0:
        return
    plt.figure()
    plt.plot(xs, ys)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default=None, help="Model directory containing adp_cycle.csv / adp_iter.csv")
    ap.add_argument("--cycle_csv", type=str, default=None)
    ap.add_argument("--iter_csv", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <log_dir>/adp_plots)")
    args = ap.parse_args()

    cycle_csv = args.cycle_csv
    iter_csv = args.iter_csv

    if args.log_dir:
        if cycle_csv is None:
            p = os.path.join(args.log_dir, "adp_cycle.csv")
            if os.path.exists(p):
                cycle_csv = p
        if iter_csv is None:
            p = os.path.join(args.log_dir, "adp_iter.csv")
            if os.path.exists(p):
                iter_csv = p

    if args.out_dir:
        out_dir = args.out_dir
    else:
        base = args.log_dir if args.log_dir else os.path.dirname(cycle_csv or iter_csv or ".")
        out_dir = os.path.join(base, "adp_plots")

    _ensure_dir(out_dir)

    if cycle_csv and os.path.exists(cycle_csv):
        cols = _read_csv(cycle_csv)
        it = cols.get("iteration")
        if it is None:
            # fallback to wall_time
            it = cols.get("wall_time", [])
        # Common plots
        for key, ylabel in [
            ("gate_ratio", "Gate pass ratio"),
            ("adp_prune_count", "ADP prune count"),
            ("prune_total_count", "Total prune count"),
            ("densify_candidates_gated", "Densify candidates (gated)"),
            ("grad_thr", "Adaptive grad threshold"),
            ("tex_thr_gate", "Texture gate threshold"),
            ("tex_thr_prune", "Texture prune threshold"),
            ("spark_thr", "Sparkle threshold"),
            ("N_after_prune", "Number of Gaussians"),
        ]:
            if key in cols:
                _plot_series(it, cols[key], f"ADP Cycle: {key}", "iteration", ylabel, os.path.join(out_dir, f"cycle_{key}.png"))

    if iter_csv and os.path.exists(iter_csv):
        cols = _read_csv(iter_csv)
        it = cols.get("iteration")
        if it is None:
            it = cols.get("wall_time", [])
        # Heuristic: plot top few "interesting" fields if present
        preferred = [
            "total_points",
            "adp_iter_tex_ema_mean", "adp_iter_spark_ema_mean",
            "adp_iter_visible_count", "adp_iter_tex_mean", "adp_iter_spark_mean",
            "adp_cycle_gate_ratio", "adp_cycle_adp_prune_count",
        ]
        for key in preferred:
            if key in cols:
                _plot_series(it, cols[key], f"ADP Iter: {key}", "iteration", key, os.path.join(out_dir, f"iter_{key}.png"))

        # Also plot any other numeric columns (up to 12) for convenience
        plotted = set(preferred)
        count = 0
        for key in cols.keys():
            if key in ("iteration", "wall_time") or key in plotted:
                continue
            # Skip non-numeric columns (heuristic: all None)
            if all(v is None for v in cols[key]):
                continue
            _plot_series(it, cols[key], f"ADP Iter: {key}", "iteration", key, os.path.join(out_dir, f"iter_{key}.png"))
            count += 1
            if count >= 12:
                break

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
