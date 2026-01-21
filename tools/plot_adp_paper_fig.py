# tools/plot_adp_paper_fig.py
# Paper-friendly ADP plotting script:
# - Produces a small set of "ready-to-drop" figures (PNG + PDF)
# - Uses matplotlib defaults (no custom colors/styles)
#
# Usage:
#   python tools/plot_adp_paper_fig.py --log_dir <model_path>
#   python tools/plot_adp_paper_fig.py --cycle_csv <path/to/adp_cycle.csv> --iter_csv <path/to/adp_iter.csv>
#
# Output:
#   <out_dir>/paper_cycle_overview.(png|pdf)
#   <out_dir>/paper_cycle_thresholds.(png|pdf)
#   <out_dir>/paper_iter_overview.(png|pdf)    (if iter_csv exists)

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

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


def _filter_xy(x: List[Optional[float]], y: List[Optional[float]]) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for xi, yi in zip(x, y):
        if xi is None or yi is None:
            continue
        xs.append(float(xi))
        ys.append(float(yi))
    return xs, ys


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _save(fig, out_base: str):
    fig.tight_layout()
    fig.savefig(out_base + ".png", dpi=250)
    fig.savefig(out_base + ".pdf")
    plt.close(fig)


def _get_x(cols: Dict[str, List[Optional[float]]]) -> List[Optional[float]]:
    if "iteration" in cols:
        return cols["iteration"]
    if "wall_time" in cols:
        return cols["wall_time"]
    # fallback: index
    n = len(next(iter(cols.values()))) if cols else 0
    return [float(i) for i in range(n)]


def _plot_cycle_overview(cols: Dict[str, List[Optional[float]]], out_dir: str):
    x = _get_x(cols)

    # Left axis: ratio(s)
    y_gate = cols.get("gate_ratio")
    # Right axis: counts / totals
    y_adp_prune = cols.get("adp_prune_count")
    y_prune_total = cols.get("prune_total_count")
    y_N = cols.get("N_after_prune")

    fig, ax1 = plt.subplots()

    # ratio line(s)
    if y_gate is not None:
        xs, ys = _filter_xy(x, y_gate)
        if xs:
            ax1.plot(xs, ys, label="gate_ratio")
            ax1.set_ylabel("ratio")

    ax1.set_xlabel("iteration")

    # counts on twin axis
    ax2 = ax1.twinx()
    any_right = False

    for key, y in [("adp_prune_count", y_adp_prune), ("prune_total_count", y_prune_total), ("N_after_prune", y_N)]:
        if y is None:
            continue
        xs, ys = _filter_xy(x, y)
        if xs:
            ax2.plot(xs, ys, label=key)
            any_right = True

    if any_right:
        ax2.set_ylabel("count")

    # merge legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if (lines or lines2):
        ax1.legend(lines + lines2, labels + labels2, loc="best")

    ax1.set_title("ADP cycle overview (gate / prune / #gaussians)")
    _save(fig, os.path.join(out_dir, "paper_cycle_overview"))


def _plot_cycle_thresholds(cols: Dict[str, List[Optional[float]]], out_dir: str):
    x = _get_x(cols)
    fig, ax = plt.subplots()

    keys = [
        "grad_thr",
        "tex_thr_gate",
        "tex_thr_prune",
        "spark_thr",
    ]
    any_line = False
    for k in keys:
        if k not in cols:
            continue
        xs, ys = _filter_xy(x, cols[k])
        if xs:
            ax.plot(xs, ys, label=k)
            any_line = True

    ax.set_xlabel("iteration")
    ax.set_ylabel("threshold value")
    ax.set_title("ADP adaptive thresholds (cycle-level)")
    if any_line:
        ax.legend(loc="best")

    _save(fig, os.path.join(out_dir, "paper_cycle_thresholds"))


def _plot_iter_overview(cols: Dict[str, List[Optional[float]]], out_dir: str):
    x = _get_x(cols)
    fig, ax = plt.subplots()

    # Prefer a small set of stable, interpretable curves
    preferred = [
        "total_points",
        "adp_iter_tex_ema_mean",
        "adp_iter_spark_ema_mean",
        "adp_cycle_gate_ratio",
        "adp_cycle_adp_prune_count",
    ]

    any_line = False
    for k in preferred:
        if k not in cols:
            continue
        xs, ys = _filter_xy(x, cols[k])
        if xs:
            ax.plot(xs, ys, label=k)
            any_line = True

    ax.set_xlabel("iteration")
    ax.set_title("ADP iteration overview (iter-level)")
    if any_line:
        ax.legend(loc="best")

    _save(fig, os.path.join(out_dir, "paper_iter_overview"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, default=None, help="Model directory containing adp_cycle.csv / adp_iter.csv")
    ap.add_argument("--cycle_csv", type=str, default=None)
    ap.add_argument("--iter_csv", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="Output directory (default: <log_dir>/adp_plots_paper)")
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

    base = args.log_dir if args.log_dir else os.path.dirname(cycle_csv or iter_csv or ".")
    out_dir = args.out_dir if args.out_dir else os.path.join(base, "adp_plots_paper")
    _ensure_dir(out_dir)

    did_any = False
    if cycle_csv and os.path.exists(cycle_csv):
        cols = _read_csv(cycle_csv)
        _plot_cycle_overview(cols, out_dir)
        _plot_cycle_thresholds(cols, out_dir)
        did_any = True

    if iter_csv and os.path.exists(iter_csv):
        cols = _read_csv(iter_csv)
        _plot_iter_overview(cols, out_dir)
        did_any = True

    if did_any:
        print(f"Saved paper-friendly plots to: {out_dir}")
    else:
        print("No valid CSV found. Provide --log_dir or --cycle_csv/--iter_csv.")


if __name__ == "__main__":
    main()
