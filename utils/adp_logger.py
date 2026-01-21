# utils/adp_logger.py
# Lightweight CSV logger for ADP (Artifact-aware Densification & Pruning) cycle statistics.
# Designed to be robust (fixed columns), append-only, and easy to plot for paper figures.

import csv
import os
import time
from typing import Dict, Any, Optional


class ADPCSVLogger:
    """Append ADP cycle stats to a CSV file with a fixed schema.

    This avoids "dynamic header" issues during long runs and stays paper-friendly.
    """
    DEFAULT_FIELDS = [
        "iteration",
        "wall_time",
        "N_before",
        "N_after_densify",
        "N_after_prune",
        "gate_total",
        "gate_pass",
        "gate_ratio",
        "densify_candidates",
        "densify_candidates_gated",
        "tex_thr_gate",
        "tex_thr_prune",
        "spark_thr",
        "grad_thr",
        "prune_official_count",
        "adp_prune_count",
        "prune_total_count",
    ]

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._initialized = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    def _init_if_needed(self):
        if self._initialized:
            return
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.DEFAULT_FIELDS)
            writer.writeheader()
        self._initialized = True

    @staticmethod
    def _to_number_or_none(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, bool):
            return float(v)
        if isinstance(v, (int, float)):
            return float(v)
        # allow numpy scalars if present
        try:
            import numpy as np  # type: ignore
            if isinstance(v, np.generic):
                return float(v)
        except Exception:
            pass
        try:
            return float(v)
        except Exception:
            return None

    def log_cycle(self, iteration: int, stats: Dict[str, Any]):
        """Append one row of cycle stats."""
        if stats is None or not isinstance(stats, dict):
            return
        self._init_if_needed()

        row = {k: None for k in self.DEFAULT_FIELDS}
        row["iteration"] = int(iteration)
        row["wall_time"] = float(time.time())

        # Map keys from gaussian_model stats dict to our fixed schema.
        # We keep names identical where possible.
        for k in self.DEFAULT_FIELDS:
            if k in ("iteration", "wall_time"):
                continue
            if k in stats:
                row[k] = self._to_number_or_none(stats.get(k))

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.DEFAULT_FIELDS)
            writer.writerow(row)
