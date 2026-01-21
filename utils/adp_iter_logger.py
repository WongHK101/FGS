# utils/adp_iter_logger.py
# Flexible (schema-on-first-use) CSV logger for ADP iteration-level scalar statistics.
# It creates the header from the first logged dict, and fills missing values with blanks.

import csv
import os
import time
from typing import Dict, Any, Optional, List


def _sanitize_key(k: str) -> str:
    # Make keys CSV/plot friendly
    return str(k).replace("/", "_").replace(" ", "_")


def _to_number_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
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


class ADPIterCSVLogger:
    """Append ADP iteration-level stats to a CSV file.
    Header is fixed after first write.
    """

    def __init__(self, csv_path: str, extra_fields: Optional[List[str]] = None):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._fieldnames: Optional[List[str]] = None
        self._initialized = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        self._extra_fields = extra_fields or []

        if self._initialized:
            # If file exists, read header to keep consistent
            try:
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        self._fieldnames = header
            except Exception:
                self._initialized = False
                self._fieldnames = None

    def _init_with_keys(self, keys: List[str]):
        base = ["iteration", "wall_time"] + self._extra_fields
        dyn = [_sanitize_key(k) for k in keys]
        # avoid duplicates
        seen = set()
        fieldnames = []
        for k in base + dyn:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
        self._fieldnames = fieldnames
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
        self._initialized = True

    def log_iter(self, iteration: int, stats: Dict[str, Any], extra: Optional[Dict[str, Any]] = None):
        if stats is None or not isinstance(stats, dict):
            return

        extra = extra or {}
        if not self._initialized or self._fieldnames is None:
            self._init_with_keys(list(stats.keys()) + list(extra.keys()))

        row = {k: None for k in self._fieldnames}
        row["iteration"] = int(iteration)
        row["wall_time"] = float(time.time())

        # extra fields (e.g., total_points)
        for k, v in (extra or {}).items():
            sk = _sanitize_key(k)
            if sk in row:
                row[sk] = _to_number_or_none(v)

        # stats fields
        for k, v in stats.items():
            sk = _sanitize_key(k)
            if sk in row:
                row[sk] = _to_number_or_none(v)

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)
