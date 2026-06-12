from datetime import datetime

import numpy as np

from .core import SCALE

GRANULARITIES = ("dow_hour", "dow", "hour")


class SeasonalBaseline:
    """Per-bucket median/MAD baselines keyed by day-of-week and/or hour."""

    def __init__(self, granularity: str = "dow_hour"):
        if granularity not in GRANULARITIES:
            raise ValueError(f"granularity must be one of {GRANULARITIES}")
        self.granularity = granularity
        self._stats: dict[tuple, tuple[float, float]] = {}
        self._global: tuple[float, float] | None = None

    def _key(self, ts: datetime) -> tuple:
        if self.granularity == "dow_hour":
            return (ts.weekday(), ts.hour)
        if self.granularity == "dow":
            return (ts.weekday(),)
        return (ts.hour,)

    @staticmethod
    def _med_mad(values: np.ndarray) -> tuple[float, float]:
        med = float(np.median(values))
        return med, float(np.median(np.abs(values - med)))

    def fit(self, timestamps, values) -> "SeasonalBaseline":
        ts = list(timestamps)
        vals = np.asarray(list(values), dtype=float)
        if vals.size == 0:
            raise ValueError("input is empty")
        if len(ts) != vals.size:
            raise ValueError("timestamps and values length mismatch")
        if np.isnan(vals).any():
            raise ValueError("input contains NaN")
        self._global = self._med_mad(vals)
        buckets: dict[tuple, list[float]] = {}
        for t, v in zip(ts, vals):
            buckets.setdefault(self._key(t), []).append(float(v))
        self._stats = {k: self._med_mad(np.asarray(v)) for k, v in buckets.items()}
        return self

    def score(self, timestamps, values) -> np.ndarray:
        if self._global is None:
            raise RuntimeError("fit() must be called before score()")
        ts = list(timestamps)
        vals = np.asarray(list(values), dtype=float)
        if len(ts) != vals.size:
            raise ValueError("timestamps and values length mismatch")
        if vals.size and np.isnan(vals).any():
            raise ValueError("input contains NaN")
        out = np.empty(vals.size)
        for i, (t, v) in enumerate(zip(ts, vals)):
            med, m = self._stats.get(self._key(t), self._global)
            out[i] = 0.0 if m == 0 else SCALE * (v - med) / m
        return out
