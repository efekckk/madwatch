import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from .core import SCALE


@dataclass(frozen=True)
class Score:
    value: float
    z: float
    is_anomaly: bool


class RollingDetector:
    """Streaming anomaly detector: scores each value against the trailing window."""

    def __init__(self, window: int = 40, threshold: float = 3.5, min_samples: int = 10):
        if window < 2:
            raise ValueError("window must be >= 2")
        if not 2 <= min_samples <= window:
            raise ValueError("min_samples must be between 2 and window")
        self.window = window
        self.threshold = threshold
        self.min_samples = min_samples
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, value) -> Score:
        v = float(value)
        if math.isnan(v):
            raise ValueError("value is NaN")
        if len(self._buf) < self.min_samples:
            self._buf.append(v)
            return Score(value=v, z=0.0, is_anomaly=False)
        arr = np.asarray(self._buf)
        med = float(np.median(arr))
        m = float(np.median(np.abs(arr - med)))
        z = 0.0 if m == 0 else SCALE * (v - med) / m
        self._buf.append(v)
        return Score(value=v, z=z, is_anomaly=abs(z) > self.threshold)

    def score(self, values) -> list[Score]:
        return [self.update(v) for v in values]
