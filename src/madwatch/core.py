import numpy as np

SCALE = 0.6745


def _validated(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        raise ValueError("input is empty")
    if np.isnan(arr).any():
        raise ValueError("input contains NaN")
    return arr


def mad(x) -> float:
    """Median Absolute Deviation of a 1-D array-like."""
    arr = _validated(x)
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))


def modified_zscore(x, scale: float = SCALE) -> np.ndarray:
    """Modified Z-Score of each element against the array's median and MAD.

    Returns zeros when MAD is 0 (constant input).
    """
    arr = _validated(x)
    med = np.median(arr)
    m = np.median(np.abs(arr - med))
    if m == 0:
        return np.zeros_like(arr)
    return scale * (arr - med) / m
