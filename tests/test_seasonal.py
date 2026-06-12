from datetime import datetime, timedelta

import numpy as np
import pytest

from madwatch import SeasonalBaseline


def weekly_series(weeks=4):
    start = datetime(2026, 1, 5)
    timestamps, values = [], []
    for day in range(weeks * 7):
        for hour in range(24):
            ts = start + timedelta(days=day, hours=hour)
            base = 100.0 if ts.weekday() < 5 else 20.0
            values.append(base + (hour % 3) + (day % 4))
            timestamps.append(ts)
    return timestamps, values


def test_weekend_level_is_not_anomalous_with_dow_baseline():
    timestamps, values = weekly_series()
    sb = SeasonalBaseline(granularity="dow").fit(timestamps, values)
    sat = datetime(2026, 2, 7, 12)
    z = sb.score([sat], [21.0])
    assert abs(z[0]) < 3.5


def test_spike_within_bucket_is_anomalous():
    timestamps, values = weekly_series()
    sb = SeasonalBaseline(granularity="dow").fit(timestamps, values)
    sat = datetime(2026, 2, 7, 12)
    z = sb.score([sat], [100.0])
    assert abs(z[0]) > 3.5


def test_dow_hour_bucket_key():
    timestamps, values = weekly_series()
    sb = SeasonalBaseline(granularity="dow_hour").fit(timestamps, values)
    monday_9 = datetime(2026, 2, 2, 9)
    z = sb.score([monday_9], [100.0])
    assert abs(z[0]) < 3.5
    z_spike = sb.score([monday_9], [200.0])
    assert abs(z_spike[0]) > 3.5


def test_unseen_bucket_falls_back_to_global():
    sb = SeasonalBaseline(granularity="hour")
    ts = [datetime(2026, 1, 5, h) for h in range(10)] * 5
    vals = list(range(10)) * 5
    sb.fit(ts, vals)
    z = sb.score([datetime(2026, 1, 5, 23)], [4.5])
    assert np.isfinite(z[0])


def test_score_before_fit_raises():
    sb = SeasonalBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        sb.score([datetime(2026, 1, 1)], [1.0])


def test_invalid_granularity_raises():
    with pytest.raises(ValueError):
        SeasonalBaseline(granularity="month")


def test_length_mismatch_raises():
    sb = SeasonalBaseline()
    with pytest.raises(ValueError, match="length"):
        sb.fit([datetime(2026, 1, 1)], [1.0, 2.0])


def test_nan_raises():
    sb = SeasonalBaseline()
    with pytest.raises(ValueError, match="NaN"):
        sb.fit([datetime(2026, 1, 1)], [float("nan")])
