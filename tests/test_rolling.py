import pytest

from madwatch import RollingDetector, Score


def warm_detector(values=None, **kw):
    det = RollingDetector(window=10, threshold=3.5, min_samples=5, **kw)
    for v in values or [10, 11, 10, 12, 11]:
        det.update(v)
    return det


def test_cold_start_never_flags():
    det = RollingDetector(window=10, threshold=3.5, min_samples=5)
    for v in [1, 1000, -50, 999]:
        s = det.update(v)
        assert s == Score(value=float(v), z=0.0, is_anomaly=False)


def test_spike_flagged_after_warmup():
    det = warm_detector()
    s = det.update(100.0)
    assert s.is_anomaly
    assert abs(s.z) > 3.5


def test_normal_value_not_flagged():
    det = warm_detector()
    s = det.update(11.0)
    assert not s.is_anomaly


def test_whale_does_not_mask_next_anomaly():
    det = warm_detector()
    det.update(1000.0)
    s = det.update(100.0)
    assert s.is_anomaly


def test_baseline_excludes_incoming_value():
    det = warm_detector()
    s = det.update(100.0)
    assert s.z == pytest.approx(0.6745 * (100.0 - 11.0) / 1.0)


def test_constant_window_zero_z():
    det = RollingDetector(window=10, threshold=3.5, min_samples=5)
    for _ in range(6):
        s = det.update(7.0)
    assert s.z == 0.0
    assert not s.is_anomaly


def test_batch_score_matches_update():
    values = [10, 11, 10, 12, 11, 10, 11, 100, 11, 10]
    a = RollingDetector(window=10, min_samples=5).score(values)
    b = []
    det = RollingDetector(window=10, min_samples=5)
    for v in values:
        b.append(det.update(v))
    assert a == b
    assert a[7].is_anomaly


def test_nan_raises():
    det = RollingDetector()
    with pytest.raises(ValueError, match="NaN"):
        det.update(float("nan"))


@pytest.mark.parametrize("kw", [{"window": 1}, {"min_samples": 1}, {"min_samples": 50}])
def test_invalid_params_raise(kw):
    with pytest.raises(ValueError):
        RollingDetector(**{"window": 40, "min_samples": 10, **kw})
