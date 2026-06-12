import numpy as np
import pytest

from madwatch import mad, modified_zscore


def test_mad_known_value():
    x = [1, 1, 2, 2, 4, 6, 9]
    assert mad(x) == 1.0


def test_mad_constant_series_is_zero():
    assert mad([5, 5, 5, 5]) == 0.0


def test_modified_zscore_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = modified_zscore(x)
    assert z.shape == (5,)
    assert z[2] == 0.0
    assert z[4] == pytest.approx(0.6745 * 2.0 / 1.0)
    assert z[0] == pytest.approx(-0.6745 * 2.0 / 1.0)


def test_modified_zscore_whale_gets_high_z():
    x = np.array([10, 11, 10, 12, 11, 10, 11, 12, 10, 11, 100], dtype=float)
    z = modified_zscore(x)
    assert abs(z[-1]) > 3.5
    assert all(abs(v) < 3.5 for v in z[:-1])


def test_modified_zscore_constant_returns_zeros():
    z = modified_zscore([3, 3, 3, 3])
    assert (z == 0).all()


def test_custom_scale():
    x = [1.0, 2.0, 3.0, 4.0, 100.0]
    z1 = modified_zscore(x, scale=0.6745)
    z2 = modified_zscore(x, scale=1.349)
    assert z2[-1] == pytest.approx(2 * z1[-1])


@pytest.mark.parametrize("fn", [mad, modified_zscore])
def test_empty_raises(fn):
    with pytest.raises(ValueError, match="empty"):
        fn([])


@pytest.mark.parametrize("fn", [mad, modified_zscore])
def test_nan_raises(fn):
    with pytest.raises(ValueError, match="NaN"):
        fn([1.0, float("nan"), 2.0])


@pytest.mark.parametrize("fn", [mad, modified_zscore])
def test_multidim_raises(fn):
    with pytest.raises(ValueError, match="1-D"):
        fn([[1.0, 2.0], [3.0, 4.0]])
