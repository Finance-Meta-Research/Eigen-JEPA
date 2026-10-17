"""Prospective OAS centering regressions; no data or scientific runs."""

import numpy as np
import pytest

from eigen_jepa.classical_covariance_v2 import oas_covariance


@pytest.mark.parametrize("offset", [1e100, 1e150, 1e300, -1e300])
def test_constant_asset_offset_does_not_change_oas_intensity(offset):
    observations = np.array([[0, 1], [0, 3], [0, 5], [0, 7]], dtype=float)
    shifted = observations.copy()
    shifted[:, 0] = offset
    # n=4, p=2, empirical diag(0,5), OAS shrinkage 0.8.
    np.testing.assert_allclose(oas_covariance(shifted), np.diag([2.0, 3.0]), rtol=1e-14)
    np.testing.assert_allclose(oas_covariance(shifted), oas_covariance(observations), rtol=1e-14)


def test_large_constant_window_is_exactly_zero():
    np.testing.assert_array_equal(oas_covariance(np.full((4, 2), 1e308)), np.zeros((2, 2)))


def test_genuinely_unrepresentable_dispersion_still_fails():
    with pytest.raises(ArithmeticError, match="representable"):
        oas_covariance(np.array([[-1e308, 0.0], [1e308, 0.0]]))
