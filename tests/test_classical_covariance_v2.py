import numpy as np
import pytest

from eigen_jepa import classical_covariance as v1
from eigen_jepa import classical_covariance_v2 as v2


def returns():
    return np.random.default_rng(213).normal(size=(30, 3)) * [1, 3, 10]


@pytest.mark.parametrize("scale", [1e-150, 1e-12, 1.0, 1e12, 1e150])
def test_oas_is_scale_equivariant_without_absolute_floor(scale):
    expected = v2.oas_covariance(returns())
    found = v2.oas_covariance(returns() * scale) / scale / scale
    np.testing.assert_allclose(found, expected, rtol=2e-13, atol=2e-13)


def test_legacy_scale_defect_is_reproduced_and_preserved():
    x = returns()
    assert not np.allclose(v1.oas_covariance(x), v1.oas_covariance(x * 1e-12) / 1e-24)
    np.testing.assert_allclose(v2.oas_covariance(x), v1.oas_covariance(x), rtol=1e-13)


def test_oas_hand_calculated_rank_one_window():
    # Empirical diag(5,0), mu=2.5, alpha=6.25, shrinkage=0.8.
    x = np.array([[-3., 0], [-1., 0], [1., 0], [3., 0]])
    np.testing.assert_allclose(v2.oas_covariance(x), np.diag([3., 2.]))
    np.testing.assert_array_equal(v2.oas_covariance(np.ones((8, 3))), np.zeros((3, 3)))


@pytest.mark.parametrize("scale", [1e-250, 1e-12, 1.0, 1e12, 1e250])
def test_both_nmse_metrics_keep_same_units(scale):
    pred, true = np.diag([2., 4.]) * scale, np.diag([1., 3.]) * scale
    assert v2.eigen_nmse(pred, true, 2) == pytest.approx(.2)
    assert v2.covariance_nmse(pred, true) == pytest.approx(.2)


def test_zero_energy_is_explicit_and_not_a_small_denominator():
    zero = np.zeros((2, 2))
    assert v2.covariance_nmse(zero, zero) == 0
    assert v2.eigen_nmse(zero, zero, 2) == 0
    assert np.isinf(v2.covariance_nmse(np.eye(2), zero))
    assert np.isinf(v2.eigen_nmse(np.eye(2), zero, 2))


def test_complete_two_asset_candidate_ladder_runs():
    x = returns()[:, :2]
    specs = v2.default_candidate_grid(2)
    assert all(spec.factor_rank is None or spec.factor_rank == 1 for spec in specs)
    best, scores = v2.select_by_validation([x[:20]], [x[20:]], specs, k=2)
    assert best in specs and len(scores) == len(specs)
    assert all(np.isfinite(value) for value in scores.values())


@pytest.mark.parametrize("cov", [np.ones(2), np.ones((2, 3)), np.diag([1., -1.]),
    np.array([[1., .2], [0., 1.]]), np.diag([1., np.nan])])
def test_invalid_covariance_is_not_silently_clipped(cov):
    with pytest.raises(ValueError):
        v2.topk_eigenvalues(cov, 1)


def test_empty_duplicate_and_zero_energy_candidates_are_rejected():
    x = returns()
    spec = v2.ClassicalCovarianceSpec("oas")
    for specs in ([], [spec, spec]):
        with pytest.raises(ValueError):
            v2.select_by_validation([x], [x], specs, k=2)
    with pytest.raises(ValueError, match="non-finite validation"):
        v2.select_by_validation([x], [np.zeros_like(x)], [spec], k=2)


def test_complex_context_cannot_be_silently_cast_by_legacy_comparator():
    with pytest.raises(ValueError, match="real"):
        v2.forecast_covariance(returns() + 1j, v2.ClassicalCovarianceSpec("sample_persistence"))


def test_complex_future_cannot_be_silently_cast_in_validation():
    with pytest.raises(ValueError, match="real"):
        v2.select_by_validation([returns()], [returns() + 1j], [v2.ClassicalCovarianceSpec("oas")], k=2)
