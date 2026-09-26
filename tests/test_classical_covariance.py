import numpy as np

from eigen_jepa.classical_covariance import (
    ClassicalCovarianceSpec,
    default_candidate_grid,
    ewma_covariance,
    factor_covariance,
    forecast_covariance,
    linear_shrinkage,
    oas_covariance,
    sample_covariance,
    select_by_validation,
    topk_eigenvalues,
)


def rng_returns(seed=0, n=80, d=5, scale=0.02):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(d, d))
    z = rng.normal(size=(n, d)) @ a.T
    return scale * z


def test_covariances_are_symmetric_psd():
    x = rng_returns()
    for cov in [sample_covariance(x), ewma_covariance(x, 0.94)]:
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > -1e-12


def test_oas_and_factor_are_psd():
    x = rng_returns(n=100, d=6)
    for cov in [oas_covariance(x), factor_covariance(x, 2)]:
        assert np.allclose(cov, cov.T)
        assert np.linalg.eigvalsh(cov).min() > -1e-12


def test_factor_rank_guard():
    x = rng_returns(n=60, d=4)
    for rank in (0, 4, 5):
        try:
            factor_covariance(x, rank)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid factor rank should fail")


def test_shrinkage_endpoints():
    x = rng_returns()
    cov = sample_covariance(x)
    assert np.allclose(linear_shrinkage(cov, 0.0), cov)
    shrunk = linear_shrinkage(cov, 1.0)
    mu = np.trace(cov) / cov.shape[0]
    assert np.allclose(shrunk, np.eye(cov.shape[0]) * mu)


def test_topk_descending_nonnegative():
    vals = topk_eigenvalues(sample_covariance(rng_returns()), 3)
    assert len(vals) == 3
    assert np.all(vals[:-1] >= vals[1:])
    assert np.all(vals >= 0)


def test_forecast_grid_is_finite():
    x = rng_returns()
    for spec in default_candidate_grid():
        cov = forecast_covariance(x, spec)
        assert cov.shape == (5, 5)
        assert np.isfinite(cov).all()


def test_validation_selection_uses_only_passed_validation_windows():
    contexts = []
    futures = []
    for seed in range(6):
        x = rng_returns(seed, n=60, d=4)
        contexts.append(x[:40])
        futures.append(x[40:])
    grid = (
        ClassicalCovarianceSpec("sample_persistence"),
        ClassicalCovarianceSpec("ewma", decay=0.94),
        ClassicalCovarianceSpec("sample_shrinkage", shrinkage_alpha=0.3),
    )
    best, scores = select_by_validation(contexts, futures, grid, k=3)
    assert best in grid
    assert len(scores) == 3
    assert all(np.isfinite(v) and v >= 0 for v in scores.values())
