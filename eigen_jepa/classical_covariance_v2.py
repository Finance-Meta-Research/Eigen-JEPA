"""Prospective scale-consistent covariance controls; no frozen runner opts in.

V1's absolute denominator floors change OAS and NMSE when return units change.
This module normalizes before quadratic arithmetic and uses explicit zero-
energy semantics. The original module, source freezes and outcomes are retained.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from . import classical_covariance as legacy
from .classical_covariance import ClassicalCovarianceSpec


def _covariance(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("covariance must be real")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square with at least two assets")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must be finite")
    scale = float(np.max(np.abs(matrix)))
    normalized = matrix / scale if scale else matrix
    if not np.allclose(normalized, normalized.T, rtol=0, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    if np.linalg.eigvalsh(normalized).min() < -1e-12:
        raise ValueError("covariance must be positive semidefinite")
    return matrix


def oas_covariance(returns: np.ndarray) -> np.ndarray:
    """OAS with dimensionless shrinkage; input scaling gives squared scaling.

    The regularization intensity is evaluated on a normalized return window,
    avoiding the quartic overflow/underflow and absolute 1e-30 cutoff in v1.
    Zero windows return zero covariance. No scikit-learn dependency is needed.
    """
    if np.iscomplexobj(returns):
        raise ValueError("returns must be real")
    x = legacy._as_2d_returns(returns)
    scale = float(np.max(np.abs(x)))
    if scale == 0:
        return np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    x = x / scale
    centered = x - x.mean(axis=0, keepdims=True)
    n, p = centered.shape
    empirical = centered.T @ centered / n
    mu = float(np.trace(empirical) / p)
    alpha = float(np.mean(empirical * empirical))
    denominator = (n + 1.0) * (alpha - mu * mu / p)
    shrinkage = 1.0 if denominator <= 0 else float(np.clip((alpha + mu * mu) / denominator, 0, 1))
    shrunk = (1 - shrinkage) * empirical + shrinkage * mu * np.eye(p)
    # Multiplying successively avoids overflowing scale**2 when the covariance
    # itself is representable (e.g. a large offset with small relative spread).
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result = (shrunk * scale) * scale
    if not np.isfinite(result).all():
        raise ArithmeticError("OAS covariance exceeds the representable float64 range")
    if np.any(shrunk != 0) and not np.any(result != 0):
        raise ArithmeticError("OAS covariance underflows the representable float64 range")
    return result


def _relative_squared_error(pred: np.ndarray, true: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(pred))), float(np.max(np.abs(true))))
    if scale == 0:
        return 0.0
    p, t = pred / scale, true / scale
    numerator = float(np.mean((p - t) ** 2))
    denominator = float(np.mean(t * t))
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return numerator / denominator


def covariance_nmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Unit-invariant relative error; nonzero prediction over zero truth is inf."""
    pred, true = _covariance(pred), _covariance(true)
    if pred.shape != true.shape:
        raise ValueError("pred and true must have identical covariance shape")
    return _relative_squared_error(pred, true)


def topk_eigenvalues(cov: np.ndarray, k: int) -> np.ndarray:
    cov = _covariance(cov)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= len(cov):
        raise ValueError("k must be an integer between 1 and covariance dimension")
    scale = float(np.max(np.abs(cov)))
    if scale == 0:
        return np.zeros(k)
    normalized = cov / scale
    eigenvalues = np.linalg.eigvalsh(0.5 * (normalized + normalized.T))
    result = np.clip(eigenvalues[::-1][:k], 0, None) * scale
    if not np.isfinite(result).all():
        raise ArithmeticError("covariance eigenvalues are not representable")
    return result


def eigen_nmse(pred_cov: np.ndarray, true_cov: np.ndarray, k: int) -> float:
    pred_cov, true_cov = _covariance(pred_cov), _covariance(true_cov)
    if pred_cov.shape != true_cov.shape:
        raise ValueError("pred and true must have identical covariance shape")
    return _relative_squared_error(topk_eigenvalues(pred_cov, k), topk_eigenvalues(true_cov, k))


def forecast_covariance(context_returns: np.ndarray, spec: ClassicalCovarianceSpec) -> np.ndarray:
    """Use v2 OAS explicitly; all other named estimators keep their v1 formula."""
    if np.iscomplexobj(context_returns):
        raise ValueError("context returns must be real")
    if spec.name == "oas":
        result = oas_covariance(context_returns)
    else:
        result = legacy.forecast_covariance(context_returns, spec)
    return _covariance(result)


def default_candidate_grid(num_assets: int) -> tuple[ClassicalCovarianceSpec, ...]:
    """Exclude impossible factor ranks before fitting small asset universes."""
    if isinstance(num_assets, bool) or not isinstance(num_assets, (int, np.integer)) or num_assets < 2:
        raise ValueError("num_assets must be an integer >= 2")
    return tuple(spec for spec in legacy.default_candidate_grid()
                 if spec.factor_rank is None or spec.factor_rank < num_assets)


def select_by_validation(
    contexts: Sequence[np.ndarray], future_returns: Sequence[np.ndarray],
    specs: Iterable[ClassicalCovarianceSpec], *, k: int,
) -> tuple[ClassicalCovarianceSpec, dict[str, float]]:
    """Evaluate only supplied validation pairs; retain every candidate score.

    Empty, repeated and non-finite candidates fail explicitly. A malformed
    comparator cannot silently disappear from a supposedly matched matrix.
    """
    contexts, future_returns, specs = list(contexts), list(future_returns), list(specs)
    if not contexts or len(contexts) != len(future_returns):
        raise ValueError("contexts/future_returns must be nonempty and length matched")
    if not specs or len({repr(spec) for spec in specs}) != len(specs):
        raise ValueError("candidate specs must be nonempty and unique")
    scores = {}
    for spec in specs:
        values = []
        for context, future in zip(contexts, future_returns):
            if np.iscomplexobj(future):
                raise ValueError("future returns must be real")
            pred = forecast_covariance(context, spec)
            true = legacy.sample_covariance(future)
            values.append(eigen_nmse(pred, true, k))
        score = float(np.mean(values))
        if not np.isfinite(score):
            raise ValueError(f"non-finite validation score for {spec!r}; inspect zero-energy targets")
        scores[repr(spec)] = score
    # Python min retains the predeclared candidate order on exact ties.
    best = min(specs, key=lambda spec: scores[repr(spec)])
    return best, scores
