from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _as_2d_returns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("returns must have shape [time, assets] with at least 2 rows/assets")
    if not np.isfinite(x).all():
        raise ValueError("returns must be finite")
    return x


def sample_covariance(returns: np.ndarray) -> np.ndarray:
    x = _as_2d_returns(returns)
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(x.shape[0] - 1, 1)
    return 0.5 * (cov + cov.T)


def ewma_covariance(returns: np.ndarray, decay: float = 0.94) -> np.ndarray:
    x = _as_2d_returns(returns)
    if not 0.0 < decay < 1.0:
        raise ValueError("decay must lie strictly between 0 and 1")
    n = x.shape[0]
    powers = np.arange(n - 1, -1, -1, dtype=np.float64)
    w = np.power(decay, powers)
    w /= w.sum()
    mean = np.sum(x * w[:, None], axis=0, keepdims=True)
    centered = x - mean
    cov = (centered * w[:, None]).T @ centered
    return 0.5 * (cov + cov.T)


def linear_shrinkage(
    cov: np.ndarray,
    alpha: float = 0.1,
    target: str = "scaled_identity",
) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
        raise ValueError("cov must be a square matrix with at least 2 assets")
    if not np.isfinite(cov).all():
        raise ValueError("cov must be finite")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    cov = 0.5 * (cov + cov.T)
    if target == "scaled_identity":
        mu = float(np.trace(cov) / cov.shape[0])
        tgt = np.eye(cov.shape[0], dtype=np.float64) * mu
    elif target == "diagonal":
        tgt = np.diag(np.diag(cov))
    else:
        raise ValueError("target must be 'scaled_identity' or 'diagonal'")
    out = (1.0 - alpha) * cov + alpha * tgt
    return 0.5 * (out + out.T)


def oas_covariance(returns: np.ndarray) -> np.ndarray:
    """Oracle-approximating shrinkage toward a scaled identity target.

    The shrinkage intensity is estimated only from the supplied return window.
    This implementation follows the standard OAS closed-form estimator and
    intentionally has no dependency on scikit-learn.
    """
    x = _as_2d_returns(returns)
    centered = x - x.mean(axis=0, keepdims=True)
    n, p = centered.shape
    emp = centered.T @ centered / float(n)
    emp = 0.5 * (emp + emp.T)
    mu = float(np.trace(emp) / p)
    alpha = float(np.mean(emp**2))
    den = float((n + 1.0) * (alpha - (mu**2) / p))
    shrinkage = 1.0 if den <= 1e-30 else min((alpha + mu**2) / den, 1.0)
    out = (1.0 - shrinkage) * emp + shrinkage * mu * np.eye(p, dtype=np.float64)
    return 0.5 * (out + out.T)


def factor_covariance(returns: np.ndarray, rank: int) -> np.ndarray:
    """Low-rank factor covariance plus non-negative idiosyncratic diagonal."""
    x = _as_2d_returns(returns)
    p = x.shape[1]
    if rank <= 0 or rank >= p:
        raise ValueError("rank must lie in [1, assets-1]")
    cov = sample_covariance(x)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = vecs[:, order]
    low = (vecs[:, :rank] * vals[:rank]) @ vecs[:, :rank].T
    residual = np.clip(np.diag(cov - low), 0.0, None)
    out = low + np.diag(residual)
    return 0.5 * (out + out.T)


def topk_eigenvalues(cov: np.ndarray, k: int) -> np.ndarray:
    cov = np.asarray(cov, dtype=np.float64)
    if k <= 0 or k > cov.shape[0]:
        raise ValueError("k must be between 1 and covariance dimension")
    vals = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    vals = np.sort(np.clip(vals, 0.0, None))[::-1]
    return vals[:k]


def covariance_nmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("pred and true must have identical shape")
    num = float(np.mean((pred - true) ** 2))
    den = float(np.mean(true**2))
    return num / max(den, 1e-18)


def eigen_nmse(pred_cov: np.ndarray, true_cov: np.ndarray, k: int) -> float:
    p = topk_eigenvalues(pred_cov, k)
    t = topk_eigenvalues(true_cov, k)
    return float(np.mean((p - t) ** 2) / max(np.mean(t**2), 1e-18))


@dataclass(frozen=True)
class ClassicalCovarianceSpec:
    name: str
    decay: float | None = None
    shrinkage_alpha: float | None = None
    shrinkage_target: str = "scaled_identity"
    factor_rank: int | None = None


def forecast_covariance(
    context_returns: np.ndarray,
    spec: ClassicalCovarianceSpec,
) -> np.ndarray:
    if spec.name == "sample_persistence":
        return sample_covariance(context_returns)
    if spec.name == "ewma":
        if spec.decay is None:
            raise ValueError("EWMA spec requires decay")
        return ewma_covariance(context_returns, spec.decay)
    if spec.name == "oas":
        return oas_covariance(context_returns)
    if spec.name == "factor":
        if spec.factor_rank is None:
            raise ValueError("factor spec requires factor_rank")
        return factor_covariance(context_returns, spec.factor_rank)
    if spec.name == "sample_shrinkage":
        if spec.shrinkage_alpha is None:
            raise ValueError("sample_shrinkage spec requires alpha")
        return linear_shrinkage(
            sample_covariance(context_returns),
            spec.shrinkage_alpha,
            spec.shrinkage_target,
        )
    if spec.name == "ewma_shrinkage":
        if spec.decay is None or spec.shrinkage_alpha is None:
            raise ValueError("ewma_shrinkage spec requires decay and alpha")
        return linear_shrinkage(
            ewma_covariance(context_returns, spec.decay),
            spec.shrinkage_alpha,
            spec.shrinkage_target,
        )
    raise ValueError(f"unknown classical covariance baseline: {spec.name}")


def select_by_validation(
    contexts: Sequence[np.ndarray],
    future_returns: Sequence[np.ndarray],
    specs: Iterable[ClassicalCovarianceSpec],
    *,
    k: int,
) -> tuple[ClassicalCovarianceSpec, dict[str, float]]:
    contexts = list(contexts)
    future_returns = list(future_returns)
    if len(contexts) == 0 or len(contexts) != len(future_returns):
        raise ValueError("contexts/future_returns must be non-empty and length matched")
    scores: dict[str, float] = {}
    best = None
    best_score = float("inf")
    for spec in specs:
        vals = []
        for ctx, fut in zip(contexts, future_returns):
            pred = forecast_covariance(ctx, spec)
            true = sample_covariance(fut)
            vals.append(eigen_nmse(pred, true, k))
        score = float(np.mean(vals))
        key = repr(spec)
        scores[key] = score
        if score < best_score:
            best_score = score
            best = spec
    assert best is not None
    return best, scores


def default_candidate_grid() -> tuple[ClassicalCovarianceSpec, ...]:
    out: list[ClassicalCovarianceSpec] = [
        ClassicalCovarianceSpec("sample_persistence"),
        ClassicalCovarianceSpec("oas"),
    ]
    for decay in (0.90, 0.94, 0.97):
        out.append(ClassicalCovarianceSpec("ewma", decay=decay))
    for rank in (1, 2, 3):
        out.append(ClassicalCovarianceSpec("factor", factor_rank=rank))
    for alpha in (0.05, 0.15, 0.30, 0.50):
        out.append(
            ClassicalCovarianceSpec("sample_shrinkage", shrinkage_alpha=alpha)
        )
    for decay in (0.90, 0.94, 0.97):
        for alpha in (0.05, 0.15, 0.30):
            out.append(
                ClassicalCovarianceSpec(
                    "ewma_shrinkage",
                    decay=decay,
                    shrinkage_alpha=alpha,
                )
            )
    return tuple(out)
