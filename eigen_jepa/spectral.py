from __future__ import annotations

from typing import Dict, Tuple

import torch


def rolling_covariance(returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute a covariance matrix for [T, N] or a batch of covariances for [B, T, N]."""
    if returns.dim() == 2:
        x = returns - returns.mean(dim=0, keepdim=True)
        denom = max(x.shape[0] - 1, 1)
        cov = x.T @ x / denom
        eye = torch.eye(cov.shape[-1], device=returns.device, dtype=returns.dtype)
        return cov + eps * eye
    if returns.dim() == 3:
        x = returns - returns.mean(dim=1, keepdim=True)
        denom = max(x.shape[1] - 1, 1)
        cov = torch.matmul(x.transpose(1, 2), x) / denom
        eye = torch.eye(cov.shape[-1], device=returns.device, dtype=returns.dtype).unsqueeze(0)
        return cov + eps * eye
    raise ValueError('returns must have shape [T, N] or [B, T, N]')


def topk_spectrum(cov: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    evals, evecs = torch.linalg.eigh(cov)
    evals = torch.flip(evals, dims=(-1,))
    evecs = torch.flip(evecs, dims=(-1,))
    return evals[..., :k], evecs[..., :, :k]


def projector_from_vecs(vecs: torch.Tensor) -> torch.Tensor:
    if vecs.dim() == 2:
        return vecs @ vecs.T
    return torch.matmul(vecs, vecs.transpose(-1, -2))


def low_rank_covariance(evals: torch.Tensor, vecs: torch.Tensor) -> torch.Tensor:
    if vecs.dim() == 2:
        return vecs @ torch.diag(evals) @ vecs.T
    return torch.matmul(vecs * evals.unsqueeze(-2), vecs.transpose(-1, -2))


def portfolio_risk(cov: torch.Tensor, weights: torch.Tensor | None = None, eps: float = 1e-8) -> torch.Tensor:
    if cov.dim() == 2:
        n = cov.shape[-1]
        if weights is None:
            weights = torch.ones(n, device=cov.device, dtype=cov.dtype) / n
        return torch.sqrt((weights.unsqueeze(0) @ cov @ weights.unsqueeze(-1)).squeeze() + eps)
    if cov.dim() == 3:
        n = cov.shape[-1]
        if weights is None:
            weights = torch.ones(n, device=cov.device, dtype=cov.dtype) / n
        w = weights.view(1, n, 1)
        return torch.sqrt((w.transpose(1, 2) @ cov @ w).squeeze(-1).squeeze(-1) + eps)
    raise ValueError('cov must have shape [N, N] or [B, N, N]')


def subspace_distance(P_hat: torch.Tensor, P_true: torch.Tensor) -> torch.Tensor:
    diff = P_hat - P_true
    return torch.sum(diff * diff, dim=(-1, -2)) if diff.dim() == 3 else torch.sum(diff * diff)


def eigengaps(evals: torch.Tensor) -> torch.Tensor:
    return evals[..., :-1] - evals[..., 1:]


def effective_rank(evals: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = evals / (evals.sum(dim=-1, keepdim=True) + eps)
    entropy = -(p * (p + eps).log()).sum(dim=-1)
    return entropy.exp()


def spectral_entropy(evals: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = evals / (evals.sum(dim=-1, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(dim=-1)


def principal_angles(U_hat: torch.Tensor, U_true: torch.Tensor) -> torch.Tensor:
    if U_hat.dim() == 2:
        U_hat = U_hat.unsqueeze(0)
    if U_true.dim() == 2:
        U_true = U_true.unsqueeze(0)
    m = torch.matmul(U_hat.transpose(-1, -2), U_true)
    s = torch.linalg.svdvals(m).clamp(0.0, 1.0)
    return torch.acos(s).mean(dim=-1)


def chordal_distance(U_hat: torch.Tensor, U_true: torch.Tensor) -> torch.Tensor:
    """Chordal distance between subspaces spanned by orthonormal bases."""
    angles = principal_angles(U_hat, U_true)
    return torch.sqrt(torch.sum(torch.sin(angles) ** 2))


def spectral_metrics(
    pred_evals: torch.Tensor,
    true_evals: torch.Tensor,
    pred_proj: torch.Tensor,
    true_proj: torch.Tensor,
    pred_drift: torch.Tensor,
    true_drift: torch.Tensor,
    pred_cov: torch.Tensor | None = None,
    true_cov: torch.Tensor | None = None,
    pred_risk: torch.Tensor | None = None,
    true_risk: torch.Tensor | None = None,
    pred_entropy: torch.Tensor | None = None,
    true_entropy: torch.Tensor | None = None,
    pred_rank: torch.Tensor | None = None,
    true_rank: torch.Tensor | None = None,
) -> Dict[str, float]:
    eps = 1e-12
    nmse = torch.mean((pred_evals - true_evals) ** 2) / (torch.mean(true_evals ** 2) + eps)
    proj = torch.mean((pred_proj - true_proj) ** 2)
    drift = torch.mean((pred_drift - true_drift) ** 2)
    gaps_true = true_evals[..., :-1] - true_evals[..., 1:]
    gaps_pred = pred_evals[..., :-1] - pred_evals[..., 1:]
    gap_rmse = torch.sqrt(torch.mean((gaps_pred - gaps_true) ** 2) + eps)
    out = {
        'eig_nmse': float(nmse.detach().cpu()),
        'proj_mse': float(proj.detach().cpu()),
        'drift_mse': float(drift.detach().cpu()),
        'gap_rmse': float(gap_rmse.detach().cpu()),
    }
    if pred_cov is not None and true_cov is not None:
        out['cov_mse'] = float(torch.mean((pred_cov - true_cov) ** 2).detach().cpu())
    if pred_risk is not None and true_risk is not None:
        out['risk_mse'] = float(torch.mean((pred_risk - true_risk) ** 2).detach().cpu())
    if pred_entropy is not None and true_entropy is not None:
        out['entropy_mse'] = float(torch.mean((pred_entropy - true_entropy) ** 2).detach().cpu())
    if pred_rank is not None and true_rank is not None:
        out['rank_mse'] = float(torch.mean((pred_rank - true_rank) ** 2).detach().cpu())
    return out
