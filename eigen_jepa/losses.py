from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    align: float = 1.0
    eig: float = 1.0
    subspace: float = 2.0
    drift: float = 0.5
    regime: float = 0.75
    gate: float = 0.05
    risk: float = 0.25
    entropy: float = 0.15
    rank: float = 0.15
    latent_reg: float = 1e-3
    tail_boost: float = 1.5


def _sample_weights(event_true: torch.Tensor, drift_true: torch.Tensor, boost: float) -> torch.Tensor:
    drift_scale = drift_true.detach()
    drift_scale = drift_scale / (drift_scale.mean().clamp_min(1e-6) + 1e-6)
    drift_scale = drift_scale.clamp(0.0, 2.0)
    return 1.0 + boost * event_true.float() + 0.35 * drift_scale


def _weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    while weight.dim() < loss.dim():
        weight = weight.unsqueeze(-1)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def spectral_jepa_loss(
    pred: Dict[str, torch.Tensor],
    target_latent: torch.Tensor,
    evals_true: torch.Tensor,
    proj_true: torch.Tensor,
    drift_true: torch.Tensor,
    regime_true: torch.Tensor,
    event_true: torch.Tensor,
    risk_true: torch.Tensor,
    entropy_true: torch.Tensor,
    rank_true: torch.Tensor,
    weights: LossWeights = LossWeights(),
    class_weights: Optional[torch.Tensor] = None,
):
    sample_w = _sample_weights(event_true, drift_true, weights.tail_boost)
    align = _weighted_mean((pred['z_hat'] - target_latent).pow(2), sample_w)
    eig = _weighted_mean((pred['eig'] - evals_true).pow(2), sample_w)
    sub = _weighted_mean((pred['proj'] - proj_true).pow(2), sample_w)
    drift = _weighted_mean((pred['drift'] - drift_true).pow(2), sample_w)
    regime = F.cross_entropy(pred['regime_logits'], regime_true, weight=class_weights)
    risk = _weighted_mean((pred['risk'] - risk_true).pow(2), sample_w)
    entropy = _weighted_mean((pred['entropy'] - entropy_true).pow(2), sample_w)
    rank = _weighted_mean((pred['rank'] - rank_true).pow(2), sample_w)
    gate_weight = sample_w * (1.0 + 1.2 * event_true.float())
    gate = _weighted_mean(
        F.binary_cross_entropy(pred['gate'].clamp(1e-4, 1 - 1e-4), event_true.float(), reduction='none'),
        gate_weight,
    )
    eye = torch.eye(pred['sub'].shape[-1], device=pred['sub'].device, dtype=pred['sub'].dtype).unsqueeze(0)
    ortho = (torch.matmul(pred['sub'].transpose(-1, -2), pred['sub']) - eye).pow(2).mean()
    reg = pred['z_hat'].pow(2).mean() + pred['z'].pow(2).mean() + pred['mem_vec'].pow(2).mean()
    total = (
        weights.align * align
        + weights.eig * eig
        + weights.subspace * sub
        + weights.drift * drift
        + weights.regime * regime
        + weights.gate * gate
        + weights.risk * risk
        + weights.entropy * entropy
        + weights.rank * rank
        + 0.1 * ortho
        + weights.latent_reg * reg
    )
    return total, {
        'loss': float(total.detach().cpu()),
        'align': float(align.detach().cpu()),
        'eig': float(eig.detach().cpu()),
        'sub': float(sub.detach().cpu()),
        'drift': float(drift.detach().cpu()),
        'regime': float(regime.detach().cpu()),
        'gate': float(gate.detach().cpu()),
        'risk': float(risk.detach().cpu()),
        'entropy': float(entropy.detach().cpu()),
        'rank': float(rank.detach().cpu()),
        'ortho': float(ortho.detach().cpu()),
        'reg': float(reg.detach().cpu()),
    }
