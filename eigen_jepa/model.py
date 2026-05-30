from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import SpectralMemory


@dataclass
class EigenJEPAConfig:
    input_dim: int
    num_assets: int
    context_len: int
    k: int = 3
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    latent_dim: int = 64
    memory_dim: int = 64


class ResidualMLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Linear(dim_in, dim_hidden), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(dim_hidden, dim_hidden), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(dim_hidden, dim_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TemporalSpectralEncoder(nn.Module):
    """Lightweight temporal encoder with local convolution and attention pooling."""

    def __init__(self, cfg: EigenJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.input_dim + 1, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.context_len, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.pre_norm = nn.LayerNorm(cfg.d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(cfg.d_model, cfg.d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool_gate = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        self.pool_proj = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
        )
        self.norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, x):
        h = self.in_proj(x) + self.pos[:, : x.shape[1]]
        h = self.pre_norm(h)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2) + h
        gate = torch.softmax(self.pool_gate(h).squeeze(-1), dim=-1)
        pooled = torch.sum(h * gate.unsqueeze(-1), dim=1)
        return self.norm(self.pool_proj(pooled))


class TargetEncoder(nn.Module):
    def __init__(self, target_dim: int, latent_dim: int):
        super().__init__()
        self.net = ResidualMLP(target_dim, max(2 * latent_dim, 64), latent_dim, depth=3, dropout=0.05)

    def forward(self, y):
        return self.net(y)


class EigenJEPA(nn.Module):
    def __init__(self, cfg: EigenJEPAConfig, target_dim: int):
        super().__init__()
        self.cfg = cfg
        self.context = TemporalSpectralEncoder(cfg)
        self.backbone = ResidualMLP(cfg.latent_dim, 2 * cfg.latent_dim, cfg.latent_dim, depth=3, dropout=cfg.dropout)
        self.query_head = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.GELU(), nn.Linear(cfg.latent_dim, cfg.memory_dim))
        self.value_head = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.GELU(), nn.Linear(cfg.latent_dim, cfg.latent_dim))
        self.gate_head = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim // 2), nn.GELU(), nn.Linear(cfg.latent_dim // 2, 1))
        self.regime_head = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim // 2), nn.GELU(), nn.Linear(cfg.latent_dim // 2, 3))
        self.eig_head = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.GELU(), nn.Linear(cfg.latent_dim, cfg.k), nn.Softplus())
        self.subspace_head = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.latent_dim), nn.GELU(), nn.Linear(cfg.latent_dim, cfg.num_assets * cfg.k))
        self.drift_head = nn.Sequential(nn.Linear(cfg.latent_dim, max(8, cfg.latent_dim // 2)), nn.GELU(), nn.Linear(max(8, cfg.latent_dim // 2), 1), nn.Softplus())
        self.risk_head = nn.Sequential(nn.Linear(cfg.latent_dim, max(8, cfg.latent_dim // 2)), nn.GELU(), nn.Linear(max(8, cfg.latent_dim // 2), 1), nn.Softplus())
        self.entropy_head = nn.Sequential(nn.Linear(cfg.latent_dim, max(8, cfg.latent_dim // 2)), nn.GELU(), nn.Linear(max(8, cfg.latent_dim // 2), 1), nn.Softplus())
        self.rank_head = nn.Sequential(nn.Linear(cfg.latent_dim, max(8, cfg.latent_dim // 2)), nn.GELU(), nn.Linear(max(8, cfg.latent_dim // 2), 1), nn.Softplus())
        self.target_encoder = TargetEncoder(target_dim, cfg.latent_dim)
        self.memory_projector = nn.Linear(cfg.latent_dim, cfg.latent_dim, bias=False)

    def encode_context(self, x):
        return self.context(x)

    def target_latent(self, target_vec):
        return self.target_encoder(target_vec)

    def forward(
        self,
        x,
        memory: Optional[SpectralMemory] = None,
        gate_override: Optional[float] = None,
        memory_scale: float = 1.0,
    ):
        z = self.context(x)
        z_backbone = self.backbone(z)
        query = self.query_head(z)
        gate = torch.sigmoid(self.gate_head(z))
        if gate_override is not None:
            gate = torch.full_like(gate, float(gate_override))
        value = self.value_head(z)

        mem_vec = torch.zeros_like(value)
        mem_info = None
        if memory is not None:
            mem_vec, mem_info = memory.retrieve(query)
            mem_vec = self.memory_projector(mem_vec) * float(memory_scale)

        z_hat = z_backbone + gate * mem_vec
        eig = self.eig_head(z_hat)

        sub_raw = self.subspace_head(z_hat).view(-1, self.cfg.num_assets, self.cfg.k)
        q = F.normalize(sub_raw, dim=1)
        proj = torch.matmul(q, q.transpose(-1, -2))
        drift = self.drift_head(z_hat).squeeze(-1)
        regime_logits = self.regime_head(z_hat)
        risk = self.risk_head(z_hat).squeeze(-1)
        entropy = self.entropy_head(z_hat).squeeze(-1)
        rank = self.rank_head(z_hat).squeeze(-1)

        return {
            'z': z,
            'z_backbone': z_backbone,
            'query': query,
            'value': value,
            'gate': gate.squeeze(-1),
            'mem_vec': mem_vec,
            'z_hat': z_hat,
            'eig': eig,
            'sub': q,
            'proj': proj,
            'drift': drift,
            'regime_logits': regime_logits,
            'risk': risk,
            'entropy': entropy,
            'rank': rank,
            'mem_info': mem_info,
        }
