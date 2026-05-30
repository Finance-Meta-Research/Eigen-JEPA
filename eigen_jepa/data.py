from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import csv
import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .spectral import (
    rolling_covariance,
    topk_spectrum,
    projector_from_vecs,
    principal_angles,
    effective_rank,
    spectral_entropy,
)


@dataclass
class MarketConfig:
    num_assets: int = 12
    total_steps: int = 900
    context_len: int = 32
    horizon: int = 8
    num_train: int = 520
    num_val: int = 140
    num_test: int = 140
    seed: int = 7
    crisis_prob: float = 0.035
    drift_prob: float = 0.015
    factor_dim: int = 3
    market_style: str = 'equity'
    event_quantile: float = 0.70
    mask_ratio: float = 0.25
    block_time: int = 4
    data_source: str = 'synthetic'  # synthetic | csv
    csv_path: str | None = None
    return_cols: Tuple[str, ...] = ()
    aux_cols: Tuple[str, ...] = ()
    date_col: str | None = None


_STYLE_PARAMS = {
    'equity': dict(crisis_prob=0.080, drift_prob=0.060, factor_scale=(0.75, 0.56, 0.42), idio=(0.34, 0.50, 1.00)),
    'crypto': dict(crisis_prob=0.120, drift_prob=0.080, factor_scale=(1.05, 0.88, 0.64), idio=(0.68, 0.90, 1.65)),
    'rates': dict(crisis_prob=0.050, drift_prob=0.040, factor_scale=(0.42, 0.31, 0.22), idio=(0.18, 0.26, 0.48)),
}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _style_cfg(cfg: MarketConfig):
    return _STYLE_PARAMS.get(cfg.market_style, _STYLE_PARAMS['equity'])


def _safe_norm(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def _load_csv_market(cfg: MarketConfig) -> Dict[str, torch.Tensor]:
    if cfg.csv_path is None:
        raise ValueError('csv_path must be provided when data_source="csv"')
    path = Path(cfg.csv_path)
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f'No rows found in {path}')

    cols = reader.fieldnames or []
    if not cfg.return_cols:
        # Prefer columns that look like returns, then any numeric columns except metadata.
        candidates = [c for c in cols if c not in {cfg.date_col} and c.lower() not in {'date', 'time', 'timestamp'}]
        ret_cols = [c for c in candidates if 'ret' in c.lower() or 'return' in c.lower()]
        if not ret_cols:
            ret_cols = candidates[: cfg.num_assets]
    else:
        ret_cols = list(cfg.return_cols)

    aux_cols = list(cfg.aux_cols)
    if not aux_cols:
        aux_cols = [c for c in cols if c not in set(ret_cols) | ({cfg.date_col} if cfg.date_col else set())]

    rets = []
    aux = []
    for row in rows:
        rets.append([float(row[c]) for c in ret_cols])
        aux_row = [float(row[c]) for c in aux_cols] if aux_cols else []
        aux.append(aux_row)

    returns = torch.tensor(np.asarray(rets, dtype=np.float32))
    if not aux_cols:
        # derive robust auxiliary channels when a CSV only contains returns
        mean_abs = returns.abs().mean(dim=1)
        realized = torch.sqrt(returns.pow(2).mean(dim=1) + 1e-8)
        stress = torch.cat([mean_abs.unsqueeze(-1), realized.unsqueeze(-1), mean_abs.unsqueeze(-1), torch.zeros_like(mean_abs.unsqueeze(-1))], dim=-1)
        aux_t = stress.float()
    else:
        aux_arr = np.asarray(aux, dtype=np.float32)
        if aux_arr.ndim == 1:
            aux_arr = aux_arr[:, None]
        if aux_arr.shape[1] < 4:
            pad = np.zeros((aux_arr.shape[0], 4 - aux_arr.shape[1]), dtype=np.float32)
            aux_arr = np.concatenate([aux_arr, pad], axis=1)
        aux_t = torch.tensor(aux_arr[:, :4], dtype=torch.float32)

    # If the CSV provides a different number of assets than cfg.num_assets, adapt to it.
    num_assets = returns.shape[1]
    if num_assets != cfg.num_assets:
        cfg.num_assets = num_assets

    # Create a simple monotone regime proxy from volatility and cross-sectional spread.
    vol = returns.abs().mean(dim=1)
    spread = returns.std(dim=1)
    stress = 0.5 * vol + 0.5 * spread
    q1, q2 = torch.quantile(stress, torch.tensor([0.55, 0.82], dtype=stress.dtype))
    regime = torch.zeros(len(returns), dtype=torch.long)
    regime[stress >= q1] = 1
    regime[stress >= q2] = 2

    return {
        'returns': returns,
        'aux': aux_t,
        'regime': regime,
        'source': 'csv',
    }


def generate_market_series(cfg: MarketConfig):
    """Generate one long regime-switching market with recurring spectral structure."""
    _set_seed(cfg.seed)
    if cfg.data_source == 'csv':
        return _load_csv_market(cfg)

    n = cfg.num_assets
    T = cfg.total_steps
    sp = _style_cfg(cfg)

    crisis_prob = sp['crisis_prob']
    drift_prob = sp['drift_prob']
    factor_scale = np.array(sp['factor_scale'], dtype=np.float32)
    idio_by_regime = {0: sp['idio'][0], 1: sp['idio'][1], 2: sp['idio'][2]}

    regimes = np.zeros(T, dtype=np.int64)
    regime = 0
    for t in range(T):
        r = np.random.rand()
        if regime == 0:
            if r < drift_prob:
                regime = 1
            elif r < drift_prob + crisis_prob:
                regime = 2
        elif regime == 1:
            if r < 0.42:
                regime = 0
            elif r < 0.42 + crisis_prob * 1.25:
                regime = 2
            else:
                regime = 1
        else:
            if r < 0.30:
                regime = 1
            elif r < 0.60:
                regime = 0
            else:
                regime = 2
        regimes[t] = regime

    base_loadings = np.random.randn(n, cfg.factor_dim)
    base_loadings = _safe_norm(base_loadings, axis=0)

    regime_loadings = []
    grid = np.linspace(0, 1, n)
    for s in range(3):
        A = base_loadings + 0.27 * np.random.randn(n, cfg.factor_dim)
        if s == 2:
            A[:, 0] += np.linspace(0.55, 1.55, n)
            A[:, 1] += np.sin(np.linspace(0, 3 * np.pi, n))
            A[:, 2] += np.cos(np.linspace(0, 4 * np.pi, n)) * 0.25
        elif s == 1:
            A[:, 1] += np.cos(np.linspace(0, 2 * np.pi, n))
            A[:, 2] += 0.20 * np.sin(np.linspace(0, 5 * np.pi, n))
        if cfg.market_style == 'crypto':
            A[:, 0] += 0.25 * np.sin(4 * np.pi * grid)
        elif cfg.market_style == 'rates':
            A[:, 1] += 0.20 * np.cos(2 * np.pi * grid)
        A = _safe_norm(A, axis=0)
        regime_loadings.append(A)

    factor_ar = np.array([0.76, 0.59, 0.46], dtype=np.float32)
    factor_vol = {
        0: factor_scale * 0.92,
        1: factor_scale * 1.10,
        2: factor_scale * 1.65,
    }

    factors = np.zeros((T, cfg.factor_dim), dtype=np.float32)
    rets = np.zeros((T, n), dtype=np.float32)
    volume = np.zeros(T, dtype=np.float32)
    realized_vol = np.zeros(T, dtype=np.float32)
    stress = np.zeros(T, dtype=np.float32)
    breadth = np.zeros(T, dtype=np.float32)

    prev_f = np.zeros(cfg.factor_dim, dtype=np.float32)
    prev_r = np.zeros(n, dtype=np.float32)
    for t in range(T):
        s = regimes[t]
        eps_f = np.random.randn(cfg.factor_dim).astype(np.float32) * factor_vol[s]
        f_t = factor_ar * prev_f + eps_f
        A = regime_loadings[s]
        idio = np.random.randn(n).astype(np.float32) * idio_by_regime[s]
        r_t = A @ f_t + idio

        if s == 2:
            shock = np.zeros(n, dtype=np.float32)
            idx = np.random.choice(n, size=max(1, n // 4), replace=False)
            amp = 1.35 if cfg.market_style != 'crypto' else 2.0
            shock[idx] = np.random.randn(len(idx)).astype(np.float32) * amp
            r_t += shock - 0.30 * np.sign(r_t) * np.abs(r_t)
        if s == 1:
            phase = 0.11 * t if cfg.market_style != 'rates' else 0.08 * t
            rotation = np.sin(phase) * np.linspace(-0.45, 0.45, n).astype(np.float32)
            r_t += 0.18 * rotation
        if cfg.market_style == 'crypto':
            r_t += 0.08 * np.sign(np.random.randn(n)).astype(np.float32) * np.abs(np.random.randn(n).astype(np.float32))
        elif cfg.market_style == 'rates':
            r_t += 0.05 * np.tanh(np.random.randn(n).astype(np.float32))

        factors[t] = f_t
        rets[t] = r_t
        volume[t] = 1.0 + 0.18 * np.abs(r_t).mean() + 0.68 * float(s == 2) + 0.18 * float(s == 1) + 0.05 * np.random.randn()
        win = rets[max(0, t - 20):t + 1]
        realized_vol[t] = win.std() if len(win) > 1 else np.abs(r_t).mean()
        corr = np.corrcoef(r_t, prev_r)[0, 1] if t > 1 and np.std(prev_r) > 1e-6 and np.std(r_t) > 1e-6 else 0.0
        stress[t] = 0.14 * np.abs(r_t).mean() + 0.12 * np.abs(corr) + 0.48 * float(s == 2) + 0.16 * float(s == 1)
        breadth[t] = (r_t > 0).mean() - 0.5
        prev_f = f_t
        prev_r = r_t

    aux = np.stack([volume, realized_vol, stress, breadth], axis=-1).astype(np.float32)
    return {
        'returns': torch.tensor(rets),
        'aux': torch.tensor(aux),
        'regime': torch.tensor(regimes),
        'factors': torch.tensor(factors),
        'source': 'synthetic',
    }


def _make_features(returns: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
    return torch.cat([returns, aux], dim=-1)


def _window_salience_score(
    returns: torch.Tensor,
    regime: torch.Tensor,
    c0: int,
    c1: int,
    f0: int,
    f1: int,
    k: int,
) -> tuple[float, float, float, float]:
    ctx_r = returns[c0:c1]
    fut_r = returns[f0:f1]
    fut_reg = regime[f0:f1]

    cov_f = rolling_covariance(fut_r)
    evals_f, evecs_f = topk_spectrum(cov_f, k)
    cov_c = rolling_covariance(ctx_r)
    evals_c, evecs_c = topk_spectrum(cov_c, k)

    drift = float(principal_angles(evecs_c, evecs_f).mean().item())
    entropy = float(spectral_entropy(evals_f).item())
    crisis_fraction = float((fut_reg == 2).float().mean().item())
    vol_ratio = float((fut_r.std() / (ctx_r.std() + 1e-6)).clamp(0.0, 6.0).item())

    drift_norm = drift / (math.pi / 2.0 + 1e-6)
    entropy_norm = entropy / (math.log(max(k, 2)) + 1e-6)
    vol_norm = math.log1p(vol_ratio) / math.log1p(6.0)

    score = 0.45 * drift_norm + 0.32 * (1.0 - entropy_norm) + 0.15 * crisis_fraction + 0.08 * vol_norm
    return score, drift, entropy, crisis_fraction


class MarketWindowDataset(Dataset):
    def __init__(
        self,
        returns: torch.Tensor,
        aux: torch.Tensor,
        regime: torch.Tensor,
        context_len: int,
        horizon: int,
        indices: np.ndarray,
        k: int = 3,
        mask_ratio: float = 0.25,
        block_time: int = 4,
        seed: int = 0,
        event_threshold: float = 0.28,
    ):
        self.returns = returns
        self.aux = aux
        self.regime = regime
        self.context_len = context_len
        self.horizon = horizon
        self.indices = indices
        self.k = k
        self.mask_ratio = mask_ratio
        self.block_time = block_time
        self.event_threshold = event_threshold
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices)

    def _mask(self, x: torch.Tensor) -> torch.Tensor:
        T, F = x.shape
        mask = torch.ones((T, F), dtype=x.dtype)
        if self.mask_ratio <= 0:
            return mask
        total_blocks = max(1, int(T * self.mask_ratio / max(self.block_time, 1)))
        for _ in range(total_blocks):
            t0 = int(self.rng.integers(0, max(1, T - self.block_time + 1)))
            if self.rng.random() < 0.55:
                mask[t0:t0 + self.block_time] = 0.0
            else:
                feat_w = int(self.rng.integers(1, max(2, F // 4 + 1)))
                f0 = int(self.rng.integers(0, max(1, F - feat_w + 1)))
                mask[t0:t0 + self.block_time, f0:f0 + feat_w] = 0.0
        return mask

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        c0 = idx - self.context_len + 1
        c1 = idx + 1
        f0 = idx + 1
        f1 = idx + 1 + self.horizon
        ctx_r = self.returns[c0:c1]
        fut_r = self.returns[f0:f1]
        ctx_aux = self.aux[c0:c1]
        fut_aux = self.aux[f0:f1]
        ctx = _make_features(ctx_r, ctx_aux)
        fut = _make_features(fut_r, fut_aux)

        cov_f = rolling_covariance(fut_r)
        evals_f, evecs_f = topk_spectrum(cov_f, self.k)
        proj_f = projector_from_vecs(evecs_f)
        cov_c = rolling_covariance(ctx_r)
        evals_c, evecs_c = topk_spectrum(cov_c, self.k)
        proj_c = projector_from_vecs(evecs_c)
        drift = principal_angles(evecs_c, evecs_f).mean()
        spec_rank = effective_rank(evals_f)
        spec_entropy = spectral_entropy(evals_f)
        salience_score, drift_raw, entropy_raw, crisis_fraction = _window_salience_score(
            self.returns, self.regime, c0, c1, f0, f1, self.k
        )
        risk_true = torch.matmul(torch.ones(self.returns.shape[1], dtype=fut_r.dtype, device=fut_r.device) / self.returns.shape[1],
                                 torch.matmul(cov_f, torch.ones(self.returns.shape[1], dtype=fut_r.dtype, device=fut_r.device) / self.returns.shape[1]))
        mask = self._mask(ctx)
        masked_ctx = ctx * mask
        mask_channel = mask.mean(dim=-1, keepdim=True)
        model_in = torch.cat([masked_ctx, mask_channel], dim=-1)
        target_vec = torch.cat([
            evals_f.flatten(),
            proj_f.flatten(),
            drift.view(1),
            risk_true.view(1),
            spec_entropy.view(1),
            spec_rank.view(1),
        ], dim=0)

        regime_true = self.regime[f0].long()
        event = torch.tensor(float(salience_score >= self.event_threshold), dtype=torch.float32)

        return {
            'x': model_in.float(),
            'mask': mask_channel.float(),
            'returns_ctx': ctx_r.float(),
            'returns_fut': fut_r.float(),
            'cov_true': cov_f.float(),
            'evecs_true': evecs_f.float(),
            'evals_true': evals_f.float(),
            'proj_true': proj_f.float(),
            'proj_ctx': proj_c.float(),
            'drift_true': drift.float(),
            'target_vec': target_vec.float(),
            'risk_true': risk_true.float(),
            'entropy_true': spec_entropy.float(),
            'rank_true': spec_rank.float(),
            'regime_true': regime_true,
            'event_true': event,
            'salience_true': torch.tensor(float(salience_score), dtype=torch.float32),
            'spec_rank_true': spec_rank.float(),
            'spec_entropy_true': spec_entropy.float(),
            'drift_raw_true': torch.tensor(float(drift_raw), dtype=torch.float32),
            'entropy_raw_true': torch.tensor(float(entropy_raw), dtype=torch.float32),
            'crisis_frac_true': torch.tensor(float(crisis_fraction), dtype=torch.float32),
            'window_index': torch.tensor(int(idx), dtype=torch.long),
        }


def build_datasets(cfg: MarketConfig, k: int = 3):
    series = generate_market_series(cfg)
    returns = series['returns']
    aux = series['aux']
    regime = series['regime']
    valid_start = cfg.context_len - 1
    valid_end = cfg.total_steps - cfg.horizon - 1
    all_idx = np.arange(valid_start, valid_end + 1)
    n = len(all_idx)
    n_train, n_val = cfg.num_train, cfg.num_val
    assert n_train + n_val + cfg.num_test <= n, (n_train, n_val, cfg.num_test, n)

    train_idx = all_idx[:n_train]
    train_scores = []
    for idx in train_idx:
        c0 = idx - cfg.context_len + 1
        c1 = idx + 1
        f0 = idx + 1
        f1 = idx + 1 + cfg.horizon
        score, _, _, _ = _window_salience_score(returns, regime, c0, c1, f0, f1, k)
        train_scores.append(score)
    event_threshold = float(np.quantile(np.asarray(train_scores, dtype=np.float32), cfg.event_quantile))

    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:n_train + n_val + cfg.num_test]
    ds = {
        'train': MarketWindowDataset(returns, aux, regime, cfg.context_len, cfg.horizon, train_idx, k=k, mask_ratio=cfg.mask_ratio, block_time=cfg.block_time, seed=cfg.seed, event_threshold=event_threshold),
        'val': MarketWindowDataset(returns, aux, regime, cfg.context_len, cfg.horizon, val_idx, k=k, mask_ratio=cfg.mask_ratio, block_time=cfg.block_time, seed=cfg.seed + 1, event_threshold=event_threshold),
        'test': MarketWindowDataset(returns, aux, regime, cfg.context_len, cfg.horizon, test_idx, k=k, mask_ratio=cfg.mask_ratio, block_time=cfg.block_time, seed=cfg.seed + 2, event_threshold=event_threshold),
        'series': series,
        'event_threshold': event_threshold,
    }
    return ds


def batch_collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out
