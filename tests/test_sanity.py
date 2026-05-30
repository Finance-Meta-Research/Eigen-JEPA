from pathlib import Path

import torch

from eigen_jepa.data import MarketConfig, build_datasets, batch_collate
from eigen_jepa.model import EigenJEPA, EigenJEPAConfig
from eigen_jepa.memory import SpectralMemory
from eigen_jepa.utils import set_seed


def test_deterministic_dataset():
    cfg = MarketConfig(num_assets=6, total_steps=80, context_len=10, horizon=4, num_train=16, num_val=8, num_test=8, seed=11, market_style='equity')
    ds1 = build_datasets(cfg, k=3)
    ds2 = build_datasets(cfg, k=3)
    ex1 = ds1['train'][0]
    ex2 = ds2['train'][0]
    assert torch.allclose(ex1['x'], ex2['x'])
    assert torch.allclose(ex1['evals_true'], ex2['evals_true'])
    assert torch.allclose(ex1['cov_true'], ex2['cov_true'])


def test_forward_and_memory_bounds():
    set_seed(5)
    cfg = MarketConfig(num_assets=6, total_steps=80, context_len=10, horizon=4, num_train=16, num_val=8, num_test=8, seed=11, market_style='equity')
    ds = build_datasets(cfg, k=3)
    batch = batch_collate([ds['train'][i] for i in range(4)])
    model = EigenJEPA(EigenJEPAConfig(input_dim=10, num_assets=6, context_len=10, k=3, d_model=24, latent_dim=24, memory_dim=24), target_dim=3 + 36 + 1 + 3)
    memory = SpectralMemory(key_dim=24, value_dim=24, max_items=8, top_k=2)
    out = model(batch['x'], memory=memory)
    assert out['eig'].shape == (4, 3)
    assert out['proj'].shape == (4, 6, 6)
    assert out['gate'].shape == (4,)
    assert out['risk'].shape == (4,)
    memory.write(out['query'].detach(), out['z_hat'].detach(), torch.ones(4), batch['regime_true'])
    assert memory.state.size <= 8
