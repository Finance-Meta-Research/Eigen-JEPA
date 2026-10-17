"""Actual legacy model/loss witness using generated tensors only."""
from dataclasses import asdict
import json

import torch

from eigen_jepa.losses import spectral_jepa_loss
from eigen_jepa.memory import SpectralMemory
from eigen_jepa.model import EigenJEPA, EigenJEPAConfig


torch.set_num_threads(1)
torch.manual_seed(94501)
config = EigenJEPAConfig(input_dim=7, num_assets=3, context_len=6, k=2,
                         d_model=8, n_heads=2, n_layers=1, dropout=0.0,
                         latent_dim=8, memory_dim=8)
model = EigenJEPA(config, target_dim=15)
memory = SpectralMemory(8, 8, max_items=3, top_k=2, temperature=0.7,
                        merge_radius=0.0, min_salience=0.0)
memory.write(torch.eye(8)[:3], torch.arange(24, dtype=torch.float32).reshape(3, 8)/10,
             torch.ones(3), torch.tensor([0, 1, 2]))
inputs = torch.linspace(-1.0, 1.0, 96).reshape(2, 6, 8)
pred = model(inputs, memory=memory)
target = model.target_latent(torch.linspace(-0.4, 0.4, 30).reshape(2, 15))
loss, parts = spectral_jepa_loss(pred, target,
    torch.tensor([[0.4, 0.2], [0.5, 0.1]]),
    torch.diag(torch.tensor([1.0, 1.0, 0.0])).repeat(2, 1, 1),
    torch.tensor([0.1, 0.2]), torch.tensor([0, 2]), torch.tensor([0.0, 1.0]),
    torch.tensor([0.3, 0.4]), torch.tensor([0.4, 0.5]), torch.tensor([1.5, 1.8]))
loss.backward()
query_gradients = {name: None if parameter.grad is None else float(parameter.grad.norm())
                   for name, parameter in model.query_head.named_parameters()}
other_gradient = float(model.backbone.net[0].weight.grad.norm())
result = {'scope': 'generated actual-model/loss derivative witness', 'torch': str(torch.__version__),
          'seed': 94501, 'config': asdict(config), 'memory_items': memory.state.size,
          'top_k': memory.top_k, 'temperature': memory.temperature,
          'query_parameter_gradient_norms': query_gradients,
          'backbone_gradient_norm': other_gradient, 'finite_loss': bool(torch.isfinite(loss)),
          'loss': parts['loss'], 'contract_passed': any(v is not None and v > 0 for v in query_gradients.values())}
print(json.dumps(result, indent=2, allow_nan=False))
raise SystemExit(0 if result['contract_passed'] else 1)
