from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class MemoryState:
    keys: torch.Tensor
    values: torch.Tensor
    salience: torch.Tensor
    regimes: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.keys.shape[0])


class SpectralMemory:
    """A sparse content-addressed memory bank for latent spectral prototypes."""

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        max_items: int = 128,
        top_k: int = 4,
        temperature: float = 0.2,
        merge_radius: float = 0.18,
        min_salience: float = 0.25,
        device: str | torch.device = 'cpu',
    ):
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.max_items = max_items
        self.top_k = top_k
        self.temperature = temperature
        self.merge_radius = merge_radius
        self.min_salience = min_salience
        self.device = torch.device(device)
        self.state = MemoryState(
            keys=torch.empty(0, key_dim, device=self.device),
            values=torch.empty(0, value_dim, device=self.device),
            salience=torch.empty(0, device=self.device),
            regimes=torch.empty(0, dtype=torch.long, device=self.device),
        )

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        self.state = MemoryState(
            keys=self.state.keys.to(self.device),
            values=self.state.values.to(self.device),
            salience=self.state.salience.to(self.device),
            regimes=self.state.regimes.to(self.device),
        )
        return self

    def clear(self):
        self.state = MemoryState(
            keys=torch.empty(0, self.key_dim, device=self.device),
            values=torch.empty(0, self.value_dim, device=self.device),
            salience=torch.empty(0, device=self.device),
            regimes=torch.empty(0, dtype=torch.long, device=self.device),
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            'keys': self.state.keys.detach().cpu(),
            'values': self.state.values.detach().cpu(),
            'salience': self.state.salience.detach().cpu(),
            'regimes': self.state.regimes.detach().cpu(),
            'meta': torch.tensor([self.key_dim, self.value_dim, self.max_items, self.top_k], dtype=torch.long),
        }

    def load_state_dict(self, sd: Dict[str, torch.Tensor]):
        self.key_dim = int(sd['meta'][0].item())
        self.value_dim = int(sd['meta'][1].item())
        self.max_items = int(sd['meta'][2].item())
        self.top_k = int(sd['meta'][3].item())
        self.state = MemoryState(
            keys=sd['keys'].to(self.device),
            values=sd['values'].to(self.device),
            salience=sd['salience'].to(self.device),
            regimes=sd['regimes'].to(self.device),
        )

    def _normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    @torch.no_grad()
    def retrieve(self, query: torch.Tensor):
        if self.state.size == 0:
            zeros = torch.zeros(query.shape[0], self.value_dim, device=query.device, dtype=query.dtype)
            return zeros, {'weights': torch.empty(query.shape[0], 0, device=query.device)}

        q = self._normalize(query)
        k = self._normalize(self.state.keys)
        sim = torch.matmul(q, k.T)
        topk = min(self.top_k, sim.shape[-1])
        vals, idx = torch.topk(sim, k=topk, dim=-1)
        w = torch.softmax(vals / max(self.temperature, 1e-6), dim=-1)
        gathered = self.state.values[idx]
        ret = torch.sum(w.unsqueeze(-1) * gathered, dim=1)
        return ret, {'weights': w, 'indices': idx, 'similarity': vals}

    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor, salience: torch.Tensor, regimes: torch.Tensor):
        if keys.numel() == 0:
            return
        keys = keys.detach().to(self.device)
        values = values.detach().to(self.device)
        salience = salience.detach().to(self.device).flatten()
        regimes = regimes.detach().to(self.device).flatten().long()
        for i in range(keys.shape[0]):
            s = float(salience[i].item())
            if s < self.min_salience:
                continue
            k = keys[i:i + 1]
            v = values[i:i + 1]
            r = regimes[i:i + 1]
            if self.state.size == 0:
                self.state = MemoryState(k.clone(), v.clone(), torch.tensor([s], device=self.device), r.clone())
                continue
            q = self._normalize(k)
            kk = self._normalize(self.state.keys)
            sim = torch.matmul(q, kk.T).squeeze(0)
            best = int(torch.argmax(sim).item())
            dist = float(1.0 - sim[best].item())
            if dist < self.merge_radius:
                old_s = float(self.state.salience[best].item())
                new_s = max(old_s, s)
                alpha = min(0.85, s / (old_s + s + 1e-6))
                self.state.keys[best] = (1 - alpha) * self.state.keys[best] + alpha * k.squeeze(0)
                self.state.values[best] = (1 - alpha) * self.state.values[best] + alpha * v.squeeze(0)
                self.state.salience[best] = torch.tensor(new_s, device=self.device)
                self.state.regimes[best] = r.squeeze(0)
            else:
                self.state.keys = torch.cat([self.state.keys, k], dim=0)
                self.state.values = torch.cat([self.state.values, v], dim=0)
                self.state.salience = torch.cat([self.state.salience, torch.tensor([s], device=self.device)], dim=0)
                self.state.regimes = torch.cat([self.state.regimes, r], dim=0)

            if self.state.size > self.max_items:
                order = torch.argsort(self.state.salience, descending=True)[: self.max_items]
                self.state = MemoryState(
                    keys=self.state.keys[order],
                    values=self.state.values[order],
                    salience=self.state.salience[order],
                    regimes=self.state.regimes[order],
                )

    def stats(self) -> Dict[str, float]:
        if self.state.size == 0:
            return {'size': 0.0, 'mean_salience': 0.0}
        return {
            'size': float(self.state.size),
            'mean_salience': float(self.state.salience.mean().item()),
        }
