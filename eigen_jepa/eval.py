from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .data import MarketConfig, build_datasets, batch_collate
from .memory import SpectralMemory
from .model import EigenJEPA, EigenJEPAConfig
from .spectral import (
    spectral_metrics,
    rolling_covariance,
    topk_spectrum,
    projector_from_vecs,
    low_rank_covariance,
    portfolio_risk,
    principal_angles,
    spectral_entropy,
    effective_rank,
)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def _f1_from_probs(probs: torch.Tensor, labels: torch.Tensor, thr: float = 0.45) -> float:
    pred = (probs >= thr).float()
    tp = float((pred * labels).sum().item())
    fp = float((pred * (1.0 - labels)).sum().item())
    fn = float(((1.0 - pred) * labels).sum().item())
    precision = tp / max(tp + fp, 1e-6)
    recall = tp / max(tp + fn, 1e-6)
    return 2 * precision * recall / max(precision + recall, 1e-6)


def _balanced_accuracy(pred_labels: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> float:
    recalls = []
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            recalls.append(float((pred_labels[mask] == c).float().mean().item()))
    return float(np.mean(recalls)) if recalls else 0.0


def _threshold_regime_from_context(batch) -> torch.Tensor:
    stress = batch['returns_ctx'].abs().mean(dim=(1,2)) + batch['returns_ctx'].std(dim=(1,2))
    q1, q2 = torch.quantile(stress, torch.tensor([0.55, 0.82], dtype=stress.dtype, device=stress.device))
    out = torch.zeros_like(batch['regime_true'])
    out[stress >= q1] = 1
    out[stress >= q2] = 2
    return out


def _baseline_predictions(batch) -> Dict[str, torch.Tensor]:
    ctx_r = batch['returns_ctx']
    fut_r = batch['returns_fut']
    cov_c = rolling_covariance(ctx_r)
    evals_c, evecs_c = topk_spectrum(cov_c, batch['evals_true'].shape[-1])
    proj_c = projector_from_vecs(evecs_c)

    half = max(2, ctx_r.shape[1] // 2)
    cov_a = rolling_covariance(ctx_r[:, :half])
    cov_b = rolling_covariance(ctx_r[:, half:])
    evals_a, evecs_a = topk_spectrum(cov_a, batch['evals_true'].shape[-1])
    evals_b, evecs_b = topk_spectrum(cov_b, batch['evals_true'].shape[-1])
    trend = (evals_b - evals_a).clamp_min(-0.75)
    pred_evals_trend = (evals_b + trend).clamp_min(1e-4)
    pred_proj_trend = projector_from_vecs(evecs_b)

    return {
        'persistence': {
            'eig': evals_c,
            'proj': proj_c,
            'evecs': evecs_c,
            'drift': principal_angles(evecs_c, evecs_c),
            'risk': portfolio_risk(low_rank_covariance(evals_c, evecs_c)),
            'entropy': spectral_entropy(evals_c),
            'rank': effective_rank(evals_c),
            'regime_logits': torch.zeros(batch['regime_true'].shape[0], 3, device=ctx_r.device),
            'gate': torch.clamp((ctx_r.abs().mean(dim=(1,2)) + ctx_r.std(dim=(1,2))), 0.0, 1.0),
        },
        'trend': {
            'eig': pred_evals_trend,
            'proj': pred_proj_trend,
            'evecs': evecs_b,
            'drift': principal_angles(evecs_a, evecs_b),
            'risk': portfolio_risk(low_rank_covariance(pred_evals_trend, evecs_b)),
            'entropy': spectral_entropy(pred_evals_trend),
            'rank': effective_rank(pred_evals_trend),
            'regime_logits': torch.stack([
                2.0 - batch['returns_ctx'].abs().mean(dim=(1,2)),
                batch['returns_ctx'].std(dim=(1,2)),
                batch['returns_ctx'].abs().mean(dim=(1,2)) + batch['returns_ctx'].std(dim=(1,2)),
            ], dim=-1),
            'gate': torch.sigmoid(1.5 * (batch['returns_ctx'].std(dim=(1,2)) + batch['returns_ctx'].abs().mean(dim=(1,2)))),
        },
    }


def _metrics_for_batch(model, batch, device, memory, gate_override: Optional[float] = None, memory_scale: float = 1.0, input_noise_std: float = 0.0):
    x = batch['x'].to(device)
    if input_noise_std > 0:
        x = x + input_noise_std * torch.randn_like(x)
    pred = model(x, memory=memory, gate_override=gate_override, memory_scale=memory_scale)
    true_cov = batch['cov_true'].to(device)
    pred_cov = low_rank_covariance(pred['eig'], pred['sub'])
    true_entropy = batch['entropy_true'].to(device)
    true_rank = batch['rank_true'].to(device)
    true_risk = batch['risk_true'].to(device)
    metrics = spectral_metrics(
        pred['eig'],
        batch['evals_true'].to(device),
        pred['proj'],
        batch['proj_true'].to(device),
        pred['drift'],
        batch['drift_true'].to(device),
        pred_cov=pred_cov,
        true_cov=true_cov,
        pred_risk=pred['risk'],
        true_risk=true_risk,
        pred_entropy=pred['entropy'],
        true_entropy=true_entropy,
        pred_rank=pred['rank'],
        true_rank=true_rank,
    )
    gate = pred['gate'].detach().cpu()
    event = batch['event_true'].cpu()
    regime = batch['regime_true'].cpu()
    lat_align = torch.mean((pred['z_hat'] - model.target_latent(batch['target_vec'].to(device))) ** 2).detach().cpu()
    pred_regime = pred['regime_logits'].argmax(dim=-1).detach().cpu()
    metrics.update({
        'lat_align': float(lat_align),
        'gate_cal': float(torch.mean(torch.abs(gate - event)).item()),
        'gate_mean': float(gate.mean().item()),
        'event_rate': float(event.mean().item()),
        'regime_acc': float((pred_regime == regime).float().mean().item()),
        'regime_bal_acc': _balanced_accuracy(pred_regime, regime),
        'tail_f1': float(_f1_from_probs(gate, event)),
    })
    rare_mask = event.bool()
    if rare_mask.any():
        metrics['rare_eig_nmse'] = float(torch.mean((pred['eig'].detach().cpu()[rare_mask] - batch['evals_true'][rare_mask]) ** 2) / (torch.mean(batch['evals_true'][rare_mask] ** 2) + 1e-12))
        metrics['rare_drift_mse'] = float(torch.mean((pred['drift'].detach().cpu()[rare_mask] - batch['drift_true'][rare_mask]) ** 2).item())
        metrics['rare_cov_mse'] = float(torch.mean((pred_cov.detach().cpu()[rare_mask] - true_cov[rare_mask].cpu()) ** 2).item())
    else:
        metrics['rare_eig_nmse'] = float('nan')
        metrics['rare_drift_mse'] = float('nan')
        metrics['rare_cov_mse'] = float('nan')
    return metrics, pred


@torch.no_grad()
def evaluate_model(model, loader, device, memory=None, gate_override: Optional[float] = None, memory_scale: float = 1.0, input_noise_std: float = 0.0):
    model.eval()
    metrics_acc = []
    examples = []
    records = []
    for batch in loader:
        metrics, pred = _metrics_for_batch(model, batch, device, memory, gate_override=gate_override, memory_scale=memory_scale, input_noise_std=input_noise_std)
        metrics_acc.append(metrics)
        if not examples:
            examples.append((batch, pred))
        for i in range(batch['event_true'].shape[0]):
            rec = {k: float(metrics[k]) if isinstance(metrics[k], (int, float, np.floating)) else metrics[k] for k in metrics.keys()}
            rec['window_index'] = int(batch['window_index'][i].item())
            rec['event_true'] = float(batch['event_true'][i].item())
            rec['regime_true'] = int(batch['regime_true'][i].item())
            rec['gate'] = float(pred['gate'][i].detach().cpu().item())
            rec['pred_regime'] = int(pred['regime_logits'][i].argmax(dim=-1).detach().cpu().item())
            records.append(rec)
    out = {}
    for k in metrics_acc[0].keys():
        vals = [m[k] for m in metrics_acc if not (isinstance(m[k], float) and np.isnan(m[k]))]
        out[k] = float(np.mean(vals)) if vals else float('nan')
    return out, examples, records


def _heuristic_gate(batch) -> torch.Tensor:
    ctx = batch['returns_ctx']
    vol = ctx.abs().mean(dim=(1, 2))
    spread = ctx.std(dim=(1, 2))
    stress = vol + spread
    stress = stress / (stress.mean().clamp_min(1e-6) + 1e-6)
    return torch.sigmoid(1.25 * (stress - 1.0))


@torch.no_grad()
def baseline_metrics(batch, device):
    bases = _baseline_predictions(batch)
    out = {}
    true_cov = batch['cov_true']
    for name, pred in bases.items():
        pred_cov = low_rank_covariance(pred['eig'], pred['evecs'])
        m = spectral_metrics(
            pred['eig'],
            batch['evals_true'],
            pred['proj'],
            batch['proj_true'],
            pred['drift'],
            batch['drift_true'],
            pred_cov=pred_cov,
            true_cov=true_cov,
            pred_risk=pred['risk'],
            true_risk=batch['risk_true'],
            pred_entropy=pred['entropy'],
            true_entropy=batch['entropy_true'],
            pred_rank=pred['rank'],
            true_rank=batch['rank_true'],
        )
        gate = _heuristic_gate(batch)
        event = batch['event_true']
        regime = batch['regime_true']
        pred_regime = torch.argmax(pred['regime_logits'], dim=-1)
        m.update({
            'gate_cal': float(torch.mean(torch.abs(gate - event)).item()),
            'gate_mean': float(gate.mean().item()),
            'event_rate': float(event.mean().item()),
            'regime_acc': float((pred_regime == regime).float().mean().item()),
            'regime_bal_acc': _balanced_accuracy(pred_regime, regime),
            'tail_f1': float(_f1_from_probs(gate, event)),
        })
        out[name] = m
    return out


@torch.no_grad()
def robustness_sweep(model, loader, device, memory=None, noise_levels: List[float] | None = None, memory_scales: List[float] | None = None):
    noise_levels = list(noise_levels) if noise_levels is not None else [0.0, 0.03, 0.06, 0.10]
    memory_scales = list(memory_scales) if memory_scales is not None else [1.0, 0.75, 0.5, 0.0]
    noise_results = []
    mem_results = []
    for nstd in noise_levels:
        metrics, _, _ = evaluate_model(model, loader, device, memory=memory, input_noise_std=nstd)
        metrics['noise_std'] = float(nstd)
        noise_results.append(metrics)
    for scale in memory_scales:
        metrics, _, _ = evaluate_model(model, loader, device, memory=memory, memory_scale=scale)
        metrics['memory_scale'] = float(scale)
        mem_results.append(metrics)
    return {'noise': noise_results, 'memory': mem_results}


def slice_metrics(records: List[Dict[str, float]], lo: float, hi: float):
    if not records:
        return {}
    idx = np.asarray([r['window_index'] for r in records], dtype=np.float32)
    tmin, tmax = idx.min(), idx.max()
    a = tmin + lo * (tmax - tmin)
    b = tmin + hi * (tmax - tmin)
    sel = [r for r in records if a <= r['window_index'] <= b]
    if not sel:
        return {}
    keys = [k for k in sel[0].keys() if isinstance(sel[0][k], (int, float, np.floating))]
    return {k: float(np.mean([r[k] for r in sel])) for k in keys}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--out_dir', type=str, default='results/eval')
    p.add_argument('--device', type=str, default='cpu')
    args = p.parse_args()
    device = torch.device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    mcfg = ckpt['cfg']
    dcfg = ckpt['data_cfg']
    cfg = MarketConfig(**dcfg)
    ds = build_datasets(cfg, k=mcfg['k'])
    loader = DataLoader(ds['test'], batch_size=32, shuffle=False, collate_fn=batch_collate)

    model = EigenJEPA(EigenJEPAConfig(**mcfg), target_dim=ckpt['target_dim']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    memory = SpectralMemory(key_dim=mcfg['memory_dim'], value_dim=mcfg['latent_dim'], max_items=int(ckpt['memory_state']['meta'][2].item()), top_k=int(ckpt['memory_state']['meta'][3].item()), device=device)
    memory.load_state_dict(ckpt['memory_state'])

    out, examples, records = evaluate_model(model, loader, device, memory=memory)
    baseline = baseline_metrics(next(iter(loader)), device)
    robustness = robustness_sweep(model, loader, device, memory=memory)
    early = slice_metrics(records, 0.0, 0.33)
    late = slice_metrics(records, 0.67, 1.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        'test': out,
        'benchmark': baseline,
        'robustness': robustness,
        'walkforward': {'early': early, 'late': late},
        'event_threshold': float(ds['event_threshold']),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True))
    torch.save({'examples': examples, 'records': records}, out_dir / 'test_records.pt')

    batch, pred = examples[0]
    true_e = batch['evals_true'][0].cpu().numpy()
    pred_e = pred['eig'][0].detach().cpu().numpy()
    true_p = batch['proj_true'][0].cpu().numpy()
    pred_p = pred['proj'][0].detach().cpu().numpy()
    true_cov = batch['cov_true'][0].cpu().numpy()
    pred_cov = low_rank_covariance(pred['eig'][0].unsqueeze(0), pred['sub'][0].unsqueeze(0))[0].detach().cpu().numpy()

    fig, axs = plt.subplots(1, 4, figsize=(15.5, 3.8))
    axs[0].plot(true_e, marker='o')
    axs[0].plot(pred_e, marker='o')
    axs[0].set_title('Top eigenvalues')
    axs[0].legend(['true', 'pred'])
    im = axs[1].imshow(true_p, aspect='auto')
    axs[1].set_title('True projector')
    plt.colorbar(im, ax=axs[1], fraction=0.046)
    im2 = axs[2].imshow(pred_p, aspect='auto')
    axs[2].set_title('Predicted projector')
    plt.colorbar(im2, ax=axs[2], fraction=0.046)
    im3 = axs[3].imshow(np.abs(true_cov - pred_cov), aspect='auto')
    axs[3].set_title('|Cov error|')
    plt.colorbar(im3, ax=axs[3], fraction=0.046)
    plt.tight_layout()
    fig.savefig(out_dir / 'spectral_example.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    (out_dir / 'memory_sweep.json').write_text(json.dumps(robustness['memory'], indent=2, sort_keys=True))
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    sizes = [r['memory_scale'] for r in robustness['memory']]
    ax.plot(sizes, [r['drift_mse'] for r in robustness['memory']], marker='o')
    ax.plot(sizes, [r['eig_nmse'] for r in robustness['memory']], marker='o')
    ax.plot(sizes, [r['tail_f1'] for r in robustness['memory']], marker='o')
    ax.set_xlabel('Memory scale')
    ax.set_ylabel('Metric')
    ax.set_title('Memory robustness sweep')
    ax.legend(['Drift MSE', 'Eig NMSE', 'Tail F1'])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'memory_sweep.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    xs = [r['noise_std'] for r in robustness['noise']]
    ax.plot(xs, [r['eig_nmse'] for r in robustness['noise']], marker='o')
    ax.plot(xs, [r['cov_mse'] for r in robustness['noise']], marker='o')
    ax.plot(xs, [r['tail_f1'] for r in robustness['noise']], marker='o')
    ax.set_xlabel('Input noise std')
    ax.set_ylabel('Metric')
    ax.set_title('Noise robustness sweep')
    ax.legend(['Eig NMSE', 'Cov MSE', 'Tail F1'])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / 'robustness_noise.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    print(json.dumps(metrics['test'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
