from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import MarketConfig, build_datasets, batch_collate
from .eval import evaluate_model, baseline_metrics, robustness_sweep, slice_metrics
from .losses import spectral_jepa_loss, LossWeights
from .memory import SpectralMemory
from .model import EigenJEPA, EigenJEPAConfig
from .plots import make_figures, write_results_table
from .spectral import rolling_covariance, topk_spectrum, projector_from_vecs, spectral_metrics, low_rank_covariance, portfolio_risk
from .utils import set_seed, save_json, build_generator, seed_worker, ensure_dir


def _dataloader(dataset, batch_size: int, shuffle: bool, seed: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=batch_collate,
        generator=build_generator(seed),
        worker_init_fn=seed_worker,
    )


def _class_weights(labels: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float().clamp_min(1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights


def _aggregate_baseline(loader, model, device):
    agg = {}
    totals = {}
    total = 0
    for batch in loader:
        b = baseline_metrics(batch, device)
        n = batch['event_true'].shape[0]
        total += n
        for method, metrics in b.items():
            agg.setdefault(method, {})
            totals.setdefault(method, 0)
            totals[method] += n
            for k, v in metrics.items():
                agg[method].setdefault(k, 0.0)
                agg[method][k] += float(v) * n
    for method, metrics in agg.items():
        denom = max(totals.get(method, 1), 1)
        for k in list(metrics.keys()):
            metrics[k] /= denom
    return agg


def run(args):
    set_seed(args.seed, deterministic=getattr(args, 'deterministic', False))
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    mask_ratio = 0.0 if args.variant == 'no_mask' else args.mask_ratio
    cfg = MarketConfig(
        num_assets=args.num_assets,
        total_steps=args.total_steps,
        context_len=args.context_len,
        horizon=args.horizon,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed,
        market_style=args.market_style,
        event_quantile=args.event_quantile,
        mask_ratio=mask_ratio,
        block_time=args.block_time,
        data_source=args.data_source,
        csv_path=args.csv_path,
        return_cols=tuple(args.return_cols) if args.return_cols else (),
        aux_cols=tuple(args.aux_cols) if args.aux_cols else (),
        date_col=args.date_col,
    )
    ds = build_datasets(cfg, k=args.k)
    train_loader = _dataloader(ds['train'], args.batch_size, True, args.seed)
    val_loader = _dataloader(ds['val'], args.batch_size, False, args.seed + 1)
    test_loader = _dataloader(ds['test'], args.batch_size, False, args.seed + 2)

    input_dim = args.num_assets + 4
    target_dim = args.k + args.num_assets * args.num_assets + 1 + 3
    model_cfg = EigenJEPAConfig(
        input_dim=input_dim,
        num_assets=args.num_assets,
        context_len=args.context_len,
        k=args.k,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        latent_dim=args.latent_dim,
        memory_dim=args.latent_dim,
    )
    model = EigenJEPA(model_cfg, target_dim=target_dim).to(args.device)
    memory = SpectralMemory(
        key_dim=args.latent_dim,
        value_dim=args.latent_dim,
        max_items=args.memory_size,
        top_k=args.memory_top_k,
        temperature=args.memory_temperature,
        merge_radius=args.merge_radius,
        min_salience=args.min_salience,
        device=args.device,
    )
    if args.variant == 'no_memory':
        memory.min_salience = 10.0
    if args.variant == 'no_regime':
        args.regime_weight = 0.0
    if args.variant == 'no_spectral':
        args.eig_weight = 0.0
        args.subspace_weight = 0.0
        args.drift_weight = 0.0
        args.risk_weight = 0.0
        args.entropy_weight = 0.0
        args.rank_weight = 0.0
    if args.variant == 'no_subspace':
        args.subspace_weight = 0.0
    if args.variant == 'no_tailweight':
        args.tail_boost = 0.0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_dir = ensure_dir(args.out_dir)
    ckpt_dir = ensure_dir(out_dir / 'checkpoints')
    ensure_dir(out_dir / 'eval')

    history = []
    best_val = float('inf')
    best_path = ckpt_dir / 'best.pt'

    class_w = _class_weights(torch.tensor([ds['train'][i]['regime_true'].item() for i in range(len(ds['train']))]), num_classes=3).to(args.device)
    weights = LossWeights(
        regime=args.regime_weight,
        gate=args.gate_weight,
        tail_boost=args.tail_boost,
        eig=args.eig_weight,
        subspace=args.subspace_weight,
        drift=args.drift_weight,
        risk=args.risk_weight,
        entropy=args.entropy_weight,
        rank=args.rank_weight,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in train_loader:
            x = batch['x'].to(args.device)
            pred = model(
                x,
                memory=memory if args.variant != 'no_memory' else None,
                gate_override=1.0 if args.variant == 'no_gate' else None,
            )
            target_latent = model.target_latent(batch['target_vec'].to(args.device))
            loss, parts = spectral_jepa_loss(
                pred,
                target_latent,
                batch['evals_true'].to(args.device),
                batch['proj_true'].to(args.device),
                batch['drift_true'].to(args.device),
                batch['regime_true'].to(args.device),
                batch['event_true'].to(args.device),
                batch['risk_true'].to(args.device),
                batch['entropy_true'].to(args.device),
                batch['rank_true'].to(args.device),
                weights=weights,
                class_weights=class_w,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running.append(parts)
            if args.variant != 'no_memory':
                salience = 0.75 * batch['salience_true'].to(args.device) + 0.25 * batch['event_true'].to(args.device) + 0.25 * batch['drift_true'].to(args.device).float().clamp(max=1.5) / 1.5
                memory.write(pred['query'].detach(), target_latent.detach(), salience.detach(), batch['regime_true'].to(args.device))

        sched.step()
        val_memory = memory if args.variant != 'no_memory' else None
        val_metrics, _, _ = evaluate_model(model, val_loader, args.device, memory=val_memory, gate_override=1.0 if args.variant == 'no_gate' else None)
        val_score = val_metrics['eig_nmse'] + val_metrics['proj_mse'] + val_metrics['drift_mse'] + val_metrics['gate_cal'] + (1.0 - val_metrics['regime_bal_acc'])
        train_loss = float(np.mean([r['loss'] for r in running]))
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_eig_nmse': val_metrics['eig_nmse'],
            'val_proj_mse': val_metrics['proj_mse'],
            'val_drift_mse': val_metrics['drift_mse'],
            'val_cov_mse': val_metrics.get('cov_mse', float('nan')),
            'val_gate_cal': val_metrics['gate_cal'],
            'val_regime_acc': val_metrics['regime_acc'],
            'val_tail_f1': val_metrics['tail_f1'],
        })
        if val_score < best_val:
            best_val = val_score
            torch.save({
                'model': model.state_dict(),
                'cfg': asdict(model_cfg),
                'data_cfg': asdict(cfg),
                'target_dim': target_dim,
                'history': history,
                'args': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                'memory_state': memory.state_dict(),
                'variant': args.variant,
            }, best_path)
        print(f"epoch={epoch:03d} train={train_loss:.4f} val_eig_nmse={val_metrics['eig_nmse']:.4f} val_drift_mse={val_metrics['drift_mse']:.4f} tail_f1={val_metrics['tail_f1']:.3f}")

    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt['model'])
    memory.load_state_dict(ckpt['memory_state'])
    test_metrics, test_examples, records = evaluate_model(
        model,
        test_loader,
        args.device,
        memory=memory if args.variant != 'no_memory' else None,
        gate_override=1.0 if args.variant == 'no_gate' else None,
    )

    benchmark = _aggregate_baseline(test_loader, model, args.device)
    robustness = robustness_sweep(model, test_loader, args.device, memory=memory if args.variant != 'no_memory' else None)
    walkforward = {
        'early': slice_metrics(records, 0.0, 0.33),
        'mid': slice_metrics(records, 0.33, 0.67),
        'late': slice_metrics(records, 0.67, 1.0),
    }
    ablations = {
        'memory_off': evaluate_model(model, test_loader, args.device, memory=None, gate_override=1.0 if args.variant == 'no_gate' else None)[0],
        'gate_off': evaluate_model(model, test_loader, args.device, memory=memory if args.variant != 'no_memory' else None, gate_override=0.0)[0],
        'memory_half': evaluate_model(model, test_loader, args.device, memory=memory if args.variant != 'no_memory' else None, memory_scale=0.5, gate_override=1.0 if args.variant == 'no_gate' else None)[0],
    }

    results = {
        'test': test_metrics,
        'benchmark': benchmark,
        'ablations': ablations,
        'robustness': robustness,
        'walkforward': walkforward,
        'history': history,
        'memory': memory.stats(),
        'event_threshold': ds['event_threshold'],
    }
    save_json(results, out_dir / 'metrics.json')
    torch.save({'examples': test_examples, 'records': records}, out_dir / 'test_records.pt')
    write_results_table(results, out_dir / 'paper' / 'results_table.tex')
    make_figures(str(out_dir), str(out_dir / 'paper'))
    return ckpt, model, memory, ds, results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='results/run1')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--num_assets', type=int, default=12)
    p.add_argument('--total_steps', type=int, default=240)
    p.add_argument('--context_len', type=int, default=20)
    p.add_argument('--horizon', type=int, default=6)
    p.add_argument('--num_train', type=int, default=140)
    p.add_argument('--num_val', type=int, default=36)
    p.add_argument('--num_test', type=int, default=36)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=8e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--d_model', type=int, default=64)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.10)
    p.add_argument('--latent_dim', type=int, default=64)
    p.add_argument('--memory_size', type=int, default=128)
    p.add_argument('--memory_top_k', type=int, default=4)
    p.add_argument('--memory_temperature', type=float, default=0.18)
    p.add_argument('--merge_radius', type=float, default=0.16)
    p.add_argument('--min_salience', type=float, default=0.20)
    p.add_argument('--variant', type=str, default='full', choices=['full', 'no_memory', 'no_gate', 'no_regime', 'no_spectral', 'no_subspace', 'no_mask', 'no_tailweight'])
    p.add_argument('--market_style', type=str, default='equity', choices=['equity', 'crypto', 'rates'])
    p.add_argument('--regime_weight', type=float, default=0.75)
    p.add_argument('--gate_weight', type=float, default=0.05)
    p.add_argument('--tail_boost', type=float, default=1.5)
    p.add_argument('--event_quantile', type=float, default=0.70)
    p.add_argument('--mask_ratio', type=float, default=0.25)
    p.add_argument('--block_time', type=int, default=4)
    p.add_argument('--data_source', type=str, default='synthetic', choices=['synthetic', 'csv'])
    p.add_argument('--csv_path', type=str, default=None)
    p.add_argument('--return_cols', type=str, nargs='*', default=[])
    p.add_argument('--aux_cols', type=str, nargs='*', default=[])
    p.add_argument('--date_col', type=str, default=None)
    p.add_argument('--eig_weight', type=float, default=1.0)
    p.add_argument('--subspace_weight', type=float, default=2.0)
    p.add_argument('--drift_weight', type=float, default=0.5)
    p.add_argument('--risk_weight', type=float, default=0.25)
    p.add_argument('--entropy_weight', type=float, default=0.15)
    p.add_argument('--rank_weight', type=float, default=0.15)
    args = p.parse_args()
    args.device = torch.device(args.device)
    run(args)


if __name__ == '__main__':
    main()
