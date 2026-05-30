from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Sequence

import torch

from .train import run as train_run
from .plots import make_figures, write_results_table
from .utils import set_seed, ensure_dir


def _build_args(base: argparse.Namespace, variant: str, seed: int, out_dir: Path, market_style: str):
    args = deepcopy(base)
    args.variant = variant
    args.deterministic = getattr(base, 'deterministic', False)
    args.seed = seed
    args.out_dir = str(out_dir)
    args.market_style = market_style
    return args


def _aggregate(items: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = items[0].keys()
    out = {}
    for k in keys:
        vals = [float(x[k]) for x in items]
        out[k] = {'mean': mean(vals), 'std': pstdev(vals) if len(vals) > 1 else 0.0}
    return out


def _aggregate_nested(items: List[Dict[str, dict]]) -> Dict[str, dict]:
    out = {}
    keys = items[0].keys()
    for key in keys:
        first = items[0][key]
        if isinstance(first, dict) and first and isinstance(next(iter(first.values())), dict):
            out[key] = {}
            for subkey in first.keys():
                vals = [it[key][subkey] for it in items]
                out[key][subkey] = _aggregate(vals)
        elif isinstance(first, dict):
            vals = [it[key] for it in items]
            out[key] = _aggregate(vals)
        else:
            out[key] = {'mean': mean([float(it[key]) for it in items]), 'std': pstdev([float(it[key]) for it in items]) if len(items) > 1 else 0.0}
    return out


def run(args, variants: Sequence[str] | None = None) -> Dict[str, object]:
    set_seed(args.seed)
    torch.device(args.device)

    out_dir = ensure_dir(args.out_dir)
    variants = list(variants) if variants is not None else ['full', 'no_memory', 'no_gate', 'no_regime', 'no_spectral', 'no_subspace', 'no_mask']
    seed_list = [args.seed + i * args.seed_stride for i in range(args.num_seeds)]

    all_runs: Dict[str, List[Dict[str, float]]] = {v: [] for v in variants}
    all_bench: Dict[str, List[Dict[str, float]]] = {}
    representative = None

    for seed in seed_list:
        for variant in variants:
            run_dir = out_dir / f'seed_{seed}' / variant
            local = _build_args(args, variant, seed, run_dir, args.market_style)
            ckpt, model, memory, ds, results = train_run(local)
            all_runs[variant].append(results['test'])
            if 'benchmark' in results:
                for k, v in results['benchmark'].items():
                    all_bench.setdefault(k, []).append(v)
            if variant == 'full' and representative is None:
                representative = {
                    'results': results,
                    'ckpt': ckpt,
                    'model': model,
                    'memory': memory,
                    'ds': ds,
                    'run_dir': run_dir,
                }

    aggregated = {variant: _aggregate(items) for variant, items in all_runs.items()}
    benchmark_aggregate = {k: _aggregate(v) for k, v in all_bench.items()}

    summary = {
        'market_style': args.market_style,
        'num_seeds': args.num_seeds,
        'seed_list': seed_list,
        'variants': variants,
        'aggregate': aggregated,
        'benchmark': benchmark_aggregate,
        'representative_run': str(representative['run_dir']) if representative else None,
    }
    (out_dir / 'metrics.json').write_text(json.dumps(summary, indent=2, sort_keys=True))

    if representative is not None:
        src_paper = representative['run_dir'] / 'paper'
        paper_dir = out_dir / 'paper'
        paper_dir.mkdir(parents=True, exist_ok=True)
        for name in ['training_curve.png', 'validation_curve.png', 'metrics_table.png', 'spectral_example.png', 'memory_sweep.png', 'robustness_noise.png', 'walkforward.png', 'benchmark_ablation.tex', 'results_table.tex']:
            src = src_paper / name
            if src.exists():
                import shutil
                shutil.copy2(src, paper_dir / name)
        write_results_table(representative['results'], paper_dir / 'results_table.tex')
        make_figures(str(representative['run_dir']), str(paper_dir))

    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='results/benchmark')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--num_seeds', type=int, default=1)
    p.add_argument('--seed_stride', type=int, default=11)
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
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
