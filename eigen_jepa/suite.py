from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from .benchmark import run as benchmark_run
from .utils import ensure_dir, set_seed


def _metric_mean(summary: Dict[str, object], variant: str, metric: str) -> float:
    return float(summary['aggregate'][variant][metric]['mean'])


def _write_markdown_table(summary: Dict[str, object], out_path: Path) -> None:
    variants = summary['variants']
    rows = ['| Style | Variant | Eig NMSE | Drift MSE | Tail F1 | Regime Acc |', '|---|---:|---:|---:|---:|---:|']
    for style, style_summary in summary['styles'].items():
        for variant in variants:
            agg = style_summary['aggregate'][variant]
            rows.append(
                f"| {style} | {variant} | {agg['eig_nmse']['mean']:.4f} ± {agg['eig_nmse']['std']:.4f} | "
                f"{agg['drift_mse']['mean']:.4f} ± {agg['drift_mse']['std']:.4f} | "
                f"{agg['tail_f1']['mean']:.4f} ± {agg['tail_f1']['std']:.4f} | "
                f"{agg['regime_acc']['mean']:.4f} ± {agg['regime_acc']['std']:.4f} |"
            )
    out_path.write_text('\n'.join(rows))


def _plot_style_comparison(summary: Dict[str, object], out_dir: Path) -> None:
    styles = list(summary['styles'].keys())
    variants = summary['variants']
    full_eig = [float(summary['styles'][s]['aggregate']['full']['eig_nmse']['mean']) for s in styles]
    full_tail = [float(summary['styles'][s]['aggregate']['full']['tail_f1']['mean']) for s in styles]
    full_gate = [float(summary['styles'][s]['aggregate']['full']['gate_cal']['mean']) for s in styles]

    x = list(range(len(styles)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar([i - width for i in x], full_eig, width=width, label='Eig NMSE')
    ax.bar(x, full_tail, width=width, label='Tail F1')
    ax.bar([i + width for i in x], full_gate, width=width, label='Gate Cal')
    ax.set_xticks(x)
    ax.set_xticklabels(styles)
    ax.set_xlabel('Market style')
    ax.set_title('Full-model comparison across market styles')
    ax.grid(True, axis='y', alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'style_comparison.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    for variant in variants:
        vals = [float(summary['styles'][s]['aggregate'][variant]['eig_nmse']['mean']) for s in styles]
        ax.plot(styles, vals, marker='o', label=variant)
    ax.set_title('Eigen-spectrum NMSE by ablation and market style')
    ax.set_xlabel('Market style')
    ax.set_ylabel('Eig NMSE')
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / 'ablation_styles.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def run(args) -> Dict[str, object]:
    set_seed(args.seed, deterministic=getattr(args, 'deterministic', False))
    styles = list(getattr(args, 'market_styles', ['equity', 'crypto', 'rates']))
    variants = list(getattr(args, 'variants', ['full', 'no_memory', 'no_gate', 'no_regime']))
    out_dir = ensure_dir(args.out_dir)

    style_summaries: Dict[str, object] = {}
    for idx, style in enumerate(styles):
        style_out = out_dir / style
        local = argparse.Namespace(**vars(args))
        local.out_dir = str(style_out)
        local.market_style = style
        local.seed = args.seed + idx * args.seed_stride
        summary = benchmark_run(local, variants=variants)
        style_summaries[style] = summary

    summary = {
        'seed': args.seed,
        'seed_stride': args.seed_stride,
        'num_seeds': args.num_seeds,
        'market_styles': styles,
        'variants': variants,
        'styles': style_summaries,
    }
    (out_dir / 'suite_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_markdown_table(summary, out_dir / 'suite_summary.md')
    _plot_style_comparison(summary, out_dir)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', type=str, default='results/suite')
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
    p.add_argument('--epochs', type=int, default=10)
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
    p.add_argument('--market_styles', type=str, nargs='*', default=['equity', 'crypto', 'rates'])
    args = p.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
