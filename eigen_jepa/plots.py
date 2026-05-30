from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def _fmt(x):
    try:
        return f"{float(x):.4f}"
    except Exception:
        return 'nan'


def _rows_from_metrics_block(name: str, block: dict):
    return [
        name,
        _fmt(block.get('eig_nmse')),
        _fmt(block.get('proj_mse')),
        _fmt(block.get('drift_mse')),
        _fmt(block.get('gap_rmse')),
        _fmt(block.get('cov_mse')),
        _fmt(block.get('gate_cal')),
        _fmt(block.get('tail_f1')),
        _fmt(block.get('regime_acc')),
    ]


def write_results_table(results: dict, out_path: str | Path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test = results.get('test', {})
    base = results.get('benchmark', {})
    lines = []
    lines.append(r'\begin{tabular}{lrrrrrrrr}')
    lines.append(r'\toprule')
    lines.append(r'Model & Eig NMSE & Proj MSE & Drift MSE & Gap RMSE & Cov MSE & Gate Cal & Tail F1 & Regime Acc \\')
    lines.append(r'\midrule')
    lines.append('Eigen-JEPA & ' + ' & '.join([
        _fmt(test.get('eig_nmse')),
        _fmt(test.get('proj_mse')),
        _fmt(test.get('drift_mse')),
        _fmt(test.get('gap_rmse')),
        _fmt(test.get('cov_mse')),
        _fmt(test.get('gate_cal')),
        _fmt(test.get('tail_f1')),
        _fmt(test.get('regime_acc')),
    ]) + r' \\')
    if base:
        lines.append('Persistence & ' + ' & '.join([
            _fmt(base.get('persistence', {}).get('eig_nmse')),
            _fmt(base.get('persistence', {}).get('proj_mse')),
            _fmt(base.get('persistence', {}).get('drift_mse')),
            _fmt(base.get('persistence', {}).get('gap_rmse')),
            _fmt(base.get('persistence', {}).get('cov_mse')),
            _fmt(base.get('persistence', {}).get('gate_cal')),
            _fmt(base.get('persistence', {}).get('tail_f1')),
            _fmt(base.get('persistence', {}).get('regime_acc')),
        ]) + r' \\')
        lines.append('Trend & ' + ' & '.join([
            _fmt(base.get('trend', {}).get('eig_nmse')),
            _fmt(base.get('trend', {}).get('proj_mse')),
            _fmt(base.get('trend', {}).get('drift_mse')),
            _fmt(base.get('trend', {}).get('gap_rmse')),
            _fmt(base.get('trend', {}).get('cov_mse')),
            _fmt(base.get('trend', {}).get('gate_cal')),
            _fmt(base.get('trend', {}).get('tail_f1')),
            _fmt(base.get('trend', {}).get('regime_acc')),
        ]) + r' \\')
    if 'memory' in results:
        lines.append(r'\midrule')
        mem = results['memory']
        lines.append(r'Memory size & \multicolumn{8}{r}{' + _fmt(mem.get('size')) + r'} \\')
        lines.append(r'Mean salience & \multicolumn{8}{r}{' + _fmt(mem.get('mean_salience')) + r'} \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    out_path.write_text('\n'.join(lines))

    ablate = results.get('ablations', {})
    ablate_path = out_path.parent / 'benchmark_ablation.tex'
    if ablate:
        ab_lines = []
        ab_lines.append(r'\begin{tabular}{lrrrrrrr}')
        ab_lines.append(r'\toprule')
        ab_lines.append(r'Variant & Eig NMSE & Proj MSE & Drift MSE & Cov MSE & Gate Cal & Tail F1 & Regime Acc \\')
        ab_lines.append(r'\midrule')
        for name, block in ablate.items():
            ab_lines.append(name.replace('_', ' ').title() + ' & ' + ' & '.join([
                _fmt(block.get('eig_nmse')),
                _fmt(block.get('proj_mse')),
                _fmt(block.get('drift_mse')),
                _fmt(block.get('cov_mse')),
                _fmt(block.get('gate_cal')),
                _fmt(block.get('tail_f1')),
                _fmt(block.get('regime_acc')),
            ]) + r' \\')
        ab_lines.append(r'\bottomrule')
        ab_lines.append(r'\end{tabular}')
        ablate_path.write_text('\n'.join(ab_lines))


def make_figures(run_dir: str, paper_dir: str):
    run = Path(run_dir)
    paper = Path(paper_dir)
    paper.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((run / 'metrics.json').read_text())
    hist = metrics.get('history', [])
    if hist:
        epochs = [h['epoch'] for h in hist]
        train_loss = [h['train_loss'] for h in hist]
        val_eig = [h['val_eig_nmse'] for h in hist]
        val_proj = [h['val_proj_mse'] for h in hist]
        val_drift = [h['val_drift_mse'] for h in hist]
        val_cov = [h['val_cov_mse'] for h in hist]
        val_tail = [h['val_tail_f1'] for h in hist]
        val_regime = [h['val_regime_acc'] for h in hist]

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(epochs, train_loss, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train loss')
        ax.set_title('Eigen-JEPA optimization')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(paper / 'training_curve.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(epochs, val_eig, marker='o')
        ax.plot(epochs, val_proj, marker='o')
        ax.plot(epochs, val_drift, marker='o')
        ax.plot(epochs, val_cov, marker='o')
        ax.plot(epochs, val_tail, marker='o')
        ax.plot(epochs, val_regime, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation metric')
        ax.set_title('Validation traces')
        ax.legend(['Eig NMSE', 'Proj MSE', 'Drift MSE', 'Cov MSE', 'Tail F1', 'Regime Acc'])
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(paper / 'validation_curve.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 3.4))
    ax.axis('off')
    rows = []
    if 'benchmark' in metrics:
        bench = metrics['benchmark']
        for name, vals in bench.items():
            rows.append([
                name,
                _fmt(vals['eig_nmse']),
                _fmt(vals['proj_mse']),
                _fmt(vals['drift_mse']),
                _fmt(vals['gap_rmse']),
                _fmt(vals.get('cov_mse')),
                _fmt(vals.get('gate_cal')),
                _fmt(vals.get('tail_f1', float('nan'))),
                _fmt(vals.get('regime_acc', float('nan'))),
            ])
    else:
        test = metrics.get('test', {})
        rows.append(['Eigen-JEPA', _fmt(test.get('eig_nmse')), _fmt(test.get('proj_mse')), _fmt(test.get('drift_mse')), _fmt(test.get('gap_rmse')), _fmt(test.get('cov_mse')), _fmt(test.get('gate_cal')), _fmt(test.get('tail_f1')), _fmt(test.get('regime_acc'))])

    tbl = ax.table(
        cellText=rows,
        colLabels=['Model', 'Eig NMSE', 'Proj MSE', 'Drift MSE', 'Gap RMSE', 'Cov MSE', 'Gate Cal', 'Tail F1', 'Regime Acc'],
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.6)
    tbl.scale(1, 1.35)
    ax.set_title('Main benchmark table')
    fig.tight_layout()
    fig.savefig(paper / 'metrics_table.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    sweep = metrics.get('robustness', {}).get('memory', [])
    if sweep:
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        sizes = [r['memory_scale'] for r in sweep]
        ax.plot(sizes, [r['drift_mse'] for r in sweep], marker='o')
        ax.plot(sizes, [r['eig_nmse'] for r in sweep], marker='o')
        ax.plot(sizes, [r['tail_f1'] for r in sweep], marker='o')
        ax.plot(sizes, [r['cov_mse'] for r in sweep], marker='o')
        ax.set_xlabel('Memory scale')
        ax.set_ylabel('Metric')
        ax.set_title('Memory robustness sweep')
        ax.legend(['Drift MSE', 'Eig NMSE', 'Tail F1', 'Cov MSE'])
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(paper / 'memory_sweep.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    noise = metrics.get('robustness', {}).get('noise', [])
    if noise:
        fig, ax = plt.subplots(figsize=(7.2, 4.3))
        xs = [r['noise_std'] for r in noise]
        ax.plot(xs, [r['eig_nmse'] for r in noise], marker='o')
        ax.plot(xs, [r['cov_mse'] for r in noise], marker='o')
        ax.plot(xs, [r['tail_f1'] for r in noise], marker='o')
        ax.set_xlabel('Input noise std')
        ax.set_ylabel('Metric')
        ax.set_title('Noise robustness sweep')
        ax.legend(['Eig NMSE', 'Cov MSE', 'Tail F1'])
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(paper / 'robustness_noise.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    walk = metrics.get('walkforward', {})
    if walk:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        blocks = ['early', 'mid', 'late']
        eig = [walk.get(b, {}).get('eig_nmse', float('nan')) for b in blocks]
        drift = [walk.get(b, {}).get('drift_mse', float('nan')) for b in blocks]
        cov = [walk.get(b, {}).get('cov_mse', float('nan')) for b in blocks]
        x = list(range(len(blocks)))
        ax.plot(x, eig, marker='o')
        ax.plot(x, drift, marker='o')
        ax.plot(x, cov, marker='o')
        ax.set_xticks(x)
        ax.set_xticklabels(blocks)
        ax.set_title('Walk-forward slices')
        ax.set_xlabel('Test segment')
        ax.set_ylabel('Metric')
        ax.legend(['Eig NMSE', 'Drift MSE', 'Cov MSE'])
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(paper / 'walkforward.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

    eval_fig = run / 'eval' / 'spectral_example.png'
    if eval_fig.exists():
        import shutil
        shutil.copy2(eval_fig, paper / 'spectral_example.png')

    if (run / 'eval' / 'memory_sweep.png').exists():
        import shutil
        shutil.copy2(run / 'eval' / 'memory_sweep.png', paper / 'memory_sweep.png')
    if (run / 'eval' / 'robustness_noise.png').exists():
        import shutil
        shutil.copy2(run / 'eval' / 'robustness_noise.png', paper / 'robustness_noise.png')
