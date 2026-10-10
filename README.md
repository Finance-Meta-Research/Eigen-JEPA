# Eigen-JEPA

Eigen-JEPA is a spectral joint-embedding predictive architecture for financial world modeling. The central idea is to forecast the future geometry of a market — covariance spectra, dominant eigenspaces, eigengaps, subspace drift, and regime transitions — rather than only pointwise returns.

This repository is a complete research package:

- synthetic regime-switching market generator
- lightweight temporal + spectral encoder
- selective memory and gating pathway
- tail-aware training objective
- baseline and ablation evaluation
- memory-budget sweep
- multi-style benchmark suite
- walk-forward and robustness diagnostics
- NeurIPS-style paper source and compiled PDF
- deterministic smoke-test and reproducibility tooling

## What is included

- `eigen_jepa/`: model, data, losses, evaluation, benchmark logic
- `paper/`: LaTeX source, refined style file, figures, compiled PDF
- `results/`: run artifacts from the packaged experiments
- `configs/`: reproducible run settings
- `scripts/`: convenience wrappers for the full suite and paper build
- `tests/`: smoke tests

## Quick start

### 1) Install

```bash
pip install -e .
```

### 2) Smoke test

```bash
python -m eigen_jepa.train --deterministic --device cpu \
  --epochs 1 --num_train 16 --num_val 8 --num_test 8 \
  --num_assets 6 --total_steps 80 --context_len 10 --horizon 4 \
  --batch_size 4 --d_model 24 --latent_dim 24 --memory_size 16 --memory_top_k 2
```

### 3) Full benchmark for one market style

```bash
python -m eigen_jepa.benchmark --deterministic --device cpu --num_seeds 3 --out_dir results/benchmark
```

### 4) Full multi-style suite

```bash
python -m eigen_jepa.suite --deterministic --device cpu --num_seeds 2 --out_dir results/suite
```

### 5) Rebuild paper

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

## Output artifacts

Each run writes:

- `checkpoints/best.pt`
- `metrics.json`
- `paper/results_table.tex`
- `paper/figures/*.png`
- `eval/metrics.json`
- `eval/spectral_example.png`
- `eval/memory_sweep.json`
- `eval/test_records.pt`

The suite runner additionally writes:

- `suite_summary.json`
- `suite_summary.md`
- `style_comparison.png`
- `ablation_styles.png`

## Reproducibility notes

- The packaged experiments use deterministic seeds.
- Train/test splits are chronological.
- Evaluation includes robustness and walk-forward slices.
- The benchmark is synthetic by design, so the package is a research baseline, not a claim of trading alpha.

## Publication checklist

Before submission, verify that you can:

- rerun the smoke test from a clean environment,
- rebuild the paper PDF from `paper/main.tex`,
- reproduce the benchmark tables and figures,
- run the ablation suite,
- and inspect the saved metrics under `results/`.

## Prospective checkpoint evaluation

The opt-in `eigen_jepa.checkpoint_evaluation_v3` entry point restores the saved
model weights, memory configuration and ablation inference policy, then evaluates
every window in an explicitly materialized development snapshot. It uses the
prospective covariance-geometry v3/cohort-v2 definitions and retains checkpoint,
input and source hashes in the report.

```bash
python -m eigen_jepa.checkpoint_evaluation_v3 \
  --checkpoint checkpoint.pt --snapshot development-windows.pt \
  --output development-evaluation.json --batch-size 32 --device cpu
```

See [snapshot creation, artifact contracts and verification](docs/development/checkpoint-policy-replay-v3-20261010.md).
The output path must be new. The mixed/negative predecessor remains closed;
this development entry point does not authorize a protected evaluation or
redefine retained scientific results. It requires complete saved policy metadata
and the existing dense float32 checkpoint/window format.

## Citation

A `CITATION.cff` file is included for convenience.

## License

Choose and add a license before public release if you plan to distribute the repository broadly.
