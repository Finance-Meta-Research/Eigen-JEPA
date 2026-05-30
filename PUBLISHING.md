# Eigen-JEPA Publishing Notes

This project is designed as a research package for a paper submission.

## What is ready

- complete codebase
- reproducibility scripts
- benchmark and ablation plumbing
- synthetic regime-switching evaluation
- walk-forward and robustness diagnostics
- compiled paper and LaTeX source

## What the paper should claim

The strongest defensible claim is that Eigen-JEPA learns and predicts market geometry more effectively than direct pointwise prediction on the synthetic and packaged benchmark suite.

Do **not** claim trading alpha unless you have external evidence.

## Final checks before release

1. Run the smoke test.
2. Rebuild the paper.
3. Confirm all figures appear in `paper/figures/`.
4. Check that `results/` contains the benchmark outputs used in the paper.
5. Verify the `README.md` quick-start commands still work.
6. Make sure the final archive contains the compiled PDF.

## What reviewers will likely care about

- chronological split discipline
- ablation clarity
- calibration and tail metrics
- robustness to corruption
- whether the geometry-focused framing is supported by the results
- whether the implementation is easy to reproduce
