# Development record: cohort evaluation v2

Date: 2026-10-10. Base: `cf3a1bd29c91c637c63dacf6813f801d8f76c337`.
Disposition: the retained mixed/negative predecessor remains closed.

## Defects found in the legacy evaluator

`eval.evaluate_model` averages batch NMSE, gap RMSE, balanced accuracy and F1.
These nonlinear reductions depend on how windows are partitioned into batches;
even additive errors give a short final batch excess weight. The evaluator then
copies every batch's aggregate errors into every individual window record, so
early/late slices and rare-event summaries can inherit other windows' errors.
The standalone CLI evaluates baselines on only the first batch. Its replacement
heuristic gate also normalizes context stress using the evaluation batch mean.

These findings are code-level validity concerns. This development revision does
not rerun, relabel or discard any retained scientific artifact, nor conclude which
historical claim changes without a source-bound artifact audit.

## Complete opt-in implementation

`eigen_jepa.evaluation_v2` introduces a separate API, leaving `eval.py`, the
historical model and every explicitly frozen source binding byte-identical.

- `window_records` computes each window's own eigenvalue squared error, target
  energy, projector/covariance/drift/risk/entropy/rank error, latent alignment,
  labels and predictions. Exact input tensor bytes, dtypes and shapes receive a
  SHA-256 identity. Window indices remain exact integers.
- `reduce_records` computes NMSE as the ratio of cohort mean error to cohort mean
  target energy plus the existing `1e-12` epsilon. It takes the square root after
  pooling gap squared errors; F1 uses pooled TP/FP/FN, and balanced accuracy uses
  pooled class recalls. Rare metrics use only actual rare-window statistics.
- `evaluation_report` retains sorted records, the explicit reduction policy,
  threshold, result schema and a digest of the complete result. Missing rare-event
  support is represented by `null` and a zero count, not a fabricated zero score.
- `evaluate_model_v2` runs one vectorized model forward per supplied batch.
- `evaluate_baselines_v2` evaluates both declared baseline predictions over the
  entire same supplied cohort, using their own per-window context-only gates.

The prediction algorithms, covariance regularization and frozen thresholds are
not retuned. The new reduction and baseline-gate policy is a scientific evaluation
change and is explicitly prospective. A successor must opt in and source-bind it;
existing source-bound runners continue using their preserved historical APIs.

## Usage with frozen inputs

```python
from torch.utils.data import DataLoader
from eigen_jepa.data import batch_collate
from eigen_jepa.evaluation_v2 import evaluate_model_v2, evaluate_baselines_v2, reduce_records

# Use only development-authorized data until a successor protocol is frozen.
# Capture once: the legacy dataset samples fresh masks on repeated reads.
frozen_windows = [dataset[i] for i in range(len(dataset))]
loader = DataLoader(frozen_windows, batch_size=32, shuffle=False, collate_fn=batch_collate)
model_report = evaluate_model_v2(model, loader, device="cpu", memory=memory)
baseline_reports = evaluate_baselines_v2(loader)
selected = [row for row in model_report["records"] if row["window_index"] < cutoff]
early_metrics = reduce_records(selected)
```

The caller must retain the frozen input snapshot, model checkpoint, configuration,
environment, source revision and dataset/split provenance in its experiment
manifest. This reporting API does not fabricate those identities or authorize
protected evaluation. Large studies should materialize the snapshot on disk rather
than holding every window in memory.

## Verification and preserved failures

Initial testing correctly found that repeated reads of the legacy dataset draw
different feature masks; the test was then corrected to freeze the actual tensors
before comparing batch partitions. No production masking policy was changed.

`python -m pytest -q tests/test_evaluation_v2.py tests/test_sanity.py`

**18 targeted tests passed**, and the complete repository suite passed **113 tests**
with CPU PyTorch 2.14.1 and Python 3.12. Exact arithmetic cases
distinguish ratio-of-means from mean-of-ratios and pooled F1 from mean F1. Actual
untrained model and baseline evaluations on five generated windows agree across
batch sizes one and three within floating-point tolerance, covering the uneven
tail. Slice-specific and rare-event reductions, invalid records, missing support
and empty cohorts are checked. No retained or protected outcome was evaluated.
The source-freeze verifier passed with all five critical bindings unchanged.
Independent source review found no blocking correctness issue.

Pending: integration and an explicitly frozen successor
protocol before scientific use. The existing purged-fold v2 PR #31 is separate.
