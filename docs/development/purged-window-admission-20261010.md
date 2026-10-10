# Development correction: admit complete purged windows

The existing Eigen-JEPA result remains closed at the mixed/negative boundary in
`FINAL_STATUS_2026-09-30.md`. This correction adds the opt-in
`eigen_jepa.purged_folds_v2` prospective API. The frozen
`eigen_jepa/real_market_folds.py` and its original blob identity remain intact.
No saved evidence, frozen protocol, manuscript, threshold,
seed, or result is changed, and no protected or training run was executed.

## Defect and corrected invariant

The date-based splitter already purges full windows, but the dataset builder
also accepts explicit user-supplied indices. It checked only that the context
endpoints were distinct. Distinct endpoints do not prevent training targets
from overlapping validation context or validation targets from overlapping test
context. Reversed split order also passed that check.

A sample ending at index `i` observes the inclusive footprint
`[i - context_len + 1, i + horizon]`. The corrected builder requires the latest
earlier target to precede the earliest later context at both split boundaries.
Thus equality at one row is rejected; adjacent non-overlapping footprints are
accepted. Within-split rolling windows remain permitted.

Integer admission now checks the original indices before casting, rejecting
fractional values, booleans, strings, nonfinite values and unrepresentable
integers. Window dimensions must be positive integers. Loading occurs with a
copied configuration so a rejected input cannot update the caller's declared
CSV shape.

## Verification

New generated-CSV regressions in `tests/test_real_market_purged_folds_v2.py` cover
both overlapping boundaries, reversed order, adjacent admissible windows,
lossy coercion, and invalid dimensions. Existing date-boundary, shape and
future-shock invariance cases remain.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest tests/test_real_market_purged_folds.py tests/test_real_market_purged_folds_v2.py -q
```

Local source compilation and direct execution of the exact NumPy admission
block passed the five overlap/order rejection fixtures and the adjacent-window
acceptance fixture. PyTorch was unavailable locally; full package and CSV/model
integration must be verified by the existing PR CI job, not claimed as locally
passed. The first PR CI run passed 112 tests, including all added regression
cases, and correctly rejected the changed frozen source blob. The fix was then
moved to the explicitly versioned API without modifying the old source gate.
This is a validity fix, not evidence for financial outperformance.

## Opt into the corrected API

For a new development study, use the following imports with the study's actual
dates, boundary, and CSV configuration:

```python
from eigen_jepa.purged_folds_v2 import (
    build_purged_fold_indices,
    build_purged_fold_datasets,
)

indices = build_purged_fold_indices(
    dates, context_len=cfg.context_len, horizon=cfg.horizon, boundary=boundary,
)
datasets = build_purged_fold_datasets(
    cfg, train_indices=indices["train"],
    validation_indices=indices["validation"], test_indices=indices["test"], k=3,
)
```

The protocol/source manifest for that new study must bind
`eigen_jepa/purged_folds_v2.py` as well as its retained dependencies. Existing
frozen runners continue to use the old source identity and must not be silently
redirected to v2.
