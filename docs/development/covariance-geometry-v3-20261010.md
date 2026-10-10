# Prospective contract: covariance geometry evaluation v3

Session opened on 2026-10-10 before implementation, under the user's renewed
request for substantive research-code improvements. Parent: PR #32 at
`7b6eae8446d85b7d281fd22979575076ccece164`. The mixed/negative predecessor remains
closed. The bounded budget is recorded in `RESEARCH_STATE.json`.

## Defect and exact scope

The legacy model normalizes each subspace column separately. It does not make
the columns orthogonal or sort its positive head slots. Thus `U U^T` is not in
general a projector, and those head slots are not the eigenvalues of the model's
covariance `C = U diag(w) U^T`. Identical unit columns with weights `[1, 1]`
produce covariance spectrum `[2, 0]` and a rank-one support projector, rather
than two unit eigenvalues and a doubled projector.

This revision will add a separate opt-in evaluation API. It will preserve `C`
and compute its actual descending spectrum and numerical-support projector from
the thin singular value decomposition of `U sqrt(diag(w))`. It will not change
any learned parameter, model forward, training loss, existing evaluator,
checkpoint format, frozen source or retained outcome. Attaching old weights to
a QR-rotated basis would change `C` and is deliberately outside this contract.

## Numerical and reporting contract

- Inputs are finite real floating tensors `w: [batch, k]` and
  `U: [batch, assets, k]`, with `1 <= k <= assets` and nonnegative weights.
- The inference-only operator uses float64 arithmetic and scales the weighted
  factor by its maximum absolute entry before thin SVD. Eigenvalues are squared
  singular values in descending order. Zero covariance is allowed.
- The numerical support contains singular vectors with singular value strictly
  larger than `rank_rtol` times the largest singular value; the default relative
  tolerance is `max(assets, k) * float64 epsilon`. The report records the actual
  tolerance. Values below the tolerance are not silently removed from `C`.
- The support projector sums only supported singular-vector outer products.
  Degenerate zero directions do not acquire arbitrary forecast subspace credit.
- Spectral, gap and projector metrics use that geometry. Covariance metrics
  describe the same factor covariance, within float64 reconstruction tolerance.
  Independently learned drift, risk, entropy and rank heads remain their own
  predictions. They are not relabeled as functions of `C`.
- The evaluator will reuse v2 per-window statistics and cohort reductions,
  retaining exact input identities, plus per-window raw-factor/covariance
  identities, support rank and a distinct complete-report schema/digest.
- No rank truncation, jitter, retry or new basis is substituted after a numerical
  failure. Nonfinite/unrepresentable arithmetic or SVD failure rejects the
  evaluation. This is an inference API, not a differentiable training repair.

## Required discriminating evidence

Test the duplicated-column witness through the real legacy model and the new
cohort evaluator, compare against independent covariance eigendecomposition,
exercise sorted orthogonal factors, zero/rank-deficient covariance, sign and
permutation invariance, extreme representable scales, explicit rank tolerance,
uneven batches, invalid input, and source/model immutability. These are generated
engineering examples only. No protected observations, training campaign or
financial performance evidence will be produced.

## Scientific and integration boundary

A prospective successor must opt into and source-bind this metric meaning and
rank policy before protected outcome access. Existing v1/v2 reports keep their
original definitions and identities. This change cannot determine the revised
scientific conclusion of a historical study without its separately authorized
artifact audit. Independent covariance-estimator PR #33 and purged-fold PR #31
remain separate. The source-freeze and scientific authorization gates remain in
force.

## Implemented API and use

`eigen_jepa.covariance_geometry_v3.covariance_factor_geometry` implements the
specified inference operator. `evaluate_model_geometry_v3` runs the unchanged
model once per batch, evaluates its derived geometry with the v2 cohort
reducer, and emits `eigen_jepa.covariance_geometry_evaluation.v3`. Its separate
`geometry_records` bind each window's original factor bytes, reconstructed
covariance bytes, numerical-support rank, spectrum and reconstruction error.
The final digest covers both the complete metric report and geometry records.

```python
from torch.utils.data import DataLoader
from eigen_jepa.data import batch_collate
from eigen_jepa.covariance_geometry_v3 import evaluate_model_geometry_v3

# Development-authorized inputs only until a separate successor is frozen.
windows = [dataset[i] for i in range(len(dataset))]
loader = DataLoader(windows, batch_size=32, shuffle=False, collate_fn=batch_collate)
report = evaluate_model_geometry_v3(model, loader, device="cpu", memory=memory)
```

The independent learned entropy/rank/risk/drift heads stay independent. The
existing cohort NMSE and gap floors remain `1e-12`; this revision does not claim
to make those historical reduction conventions invariant to financial units.
Directly compare geometry only under a common predeclared metric meaning.
Exact raw-factor hashes may differ between floating-point batch computations,
even when numerical metrics agree within tolerance. Input hashes identify the
same frozen windows; factor hashes identify the outputs actually scored.

Thin SVD costs approximately `O(batch * assets * k^2)`; retaining/reconstructing
the covariance and projector costs `O(batch * assets^2 * k)` and
`O(batch * assets^2)` storage. The report retains per-window records in memory,
as v2 does. Large cohorts require a separately specified streaming artifact
writer. A full differentiable orthonormal model/training repair remains outside
this evaluation-only change.

## Retained engineering verification

The first focused invocation passed **55 tests in 13.86 s**. Four additional
regressions then addressed supported-eigenvalue underflow, an unrepresentable
eigenvalue despite finite covariance entries, integer tolerance overflow, and
raw-factor identity under an equivalent sign change. Formatting was applied.
The final invocation passed **59 tests in 48.57 s** with warnings treated as
errors: **36 new geometry tests plus 23 existing cohort, sanity and source-freeze
tests**. Python 3.12 / CPU PyTorch 2.14.1; one configured OMP/MKL thread.

```text
PYTHONPATH=/workspace/scratch/2a0c1e6ddd92/torch-deps:. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -W error tests/test_covariance_geometry_v3.py tests/test_evaluation_v2.py tests/test_sanity.py tests/test_real_market_source_freeze.py
59 passed in 48.57s

python scripts/verify_real_market_source_freeze.py
SOURCE_FREEZE_VERIFIED_NOT_AUTHORIZED
SOURCE_COMMIT=057d723bdb693233ac47bcd4bd351fc8f36fc281
CRITICAL_BINDINGS=5
```

The actual legacy-model duplicated-column fixture has v2 eigenvalue NMSE `0.5`
and projector MSE `1/9`. V3 scores the known covariance spectrum/projector with
errors below `1e-13`, preserving its covariance and all model state tensors.
This is a constructed arithmetic witness, not forecasting improvement. The
independent NumPy dense eigendecomposition oracle, scale cases from `1e-150` to
`1e150` in factor amplitude, rank/zero cases, uneven-batch inputs, fallback
rejection and source identities are covered. Ruff lint/format checks and patch
whitespace checks passed. No test failure or scientific outcome was discarded.

Two of the ten authorized local targeted test invocations were used before
independent review. All generated model cases contain five materialized training
windows; no optimizer, training campaign, retained outcome or protected data
was used. A new hosted workflow runs only these bounded correctness suites,
including when the PR is stacked on #32. It does not dispatch scientific runs.
Machine-readable source/test receipt:
`covariance-geometry-v3-20261010-receipt.json`.

Independent agent review checked covariance preservation, the singular-value
rank policy, untruncated eigenvalues, numerical edge cases and the model-state
preservation witness. No blocking issue or source change was requested. This
is an engineering source review, not scientific peer review. Publication is a
distinct stacked draft on #32; neither #32, #33 nor main is rewritten.

Published as [draft PR #34](https://github.com/Finance-Meta-Research/Eigen-JEPA/pull/34)
with implementation commit `fac569f355887410649edaa871a773f21318e8d7` and tree
`6836d4cadcb8b13cb73cc1c5b0c7aa50d107b282`. All 451 blob identities and modes
matched the exact parent plus the six intended changes. This append records
publication without modifying tested source; the PR retains current hosted
verification status. The bounded engineering session is closed with integration
pending. The scientific predecessor remains closed at its original disposition.
