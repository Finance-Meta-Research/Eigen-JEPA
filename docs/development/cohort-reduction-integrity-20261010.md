# Development correction: finite cohort reductions and owned receipts

Publication parent: `a3896fe974e4517149444f8bc6bb12d3c5bdd85b`, existing draft PR #34,
which is stacked on #32. The original defect was reproduced on #32 at
`7b6eae8446d85b7d281fd22979575076ccece164`; its `evaluation_v2` source is
byte-identical in #34. This repairs that opt-in API. The mixed/negative predecessor stays
closed; the historical evaluator, model, losses and frozen source files are not
changed.

## Reproduced defects

The reducer admitted finite nonnegative per-window statistics, then used sums
inside ordinary means. Two finite error/energy records of `1e308` produced an
infinite mean error and NaN NMSE, although the correct results are `1e308` and 1.
The same issue affected rare-window, gap and latent-alignment reductions.

The report also retained the caller's original row dictionaries. Changing an
input row after generating a report changed its retained evidence without
changing the already computed metrics or SHA-256 digest. Two reports constructed
from the same rows could alter one another.

## Correction and discriminating tests

Nonnegative statistics are normalized before averaging and restored to their
units afterward. This prevents overflow caused solely by summing finite nonnegative records. A genuinely nonzero
mean or NMSE that is not representable is rejected explicitly rather than
reported as a false zero or infinity. Cohort and rare-window NMSE use the same
helper. The declared `1e-12` denominator epsilon, gap epsilon, equal-window
weighting and classification policies are unchanged; this revision does not
claim unit-invariant NMSE or introduce a new estimator.

Reports copy each admitted scalar-valued row before validation and hashing.
Caller edits and independently generated reports no longer share row storage.
The returned report is ordinary mutable data; deliberate edits to that report
still require recomputing or verifying its digest.

The ten new regression cases produced **8 failures and 2 passes** against the
parent. Corrected source plus existing cohort and source-freeze checks passed
**31 tests**. Fixtures cover repeated/unequal finite extremes, rare slices,
the unchanged zero-energy policy, explicit overflow/underflow errors, and digest
stability after caller mutation. Exact versions, hashes and raw outputs are in
`research/verification/cohort_reduction_integrity_20261010/receipt.json`.

## Limits and next action

This repair concerns reductions of already admitted per-window sufficient
statistics. It does not replace the model's factor representation, alter
training, or establish that every earlier per-window arithmetic operation is
representable. Existing source-bound runners remain unchanged. No retained or
protected outcome, market dataset, scientific training campaign or paid compute
was used. The five critical source-freeze bindings remain unchanged.

Review and integrate after #34 and its #32 dependency. Any future study must explicitly select and
source-bind this prospective API and satisfy its scientific execution gate.


## Fresh dependency integration and independent review

PR #34 introduced a distinct covariance-geometry v3 API while this repair was
being reviewed. Its source, tests, development note and prior state history are
retained unchanged. V3 delegates cohort reduction to v2, so the final bounded
invocation includes both APIs: **69 tests passed in 6.26 s**, with warnings
treated as errors. This comprises the ten new reduction/ownership regressions,
36 geometry cases and 23 existing cohort, sanity and source-freeze checks.
The earlier 31-test success remains in `pre_geometry_stack_pytest.log`.

Two first integration invocations loaded the parent evaluator bytes after
timestamp-preserving file copies; each reported 8 failures and 61 passes. Their
outputs and the second invocation's differing before/after source hashes are
retained. They are not verification of the proposed source. Writing the source
bytes with a fresh timestamp and guarding its identity before and after the
final invocation produced the passing result above. No mathematical change was
made in response to those workspace-source mismatches. The final tested source
SHA-256 is `9e26557139d7e592b713ddc0e38f83c0c06f6c862e3673da5ffe3046b7befd95`.

Independent engineering source review at that hash checked eight additional
Decimal mean/NMSE cases and row ownership. No blocking issue remained within
the explicit policy that an unrepresentable required mean is refused; the
repair does not promise recovery of every representable final ratio when an
intermediate required mean is unrepresentable. This review is not scientific
validation. All five frozen source bindings and the closed predecessor remain
unchanged. This stacked draft does not merge either dependency or open a study.
