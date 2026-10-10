# Prospective checkpoint-policy replay and development snapshot evaluation

Session opened 2026-10-10 before implementation. Parent: Eigen-JEPA PR #35 at
`0bebf465d4e66f08acd683c669396791b37d4198`, incorporating #34 and #32.
The mixed/negative predecessor remains closed. The canonical state contains the
finite correctness budget; no training campaign or protected evaluation is part
of this session.

## Existing executable defect

The legacy standalone `eigen_jepa.eval.main` reconstructs SpectralMemory using
only dimensions, capacity and top-k metadata. `memory.state_dict()` omits
temperature, merge radius and minimum salience, while the training checkpoint
does retain those settings in `args`. The standalone path therefore substitutes
constructor defaults. It also ignores the saved variant: in particular,
`no_gate` inference requires the training path's fixed gate of 1.0, and
`no_memory` requires omitting retrieval. Loading matching learned weights is
insufficient to reproduce the saved model's inference behavior.

## Implementation contract

Add a separate opt-in `eigen_jepa.checkpoint_evaluation_v3` module. Preserve the
legacy CLI and every existing source-bound model, memory, training and evaluation
file byte-for-byte. The new path will:

- Read the exact checkpoint bytes once with `torch.load(weights_only=True)`.
  Require the saved model configuration, variant and memory parameters rather
  than inventing missing inference settings. Validate finite dense state tensors,
  dimensions, capacity and metadata agreement before inference. Support the
  repository's existing float32 model/checkpoint format explicitly.
- Reconstruct the actual EigenJEPA and SpectralMemory, load the exact tensors,
  restore saved memory temperature/merge/minimum-salience configuration, and use
  the recorded no-memory/no-gate inference semantics. Other declared training
  ablations retain their learned checkpoint weights and ordinary inference path.
- Consume a separately materialized tensor snapshot explicitly marked
  `development`, with a supplied source reference. Validate window fields, shapes,
  identities and finite values; do not regenerate masks or read an arbitrary
  checkpoint CSV path. A snapshot source label is not independent provenance
  verification or protected-run authorization.
- Run the existing covariance-geometry v3 evaluator on every supplied window.
  Record checkpoint/snapshot byte hashes, reconstructed inference policy, source
  module identities, configuration and environment, batch size and the full v3
  evidence report under a distinct checkpoint-report schema and digest.
- Provide an executable CLI, reject ambiguous/invalid artifacts before emitting
  a result, and preserve an existing output file. These are model-evaluation
  artifacts; a baseline comparison, scientific result or finished paper is not
  implied.

## Discriminating verification plan

Exercise the actual legacy CLI factory with a newly generated untrained checkpoint
and intercept its first evaluation call to compare loaded policy/predictions
against the original in-memory model. Retain the reproduction without rerunning
historical outputs. Check full/no-gate/no-memory inference against independent
direct-model calls, including an occupied memory with nondefault temperature.
Round-trip a real serialized checkpoint and frozen window snapshot through the new
CLI; recompute hashes/digest and verify output preservation, malformed/mismatched
state, invalid snapshots, missing metadata and duplicate IDs. Include the 69
existing bounded geometry/cohort/source-freeze checks. Record failures, exact
source hashes, environments and independent review before publication.

## Scientific boundary

This restores the declared checkpoint's inference policy and makes prospective
evaluation executable; it does not reinterpret any retained paper result.
The saved research, real-market authorization and source-freeze gates remain in
force. Any scientific successor still needs explicit metric/source/input-policy
selection and its own pre-outcome protocol. The new development-mode input label
is an admission contract, not evidence that external data have been independently
verified or that held-out outcomes may be opened.

## Implemented development entry point

The new module is `eigen_jepa.checkpoint_evaluation_v3`. The existing model,
memory, data, training, both older evaluators and every protected source binding
remain byte-identical to the parent. The README points to this opt-in path.

Materialize windows from an already selected, independently authorized development
population once. For example, with such a `development_dataset` already in memory:

```python
from eigen_jepa.checkpoint_evaluation_v3 import save_development_snapshot

save_development_snapshot(
    "development-windows.pt",
    (development_dataset[i] for i in range(len(development_dataset))),
    source_reference="your existing development population/source record",
)
```

The helper owns CPU copies of the supplied tensors and creates a new file. It
does not select a split or establish data provenance. In particular, changing a
label to `development` does not authorize access to a protected population.
The helper validates tensor types/finiteness; evaluation validates the complete
window contract against the checkpoint before producing a report.

```bash
python -m eigen_jepa.checkpoint_evaluation_v3 \
  --checkpoint checkpoint.pt --snapshot development-windows.pt \
  --output development-evaluation.json --batch-size 32 --device cpu
```

The output parent directory must exist and the output file must be new. Optional
`--event-threshold` and `--rank-rtol` select and record the existing prospective
evaluator's policies. Repeat a comparison with the same snapshot and batch size;
ordinary floating-point batch-shape differences remain possible. The report has
its own canonical JSON SHA-256 and embeds the complete geometry-v3 report,
including its separate digest, per-window statistics and factor/covariance hashes.
It is a model report, not a baseline comparison.

### Artifact admission

Training checkpoints must retain the complete `EigenJEPAConfig`, `target_dim`,
JSON-compatible `data_cfg`, `variant`, `args`, model state and memory state. The
variant in `args` must agree with the top-level variant. Memory capacity/top-k
must agree with metadata; temperature is positive and merge radius nonnegative.
Model and memory floating tensors must be dense, finite float32 tensors with
the declared dimensions. The current target dimension is `k + assets**2 + 4`.
No permissive load, implicit float64-to-float32 conversion, or missing-policy
default is used. The original `torch.device` value saved in training arguments
is compatible with the exercised weights-only loader; unused arguments are not
executed or incorporated as executable configuration.

Snapshots use schema `eigen_jepa.development_windows.v1`, mode `development`, a
nonempty supplied `source_reference`, and a nonempty list of tensor mappings.
Required floating fields are float32: `x[context_len, input_dim+1]`,
`returns_ctx[context_len, assets]`, `target_vec[target_dim]`, `evals_true[k]`,
`cov_true[assets, assets]`, `proj_true[assets, assets]` and scalar `drift_true`,
`risk_true`, `entropy_true`, `rank_true`, `event_true`. Scalar `window_index` and
`regime_true` must be int64; identities are nonnegative/unique, events binary and
regimes in 0/1/2. Additional tensor fields may be retained, but all windows must
have the same field names, shapes and dtypes and contain finite values.
The exact snapshot byte hash binds even extra fields. Supplied provenance is
explicitly marked unverified and does not grant protected-run authorization.

### Actual legacy-CLI reproduction

Before implementing the new module, `python tests/test_checkpoint_evaluation_v3.py`
constructed untrained float32 fixtures and invoked the actual legacy CLI,
intercepting its first evaluation call before any legacy result files were
written. Each fixture had 3 assets, 8 context steps and a generated 60-step series;
the occupied memory used temperature 0.7, merge radius 0.01 and minimum salience
0.6. The CLI reloaded temperature 0.2 and merge radius 0.18.

- `full`: maximum latent prediction difference from the original in-memory model
  was `0.35594671964645386` on the same exact first batch.
- `no_gate`: expected gate 1.0 reloaded as approximately 0.11920293; maximum
  latent difference was `3.9660000801086426`.
- `no_memory`: the CLI incorrectly supplied a memory object, but the valid
  training-format fixture had empty memory, so its latent difference was exactly
  zero. This is a policy omission, not evidence of changed empty-memory outputs.

These three legacy failure controls remain executable and the legacy source is
preserved. All eight declared variants are checked against the original model
and original memory with independently supplied inference arguments. The actual
new CLI serializes a complete five-window result with an uneven final batch and
recomputed artifact/source/result hashes. Rejections cover ambiguous saved policy,
state shape/dtype/finiteness, malformed windows, duplicate identities, invalid
labels, unsupported deserialization and existing output paths.

### Local verification so far

- Generated legacy CLI witness: successful reproduction before implementation.
- New focused suite: **47 passed in 27.28 seconds**.
- Combined new and existing suite: **116 passed in 16.03 seconds** (47 new and
  69 existing geometry, cohort, sanity and source-freeze checks).
- Local runtime: Python 3.12.14, CPU PyTorch 2.14.1+cpu, NumPy 2.3.5,
  pytest 9.1.1, with OMP/OpenBLAS/MKL thread counts set to 1.
- Ruff with explicit isolated `E4,E7,E9,F,I,UP` selection and whitespace checks
  passed. An initial ambient-workspace lint invocation reported import formatting
  and a preferred exception-type style rule; import formatting was repaired.
  Artifact validation intentionally uses ValueError. This was not a repository
  functional-test failure or a scientific outcome.

Independent review and hosted verification are recorded as they complete in the
canonical state and accompanying receipt. No training campaign, protected
population, historical checkpoint/outcome or paid compute was used.

### Independent review and final local source

Review identified an ambient-default edge in the programmatic API. Model
construction is now explicitly CPU/float32 under a preserved CPU random stream.
An independent reviewer then reproduced an actual remaining invocation failure:
under `with torch.device("meta")`, the DataLoader iterator created a meta seed
tensor and raised `RuntimeError: Tensor.item() cannot be called on meta tensors`.
The correction binds CPU factories during both loader construction and the
entire evaluator iteration. Explicit model/batch moves still honor the requested
device. The extended regression compares the complete report with ordinary
defaults versus simultaneous float64/meta defaults, and verifies that the
caller's settings remain intact.

The reviewer inspected the final policy reconstruction, all eight original-model
oracles, artifact shape/dtype admission, same-byte hashes and real CLI invocation;
no blocking finding remains. The final exact-source suite passed **117 tests in
30.36 seconds** (48 new plus 69 existing). All recorded runtime/test source hashes
were identical before and after execution. Earlier 116-test and initial
117-test verification are retained, with the latter explicitly labeled as the
factory-only candidate before the complete iterator regression.

Verification artifacts are under
`research/verification/checkpoint_policy_replay_20261010/`. The final executable
source receipt is `local_117_final_receipt.json`; its raw output is
`local_117_final_tests.txt`. The session used seven root test/reproduction
invocations and one independent review probe, below the predeclared limit of 16.
All new fixtures use five model windows and a generated 60-step series; existing
checks use at most the predeclared 80-step bound. No optimizer campaign or
protected evaluation was introduced by these corrections.

### GitHub publication and hosted evidence

Published draft: https://github.com/Finance-Meta-Research/Eigen-JEPA/pull/36 .
Implementation commit `5077684e7b03bbaadd2122c317a3497ed95fe008`, tree
`f15582a652525bd026410efba3c440a127b0aa40`. Every one of the 472 implementation
leaf paths/modes matched the exact parent plus the 12 declared changes; all
460 other parent files remained identical. The draft stacks on #35 and preserves
the separate #31/#33 proposals and every earlier recorded scientific boundary.

Both workflows verified an explicit checkout of the implementation head:

- Development checkpoint replay run **38030239548**, job **114149452905**:
  **117 passed in 7.09 seconds**; Python 3.11.17, PyTorch 2.14.1+cpu,
  NumPy 2.4.6 and pytest 9.1.1.
- Submission claim gate run **38030239556**, job **114149453102**: existing
  frozen synthetic-v2 identity, mixed/negative claim boundary and no-outcome/
  no-self-authorization verification passed; **6 regressions passed** with
  pytest 8.4.2.

The durable hosted receipt is `github_5077684_receipt.json` alongside the local
evidence. This follow-up changes only state, this record and verification receipts;
reviewed source, tests and workflow remain unchanged. The draft awaits GitHub
review/integration. No scientific conclusion or authorization changes.
