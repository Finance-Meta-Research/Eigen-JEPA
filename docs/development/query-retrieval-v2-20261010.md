# Prospective successor development contract: query-gradient retrieval v2

Recorded 2026-10-10 09:57:23 UTC, before implementation or reproduction. Primary parent:
Eigen-JEPA PR #36 at `e0f71481524e2af88029106bde7476fd95a34b27`, preserving
#35, #34 and #32. The live default advanced to
`a20efd8ad82e188a4803072b74c6d95a5cb09186` with only
`scripts/verify_research_evidence.py` and its six regression tests; preserve those
exact additions through an additional integration parent. Separate PR #31/#33 and
every frozen source remain unchanged.

New session `20261010-0945-query-gradient-v2` has a fresh finite budget. Every
previous session remains closed. The retained scientific predecessor remains
**MIXED / NEGATIVE BOUNDARY, CLOSED**.

## New bounded hypothesis and existing code boundary

The legacy `SpectralMemory.retrieve` is entirely under `torch.no_grad`.
`EigenJEPA.query_head` feeds only this read and detached memory writes. Therefore
the standard differentiable objective cannot train its query parameters through
retrieval. This observation must be reproduced with the actual model/loss before
the new implementation is accepted; it does not establish why historical financial
performance was adverse.

Successor development hypothesis `eigen-jepa-query-learning-v2-20261010`:
when at least two selected stored neighbors have distinct relevant keys/values,
differentiating the selected-neighbor read supplies the loss gradient to the query
head that the forward map mathematically implies. This is a testable algorithmic
claim, not a forecast of financial superiority or a new novelty claim.

## Exact prospective model and training contract

Add a separate opt-in module `eigen_jepa.query_retrieval_v2`, explicit v2 model
class/config identity and v2 checkpoint identity. Do not alter legacy
`model.py`, `memory.py`, `losses.py`, `train.py`, evaluators or their frozen
bindings. The new model has exactly the same allocated parameter names/shapes and
initialization as EigenJEPA.

The memory read keeps the original equations and selection:
`q_n=q/(||q||_2+1e-8)`, similarly normalized stored keys, dot-product similarity,
the original top-k selection, and softmax over selected similarities divided by
`max(temperature,1e-6)`. Its output is the weighted sum of selected stored values.
The intentional change is autograd through the query, normalization and selected
softmax weights. No gradient differentiates the discrete top-k indices. Stored
keys and values are detached, owned read snapshots; writes keep the inherited
detached prototype policy. A later bank write must not change an already-created
read graph.

Finite real dense query/state tensors, compatible dimensions/dtype/device and a
valid positive neighbor count are required. The inherited numerical formula is
retained; unsupported/unrepresentable normalization or read arithmetic fails
explicitly rather than inventing a fallback. No learned temperature, new loss,
ranking surrogate, rank repair, covariance estimator or train/test split change
is introduced.

The v2 model accepts only the v2 memory (or an explicit memory-off control). Its
configuration and saved checkpoint state name the model/retrieval version and
retain all memory inference settings, model configuration, model dtype and exact
model/memory tensors. Loading must verify the v2 identity and compatible policy;
a legacy checkpoint is not silently promoted. Checkpoint serialization is model/
memory restoration, not exact optimizer/RNG/loader resumption.

Limitations are part of the claim: one neighbor gives constant unit weight; equal
stored values can give zero query gradient; ties and selection boundaries are
nondifferentiable; a top-k read does not differentiate membership; empty memory
cannot train a retrieval query; and a nonzero gradient does not establish learning
utility. Existing negative outcomes retain their original stop-gradient policy.

## Discriminating and end-to-end evidence

First retain the actual-model legacy no-query-gradient witness. Test a hand-derived
two-neighbor softmax/cosine derivative and float64 finite-difference/gradcheck
away from ties, selected-neighbor behavior, detached stored tensors, writes between
forward/backward, one-neighbor/equal-value/tie/empty limits, and exact ordinary
forward/initial-parameter equivalence to the retained legacy classes. Exercise
the actual spectral loss and optimizer on generated windows. A newly generated
v2 checkpoint/config must round-trip into the v2 model with matching predictions
and must reject mismatched/legacy identity.

Retain one generated paired demonstration with identical initial parameter tensors
and input/memory tensors for legacy and v2 arms, at most two optimizer updates per
arm. Record raw query gradients and tensor/checkpoint identities; the demonstration
tests executed gradient semantics and makes no comparative financial claim.

## Finite budget and protected boundary

At most five local test/reproduction commands, 240 seconds cumulative command wall
time, at most two review repair cycles, one independent reviewer test/probe command
of at most 90 seconds, and one retained generated demo attempt of at most 60 seconds.
New model cases have at most four examples, context length eight, latent width
eight and two updates per arm; existing bounded inherited tests retain their prior
fixture sizes. One CPU numerical thread; no paid/provider/GPU work, protected
finance outcomes, frozen seed matrix or scientific campaign. Stop optional tests
after sufficient verification.

Source hashes identify source files and require a fresh interpreter with an
unchanged checkout. They are not secure attestation of cached/monkeypatched/live-
mutated execution. Retain all failed attempts and independent review decisions.
Publish a reviewed distinct stacked draft with the exact additional default-parent
changes, full tree/blob verification and actual CI readback. Only an explicitly
versioned future study may test whether this training change helps forecasting.

