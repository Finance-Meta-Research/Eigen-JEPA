# Explicit query-gradient model v2

The opt-in implementation is `eigen_jepa.query_retrieval_v2`. Its prospective
contract and successor hypothesis were recorded in
[`query-retrieval-v2-20261010.md`](query-retrieval-v2-20261010.md) before the baseline
probe or implementation. The completed predecessor remains **MIXED / NEGATIVE
BOUNDARY, CLOSED**. This module does not replace the legacy model, trainer,
checkpoint replay, losses or frozen scientific source.

## What changes mathematically

The legacy memory read runs inside `torch.no_grad`. As a result, its query head
has no gradient from the actual spectral loss. The v2 read computes the same
normalized cosine similarities, discrete top-k selection, softmax weights and
weighted stored values, while permitting derivatives through the query and its
selected weights. Parameter names, shapes, initialization and ordinary forward
results remain identical before an optimizer update.

Every read owns detached copies of the stored keys and values. A later prototype
write cannot change the content used by an already-created backward graph.
Prototype write equations remain inherited from `SpectralMemory`. The extra read
storage is `O(M * (key_dim + value_dim))`; the method does not make writes atomic.

The implementation admits finite dense float32/float64 queries and compatible
bank tensors, positive dimensions/capacity/top-k/temperature, nonnegative finite
merge radius and salience threshold, and stored regime labels 0–2. It rejects
overflow in the retained cosine normalization or output instead of inventing a
different forward calculation. Optimizer callers must still check loss/gradient
finiteness before mutation.

One neighbor has constant weight. Identical stored values can cancel the query
gradient. Empty memory has no read gradient. Top-k membership and ties are
nondifferentiable; the implementation differentiates the selected branch only.
A nonzero gradient is an algorithmic property, not evidence of predictive value.

## Callable model and checkpoint API

```python
import torch

from eigen_jepa.query_retrieval_v2 import (
    QueryGradientConfigV2, QueryGradientEigenJEPAV2, QueryGradientMemoryV2,
    save_query_checkpoint_v2, load_query_checkpoint_v2,
)

cfg = QueryGradientConfigV2(
    input_dim=7, num_assets=3, context_len=6, k=2,
    d_model=8, n_heads=2, n_layers=1, dropout=0.0,
    latent_dim=8, memory_dim=8,
)
model = QueryGradientEigenJEPAV2(cfg, target_dim=15)
memory = QueryGradientMemoryV2(8, 8, top_k=2, temperature=0.7)
memory.write(torch.eye(8)[:2], torch.arange(16).float().reshape(2, 8),
             torch.ones(2), torch.zeros(2, dtype=torch.long))
x = torch.zeros(2, 6, 8)  # Generated API example, not market evidence.
prediction = model(x, memory)
digest = save_query_checkpoint_v2("fresh-query-v2.pt", model, memory)
restored = load_query_checkpoint_v2("fresh-query-v2.pt")
restored_prediction = restored.model(x, restored.memory)
```

`memory=None` is the explicit model memory-off control. A legacy memory object is
refused by the v2 model. The v2 checkpoint helper requires the explicit v2 bank,
including an empty one when appropriate. `query_config_v2(mapping)` validates a
complete serialized v2 configuration and refuses implicit promotion of a legacy
configuration.

The fresh-file checkpoint records model and retrieval identities, configuration,
target dimension, dtype, exact model/memory tensors and every memory policy
setting. Loading reads, hashes and decodes one byte snapshot using weights-only
Torch loading, validates identities/shapes/policy, restores owned tensors, and
preserves the caller's CPU RNG. The returned model is in evaluation mode. Use
`.train()` explicitly for a new, separately configured training continuation.

The format records the default forward call (`gate_override=None`,
`memory_scale=1`). Per-call ablation overrides, optimizer state, random stream and
data-loader continuation are outside this format. Source/checkpoint hashes
identify files and require a fresh interpreter with an unchanged checkout; they
are not secure attestation of arbitrary cached or monkeypatched execution.

## Generated verification and paired demonstration

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m pytest -q -W error tests/test_query_retrieval_v2.py tests/test_sanity.py \
  tests/test_checkpoint_evaluation_v3.py tests/test_research_evidence_integrity.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python scripts/demo_query_retrieval_v2.py --output /tmp/eigen-query-v2-demo
```

The initial gate passed 93 cases: 37 new query-gradient/checkpoint cases plus 56
relevant inherited cases, including all six latest-default evidence-verifier
tests. New cases compare a hand-derived two-neighbor derivative and float64
gradcheck, stored-content detachment, writes between forward/backward, declared
constant/tied limits, identical initial parameters/forward, actual spectral-loss
gradients and an SGD update, plus checkpoint identity/dtype/state admission.

The fixed demonstration uses two generated windows, context six, latent width
eight, and two updates per arm. Both arms begin with identical model and bank
tensors. Each step explicitly pairs the existing target encoder's dropout RNG.
It retains initial inputs/model/memory, both configurations, raw gradients from
each arm/step, a legacy final tensor receipt, and a v2 checkpoint with exact
generated forward replay. It compares gradient semantics; its two losses are not
a financial performance benchmark.

All attempts, baseline failures, command/source receipts, independent review and
the single retained demo are under
`research/verification/query_retrieval_v2_20261010/`. A future scientific study
must explicitly freeze this new training policy, its baselines, tuning budget and
protected evaluation before evaluating forecasting benefit. Earlier adverse
outcomes are neither repaired nor reclassified by this engineering result.
