# Eigen-JEPA submission claim audit — 2026-09-06

## Scope

This audit is a packaging and claim-to-evidence control for the frozen final-rigor v2 synthetic-equity result. It does **not** execute a model, inspect a new scientific outcome, change a seed/split/threshold, introduce a significance gate, authorize submission, or promote the separately frozen real-market confirmation lane.

Evidence base: `c9dee8572f4aaf43251e4e63268040ea89bdc644`. Frozen v2 execution: run `33988159305`, artifact `9975833698`, digest `sha256:5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4`. Exact seeds are `7, 19, 31, 43, 59`; exact variants are `full`, `no_memory`, `no_gate`, `no_regime`; all 20 seed-by-variant runs are retained.

## Claim-to-evidence reconciliation

The manuscript may report the retained descriptive values and directions. Full has mean Eig NMSE `0.305095 ± 0.076342` versus `0.345191 ± 0.035890` for `no_memory`, and mean Gate Cal `0.500257 ± 0.035723` versus `0.521784 ± 0.048304`. These are descriptive comparisons only because final-rigor v2 did not preregister an inferential significance test or a practical-effect threshold.

Adverse and null evidence remains first-class. Tail F1 is exactly tied between full and `no_memory` at every retained seed. `no_gate` has better/lower mean Drift MSE (`0.020178 ± 0.008655`) than full (`0.022995 ± 0.012830`). `no_regime` has the best/lower mean Eig NMSE (`0.285184 ± 0.088981`) among the four paper-facing variants. These observations prevent a full-model-dominance or universal-mechanism claim.

The final-rigor v2 evidence does **not** support trading alpha, deployable trading performance, real-market validation, joint equity/crypto/rates validation, statistically significant component effects, a Tail-F1 benefit from memory, a universal drift-prediction benefit from gating, or superiority over DCC, covariance-shrinkage estimators, random-matrix filters, or other classical methods that were not matched experimental baselines.

## Submission surface

The canonical entry point is `paper/main.tex`, which delegates to `paper/conference_v2.tex`. The final manuscript must continue to consume `paper/final_rigor_v2_table.tex` and `paper/final_rigor_v2_paired_table.tex`; historical representative outputs such as `paper/results_table.tex` and `paper/benchmark_ablation.tex` are forbidden as final-v2 evidence.

`paper/SUBMISSION_READINESS_20260906.json` binds the manuscript, protocol, retained evidence, tables, and publishing notes to exact Git blob identities. `scripts/verify_submission_claim_gate.py` fails closed if those identities drift, if the manuscript reintroduces historical final-table inputs, if the frozen run/artifact/seeds/variants change, if new outcome access is asserted, or if the machine gate is changed to self-authorize submission.

## Current status

Scientific status: **frozen synthetic final-rigor v2, mixed/descriptive**. Submission status: **not self-authorized by this gate**. The canonical paper PDF has a retained exact-head build from the parent paper lane, but any later manuscript change requires a fresh source-bound PDF build. Venue page limit, anonymity, template, and portal checks remain target-dependent.

Any real-market confirmation remains a separate pre-outcome lane. Nothing in this audit may be interpreted as authorization to access its outcomes or as evidence that the synthetic v2 result generalizes to real markets.
