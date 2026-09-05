# Eigen-JEPA final-rigor v2 retained result summary — 2026-09-06

## Evidence identity

- Frozen protocol: `eigen-jepa-final-rigor-v2-20260905`.
- Exact seeds: `7, 19, 31, 43, 59`; variants: `full`, `no_memory`, `no_gate`, `no_regime`.
- Execution head: `9369a5f2b0b972af846fa70baca027451271e08c`; GitHub Actions run `33988159305`.
- Retained workflow artifact: `9975833698`; artifact digest `sha256:5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4`.
- Frozen verifier result: `FINAL_RIGOR_V2_PASS: exact seeds, variants, 20 retained run-metric files, and 72 finite aggregate summaries verified.`

This file records observed v2 outcomes. It does not alter the frozen protocol, rerun a favorable subset, or create a post-outcome significance threshold.

## Five-seed aggregate

| Variant | Eig NMSE ↓ | Drift MSE ↓ | Cov MSE ↓ | Gate Cal ↓ | Tail F1 ↑ | Regime Acc ↑ | Rare Eig NMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full` | 0.305095 ± 0.076342 | 0.022995 ± 0.012830 | 0.091642 ± 0.048656 | 0.500257 ± 0.035723 | 0.553932 ± 0.191267 | 0.616667 ± 0.211886 | 0.372357 ± 0.162879 |
| `no_memory` | 0.345191 ± 0.035890 | 0.029290 ± 0.009450 | 0.092045 ± 0.046544 | 0.521784 ± 0.048304 | 0.553932 ± 0.191267 | 0.612500 ± 0.217546 | 0.388451 ± 0.163679 |
| `no_gate` | 0.322545 ± 0.098770 | 0.020178 ± 0.008655 | 0.090629 ± 0.046908 | 0.550000 ± 0.176088 | 0.553932 ± 0.191267 | 0.616667 ± 0.211886 | 0.391944 ± 0.170947 |
| `no_regime` | 0.285184 ± 0.088981 | 0.022839 ± 0.012889 | 0.094715 ± 0.047779 | 0.500099 ± 0.035875 | 0.553932 ± 0.191267 | 0.495833 ± 0.290593 | 0.344453 ± 0.179640 |

## Paired descriptive interpretation

The v2 protocol precommitted exact execution and aggregation, but it did not precommit a null-hypothesis significance test or practical-effect threshold. The comparisons below are therefore descriptive paired effects across the five frozen seeds; they must not be relabelled as post-hoc statistical significance.

- **Memory:** full has lower mean Eig NMSE than `no_memory` by `0.040095` and wins that paired comparison on 4/5 seeds. Full also has lower gate-calibration error by `0.021528` on average and wins 5/5 seeds. Drift MSE is lower by `0.006294` on average but only 3/5 seeds. **Tail F1 is exactly tied at every seed**, so the retained v2 evidence does not support the paper's current expectation that removing memory should weaken tail-event F1.
- **Gate:** full has lower mean Eig NMSE than `no_gate` by `0.017450`, but the seedwise effect is mixed (3/5). Full has **higher/worse** mean drift MSE by `0.002817`. Tail F1 and regime accuracy are exactly tied at every seed. Gate-calibration error is lower for full on average by `0.049743`, but only on 2/5 seeds because two seeds favor full strongly while three do not.
- **Regime supervision:** `no_regime` has lower mean Eig NMSE than full by `0.019912`; full wins 3/5 seeds but loses the aggregate because of a large adverse seed-7 difference. Full improves regime accuracy by `0.120833` on average, with 2 wins and 3 ties. Tail F1 is again identical at every seed.

These outcomes are mixed. They support a narrow claim that the memory-enabled full model improves some spectral and calibration metrics relative to `no_memory`, but they do **not** establish full-model dominance, tail-F1 benefit, or a general mechanism-success result. In particular, `no_regime` has the best five-seed mean Eig NMSE among the four paper-facing variants, and `no_gate` has the best mean drift MSE.

## Manuscript correction boundary

The current manuscript predates this five-seed package and uses a representative/single-run table plus prospective language such as metrics that “should” improve. Those statements must not be presented as the final v2 evidence. A conference-facing revision should replace the representative ablation claims with the five-seed aggregate above, explicitly report the identical Tail F1 across variants, and narrow the mechanism conclusion to mixed descriptive evidence. The v2 run uses the frozen **synthetic equity** configuration only; it does not by itself validate equity, crypto, and rates jointly or establish real-market performance.

## Retention / next scientific gate

Retain the complete workflow artifact and the aggregate irrespective of whether the result is favorable. The next scientific action is manuscript reconciliation, followed by a separately frozen real-data or cross-market confirmation protocol if broader financial claims are desired. Do not change the v2 seeds, variants, or thresholds in response to these outcomes.
