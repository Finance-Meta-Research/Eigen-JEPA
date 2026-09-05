# Eigen-JEPA manuscript reconciliation — final-rigor v2

This patch text is outcome-bound to the frozen five-seed v2 package from GitHub Actions run `33988159305` and artifact digest `sha256:5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4`. It does not add an unregistered significance test or convert descriptive effects into causal proof.

## Abstract replacement for the empirical-claims portion

We evaluate Eigen-JEPA under a frozen five-seed synthetic-equity protocol with four paper-facing variants: the full model, memory disabled, gate disabled, and regime supervision disabled. The exact run retained all 20 seed-by-variant metric files and passed the frozen completeness verifier. Across seeds 7, 19, 31, 43, and 59, the full model obtained Eig NMSE `0.305095 ± 0.076342`, Drift MSE `0.022995 ± 0.012830`, Gate Cal `0.500257 ± 0.035723`, and Tail F1 `0.553932 ± 0.191267`. Removing memory worsened mean Eig NMSE to `0.345191 ± 0.035890` and Gate Cal to `0.521784 ± 0.048304`, but Tail F1 was identical to the full model at every retained seed. The gate-disabled variant achieved lower mean Drift MSE (`0.020178 ± 0.008655`) than the full model, while the no-regime variant achieved the lowest mean Eig NMSE (`0.285184 ± 0.088981`) among the four frozen variants. These results provide mixed mechanism evidence rather than full-model dominance: selective memory improves some spectral and calibration summaries, but the frozen experiment does not establish a tail-F1 benefit, universal gating benefit, or superiority across all reported metrics. The study remains a controlled synthetic benchmark and does not claim trading alpha or real-market validation.

## Results paragraph

The five-seed v2 package changes the interpretation of the earlier representative-run results. Relative to `no_memory`, the full model reduced mean Eig NMSE by `0.040095` and won the paired Eig NMSE comparison on four of five seeds. It also reduced gate-calibration error by `0.021528` on average and did so on all five seeds. The Drift MSE difference favored the full model by `0.006294` on average but was seedwise mixed (three wins, two losses). Crucially, Tail F1 was exactly unchanged by memory removal at every frozen seed. Memory therefore has descriptive support for spectral-error and calibration improvements in this protocol, but not for the manuscript's stronger prospective claim that it improves tail-event F1.

The gate and regime-supervision ablations are likewise mixed. The full model reduced mean Eig NMSE relative to `no_gate` by `0.017450`, but `no_gate` achieved lower mean Drift MSE by `0.002817`; Tail F1 and regime accuracy were identical between those two variants at every seed. Disabling regime supervision reduced mean Eig NMSE from `0.305095` to `0.285184`, although the full model improved mean regime accuracy by `0.120833`. Because the frozen protocol did not precommit a null-hypothesis significance test or practical-effect threshold, these comparisons should remain descriptive. They do not justify a post-hoc claim of statistical significance.

## Replacement for “What the results really say”

The final-rigor v2 evidence is a mixed mechanism result. Selective memory is associated with better mean eigenspectrum error and gate calibration than the matched `no_memory` variant, but the expected tail-F1 advantage does not appear: all four paper-facing variants have the same Tail F1 at every retained seed. The full model also does not dominate the other ablations across spectral metrics; `no_regime` has the best mean Eig NMSE and `no_gate` has the best mean Drift MSE. The defensible conclusion is therefore narrower than the current manuscript's prospective mechanism language: the architecture exhibits some component-specific benefits under the frozen synthetic-equity benchmark, alongside null and adverse ablation effects that must be retained.

## Required manuscript deletions/narrowing

The current manuscript should not describe the final empirical study as jointly validating equity, crypto, and rates: v2 freezes `market_style = equity` and synthetic data only. Replace future-tense claims that diagnostics “should” show a favorable pattern with observed v2 statements. Do not use the historical representative-run `results_table.tex` or `benchmark_ablation.tex` as the final five-seed evidence. Do not claim that memory improves tail recovery, that the full model provides the best overall trade-off, or that gating improves drift prediction unless a separately frozen follow-up directly supports those claims.

## Conference-facing claim boundary

Supported: a reproducible five-seed, four-variant synthetic-equity execution; retention of all 20 runs; mixed descriptive ablation evidence; better full-model mean Eig NMSE and Gate Cal versus `no_memory`; no Tail F1 differentiation across the frozen variants; and adverse/null ablation findings that constrain the architecture claim.

Not supported: real-market performance, trading alpha, joint equity/crypto/rates validation, full-model dominance across metrics, statistically significant component effects, or a general conclusion about JEPA-style finance models.
