# Eigen-JEPA final-rigor v2 seed-level retained evidence

This ledger is a repository-retained, seed-level projection of the completed frozen final-rigor v2 workflow artifact. It does **not** rerun the experiment, select favorable seeds, add a post-outcome significance test, or create a practical-effect threshold.

## Frozen evidence identity

- Protocol: `eigen-jepa-final-rigor-v2-20260905`
- Exact seeds: `7, 19, 31, 43, 59`
- Exact variants: `full`, `no_memory`, `no_gate`, `no_regime`
- Execution head: `9369a5f2b0b972af846fa70baca027451271e08c`
- GitHub Actions run: `33988159305`
- Retained artifact: `9975833698`
- Artifact digest: `sha256:5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4`
- Seed ledger SHA-256 at creation: `1d9b83065cf026bdeb8156d49ef0c87b36efb7d8cacabeb1326fa78311ca55ed`

The canonical machine-readable ledger is `SEED_LEVEL_METRICS_20260906.json`. Every row carries the SHA-256 of the raw retained `metrics.json` file from the workflow artifact. `scripts/verify_final_rigor_v2_seed_ledger.py` recomputes the 5-seed means/population standard deviations and paired arithmetic; when given the downloaded artifact root, it also verifies all 20 raw metric-file hashes and exact metric values.

## Seed-level manuscript metrics

| Seed | Variant | Eig NMSE ↓ | Drift MSE ↓ | Gate Cal ↓ | Tail F1 ↑ |
|---:|---|---:|---:|---:|---:|
| 7 | full | 0.294799 | 0.011121 | 0.450591 | 0.760218 |
| 7 | no_memory | 0.346328 | 0.038501 | 0.470345 | 0.760218 |
| 7 | no_gate | 0.273819 | 0.009204 | 0.375000 | 0.760218 |
| 7 | no_regime | 0.194135 | 0.010562 | 0.449506 | 0.760218 |
| 19 | full | 0.389187 | 0.013191 | 0.491439 | 0.647388 |
| 19 | no_memory | 0.391489 | 0.012919 | 0.491668 | 0.647388 |
| 19 | no_gate | 0.388310 | 0.013269 | 0.479167 | 0.647388 |
| 19 | no_regime | 0.389683 | 0.013201 | 0.491727 | 0.647388 |
| 31 | full | 0.346811 | 0.046043 | 0.477776 | 0.710201 |
| 31 | no_memory | 0.371455 | 0.038797 | 0.486351 | 0.710201 |
| 31 | no_gate | 0.454295 | 0.032046 | 0.375000 | 0.710201 |
| 31 | no_regime | 0.347654 | 0.045883 | 0.478182 | 0.710201 |
| 43 | full | 0.329653 | 0.027494 | 0.535582 | 0.285714 |
| 43 | no_memory | 0.329433 | 0.027588 | 0.577568 | 0.285714 |
| 43 | no_gate | 0.330766 | 0.028055 | 0.750000 | 0.285714 |
| 43 | no_regime | 0.329664 | 0.027532 | 0.535356 | 0.285714 |
| 59 | full | 0.165026 | 0.017129 | 0.545895 | 0.366138 |
| 59 | no_memory | 0.287247 | 0.028643 | 0.582990 | 0.366138 |
| 59 | no_gate | 0.165534 | 0.018318 | 0.770833 | 0.366138 |
| 59 | no_regime | 0.164782 | 0.017015 | 0.545726 | 0.366138 |

The identical Tail F1 within every seed is preserved as an explicit verifier invariant because it directly limits the strongest mechanism claim: the retained v2 experiment does not show a tail-F1 benefit from memory, gating, or regime supervision.

## Interpretation boundary

The seed-level rows make the mixed result easier to audit. They do not change the frozen aggregate or its scientific interpretation. In particular, the ledger preserves the adverse seed-7 regime-supervision result, the seed-31 gate-disabled spectral degradation, the seed-43 near-tie between full and no-memory Eig NMSE, and the strong seed-59 full-vs-no-memory Eig NMSE gap. No seed can be omitted without the verifier failing.

For conference-facing use, the aggregate table remains the primary result. The seed-level table is appropriate as supplementary evidence supporting transparency, reproducibility, and the claim boundary.
