# Eigen-JEPA Publishing Notes

This project is designed as a research package for a paper submission.

## Current evidence-bound submission source

The canonical manuscript entry point is now `paper/main.tex`. It intentionally delegates to `paper/conference_v2.tex`, the evidence-bound conference manuscript reconciled to the frozen five-seed final-rigor v2 package. This prevents the older representative-run manuscript from silently becoming the default submission surface again.

The retained v2 evidence comes from GitHub Actions run `33988159305`, artifact `9975833698`, digest `sha256:5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4`. The frozen protocol is `protocols/final_rigor_v2_20260905.json` and the manuscript-facing aggregate table is `paper/final_rigor_v2_table.tex`.

## What is ready

- complete codebase and deterministic execution plumbing
- frozen final-rigor v2 protocol with exact source/environment/execution identities
- five precommitted seeds: `7, 19, 31, 43, 59`
- four paper-facing variants: `full`, `no_memory`, `no_gate`, `no_regime`
- all 20 seed-by-variant runs retained and verified for completeness
- evidence-bound result summary and LaTeX table
- canonical `paper/main.tex` entry point that resolves only to the evidence-bound conference manuscript
- CI regression coverage that fails if the canonical entry point or core frozen-v2 claim boundary drifts back to stale representative-run language

## What the paper may claim

The strongest defensible empirical claim is narrow: under the frozen synthetic-equity v2 benchmark, the full model has better mean Eig NMSE and Gate Cal than the matched `no_memory` variant, while the overall ablation evidence is mixed. Tail F1 is identical across all four paper-facing variants at every retained seed; `no_gate` has better mean Drift MSE than the full model; and `no_regime` has the best mean Eig NMSE among the four variants.

Because the frozen v2 protocol did not preregister a significance test or practical-effect threshold, these component comparisons are descriptive. Do not add post-hoc statistical-significance language.

Do **not** claim:

- trading alpha or deployable trading performance;
- real-market validation;
- joint equity/crypto/rates validation from v2;
- full-model dominance across metrics;
- a demonstrated Tail-F1 benefit from memory;
- a universal drift-prediction benefit from gating;
- statistically significant component effects without a separately frozen follow-up.

## Final checks before release

1. Re-run or verify the exact frozen v2 evidence package rather than substituting historical representative-run outputs.
2. Build `paper/main.tex`; it must resolve to `paper/conference_v2.tex`, with `paper/final_rigor_v2_table.tex` available in the same paper directory.
3. Confirm the table values match `results/final_rigor_v2/RESULT_SUMMARY_20260906.md` and the retained v2 artifact identity.
4. Confirm no prose reintroduces historical representative-run, multi-market, or future-tense mechanism claims as observed results.
5. Verify the README quick-start commands still work.
6. Add venue-specific formatting only after the evidence-bound text is stable; formatting changes must not broaden claims.
7. Freeze any real-market or cross-market follow-up protocol before inspecting its outcomes.

## What reviewers will likely care about

- chronological split discipline;
- why historical seed-7 evidence is excluded from the final aggregate;
- exact seed and variant retention;
- mixed ablation findings, especially the Tail-F1 tie;
- absence of post-hoc significance claims;
- source/environment/artifact provenance;
- whether broader real-market claims are clearly separated from the synthetic benchmark.
