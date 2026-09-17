# Eigen-JEPA paper rigor gate

## Current publication status

**Canonical final-rigor v2 gate: PASS on the retained frozen evidence package.**

The older `results/final_rigor/metrics.json` package is a historical single-seed artifact and still fails the generic robustness gate if checked on its own. It is **not** the canonical paper-facing evidence source and must not be used to support submission claims.

The canonical publication evidence is the frozen v2 package:

- protocol: `protocols/final_rigor_v2_20260905.json`;
- seeds: `7, 19, 31, 43, 59`;
- variants: `full`, `no_memory`, `no_gate`, `no_regime`;
- retained run count: 20 seed-by-variant metric files;
- retained execution head: `9369a5f2b0b972af846fa70baca027451271e08c`;
- GitHub Actions run: `33988159305`;
- retained artifact: `9975833698`;
- artifact digest: `sha256:5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4`;
- observed result summary: `results/final_rigor_v2/RESULT_SUMMARY_20260906.md`.

The retained verifier result recorded with that package is:

```text
FINAL_RIGOR_V2_PASS: exact seeds, variants, 20 retained run-metric files, and 72 finite aggregate summaries verified.
```

## Canonical machine gate

For publication-facing evidence, run:

```bash
python scripts/check_final_rigor_v2.py
```

This checker fails closed unless the exact preregistered v2 seed list, variant order, market style, aggregate structure, all 20 retained run files, and finite aggregate summaries are present. It then calls the generic rigor gate on the v2 metrics with `--min-seeds 5` and requires a literal `RIGOR_GATE_PASS` before emitting `FINAL_RIGOR_V2_PASS`.

The older command remains useful only as a generic checker when pointed at a chosen metrics file:

```bash
python scripts/check_rigor_gate.py --metrics <metrics.json> --min-seeds <N>
```

Do not run the generic checker against the legacy single-seed `results/final_rigor/metrics.json` and interpret that expected failure as the current paper status.

## Evidence and claim boundary

The v2 evidence is multi-seed but scientifically mixed. The paper may make only the bounded descriptive claims documented in `PUBLISHING.md` and the retained result summary.

In particular:

- the full model has better mean Eig NMSE and Gate Cal than `no_memory` under the frozen synthetic-equity v2 benchmark;
- relative to `no_memory`, full has lower Eig NMSE on 4/5 seeds and lower Gate Cal on 5/5 seeds;
- Tail F1 is identical across all four paper-facing variants at every retained seed;
- `no_gate` has better mean Drift MSE than the full model;
- `no_regime` has the best mean Eig NMSE among the four paper-facing variants;
- the frozen v2 protocol did not preregister a significance test or practical-effect threshold, so component comparisons remain descriptive.

Do **not** claim trading alpha, deployable trading performance, real-market validation, cross-market validation, full-model dominance, statistical significance, or superiority to unevaluated classical covariance baselines.

## Historical single-seed artifact

The legacy `results/final_rigor/metrics.json` declares one representative seed (`7`). Its zero standard deviations are single-seed aggregation artifacts, not estimates of run-to-run variability. That package remains useful for provenance/history only and must not become the canonical submission surface again.

## Final release checks

1. Run `python scripts/check_final_rigor_v2.py` against the retained v2 package.
2. Regenerate `paper/final_rigor_v2_paired_table.tex` with `python scripts/build_final_rigor_v2_paired_table.py` and require a clean git diff.
3. Build `paper/main.tex`; it must resolve to `paper/conference_v2.tex`.
4. Confirm aggregate values match `results/final_rigor_v2/RESULT_SUMMARY_20260906.md` and paired directions match `results/final_rigor_v2/SEED_LEVEL_METRICS_20260906.json`.
5. Confirm no prose reintroduces representative-run, multi-market, statistical-significance, or future-tense mechanism claims as observed v2 results.
6. Keep any real-market confirmation pre-outcome until its separately frozen provider snapshot, normalized input receipt, fold/source plan, and independent approval are complete.

## Remaining blocker outside this gate

Passing the canonical v2 rigor gate does **not** mean the paper is fully submission-ready. A fresh submission-PDF build plus venue-specific page-limit, anonymity, template, and portal checks are still required. The separately frozen real-market confirmation remains a distinct pre-outcome scientific gate and is not required to preserve the bounded synthetic-v2 result.
