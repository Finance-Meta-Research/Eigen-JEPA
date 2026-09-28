# arXiv Submission Readiness — Eigen-JEPA

**Status:** preprint packaging candidate; scientific conclusion remains mixed/descriptive.

This checklist prepares the frozen five-seed synthetic study for a possible arXiv preprint without changing any outcome, seed, model, metric, threshold, or interpretation.

## Scientific lock

- [x] Frozen five-seed protocol retained: seeds {7, 19, 31, 43, 59}.
- [x] Four paper-facing variants retained: full, no_memory, no_gate, no_regime.
- [x] All 20 seed-by-variant runs retained in the final-rigor v2 evidence lane.
- [x] Null/adverse outcomes retained: Tail F1 does not separate full from no_memory; no_gate and no_regime each improve at least one central error metric.
- [x] No post-outcome significance test or practical-effect threshold introduced.
- [x] Synthetic-only scope stated explicitly.
- [x] Classical covariance comparator **control plane** is implemented and pre-outcome reviewed on current main: sample persistence, OAS, EWMA, low-rank factor covariance, sample shrinkage, and EWMA+shrinkage across an exact 21-candidate validation-only grid.
- [x] The classical comparator protocol remains pre-outcome and unauthorized; merging its implementation did not open or inspect a real-market test result.
- [x] The current synthetic paper makes no claim of superiority to DCC, shrinkage, OAS, factor-covariance, random-matrix, or other classical covariance methods.
- [ ] If a future manuscript version adds any classical real-market comparison, freeze/authorize/execute/report that separately before inclusion; do not infer the result from the existence of the comparator code.
- [ ] Do not use any future real-market result to rewrite the frozen synthetic v2 conclusion.

## arXiv preprint gate

- [ ] Freeze final author list and obtain approval from every listed author.
- [ ] Choose the scientifically appropriate primary arXiv category; do not select a category solely because an endorsement is available.
- [ ] Confirm submitter eligibility for that primary category and any cross-lists.
- [ ] Select the license/copyright option with author approval.
- [ ] Rebuild the canonical manuscript from the exact release commit.
- [ ] Verify generated tables against retained final-rigor v2 artifacts and paired seed-level data.
- [ ] Verify bibliography entries against primary sources.
- [ ] Search the source package for placeholders, TODOs, private paths, credentials, and stale representative-run results.
- [ ] Remove repository material that is not required to compile the paper; do not package checkpoints or raw private data into the arXiv source upload.
- [ ] Confirm the abstract and conclusion say: mixed component-specific synthetic evidence, no trading-alpha claim, no real-market validation, no full-model dominance.
- [ ] Confirm no wording implies classical-baseline superiority merely because the pre-outcome comparator infrastructure exists.
- [ ] Human visual inspection of every PDF page, equations, tables, references, and headers.
- [ ] Record final Git commit, source-package SHA-256, and PDF SHA-256 in a release receipt.
- [ ] Submit manually and retain the arXiv submission receipt/version identifier.

## Prohibited upgrades

The frozen study does **not** establish trading alpha, portfolio performance, real-market generalization, statistical significance of component effects, universal benefit from selective memory, or superiority to classical covariance estimators. The merged classical-baseline infrastructure is a prospective control plane, not positive or negative comparator evidence. Any future study testing those questions must remain separately versioned.
