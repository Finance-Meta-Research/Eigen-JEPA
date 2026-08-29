# Eigen-JEPA paper rigor gate

## Current status: FAIL

The checked-in paper-facing artifact `results/final_rigor/metrics.json` currently declares:

- `num_seeds: 1`
- `seed_list: [7]`
- one representative run under `results/final_rigor/seed_7/full`

That is not sufficient evidence for robustness across random seeds. Standard deviations of `0.0` in this artifact are therefore single-seed aggregation artifacts, not empirical estimates of run-to-run variability.

## Machine gate

Run:

```bash
python scripts/check_rigor_gate.py
```

The gate intentionally fails until the canonical paper-facing result package contains at least three unique seeds. Five seeds are preferred for headline robustness claims when compute permits.

The check also verifies that:

- `num_seeds` exactly matches the unique seed list;
- the aggregate result mapping exists;
- every paper-facing aggregate mean is finite;
- every aggregate standard deviation is finite and non-negative.

## Required evidence before changing this status

1. Freeze the seed list before execution.
2. Run the full model and every paper-facing ablation/control under identical data splits, budgets, and evaluation code.
3. Preserve failed and adverse seeds rather than silently dropping them.
4. Regenerate `results/final_rigor/metrics.json` from the complete run set.
5. Run `python scripts/check_rigor_gate.py` and require `RIGOR_GATE_PASS`.
6. Recheck all manuscript statements against the multi-seed aggregate, including null or baseline-winning outcomes.

Do not lower `--min-seeds` merely to make the current artifact pass for publication.
