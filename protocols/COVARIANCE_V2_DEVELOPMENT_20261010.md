# Prospective covariance v2 numerical correction

State: SMOKE_PASSED (29 v1/v2 covariance tests); no new scientific outcomes. The original
`classical_covariance.py`, frozen runner/source identities and retained results
are unchanged. V2 is an explicit import, not a replacement in any frozen plan.

## Reproduced defects

1. V1 OAS compares its quartic denominator to an absolute 1e-30 floor. On a
   fixed 30-by-3 synthetic matrix, converting return units by 1e-12 changes the
   normalized covariance by relative norm 0.7248371627. The estimator should
   change by the square of the unit factor only.
2. V1 normalized eigenvalue error on predicted diag(2,4), true diag(1,3) is
   0.2; scaling both matrices by 1e-12 changes it to 1e-6. This metric should be
   dimensionless.
3. The v1 default factor ladder includes impossible ranks for two-asset inputs.

## V2 contract

OAS computes shrinkage on normalized returns, then restores covariance units.
NMSE divides jointly normalized squared errors by normalized target energy.
Zero/zero error is zero; nonzero error against zero energy is infinity and is
explicitly rejected during validation selection. Covariances must be real,
finite, symmetric and positive semidefinite. The candidate grid takes the
asset count and excludes impossible factor ranks before selection. All retained
candidate scores are required to be finite and candidate identities unique.

The hand-verifiable OAS example `[-3,0],[-1,0],[1,0],[3,0]` has empirical
covariance diag(5,0), shrinkage 0.8 and OAS covariance diag(3,2). Regression tests
also cover scaling from 1e-150 to 1e150 and dimensionless errors from 1e-250 to
1e250. Existing methods retain their v1 formulas except for the opt-in OAS and
metric corrections.

Run `python -m pytest tests/test_classical_covariance.py tests/test_classical_covariance_v2.py -q`.

Any adoption into real-market confirmation requires a new source-bound protocol
and the existing independent pre-outcome review. No held-out outcomes, provider
data, comparator ranking or market-performance claim is produced here.
