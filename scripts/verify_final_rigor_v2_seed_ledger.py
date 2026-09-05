#!/usr/bin/env python3
"""Verify the retained Eigen-JEPA final-rigor v2 seed-level evidence ledger.

The default mode is repository-only and verifies the frozen 5x4 seed/variant
surface, finite manuscript metrics, aggregate arithmetic, paired-difference
arithmetic, evidence identity, and the observed Tail-F1 equality invariant.

When --artifact-root is supplied, the verifier also SHA-256 checks each of the
20 retained raw metrics.json files and confirms that the ledger values exactly
match the raw ``test`` metrics. This mode is intended for the downloaded
workflow artifact from run 33988159305.

This verifier does not add a significance test or a practical-effect threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

EXPECTED_SCHEMA = "eigen-jepa-final-rigor-v2-seed-ledger-v1"
EXPECTED_PROTOCOL = "eigen-jepa-final-rigor-v2-20260905"
EXPECTED_EXECUTION_HEAD = "9369a5f2b0b972af846fa70baca027451271e08c"
EXPECTED_RUN = 33988159305
EXPECTED_ARTIFACT = 9975833698
EXPECTED_ARTIFACT_DIGEST = (
    "sha256:5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4"
)
EXPECTED_SEEDS = [7, 19, 31, 43, 59]
EXPECTED_VARIANTS = ["full", "no_memory", "no_gate", "no_regime"]
EXPECTED_METRICS = [
    "eig_nmse",
    "drift_mse",
    "cov_mse",
    "gate_cal",
    "tail_f1",
    "regime_acc",
    "rare_eig_nmse",
]
EXPECTED_AGGREGATES = {
    "full": {
        "eig_nmse": (0.30509522929787636, 0.07634160925041349),
        "drift_mse": (0.022995439885805054, 0.012829706353414978),
        "cov_mse": (0.09164243824779987, 0.04865604267098693),
        "gate_cal": (0.5002565979957581, 0.03572267818760646),
        "tail_f1": (0.5539317512162669, 0.19126704627626054),
        "regime_acc": (0.6166666666666667, 0.21188636891818535),
        "rare_eig_nmse": (0.372357173760732, 0.16287924705110376),
    },
    "no_memory": {
        "eig_nmse": (0.34519054790337883, 0.0358899534433878),
        "drift_mse": (0.029289797817667322, 0.009450429620198425),
        "cov_mse": (0.09204494506120682, 0.04654434948648592),
        "gate_cal": (0.5217843314011892, 0.048303938748600395),
        "tail_f1": (0.5539317512162669, 0.19126704627626054),
        "regime_acc": (0.6125, 0.21754629137011022),
        "rare_eig_nmse": (0.38845080261429155, 0.16367883256113278),
    },
    "no_gate": {
        "eig_nmse": (0.3225448320309321, 0.09876964814159545),
        "drift_mse": (0.020178189656386774, 0.008654829351451567),
        "cov_mse": (0.09062919213126104, 0.04690821816851445),
        "gate_cal": (0.55, 0.17608788840929535),
        "tail_f1": (0.5539317512162669, 0.19126704627626054),
        "regime_acc": (0.6166666666666667, 0.21188636891818535),
        "rare_eig_nmse": (0.39194369589289024, 0.17094665847852847),
    },
    "no_regime": {
        "eig_nmse": (0.2851836445430915, 0.08898060535403471),
        "drift_mse": (0.022838871367275713, 0.012888749225234343),
        "cov_mse": (0.09471529349684715, 0.04777853547416667),
        "gate_cal": (0.5000993549823761, 0.03587470671674477),
        "tail_f1": (0.5539317512162669, 0.19126704627626054),
        "regime_acc": (0.4958333333333334, 0.2905932629027116),
        "rare_eig_nmse": (0.34445292403300604, 0.17963972378223658),
    },
}


def fail(message: str) -> None:
    raise SystemExit(f"FINAL_RIGOR_V2_SEED_LEDGER_FAIL: {message}")


def close(a: float, b: float, tol: float = 1e-12) -> bool:
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)


def artifact_path(root: Path, source_path: str) -> Path:
    prefix = "results/final_rigor_v2/"
    if not source_path.startswith(prefix):
        fail(f"unexpected retained source path: {source_path}")
    return root / source_path[len(prefix):]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ledger",
        nargs="?",
        default="results/final_rigor_v2/SEED_LEVEL_METRICS_20260906.json",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Root of the downloaded final-rigor v2 workflow artifact.",
    )
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    data = json.loads(ledger_path.read_text())

    if data.get("schema_version") != EXPECTED_SCHEMA:
        fail("schema identity drift")
    if data.get("source_protocol") != EXPECTED_PROTOCOL:
        fail("protocol identity drift")
    if data.get("execution_head") != EXPECTED_EXECUTION_HEAD:
        fail("execution-head drift")
    if data.get("workflow_run_id") != EXPECTED_RUN:
        fail("workflow-run drift")
    if data.get("artifact_id") != EXPECTED_ARTIFACT:
        fail("artifact-id drift")
    if data.get("artifact_digest") != EXPECTED_ARTIFACT_DIGEST:
        fail("artifact-digest drift")
    if data.get("seeds") != EXPECTED_SEEDS:
        fail("seed set/order drift")
    if data.get("variants") != EXPECTED_VARIANTS:
        fail("variant set/order drift")
    if data.get("manuscript_metrics") != EXPECTED_METRICS:
        fail("manuscript metric set/order drift")

    boundary = data.get("interpretation_boundary", {})
    if boundary.get("post_outcome_significance_test_added") is not False:
        fail("post-outcome significance test introduced")
    if boundary.get("practical_effect_threshold_added") is not False:
        fail("post-outcome practical-effect threshold introduced")
    if boundary.get("all_seed_rows_retained") is not True:
        fail("ledger no longer attests complete retained rows")

    rows = data.get("rows")
    if not isinstance(rows, list) or len(rows) != 20:
        fail("expected exactly 20 seed-by-variant rows")

    row_map = {}
    for row in rows:
        key = (row.get("seed"), row.get("variant"))
        if key in row_map:
            fail(f"duplicate row {key}")
        if key[0] not in EXPECTED_SEEDS or key[1] not in EXPECTED_VARIANTS:
            fail(f"unexpected row {key}")
        sha = row.get("source_sha256", "")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            fail(f"invalid SHA-256 for {key}")
        for metric in EXPECTED_METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                fail(f"non-finite {metric} for {key}")
        row_map[key] = row

    expected_keys = {(s, v) for s in EXPECTED_SEEDS for v in EXPECTED_VARIANTS}
    if set(row_map) != expected_keys:
        fail("seed-by-variant surface is incomplete")

    aggregates = data.get("aggregates", {})
    for variant in EXPECTED_VARIANTS:
        for metric in EXPECTED_METRICS:
            xs = [float(row_map[(seed, variant)][metric]) for seed in EXPECTED_SEEDS]
            mean = sum(xs) / len(xs)
            pop_sd = statistics.pstdev(xs)
            recorded = aggregates.get(variant, {}).get(metric, {})
            if not close(recorded.get("mean"), mean):
                fail(f"aggregate mean drift: {variant}/{metric}")
            if not close(recorded.get("population_sd"), pop_sd):
                fail(f"aggregate population SD drift: {variant}/{metric}")
            exp_mean, exp_sd = EXPECTED_AGGREGATES[variant][metric]
            if not close(mean, exp_mean) or not close(pop_sd, exp_sd):
                fail(f"retained evidence drift: {variant}/{metric}")

    paired = data.get("paired_descriptive_differences", {})
    for ablation in ["no_memory", "no_gate", "no_regime"]:
        for metric in EXPECTED_METRICS:
            expected = [
                float(row_map[(seed, "full")][metric])
                - float(row_map[(seed, ablation)][metric])
                for seed in EXPECTED_SEEDS
            ]
            recorded = paired.get(ablation, {}).get(metric, {})
            got = recorded.get("paired_full_minus_ablation")
            if not isinstance(got, list) or len(got) != 5:
                fail(f"paired differences missing: {ablation}/{metric}")
            if any(not close(a, b) for a, b in zip(got, expected)):
                fail(f"paired differences drift: {ablation}/{metric}")
            if not close(recorded.get("mean_full_minus_ablation"), sum(expected) / 5):
                fail(f"paired mean drift: {ablation}/{metric}")

    for seed in EXPECTED_SEEDS:
        values = [row_map[(seed, variant)]["tail_f1"] for variant in EXPECTED_VARIANTS]
        if not all(close(values[0], value) for value in values[1:]):
            fail(f"Tail-F1 equality invariant changed at seed {seed}")

    if args.artifact_root is not None:
        for row in rows:
            source = artifact_path(args.artifact_root, row["source_path"])
            if not source.is_file():
                fail(f"retained raw metric file missing: {source}")
            raw = source.read_bytes()
            if hashlib.sha256(raw).hexdigest() != row["source_sha256"]:
                fail(f"raw metric SHA-256 mismatch: {source}")
            metric_doc = json.loads(raw)
            test = metric_doc.get("test", {})
            for metric in EXPECTED_METRICS:
                if not close(test.get(metric), row[metric]):
                    fail(f"raw metric value mismatch: {source}/{metric}")

    print(
        "FINAL_RIGOR_V2_SEED_LEDGER_PASS: "
        "20 retained rows, exact evidence identity, aggregates, paired arithmetic, "
        "and Tail-F1 equality invariant verified."
    )


if __name__ == "__main__":
    main()
