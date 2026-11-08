#!/usr/bin/env python3
"""Read-only verification of every saved final-rigor v2 aggregate.

This supplements, rather than changes, the historical scientific gate. It
requires the original hash-bound raw metrics and recomputes recorded arithmetic.
It neither executes a model nor authenticates how the original run was trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

SEEDS = [7, 19, 31, 43, 59]
VARIANTS = ["full", "no_memory", "no_gate", "no_regime"]
METRICS = {
    "cov_mse", "drift_mse", "eig_nmse", "entropy_mse", "event_rate", "gap_rmse",
    "gate_cal", "gate_mean", "lat_align", "proj_mse", "rank_mse", "rare_cov_mse",
    "rare_drift_mse", "rare_eig_nmse", "regime_acc", "regime_bal_acc", "risk_mse", "tail_f1",
}
BASELINE_METRICS = METRICS - {"lat_align", "rare_cov_mse", "rare_drift_mse", "rare_eig_nmse"}
BASELINES = {"persistence", "trend"}
UNIT_INTERVAL = {"event_rate", "gate_cal", "gate_mean", "regime_acc", "regime_bal_acc", "tail_f1"}
LEDGER_SHA256 = "a5b4586ce8ee4dbd377fdd55117003680a5cb10c8256690da7bb195aa30d6c8c"
ROOT = Path(__file__).resolve().parents[1]
# Empty-event walkforward slices in the original seed-43 archive contain these
# NaNs. They are preserved, explicitly reported, and never admitted as metrics.
RETAINED_NAN_PATHS = frozenset(
    f"walkforward/{window}/{metric}"
    for window in ("mid", "late")
    for metric in ("rare_cov_mse", "rare_drift_mse", "rare_eig_nmse")
)


class EvidenceError(ValueError):
    """A retained package is malformed, changed, or arithmetically inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(raw: bytes, *, retained_nan_paths: frozenset[str] = frozenset()) -> dict:
    try:
        data = json.loads(raw, object_pairs_hook=_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid JSON: {exc}") from exc
    require(isinstance(data, dict), "JSON root must be an object")
    found_nan = set()

    def inspect(value: object, path: str = "") -> None:
        if isinstance(value, float) and not math.isfinite(value):
            require(math.isnan(value) and path in retained_nan_paths,
                    f"non-finite JSON value: {path}")
            found_nan.add(path)
        elif isinstance(value, dict):
            for key, item in value.items():
                inspect(item, f"{path}/{key}".lstrip("/"))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                inspect(item, f"{path}/{index}".lstrip("/"))

    inspect(data)
    require(found_nan == retained_nan_paths, "retained auxiliary NaN surface drift")
    return data


def exact_keys(data: object, expected: set[str], label: str) -> None:
    require(isinstance(data, dict) and set(data) == expected, f"{label}: field set drift")


def number(value: object, label: str, *, unit_interval: bool = False) -> float:
    require(type(value) in (int, float), f"{label}: expected a numeric scalar, not bool/string")
    try:
        valid = math.isfinite(value) and value >= 0
    except (OverflowError, TypeError):
        valid = False
    require(valid, f"{label}: expected a finite nonnegative value")
    if unit_interval:
        require(value <= 1, f"{label}: expected a value in [0, 1]")
    return float(value)


def metric_values(data: object, expected: set[str], label: str) -> None:
    exact_keys(data, expected, label)
    for metric, value in data.items():
        number(value, f"{label}/{metric}", unit_interval=metric in UNIT_INTERVAL)


def check_summary(summary: object, values: list[float], label: str) -> dict:
    exact_keys(summary, {"mean", "std"}, label)
    observed = {key: number(value, f"{label}/{key}") for key, value in summary.items()}
    recomputed = {"mean": statistics.mean(values), "std": statistics.pstdev(values)}
    for field in recomputed:
        require(math.isclose(observed[field], recomputed[field], rel_tol=0.0, abs_tol=1e-12),
                f"{label}/{field}: recorded {observed[field]!r} != recomputed {recomputed[field]!r}")
    return {"metric": label, "values_aggregated": len(values), "independent_seeds": len(SEEDS),
            "recorded": observed, "recomputed": recomputed}


def validate_payload(summary: dict, runs: dict[tuple[int, str], dict]) -> list[dict]:
    """Validate parsed data; generated fixtures can exercise this without outcomes."""
    exact_keys(summary, {"aggregate", "benchmark", "market_style", "num_seeds",
                         "representative_run", "seed_list", "variants"}, "summary")
    require(type(summary["num_seeds"]) is int and summary["num_seeds"] == 5, "num_seeds drift")
    require(isinstance(summary["seed_list"], list)
            and all(type(seed) is int for seed in summary["seed_list"])
            and summary["seed_list"] == SEEDS, "seed set/order/type drift")
    require(summary["variants"] == VARIANTS, "variant set/order drift")
    require(summary["market_style"] == "equity", "market_style drift")
    require(summary["representative_run"] == "results/final_rigor_v2/seed_7/full",
            "representative_run drift")
    require(set(runs) == {(seed, variant) for seed in SEEDS for variant in VARIANTS},
            "expected exactly 20 seed-by-variant runs")
    exact_keys(summary["aggregate"], set(VARIANTS), "aggregate")
    exact_keys(summary["benchmark"], BASELINES, "benchmark")
    for (seed, variant), run in runs.items():
        label = f"seed_{seed}/{variant}"
        require(isinstance(run, dict), f"{label}: run must be an object")
        metric_values(run.get("test"), METRICS, f"{label}/test")
        history = run.get("history")
        require(isinstance(history, list) and len(history) == 8, f"{label}: expected eight epochs")
        for epoch, row in enumerate(history, 1):
            require(isinstance(row, dict) and type(row.get("epoch")) is int
                    and row["epoch"] == epoch, f"{label}: epoch sequence drift")
            require(len(row) > 1, f"{label}: empty epoch metrics")
            for field, value in row.items():
                number(value, f"{label}/history/{epoch}/{field}")
        exact_keys(run.get("benchmark"), BASELINES, f"{label}/benchmark")
        for baseline in sorted(BASELINES):
            metric_values(run["benchmark"][baseline], BASELINE_METRICS, f"{label}/{baseline}")

    checks = []
    for variant in VARIANTS:
        exact_keys(summary["aggregate"][variant], METRICS, f"aggregate/{variant}")
        for metric in sorted(METRICS):
            values = [runs[seed, variant]["test"][metric] for seed in SEEDS]
            checks.append(check_summary(summary["aggregate"][variant][metric], values,
                                        f"aggregate/{variant}/{metric}"))
    # The frozen benchmark.py appends baseline values once per seed/variant.
    # Reproduce that 20-row arithmetic and explicitly retain only five independent
    # seeds; repeated variants are not extra experimental replication.
    for baseline in sorted(BASELINES):
        exact_keys(summary["benchmark"][baseline], BASELINE_METRICS, f"benchmark/{baseline}")
        for metric in sorted(BASELINE_METRICS):
            values = [runs[seed, variant]["benchmark"][baseline][metric]
                      for seed in SEEDS for variant in VARIANTS]
            for seed in SEEDS:
                per_seed = [runs[seed, variant]["benchmark"][baseline][metric] for variant in VARIANTS]
                require(all(value == per_seed[0] for value in per_seed),
                        f"benchmark/{baseline}/{metric}: variant-dependent baseline at seed {seed}")
            checks.append(check_summary(summary["benchmark"][baseline][metric], values,
                                        f"benchmark/{baseline}/{metric}"))
    return checks


def verify_package(artifact_root: Path, ledger_path: Path) -> dict:
    ledger_raw = ledger_path.read_bytes()
    require(hashlib.sha256(ledger_raw).hexdigest() == LEDGER_SHA256, "retained seed-ledger SHA-256 drift")
    ledger = read_json(ledger_raw)
    rows = {(row["seed"], row["variant"]): row for row in ledger["rows"]}
    runs, source_hashes, retained_nan = {}, {}, []
    for seed in SEEDS:
        for variant in VARIANTS:
            relative = f"seed_{seed}/{variant}/metrics.json"
            raw = (artifact_root / relative).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            require(digest == rows[seed, variant]["source_sha256"], f"raw SHA-256 drift: {relative}")
            allowed_nan = RETAINED_NAN_PATHS if seed == 43 else frozenset()
            runs[seed, variant] = read_json(raw, retained_nan_paths=allowed_nan)
            retained_nan.extend(f"{relative}:{path}" for path in sorted(allowed_nan))
            source_hashes[relative] = digest
    aggregate_raw = (artifact_root / "metrics.json").read_bytes()
    checks = validate_payload(read_json(aggregate_raw), runs)
    return {"status": "RETAINED_V2_PACKAGE_PASS", "schema_version": 1,
            "protocol": ledger["source_protocol"], "execution_head": ledger["execution_head"],
            "workflow_run_id": ledger["workflow_run_id"], "artifact_id": ledger["artifact_id"],
            "archive_digest": ledger["artifact_digest"], "ledger_sha256": LEDGER_SHA256,
            "aggregate_sha256": hashlib.sha256(aggregate_raw).hexdigest(),
            "raw_file_sha256": source_hashes, "model_summary_count": 72,
            "baseline_summary_count": 28, "aggregate_checks": checks,
            "nonfinite_auxiliary_diagnostics": retained_nan,
            "auxiliary_nan_boundary": "24 original seed-43 walkforward rare-event NaNs are preserved outside the audited test/baseline/history metrics; they are not usable numerical evidence.",
            "standard_deviation": "population", "absolute_tolerance": 1e-12,
            "scientific_runs": 0, "protected_outcomes_accessed": False,
            "interpretation": "Saved-file identity and arithmetic only; mixed/negative predecessor remains closed. No independent training reproduction, significance test, practical-effect threshold, or successor execution authorization."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "results/final_rigor_v2")
    parser.add_argument("--ledger", type=Path, default=ROOT / "results/final_rigor_v2/SEED_LEVEL_METRICS_20260906.json")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable receipt to stdout")
    args = parser.parse_args()
    try:
        receipt = verify_package(args.artifact_root, args.ledger)
    except (EvidenceError, OSError) as exc:
        raise SystemExit(f"RETAINED_V2_PACKAGE_FAIL: {exc}") from exc
    if args.json:
        print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("RETAINED_V2_PACKAGE_PASS: 20 hash-bound raw files; 72 model and 28 baseline means/population SDs recomputed; five independent seeds; no training or new outcomes.")


if __name__ == "__main__":
    main()
