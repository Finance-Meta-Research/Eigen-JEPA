from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPECTED_FOLDS = ("F1", "F2", "F3")
EXPECTED_SEEDS = ("7", "19", "31", "43", "59")
BLOCK_LENGTH = 20
RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_906
CONFIDENCE_LEVEL = 0.95
MIN_RELATIVE_REDUCTION = 0.05


class AnalysisInputError(ValueError):
    pass


def _finite_nonnegative_array(values: Any, *, label: str, positive: bool = False) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AnalysisInputError(f"{label}: expected numeric array") from exc
    if array.ndim != 1 or array.size == 0:
        raise AnalysisInputError(f"{label}: expected non-empty 1D array")
    if not np.all(np.isfinite(array)):
        raise AnalysisInputError(f"{label}: contains non-finite values")
    if positive:
        if np.any(array <= 0):
            raise AnalysisInputError(f"{label}: all values must be > 0")
    elif np.any(array < 0):
        raise AnalysisInputError(f"{label}: all values must be >= 0")
    return array


def validate_payload(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != 1:
        raise AnalysisInputError("schema_version must equal 1")
    folds = payload.get("folds")
    if not isinstance(folds, Mapping) or tuple(sorted(folds)) != tuple(sorted(EXPECTED_FOLDS)):
        raise AnalysisInputError(f"folds must be exactly {EXPECTED_FOLDS}")

    parsed: dict[str, dict[str, Any]] = {}
    seen_window_ids: set[int] = set()
    for fold_id in EXPECTED_FOLDS:
        fold = folds[fold_id]
        if not isinstance(fold, Mapping):
            raise AnalysisInputError(f"{fold_id}: fold must be an object")

        window_ids_raw = fold.get("window_indices")
        if not isinstance(window_ids_raw, list) or not window_ids_raw:
            raise AnalysisInputError(f"{fold_id}: window_indices must be a non-empty list")
        if any(not isinstance(value, int) for value in window_ids_raw):
            raise AnalysisInputError(f"{fold_id}: window_indices must contain integers")
        window_ids = np.asarray(window_ids_raw, dtype=np.int64)
        if np.any(window_ids[1:] <= window_ids[:-1]):
            raise AnalysisInputError(f"{fold_id}: window_indices must be strictly increasing")
        overlap = seen_window_ids.intersection(int(x) for x in window_ids)
        if overlap:
            raise AnalysisInputError(f"{fold_id}: window indices overlap another fold: {sorted(overlap)[:5]}")
        seen_window_ids.update(int(x) for x in window_ids)

        target_energy = _finite_nonnegative_array(
            fold.get("target_energy"), label=f"{fold_id}.target_energy", positive=True
        )
        if len(target_energy) != len(window_ids):
            raise AnalysisInputError(f"{fold_id}: target_energy length mismatch")

        variant_arrays: dict[str, np.ndarray] = {}
        for variant in ("full", "no_memory"):
            by_seed = fold.get(variant)
            if not isinstance(by_seed, Mapping) or tuple(sorted(by_seed)) != tuple(sorted(EXPECTED_SEEDS)):
                raise AnalysisInputError(
                    f"{fold_id}.{variant}: seeds must be exactly {EXPECTED_SEEDS}"
                )
            seed_arrays = []
            for seed in EXPECTED_SEEDS:
                values = _finite_nonnegative_array(
                    by_seed[seed], label=f"{fold_id}.{variant}.{seed}"
                )
                if len(values) != len(window_ids):
                    raise AnalysisInputError(
                        f"{fold_id}.{variant}.{seed}: length mismatch"
                    )
                seed_arrays.append(values)
            variant_arrays[variant] = np.stack(seed_arrays, axis=0)

        parsed[fold_id] = {
            "window_indices": window_ids,
            "target_energy": target_energy,
            "full": variant_arrays["full"],
            "no_memory": variant_arrays["no_memory"],
        }
    return parsed


def _nmse(error: np.ndarray, target_energy: np.ndarray) -> float:
    denom = float(np.sum(target_energy))
    if not math.isfinite(denom) or denom <= 0:
        raise AnalysisInputError("pooled target energy must be positive and finite")
    return float(np.sum(error) / denom)


def observed_effect(parsed: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, float]:
    full_errors = []
    no_memory_errors = []
    energies = []
    for fold_id in EXPECTED_FOLDS:
        fold = parsed[fold_id]
        full_errors.append(np.mean(fold["full"], axis=0))
        no_memory_errors.append(np.mean(fold["no_memory"], axis=0))
        energies.append(fold["target_energy"])
    full = np.concatenate(full_errors)
    no_memory = np.concatenate(no_memory_errors)
    energy = np.concatenate(energies)
    full_nmse = _nmse(full, energy)
    no_memory_nmse = _nmse(no_memory, energy)
    if no_memory_nmse <= 0:
        raise AnalysisInputError("no_memory pooled NMSE must be > 0")
    difference = full_nmse - no_memory_nmse
    relative_reduction = 1.0 - full_nmse / no_memory_nmse
    return {
        "full_nmse": full_nmse,
        "no_memory_nmse": no_memory_nmse,
        "difference_full_minus_no_memory": difference,
        "relative_reduction": relative_reduction,
    }


def _moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        raise AnalysisInputError("cannot bootstrap an empty fold")
    block = min(block_length, n)
    max_start = n - block
    pieces = []
    selected = 0
    while selected < n:
        start = int(rng.integers(0, max_start + 1))
        idx = np.arange(start, start + block, dtype=np.int64)
        pieces.append(idx)
        selected += len(idx)
    return np.concatenate(pieces)[:n]


def bootstrap_difference(
    parsed: Mapping[str, Mapping[str, np.ndarray]],
    *,
    block_length: int = BLOCK_LENGTH,
    resamples: int = RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    if block_length <= 0:
        raise AnalysisInputError("block_length must be positive")
    if resamples <= 0:
        raise AnalysisInputError("resamples must be positive")
    rng = np.random.default_rng(seed)
    diffs = np.empty(resamples, dtype=np.float64)
    n_seeds = len(EXPECTED_SEEDS)

    for draw in range(resamples):
        sampled_seed_rows = rng.integers(0, n_seeds, size=n_seeds)
        pooled_full = []
        pooled_no_memory = []
        pooled_energy = []
        for fold_id in EXPECTED_FOLDS:
            fold = parsed[fold_id]
            time_idx = _moving_block_indices(
                len(fold["window_indices"]), block_length, rng
            )
            full = np.mean(fold["full"][sampled_seed_rows][:, time_idx], axis=0)
            no_memory = np.mean(
                fold["no_memory"][sampled_seed_rows][:, time_idx], axis=0
            )
            pooled_full.append(full)
            pooled_no_memory.append(no_memory)
            pooled_energy.append(fold["target_energy"][time_idx])

        energy = np.concatenate(pooled_energy)
        full_nmse = _nmse(np.concatenate(pooled_full), energy)
        no_memory_nmse = _nmse(np.concatenate(pooled_no_memory), energy)
        diffs[draw] = full_nmse - no_memory_nmse

    return diffs


def analyze(payload: Mapping[str, Any]) -> dict[str, Any]:
    parsed = validate_payload(payload)
    observed = observed_effect(parsed)
    bootstrap = bootstrap_difference(parsed)
    alpha = 1.0 - CONFIDENCE_LEVEL
    ci_low, ci_high = np.quantile(
        bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0], method="linear"
    )
    passes_effect = observed["relative_reduction"] >= MIN_RELATIVE_REDUCTION
    passes_interval = float(ci_high) < 0.0
    primary_pass = bool(passes_effect and passes_interval)

    return {
        "schema_version": 1,
        "analysis_id": "eigen-jepa-real-market-confirmation-analysis-v1",
        "analysis_status": "PRIMARY_PASS" if primary_pass else "PRIMARY_FAIL",
        "primary_endpoint": "pooled_eig_nmse",
        "observed": observed,
        "uncertainty": {
            "method": "two-stage paired bootstrap: resample frozen seeds with replacement; independently moving-block-resample chronological windows within each frozen fold",
            "confidence_level": CONFIDENCE_LEVEL,
            "block_length_windows": BLOCK_LENGTH,
            "resamples": RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "difference_ci_full_minus_no_memory": [
                float(ci_low),
                float(ci_high),
            ],
        },
        "decision_rule": {
            "minimum_relative_nmse_reduction": MIN_RELATIVE_REDUCTION,
            "relative_reduction_pass": bool(passes_effect),
            "ci_upper_below_zero_pass": bool(passes_interval),
            "primary_pass": primary_pass,
        },
        "claim_boundary": (
            "PRIMARY_PASS supports only the preregistered full-vs-no_memory "
            "eigenspectrum-error comparison on the frozen real-market confirmation "
            "package. PRIMARY_FAIL must be reported as failed confirmation. This "
            "analysis does not establish trading alpha, profitability, causality, "
            "universal cross-asset superiority, or a Tail-F1 benefit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze(payload)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(result["analysis_status"])


if __name__ == "__main__":
    main()
