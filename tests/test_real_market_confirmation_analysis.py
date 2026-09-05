import copy

import numpy as np
import pytest

from scripts.analyze_real_market_confirmation import (
    AnalysisInputError,
    EXPECTED_FOLDS,
    EXPECTED_SEEDS,
    analyze,
    bootstrap_difference,
    observed_effect,
    validate_payload,
)


def _payload(full_scale=0.80, no_memory_scale=1.0, windows=24):
    folds = {}
    offset = 0
    for fold_no, fold_id in enumerate(EXPECTED_FOLDS):
        idx = list(range(offset, offset + windows))
        offset += windows
        target = [1.0 + 0.01 * ((i + fold_no) % 7) for i in range(windows)]
        full = {}
        no_memory = {}
        for seed_no, seed in enumerate(EXPECTED_SEEDS):
            seed_factor = 1.0 + 0.01 * (seed_no - 2)
            base = np.asarray(target) * seed_factor
            no_memory[seed] = (no_memory_scale * base).tolist()
            full[seed] = (full_scale * base).tolist()
        folds[fold_id] = {
            "window_indices": idx,
            "target_energy": target,
            "full": full,
            "no_memory": no_memory,
        }
    return {"schema_version": 1, "folds": folds}


def test_observed_effect_is_paired_and_pooled_over_all_folds_and_seeds():
    parsed = validate_payload(_payload(full_scale=0.8, no_memory_scale=1.0))
    effect = observed_effect(parsed)
    assert effect["full_nmse"] == pytest.approx(0.8)
    assert effect["no_memory_nmse"] == pytest.approx(1.0)
    assert effect["difference_full_minus_no_memory"] == pytest.approx(-0.2)
    assert effect["relative_reduction"] == pytest.approx(0.2)


def test_bootstrap_is_deterministic_for_frozen_seed():
    parsed = validate_payload(_payload())
    a = bootstrap_difference(parsed, block_length=5, resamples=100, seed=123)
    b = bootstrap_difference(parsed, block_length=5, resamples=100, seed=123)
    assert np.array_equal(a, b)
    assert np.all(a < 0)


def test_primary_pass_requires_effect_size_and_interval():
    result = analyze(_payload(full_scale=0.8, no_memory_scale=1.0))
    assert result["analysis_status"] == "PRIMARY_PASS"
    assert result["decision_rule"]["relative_reduction_pass"] is True
    assert result["decision_rule"]["ci_upper_below_zero_pass"] is True


def test_small_improvement_fails_practical_effect_gate_even_if_direction_is_consistent():
    result = analyze(_payload(full_scale=0.98, no_memory_scale=1.0))
    assert result["analysis_status"] == "PRIMARY_FAIL"
    assert result["decision_rule"]["relative_reduction_pass"] is False


def test_rejects_seed_dropping_and_fold_overlap():
    payload = _payload()
    del payload["folds"]["F1"]["full"]["59"]
    with pytest.raises(AnalysisInputError, match="seeds must be exactly"):
        validate_payload(payload)

    payload = _payload()
    payload["folds"]["F2"]["window_indices"][0] = payload["folds"]["F1"]["window_indices"][-1]
    payload["folds"]["F2"]["window_indices"].sort()
    with pytest.raises(AnalysisInputError, match="overlap another fold"):
        validate_payload(payload)


def test_rejects_target_or_error_tampering_shapes_and_nonfinite_values():
    payload = _payload()
    payload["folds"]["F1"]["target_energy"][0] = 0.0
    with pytest.raises(AnalysisInputError, match="must be > 0"):
        validate_payload(payload)

    payload = _payload()
    payload["folds"]["F1"]["full"]["7"][0] = float("nan")
    with pytest.raises(AnalysisInputError, match="non-finite"):
        validate_payload(payload)

    payload = _payload()
    payload["folds"]["F1"]["no_memory"]["7"].pop()
    with pytest.raises(AnalysisInputError, match="length mismatch"):
        validate_payload(payload)
