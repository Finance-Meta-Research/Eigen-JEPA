import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_real_market_confirmation import EXPECTED_FOLDS, EXPECTED_SEEDS, analyze
from scripts.verify_real_market_confirmation_result import (
    ResultVerificationError,
    verify_protocol_analysis_freeze,
    verify_result,
)


def _payload(full_scale=0.8, no_memory_scale=1.0, windows=24):
    folds = {}
    offset = 0
    for fold_no, fold_id in enumerate(EXPECTED_FOLDS):
        indices = list(range(offset, offset + windows))
        offset += windows
        target = [1.0 + 0.01 * ((i + fold_no) % 7) for i in range(windows)]
        full = {}
        no_memory = {}
        for seed_no, seed in enumerate(EXPECTED_SEEDS):
            seed_factor = 1.0 + 0.01 * (seed_no - 2)
            base = [value * seed_factor for value in target]
            full[seed] = [full_scale * value for value in base]
            no_memory[seed] = [no_memory_scale * value for value in base]
        folds[fold_id] = {
            "window_indices": indices,
            "target_energy": target,
            "full": full,
            "no_memory": no_memory,
        }
    return {"schema_version": 1, "folds": folds}


def test_repository_protocol_binds_exact_analysis_and_verifier_bytes():
    protocol = json.loads(
        Path("protocols/real_market_confirmation_v1_candidate_20260906.json").read_text(
            encoding="utf-8"
        )
    )
    verify_protocol_analysis_freeze(
        protocol,
        analysis_script_path=Path("scripts/analyze_real_market_confirmation.py"),
        verifier_path=Path("scripts/verify_real_market_confirmation_result.py"),
    )


def test_exact_deterministic_recomputation_passes():
    payload = _payload()
    retained = analyze(payload)
    recomputed = verify_result(payload, retained)
    assert recomputed == retained
    assert retained["analysis_status"] == "PRIMARY_PASS"


def test_metric_or_decision_tampering_fails_closed():
    payload = _payload()
    retained = analyze(payload)

    metric_tamper = copy.deepcopy(retained)
    metric_tamper["observed"]["full_nmse"] += 1e-6
    with pytest.raises(ResultVerificationError, match="deterministic recomputation"):
        verify_result(payload, metric_tamper)

    gate_tamper = copy.deepcopy(retained)
    gate_tamper["decision_rule"]["primary_pass"] = False
    with pytest.raises(ResultVerificationError, match="deterministic recomputation"):
        verify_result(payload, gate_tamper)


def test_nonfinite_result_fails_before_comparison():
    payload = _payload()
    retained = analyze(payload)
    retained["observed"]["full_nmse"] = float("nan")
    with pytest.raises(ResultVerificationError, match="non-finite"):
        verify_result(payload, retained)
