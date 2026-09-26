import copy
import json
from pathlib import Path

import pytest

from scripts.run_real_market_classical_baselines import (
    BASELINE_STATUS,
    EXPECTED_FAMILIES,
    ConfirmationRunnerError,
    _families,
    assert_baseline_execution_authorized,
    validate_baseline_protocol,
)


def _candidate_protocol():
    return json.loads(
        Path(
            "protocols/real_market_classical_baseline_ladder_v1_candidate_20260926.json"
        ).read_text(encoding="utf-8")
    )


def test_candidate_baseline_protocol_matches_implementation_grid():
    protocol = _candidate_protocol()
    validate_baseline_protocol(protocol)
    families = _families()
    assert tuple(families) == EXPECTED_FAMILIES
    assert sum(len(v) for v in families.values()) == 21


def test_candidate_baseline_protocol_cannot_access_test_outcomes():
    protocol = _candidate_protocol()
    assert protocol["status"] != BASELINE_STATUS
    with pytest.raises(ConfirmationRunnerError, match="baseline protocol status must be"):
        assert_baseline_execution_authorized(protocol)


def test_status_flip_without_source_binding_still_fails_closed():
    protocol = copy.deepcopy(_candidate_protocol())
    protocol["status"] = BASELINE_STATUS
    protocol["execution_authorized"] = True
    with pytest.raises(ConfirmationRunnerError, match="source_commit must be bound"):
        assert_baseline_execution_authorized(protocol)


def test_family_drift_is_rejected():
    protocol = copy.deepcopy(_candidate_protocol())
    protocol["reporting_families_exact"] = list(EXPECTED_FAMILIES[:-1])
    with pytest.raises(ConfirmationRunnerError, match="baseline family drift"):
        validate_baseline_protocol(protocol)
