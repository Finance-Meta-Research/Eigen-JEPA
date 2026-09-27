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


def _authorized_protocol():
    protocol = copy.deepcopy(_candidate_protocol())
    protocol["status"] = BASELINE_STATUS
    protocol["execution_authorized"] = True
    protocol["implementation"]["source_commit"] = "a" * 40
    protocol["implementation"]["reference_protocol_sha256"] = "b" * 64
    return protocol


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
    with pytest.raises(ConfirmationRunnerError, match="source_commit must be"):
        assert_baseline_execution_authorized(protocol)


def test_bound_source_commit_must_match_running_checkout():
    protocol = _authorized_protocol()
    with pytest.raises(ConfirmationRunnerError, match="does not match bound source_commit"):
        assert_baseline_execution_authorized(
            protocol,
            actual_source_commit="c" * 40,
            actual_git_dirty=False,
            actual_reference_protocol_sha256="b" * 64,
        )


def test_dirty_checkout_fails_before_test_outcome_execution():
    protocol = _authorized_protocol()
    with pytest.raises(ConfirmationRunnerError, match="working tree must be clean"):
        assert_baseline_execution_authorized(
            protocol,
            actual_source_commit="a" * 40,
            actual_git_dirty=True,
            actual_reference_protocol_sha256="b" * 64,
        )


def test_reference_protocol_hash_must_be_bound():
    protocol = _authorized_protocol()
    protocol["implementation"]["reference_protocol_sha256"] = None
    with pytest.raises(ConfirmationRunnerError, match="reference_protocol_sha256 must be"):
        assert_baseline_execution_authorized(
            protocol,
            actual_source_commit="a" * 40,
            actual_git_dirty=False,
            actual_reference_protocol_sha256="b" * 64,
        )


def test_reference_protocol_hash_must_match_runtime_file():
    protocol = _authorized_protocol()
    with pytest.raises(ConfirmationRunnerError, match="reference-protocol SHA-256"):
        assert_baseline_execution_authorized(
            protocol,
            actual_source_commit="a" * 40,
            actual_git_dirty=False,
            actual_reference_protocol_sha256="c" * 64,
        )


def test_fully_bound_clean_checkout_can_pass_authority_gate():
    protocol = _authorized_protocol()
    assert_baseline_execution_authorized(
        protocol,
        actual_source_commit="a" * 40,
        actual_git_dirty=False,
        actual_reference_protocol_sha256="b" * 64,
    ) is None


def test_family_drift_is_rejected():
    protocol = copy.deepcopy(_candidate_protocol())
    protocol["reporting_families_exact"] = list(EXPECTED_FAMILIES[:-1])
    with pytest.raises(ConfirmationRunnerError, match="baseline family drift"):
        validate_baseline_protocol(protocol)


def test_reference_protocol_path_drift_is_rejected():
    protocol = copy.deepcopy(_candidate_protocol())
    protocol["implementation"]["reference_protocol"] = "protocols/other.json"
    with pytest.raises(ConfirmationRunnerError, match="reference protocol path"):
        validate_baseline_protocol(protocol)
