import copy
import json
from pathlib import Path

import pytest

from scripts.verify_real_market_source_freeze import SourceFreezeError, verify_source_freeze


FREEZE_PATH = Path("protocols/real_market_source_freeze_candidate_20260906.json")


def _freeze():
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def test_current_repository_matches_frozen_preoutcome_source_basis():
    result = verify_source_freeze(_freeze(), repo_root=Path("."))
    assert result["status"] == "SOURCE_FREEZE_VERIFIED_NOT_AUTHORIZED"
    assert result["execution_authorized"] is False
    assert result["outcome_access_authorized"] is False
    assert result["source_commit"] == "057d723bdb693233ac47bcd4bd351fc8f36fc281"
    assert len(result["critical_code_bindings"]) == 5
    assert result["critical_code_bindings"]["scripts/run_real_market_confirmation.py"][
        "sha256"
    ]
    assert result["critical_code_bindings"]["scripts/analyze_real_market_confirmation.py"][
        "sha256"
    ]
    assert result["critical_code_bindings"][
        "scripts/verify_real_market_confirmation_result.py"
    ]["sha256"]


def test_source_freeze_cannot_authorize_execution_or_outcome_access():
    freeze = _freeze()
    freeze["execution_authorized"] = True
    with pytest.raises(SourceFreezeError, match="execution_authorized=false"):
        verify_source_freeze(freeze, repo_root=Path("."))

    freeze = _freeze()
    freeze["outcome_access_authorized"] = True
    with pytest.raises(SourceFreezeError, match="outcome_access_authorized=false"):
        verify_source_freeze(freeze, repo_root=Path("."))


def test_source_freeze_rejects_mutated_or_missing_critical_binding():
    freeze = _freeze()
    freeze["critical_code_bindings"]["scripts/prepare_real_market_input.py"][
        "git_blob_sha1"
    ] = "0" * 40
    with pytest.raises(SourceFreezeError, match="frozen Git blob drift"):
        verify_source_freeze(freeze, repo_root=Path("."))

    freeze = _freeze()
    freeze["critical_code_bindings"].pop("eigen_jepa/real_market_folds.py")
    with pytest.raises(SourceFreezeError, match="critical_code_bindings must contain exactly"):
        verify_source_freeze(freeze, repo_root=Path("."))


def test_source_freeze_rejects_nonimmutable_source_commit():
    freeze = _freeze()
    freeze["source_commit"] = "main"
    with pytest.raises(SourceFreezeError, match="source_commit must be exact lowercase 40-hex"):
        verify_source_freeze(freeze, repo_root=Path("."))


def test_source_freeze_rejects_protocol_or_dependency_identity_drift():
    freeze = _freeze()
    freeze["candidate_protocol"]["git_blob_sha1"] = "f" * 40
    with pytest.raises(SourceFreezeError, match="frozen Git blob drift"):
        verify_source_freeze(freeze, repo_root=Path("."))

    freeze = _freeze()
    freeze["dependency_lock"]["sha256"] = "0" * 64
    with pytest.raises(SourceFreezeError, match="dependency lock SHA-256 drifted"):
        verify_source_freeze(freeze, repo_root=Path("."))
