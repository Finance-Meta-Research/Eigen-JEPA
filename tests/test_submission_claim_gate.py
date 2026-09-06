from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import verify_submission_claim_gate as gate


ROOT = Path(__file__).resolve().parents[1]


def test_current_submission_claim_gate_passes() -> None:
    gate.verify(ROOT)


def test_outcome_access_cannot_be_enabled() -> None:
    manifest = gate.load_manifest(ROOT / "paper" / "SUBMISSION_READINESS_20260906.json")
    mutated = copy.deepcopy(manifest)
    mutated["new_outcome_access_performed_by_this_gate"] = True
    with pytest.raises(ValueError, match="must not perform new outcome access"):
        gate.validate_manifest(mutated)


def test_submission_cannot_self_authorize() -> None:
    manifest = gate.load_manifest(ROOT / "paper" / "SUBMISSION_READINESS_20260906.json")
    mutated = copy.deepcopy(manifest)
    mutated["submission_ready"] = True
    with pytest.raises(ValueError, match="must not self-authorize submission"):
        gate.validate_manifest(mutated)


def test_frozen_seed_set_cannot_drift() -> None:
    manifest = gate.load_manifest(ROOT / "paper" / "SUBMISSION_READINESS_20260906.json")
    mutated = copy.deepcopy(manifest)
    mutated["final_rigor_v2"]["seeds"] = [7, 19, 31, 43]
    with pytest.raises(ValueError, match="frozen v2 identity changed: seeds"):
        gate.validate_manifest(mutated)


def test_historical_result_table_cannot_reenter_manuscript() -> None:
    manuscript = (ROOT / "paper" / "conference_v2.tex").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="historical output reintroduced"):
        gate.validate_manuscript(manuscript + "\n\\input{results_table.tex}\n")


def test_git_blob_binding_detects_byte_drift(tmp_path: Path) -> None:
    payload = b"retained evidence\n"
    path = tmp_path / "evidence.txt"
    path.write_bytes(payload)
    manifest = {
        "evidence_files": [
            {"path": "evidence.txt", "git_blob_sha1": gate.git_blob_sha1(payload)}
        ]
    }
    gate.validate_blob_bindings(tmp_path, manifest)
    path.write_bytes(payload + b"changed\n")
    with pytest.raises(ValueError, match="bound evidence blob drifted"):
        gate.validate_blob_bindings(tmp_path, manifest)
