import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_real_market_freeze_bundle import (
    EXPECTED_ASSETS,
    EXPECTED_STATUS,
    FreezeBundleError,
    build_review_request,
    validate_candidate_protocol,
    verify_implementation_bindings,
    write_json_once,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    runner = b"runner-v1\n"
    analysis = b"analysis-v1\n"
    verifier = b"verifier-v1\n"
    prepare = b"prepare-v1\n"
    folds_impl = b"folds-v1\n"
    _write(repo / "scripts/run_real_market_confirmation.py", runner)
    _write(repo / "scripts/analyze_real_market_confirmation.py", analysis)
    _write(repo / "scripts/verify_real_market_confirmation_result.py", verifier)
    _write(repo / "scripts/prepare_real_market_input.py", prepare)
    _write(repo / "eigen_jepa/real_market_folds.py", folds_impl)

    protocol = {
        "schema_version": 1,
        "protocol_id": "candidate-test",
        "status": EXPECTED_STATUS,
        "asset_universe": {"symbols_exact": list(EXPECTED_ASSETS)},
        "candidate_date_span": {"start": "2010-01-04", "end": "2025-12-31"},
        "data_freeze_requirements": {"execution_authorized": False},
        "evaluation_design": {
            "folds": [{"id": "F1"}, {"id": "F2"}, {"id": "F3"}],
            "fold_runner": {
                "path": "scripts/run_real_market_confirmation.py",
                "sha256": _sha(runner),
            },
        },
        "analysis_freeze_requirements": {
            "analysis_script": "scripts/analyze_real_market_confirmation.py",
            "analysis_script_sha256": _sha(analysis),
            "result_verifier": "scripts/verify_real_market_confirmation_result.py",
            "result_verifier_sha256": _sha(verifier),
        },
        "hypotheses": {"primary": {"endpoint": "pooled_eig_nmse"}},
        "claim_boundary": "bounded claim",
    }
    protocol_path = repo / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, sort_keys=True), encoding="utf-8")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    csv_path = _write(repo / "returns.csv", b"date,SPY\n2010-01-04,0.0\n2025-12-31,0.0\n")
    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    receipt = {
        "schema_version": 1,
        "status": "PREPARED_INPUT_ONLY_NOT_AUTHORIZED",
        "dataset": {
            "sha256": csv_sha,
            "rows": 2,
            "date_start": "2010-01-04",
            "date_end": "2025-12-31",
            "return_cols": list(EXPECTED_ASSETS),
            "num_assets": 12,
        },
    }
    receipt_path = repo / "input_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    plan = {
        "schema_version": 1,
        "status": "PREOUTCOME_FOLD_PLAN_ONLY",
        "protocol_sha256": protocol_sha,
        "normalized_return_csv_sha256": csv_sha,
        "folds": {
            "F1": {"test_count": 10},
            "F2": {"test_count": 11},
            "F3": {"test_count": 12},
        },
    }
    plan_path = repo / "fold_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    raw_path = _write(repo / "raw_source.bin", b"immutable-provider-snapshot")
    lock_path = _write(repo / "requirements.lock", b"numpy==2.0.0\n")
    return {
        "repo": repo,
        "protocol": protocol,
        "protocol_path": protocol_path,
        "csv_path": csv_path,
        "receipt_path": receipt_path,
        "plan_path": plan_path,
        "raw_path": raw_path,
        "lock_path": lock_path,
    }


def _build(fx):
    return build_review_request(
        protocol_path=fx["protocol_path"],
        raw_source_path=fx["raw_path"],
        normalized_csv_path=fx["csv_path"],
        input_receipt_path=fx["receipt_path"],
        fold_plan_path=fx["plan_path"],
        dependency_lock_path=fx["lock_path"],
        source_commit="1" * 40,
        provider_identity="provider-a",
        provider_snapshot_or_retrieval_id="snapshot-20260906",
        repo_root=fx["repo"],
    )


def test_valid_bundle_is_review_only_and_binds_all_material_inputs(tmp_path):
    fx = _fixture(tmp_path)
    request = _build(fx)
    assert request["status"] == "PREOUTCOME_REVIEW_REQUEST_NOT_AUTHORIZED"
    assert request["execution_authorized"] is False
    assert request["independent_review_required"]["approved"] is False
    assert request["data_freeze"]["normalized_return_csv"]["sha256"] == hashlib.sha256(
        fx["csv_path"].read_bytes()
    ).hexdigest()
    assert request["source_freeze"]["source_commit"] == "1" * 40
    assert request["source_freeze"]["implementation_files"][
        "scripts/run_real_market_confirmation.py"
    ]["protocol_bound"] is True
    assert "model-result" in request["builder_attestation"][1]


def test_rejects_protocol_whose_authorization_boundary_was_already_flipped(tmp_path):
    fx = _fixture(tmp_path)
    protocol = copy.deepcopy(fx["protocol"])
    protocol["status"] = "FROZEN_PRE_OUTCOME_AUTHORIZED"
    protocol["data_freeze_requirements"]["execution_authorized"] = True
    fx["protocol_path"].write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(FreezeBundleError, match="protocol status must remain"):
        _build(fx)


def test_rejects_input_receipt_that_does_not_bind_csv_bytes(tmp_path):
    fx = _fixture(tmp_path)
    receipt = json.loads(fx["receipt_path"].read_text(encoding="utf-8"))
    receipt["dataset"]["sha256"] = "0" * 64
    fx["receipt_path"].write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(FreezeBundleError, match="does not bind the supplied normalized CSV"):
        _build(fx)


def test_rejects_fold_plan_built_from_different_protocol_or_with_outcome_fields(tmp_path):
    fx = _fixture(tmp_path)
    plan = json.loads(fx["plan_path"].read_text(encoding="utf-8"))
    plan["protocol_sha256"] = "0" * 64
    fx["plan_path"].write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(FreezeBundleError, match="not built from the supplied candidate protocol"):
        _build(fx)

    fx = _fixture(tmp_path / "second")
    plan = json.loads(fx["plan_path"].read_text(encoding="utf-8"))
    plan["folds"]["F2"]["eig_nmse"] = 0.123
    fx["plan_path"].write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(FreezeBundleError, match="outcome-like fields"):
        _build(fx)


def test_rejects_frozen_implementation_byte_drift(tmp_path):
    fx = _fixture(tmp_path)
    (fx["repo"] / "scripts/analyze_real_market_confirmation.py").write_text(
        "analysis-mutated\n", encoding="utf-8"
    )
    with pytest.raises(FreezeBundleError, match="frozen implementation drift"):
        _build(fx)


def test_rejects_nonimmutable_source_identity(tmp_path):
    fx = _fixture(tmp_path)
    with pytest.raises(FreezeBundleError, match="source_commit must be exact lowercase 40-hex"):
        build_review_request(
            protocol_path=fx["protocol_path"],
            raw_source_path=fx["raw_path"],
            normalized_csv_path=fx["csv_path"],
            input_receipt_path=fx["receipt_path"],
            fold_plan_path=fx["plan_path"],
            dependency_lock_path=fx["lock_path"],
            source_commit="ABC",
            provider_identity="provider-a",
            provider_snapshot_or_retrieval_id="snapshot",
            repo_root=fx["repo"],
        )


def test_review_request_is_non_overwriting_retained_evidence(tmp_path):
    path = tmp_path / "request.json"
    digest = write_json_once(path, {"status": "x"})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_json_once(path, {"status": "y"})


def test_current_repository_candidate_still_matches_its_frozen_execution_tools():
    protocol_path = Path("protocols/real_market_confirmation_v1_candidate_20260906.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    validate_candidate_protocol(protocol)
    bindings = verify_implementation_bindings(protocol, repo_root=Path("."))
    assert bindings["scripts/run_real_market_confirmation.py"]["protocol_bound"] is True
    assert bindings["scripts/analyze_real_market_confirmation.py"]["protocol_bound"] is True
    assert bindings["scripts/verify_real_market_confirmation_result.py"]["protocol_bound"] is True
