#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

EXPECTED_STATUS = "CANDIDATE_PRE_DATA_FREEZE_NOT_AUTHORIZED"
OUTPUT_STATUS = "PREOUTCOME_REVIEW_REQUEST_NOT_AUTHORIZED"
EXPECTED_ASSETS = (
    "SPY", "IWM", "QQQ", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"
)
EXPECTED_FOLDS = ("F1", "F2", "F3")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
OUTCOME_KEYS = {
    "eig_nmse", "tail_f1", "drift_mse", "gate_cal", "regime_bal_acc",
    "window_rows", "test_metrics", "model_results", "primary_result", "p_value",
}


class FreezeBundleError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FreezeBundleError(f"missing required file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FreezeBundleError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FreezeBundleError(f"{path}: expected a JSON object")
    return payload


def require_hex(value: Any, *, length: int, label: str) -> str:
    pattern = HEX40 if length == 40 else HEX64
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise FreezeBundleError(f"{label} must be exact lowercase {length}-hex")
    return value


def validate_candidate_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise FreezeBundleError("protocol schema_version must equal 1")
    if protocol.get("status") != EXPECTED_STATUS:
        raise FreezeBundleError(
            f"protocol status must remain {EXPECTED_STATUS!r} while building a review request"
        )
    freeze = protocol.get("data_freeze_requirements")
    if not isinstance(freeze, Mapping) or freeze.get("execution_authorized") is not False:
        raise FreezeBundleError("candidate protocol must explicitly keep execution_authorized=false")
    if tuple(protocol.get("asset_universe", {}).get("symbols_exact", ())) != EXPECTED_ASSETS:
        raise FreezeBundleError("asset universe drifted from the prospectively fixed 12-ETF panel")
    folds = protocol.get("evaluation_design", {}).get("folds")
    if not isinstance(folds, list) or tuple(fold.get("id") for fold in folds) != EXPECTED_FOLDS:
        raise FreezeBundleError(f"fold identity drift: expected {EXPECTED_FOLDS}")

    runner = protocol.get("evaluation_design", {}).get("fold_runner", {})
    if runner.get("path") != "scripts/run_real_market_confirmation.py":
        raise FreezeBundleError("unexpected frozen fold runner path")
    require_hex(runner.get("sha256"), length=64, label="fold runner sha256")

    analysis = protocol.get("analysis_freeze_requirements", {})
    if analysis.get("analysis_script") != "scripts/analyze_real_market_confirmation.py":
        raise FreezeBundleError("unexpected frozen analysis script path")
    if analysis.get("result_verifier") != "scripts/verify_real_market_confirmation_result.py":
        raise FreezeBundleError("unexpected frozen result verifier path")
    require_hex(analysis.get("analysis_script_sha256"), length=64, label="analysis script sha256")
    require_hex(analysis.get("result_verifier_sha256"), length=64, label="result verifier sha256")


def validate_input_receipt(
    receipt: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    normalized_csv_sha256: str,
    raw_source_sha256: str,
    provider_identity: str,
    provider_snapshot_or_retrieval_id: str,
) -> None:
    if receipt.get("schema_version") != 1 or receipt.get("status") != "PREPARED_INPUT_ONLY_NOT_AUTHORIZED":
        raise FreezeBundleError("input receipt must be a PREPARED_INPUT_ONLY_NOT_AUTHORIZED v1 receipt")
    dataset = receipt.get("dataset")
    if not isinstance(dataset, Mapping):
        raise FreezeBundleError("input receipt is missing dataset metadata")
    if dataset.get("sha256") != normalized_csv_sha256:
        raise FreezeBundleError("input receipt does not bind the supplied normalized CSV bytes")
    if tuple(dataset.get("return_cols", ())) != EXPECTED_ASSETS:
        raise FreezeBundleError("input receipt return columns do not match the frozen asset order")
    if dataset.get("num_assets") != len(EXPECTED_ASSETS):
        raise FreezeBundleError("input receipt must bind exactly 12 assets")

    span = protocol.get("candidate_date_span", {})
    if dataset.get("date_start") != span.get("start"):
        raise FreezeBundleError("normalized input start date drifted from the candidate protocol")
    if dataset.get("date_end") != span.get("end"):
        raise FreezeBundleError("normalized input end date drifted from the candidate protocol")

    source = receipt.get("source_provenance")
    if not isinstance(source, Mapping):
        raise FreezeBundleError("input receipt must bind raw-source provenance before review")
    if source.get("raw_source_sha256") != raw_source_sha256:
        raise FreezeBundleError("input receipt raw-source hash does not match supplied source bytes")
    if source.get("provider_identity") != provider_identity:
        raise FreezeBundleError("input receipt provider identity drift")
    if source.get("provider_snapshot_or_retrieval_id") != provider_snapshot_or_retrieval_id:
        raise FreezeBundleError("input receipt provider snapshot/retrieval identity drift")


def _reject_outcome_fields(value: Any, *, path: str = "folds") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            if key_text in OUTCOME_KEYS:
                raise FreezeBundleError(f"{path}.{key_text}: outcome-like field found in pre-outcome fold plan")
            _reject_outcome_fields(nested, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_outcome_fields(nested, path=f"{path}[{index}]")


def validate_fold_plan(
    plan: Mapping[str, Any],
    *,
    protocol_sha256: str,
    normalized_csv_sha256: str,
) -> None:
    if plan.get("schema_version") != 1 or plan.get("status") != "PREOUTCOME_FOLD_PLAN_ONLY":
        raise FreezeBundleError("fold plan must be a PREOUTCOME_FOLD_PLAN_ONLY v1 artifact")
    if plan.get("protocol_sha256") != protocol_sha256:
        raise FreezeBundleError("fold plan was not built from the supplied candidate protocol bytes")
    if plan.get("normalized_return_csv_sha256") != normalized_csv_sha256:
        raise FreezeBundleError("fold plan was not built from the supplied normalized CSV bytes")
    folds = plan.get("folds")
    if not isinstance(folds, Mapping) or tuple(folds.keys()) != EXPECTED_FOLDS:
        raise FreezeBundleError(f"fold plan must contain ordered folds {EXPECTED_FOLDS}")
    for fold_id in EXPECTED_FOLDS:
        if not isinstance(folds.get(fold_id), Mapping):
            raise FreezeBundleError(f"{fold_id}: missing fold summary")
    _reject_outcome_fields(folds)


def verify_implementation_bindings(protocol: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    runner = protocol["evaluation_design"]["fold_runner"]
    analysis = protocol["analysis_freeze_requirements"]
    expected = {
        runner["path"]: runner["sha256"],
        analysis["analysis_script"]: analysis["analysis_script_sha256"],
        analysis["result_verifier"]: analysis["result_verifier_sha256"],
    }
    observed: dict[str, Any] = {}
    for relative, expected_sha in expected.items():
        actual = sha256_file(repo_root / relative)
        if actual != expected_sha:
            raise FreezeBundleError(
                f"frozen implementation drift for {relative}: expected {expected_sha}, observed {actual}"
            )
        observed[relative] = {"sha256": actual, "protocol_bound": True}

    for relative in ("scripts/prepare_real_market_input.py", "eigen_jepa/real_market_folds.py"):
        observed[relative] = {
            "sha256": sha256_file(repo_root / relative),
            "protocol_bound": False,
            "role": "pre-outcome data/fold construction dependency retained for independent review",
        }
    return observed


def build_review_request(
    *,
    protocol_path: Path,
    raw_source_path: Path,
    normalized_csv_path: Path,
    input_receipt_path: Path,
    fold_plan_path: Path,
    dependency_lock_path: Path,
    source_commit: str,
    provider_identity: str,
    provider_snapshot_or_retrieval_id: str,
    repo_root: Path,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    validate_candidate_protocol(protocol)
    source_commit = require_hex(source_commit, length=40, label="source_commit")
    provider_identity = provider_identity.strip()
    provider_snapshot_or_retrieval_id = provider_snapshot_or_retrieval_id.strip()
    if not provider_identity:
        raise FreezeBundleError("provider_identity must be non-empty")
    if not provider_snapshot_or_retrieval_id:
        raise FreezeBundleError("provider_snapshot_or_retrieval_id must be non-empty")

    protocol_sha = sha256_file(protocol_path)
    raw_sha = sha256_file(raw_source_path)
    csv_sha = sha256_file(normalized_csv_path)
    receipt_sha = sha256_file(input_receipt_path)
    plan_sha = sha256_file(fold_plan_path)
    dependency_sha = sha256_file(dependency_lock_path)

    receipt = load_json(input_receipt_path)
    validate_input_receipt(
        receipt,
        protocol=protocol,
        normalized_csv_sha256=csv_sha,
        raw_source_sha256=raw_sha,
        provider_identity=provider_identity,
        provider_snapshot_or_retrieval_id=provider_snapshot_or_retrieval_id,
    )
    plan = load_json(fold_plan_path)
    validate_fold_plan(plan, protocol_sha256=protocol_sha, normalized_csv_sha256=csv_sha)
    implementation = verify_implementation_bindings(protocol, repo_root=repo_root)

    return {
        "schema_version": 1,
        "status": OUTPUT_STATUS,
        "execution_authorized": False,
        "protocol": {
            "protocol_id": protocol.get("protocol_id"),
            "sha256": protocol_sha,
            "candidate_status": protocol.get("status"),
        },
        "data_freeze": {
            "provider_identity": provider_identity,
            "provider_snapshot_or_retrieval_id": provider_snapshot_or_retrieval_id,
            "raw_source": {
                "path": raw_source_path.as_posix(),
                "sha256": raw_sha,
                "bytes": raw_source_path.stat().st_size,
            },
            "normalized_return_csv": {
                "path": normalized_csv_path.as_posix(),
                "sha256": csv_sha,
                "bytes": normalized_csv_path.stat().st_size,
                "rows": receipt["dataset"].get("rows"),
                "date_start": receipt["dataset"].get("date_start"),
                "date_end": receipt["dataset"].get("date_end"),
                "return_cols": receipt["dataset"].get("return_cols"),
            },
            "input_receipt": {"path": input_receipt_path.as_posix(), "sha256": receipt_sha},
            "fold_plan": {
                "path": fold_plan_path.as_posix(),
                "sha256": plan_sha,
                "folds": list(EXPECTED_FOLDS),
            },
            "raw_to_normalized_lineage_verified": True,
        },
        "source_freeze": {
            "source_commit": source_commit,
            "dependency_lock": {"path": dependency_lock_path.as_posix(), "sha256": dependency_sha},
            "implementation_files": implementation,
        },
        "analysis_freeze": {
            "primary_comparison": protocol.get("hypotheses", {}).get("primary"),
            "analysis_script_sha256": protocol["analysis_freeze_requirements"]["analysis_script_sha256"],
            "result_verifier_sha256": protocol["analysis_freeze_requirements"]["result_verifier_sha256"],
        },
        "independent_review_required": {
            "approved": False,
            "review_must_precede_any_model_or_test_outcome_access": True,
            "reviewer_must_confirm": [
                "provider/source snapshot identity and adjustment semantics",
                "raw-to-normalized lineage plus raw/normalized hashes and input receipt",
                "fold-plan identity and purged chronological boundaries",
                "prospective source commit and dependency lock",
                "frozen runner, analysis, and result-verifier byte identities",
                "primary endpoint, practical-effect rule, bootstrap procedure, and claim boundary",
            ],
        },
        "claim_boundary": protocol.get("claim_boundary"),
        "builder_attestation": [
            "This artifact is a review request only and cannot authorize execution.",
            "The builder accepts no model-result or test-metric input and does not inspect outcome artifacts.",
            "Any later authorization must bind this exact review-request SHA-256 and a separate independent approval receipt.",
        ],
    }


def write_json_once(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained review request: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable pre-outcome Eigen-JEPA real-market freeze bundle for independent review. "
            "This tool never authorizes model/test execution."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--raw-source", type=Path, required=True)
    parser.add_argument("--normalized-csv", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--fold-plan", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--provider-identity", required=True)
    parser.add_argument("--provider-snapshot-or-retrieval-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    request = build_review_request(
        protocol_path=args.protocol,
        raw_source_path=args.raw_source,
        normalized_csv_path=args.normalized_csv,
        input_receipt_path=args.input_receipt,
        fold_plan_path=args.fold_plan,
        dependency_lock_path=args.dependency_lock,
        source_commit=args.source_commit,
        provider_identity=args.provider_identity,
        provider_snapshot_or_retrieval_id=args.provider_snapshot_or_retrieval_id,
        repo_root=args.repo_root,
    )
    digest = write_json_once(args.out, request)
    print("REAL_MARKET_PREOUTCOME_REVIEW_REQUEST_READY_NOT_AUTHORIZED")
    print(f"REVIEW_REQUEST_SHA256={digest}")


if __name__ == "__main__":
    main()
