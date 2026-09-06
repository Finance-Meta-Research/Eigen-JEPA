#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

EXPECTED_STATUS = "FROZEN_PREOUTCOME_SOURCE_BASIS_NOT_AUTHORIZED"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_CRITICAL_PATHS = (
    "scripts/prepare_real_market_input.py",
    "eigen_jepa/real_market_folds.py",
    "scripts/run_real_market_confirmation.py",
    "scripts/analyze_real_market_confirmation.py",
    "scripts/verify_real_market_confirmation_result.py",
)


class SourceFreezeError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SourceFreezeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SourceFreezeError(f"{path}: expected JSON object")
    return payload


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise SourceFreezeError(f"missing frozen file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    if not path.is_file():
        raise SourceFreezeError(f"missing frozen file: {path}")
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _require_hex40(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX40.fullmatch(value) is None:
        raise SourceFreezeError(f"{label} must be exact lowercase 40-hex")
    return value


def _require_hex64(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise SourceFreezeError(f"{label} must be exact lowercase 64-hex")
    return value


def _verify_blob_binding(repo_root: Path, relative_path: str, expected: Any) -> dict[str, str]:
    expected_sha = _require_hex40(expected, label=f"{relative_path} git_blob_sha1")
    actual = _git_blob_sha1(repo_root / relative_path)
    if actual != expected_sha:
        raise SourceFreezeError(
            f"frozen Git blob drift for {relative_path}: expected {expected_sha}, observed {actual}"
        )
    return {"git_blob_sha1": actual}


def verify_source_freeze(
    freeze: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    if freeze.get("schema_version") != 1:
        raise SourceFreezeError("source freeze schema_version must equal 1")
    if freeze.get("status") != EXPECTED_STATUS:
        raise SourceFreezeError(f"source freeze status must equal {EXPECTED_STATUS}")
    if freeze.get("execution_authorized") is not False:
        raise SourceFreezeError("source freeze must keep execution_authorized=false")
    if freeze.get("outcome_access_authorized") is not False:
        raise SourceFreezeError("source freeze must keep outcome_access_authorized=false")

    source_commit = _require_hex40(freeze.get("source_commit"), label="source_commit")

    protocol_binding = freeze.get("candidate_protocol")
    if not isinstance(protocol_binding, Mapping):
        raise SourceFreezeError("candidate_protocol binding is required")
    protocol_path = protocol_binding.get("path")
    if protocol_path != "protocols/real_market_confirmation_v1_candidate_20260906.json":
        raise SourceFreezeError("unexpected candidate protocol path")
    protocol_observed = _verify_blob_binding(
        repo_root, protocol_path, protocol_binding.get("git_blob_sha1")
    )
    protocol = _load_json(repo_root / protocol_path)
    if protocol.get("status") != "CANDIDATE_PRE_DATA_FREEZE_NOT_AUTHORIZED":
        raise SourceFreezeError("candidate protocol authorization boundary drifted")
    if protocol.get("data_freeze_requirements", {}).get("execution_authorized") is not False:
        raise SourceFreezeError("candidate protocol must keep execution_authorized=false")

    dependency = freeze.get("dependency_lock")
    if not isinstance(dependency, Mapping):
        raise SourceFreezeError("dependency_lock binding is required")
    dependency_path = dependency.get("path")
    if dependency_path != protocol.get("source_freeze_requirements", {}).get("dependency_lock_path"):
        raise SourceFreezeError("dependency lock path disagrees with candidate protocol")
    dependency_observed = _verify_blob_binding(
        repo_root, dependency_path, dependency.get("git_blob_sha1")
    )
    expected_dependency_sha256 = _require_hex64(
        dependency.get("sha256"), label="dependency lock sha256"
    )
    actual_dependency_sha256 = _sha256_file(repo_root / dependency_path)
    protocol_dependency_sha256 = protocol.get("source_freeze_requirements", {}).get(
        "dependency_lock_sha256"
    )
    if actual_dependency_sha256 != expected_dependency_sha256:
        raise SourceFreezeError("dependency lock SHA-256 drifted from source freeze")
    if actual_dependency_sha256 != protocol_dependency_sha256:
        raise SourceFreezeError("dependency lock SHA-256 disagrees with candidate protocol")
    dependency_observed["sha256"] = actual_dependency_sha256

    bindings = freeze.get("critical_code_bindings")
    if not isinstance(bindings, Mapping) or tuple(bindings.keys()) != EXPECTED_CRITICAL_PATHS:
        raise SourceFreezeError(
            f"critical_code_bindings must contain exactly ordered paths {EXPECTED_CRITICAL_PATHS}"
        )

    observed: dict[str, dict[str, str]] = {}
    for relative_path in EXPECTED_CRITICAL_PATHS:
        entry = bindings.get(relative_path)
        if not isinstance(entry, Mapping):
            raise SourceFreezeError(f"missing binding object for {relative_path}")
        observed[relative_path] = _verify_blob_binding(
            repo_root, relative_path, entry.get("git_blob_sha1")
        )

    # The candidate protocol separately binds the execution/analysis controls by SHA-256.
    runner = protocol.get("evaluation_design", {}).get("fold_runner", {})
    analysis = protocol.get("analysis_freeze_requirements", {})
    protocol_sha_bindings = {
        runner.get("path"): runner.get("sha256"),
        analysis.get("analysis_script"): analysis.get("analysis_script_sha256"),
        analysis.get("result_verifier"): analysis.get("result_verifier_sha256"),
    }
    expected_sha_paths = {
        "scripts/run_real_market_confirmation.py",
        "scripts/analyze_real_market_confirmation.py",
        "scripts/verify_real_market_confirmation_result.py",
    }
    if set(protocol_sha_bindings) != expected_sha_paths:
        raise SourceFreezeError("candidate protocol execution/analysis path bindings drifted")
    for relative_path, expected_sha in protocol_sha_bindings.items():
        expected_sha = _require_hex64(expected_sha, label=f"{relative_path} protocol sha256")
        actual_sha = _sha256_file(repo_root / relative_path)
        if actual_sha != expected_sha:
            raise SourceFreezeError(
                f"candidate protocol SHA-256 drift for {relative_path}: expected {expected_sha}, observed {actual_sha}"
            )
        observed[relative_path]["sha256"] = actual_sha

    return {
        "status": "SOURCE_FREEZE_VERIFIED_NOT_AUTHORIZED",
        "execution_authorized": False,
        "outcome_access_authorized": False,
        "source_commit": source_commit,
        "candidate_protocol": {
            "path": protocol_path,
            **protocol_observed,
        },
        "dependency_lock": {
            "path": dependency_path,
            **dependency_observed,
        },
        "critical_code_bindings": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the prospectively frozen Eigen-JEPA real-market source basis without authorizing outcome access."
    )
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("protocols/real_market_source_freeze_candidate_20260906.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    freeze = _load_json(args.freeze)
    result = verify_source_freeze(freeze, repo_root=args.repo_root)
    print(result["status"])
    print(f"SOURCE_COMMIT={result['source_commit']}")
    print(f"CRITICAL_BINDINGS={len(result['critical_code_bindings'])}")


if __name__ == "__main__":
    main()
