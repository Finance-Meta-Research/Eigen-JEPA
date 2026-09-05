#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from scripts.analyze_real_market_confirmation import (
    BLOCK_LENGTH,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    MIN_RELATIVE_REDUCTION,
    RESAMPLES,
    AnalysisInputError,
    analyze,
)

EXPECTED_ANALYSIS_ID = "eigen-jepa-real-market-confirmation-analysis-v1"
EXPECTED_PRIMARY_ENDPOINT = "pooled_eig_nmse"


class ResultVerificationError(ValueError):
    pass


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ResultVerificationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultVerificationError(f"{path}: expected a JSON object")
    return payload


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _require_finite_tree(value: Any, *, path: str = "result") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResultVerificationError(f"{path}: contains non-finite float")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_tree(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_tree(item, path=f"{path}.{key}")
        return
    raise ResultVerificationError(f"{path}: unsupported value type {type(value).__name__}")


def verify_protocol_analysis_freeze(
    protocol: Mapping[str, Any],
    *,
    analysis_script_path: Path,
    verifier_path: Path,
) -> None:
    freeze = protocol.get("analysis_freeze_requirements")
    if not isinstance(freeze, Mapping):
        raise ResultVerificationError("protocol missing analysis_freeze_requirements")

    expected = {
        "block_length_windows": BLOCK_LENGTH,
        "resamples": RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
    }
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise ResultVerificationError(
                f"protocol analysis freeze drift for {key}: expected {value!r}, "
                f"observed {freeze.get(key)!r}"
            )

    script_sha = _sha256_bytes(analysis_script_path)
    if freeze.get("analysis_script_sha256") != script_sha:
        raise ResultVerificationError(
            "analysis script bytes do not match protocol analysis_script_sha256"
        )

    verifier_sha = _sha256_bytes(verifier_path)
    if freeze.get("result_verifier_sha256") != verifier_sha:
        raise ResultVerificationError(
            "result verifier bytes do not match protocol result_verifier_sha256"
        )

    primary = protocol.get("hypotheses", {}).get("primary")
    if not isinstance(primary, Mapping):
        raise ResultVerificationError("protocol missing primary hypothesis")
    if primary.get("endpoint") != EXPECTED_PRIMARY_ENDPOINT:
        raise ResultVerificationError("protocol primary endpoint drift")
    practical_rule = str(primary.get("practical_effect_rule", ""))
    if "5 percent" not in practical_rule:
        raise ResultVerificationError("protocol no longer states the frozen 5 percent effect gate")


def verify_result(
    analysis_input: Mapping[str, Any],
    retained_result: Mapping[str, Any],
) -> dict[str, Any]:
    _require_finite_tree(retained_result)
    if retained_result.get("schema_version") != 1:
        raise ResultVerificationError("result schema_version must equal 1")
    if retained_result.get("analysis_id") != EXPECTED_ANALYSIS_ID:
        raise ResultVerificationError("analysis_id drift")
    if retained_result.get("primary_endpoint") != EXPECTED_PRIMARY_ENDPOINT:
        raise ResultVerificationError("primary endpoint drift")
    if retained_result.get("analysis_status") not in {"PRIMARY_PASS", "PRIMARY_FAIL"}:
        raise ResultVerificationError("analysis_status must be PRIMARY_PASS or PRIMARY_FAIL")

    try:
        recomputed = analyze(analysis_input)
    except (AnalysisInputError, ValueError) as exc:
        raise ResultVerificationError(f"analysis input rejected during recomputation: {exc}") from exc

    if _canonical_json(retained_result) != _canonical_json(recomputed):
        raise ResultVerificationError(
            "retained result does not exactly equal deterministic recomputation from retained input"
        )

    decision = retained_result.get("decision_rule")
    if not isinstance(decision, Mapping):
        raise ResultVerificationError("missing decision_rule")
    if decision.get("minimum_relative_nmse_reduction") != MIN_RELATIVE_REDUCTION:
        raise ResultVerificationError("minimum relative reduction drift")
    expected_pass = bool(
        decision.get("relative_reduction_pass") is True
        and decision.get("ci_upper_below_zero_pass") is True
    )
    if decision.get("primary_pass") is not expected_pass:
        raise ResultVerificationError("primary_pass is inconsistent with its two frozen gates")
    expected_status = "PRIMARY_PASS" if expected_pass else "PRIMARY_FAIL"
    if retained_result.get("analysis_status") != expected_status:
        raise ResultVerificationError("analysis_status is inconsistent with frozen decision gates")

    uncertainty = retained_result.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise ResultVerificationError("missing uncertainty block")
    expected_uncertainty = {
        "confidence_level": CONFIDENCE_LEVEL,
        "block_length_windows": BLOCK_LENGTH,
        "resamples": RESAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for key, value in expected_uncertainty.items():
        if uncertainty.get(key) != value:
            raise ResultVerificationError(f"uncertainty setting drift for {key}")

    boundary = str(retained_result.get("claim_boundary", ""))
    for forbidden_expansion in ("trading alpha", "profitability", "causality", "Tail-F1"):
        if forbidden_expansion not in boundary:
            raise ResultVerificationError(
                f"claim boundary no longer explicitly excludes {forbidden_expansion}"
            )
    return recomputed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed unless a retained Eigen-JEPA real-market confirmation result is "
            "the exact deterministic output of the predeclared analysis implementation."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/real_market_confirmation_v1_candidate_20260906.json"),
    )
    parser.add_argument(
        "--analysis-script",
        type=Path,
        default=Path("scripts/analyze_real_market_confirmation.py"),
    )
    args = parser.parse_args()

    protocol = _load_json(args.protocol)
    analysis_input = _load_json(args.input)
    retained_result = _load_json(args.result)
    verify_protocol_analysis_freeze(
        protocol,
        analysis_script_path=args.analysis_script,
        verifier_path=Path(__file__),
    )
    recomputed = verify_result(analysis_input, retained_result)

    print("REAL_MARKET_CONFIRMATION_RESULT_VERIFIED")
    print(f"INPUT_SHA256={_sha256_bytes(args.input)}")
    print(f"RESULT_SHA256={_sha256_bytes(args.result)}")
    print(f"ANALYSIS_STATUS={recomputed['analysis_status']}")


if __name__ == "__main__":
    main()
