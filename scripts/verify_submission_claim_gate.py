#!/usr/bin/env python3
"""Fail-closed verification for the evidence-bound Eigen-JEPA submission surface.

This script performs no scientific execution and reads no new outcome source. It only
reconciles frozen retained evidence and manuscript/package identities.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "paper" / "SUBMISSION_READINESS_20260906.json"

EXPECTED_BASE = "c9dee8572f4aaf43251e4e63268040ea89bdc644"
EXPECTED_RUN = 33988159305
EXPECTED_ARTIFACT = 9975833698
EXPECTED_DIGEST = "sha256:5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4"
EXPECTED_SEEDS = [7, 19, 31, 43, 59]
EXPECTED_VARIANTS = ["full", "no_memory", "no_gate", "no_regime"]


def fail(message: str) -> None:
    raise ValueError(f"SUBMISSION_CLAIM_GATE_FAIL: {message}")


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("evidence_base_head_sha") != EXPECTED_BASE:
        fail("evidence base head changed")
    if manifest.get("scientific_execution_performed_by_this_gate") is not False:
        fail("gate must not perform scientific execution")
    if manifest.get("new_outcome_access_performed_by_this_gate") is not False:
        fail("gate must not perform new outcome access")
    if manifest.get("real_market_confirmation_promoted_to_evidence") is not False:
        fail("real-market confirmation must remain separate from synthetic-v2 evidence")
    if manifest.get("submission_ready") is not False:
        fail("machine gate must not self-authorize submission")
    if manifest.get("release_authorization_asserted") is not False:
        fail("release authorization must not be fabricated")

    rigor = manifest.get("final_rigor_v2", {})
    expected = {
        "workflow_run_id": EXPECTED_RUN,
        "artifact_id": EXPECTED_ARTIFACT,
        "artifact_digest": EXPECTED_DIGEST,
        "seeds": EXPECTED_SEEDS,
        "variants": EXPECTED_VARIANTS,
        "retained_seed_variant_runs": 20,
    }
    for key, value in expected.items():
        if rigor.get(key) != value:
            fail(f"frozen v2 identity changed: {key}")

    boundary = manifest.get("claim_boundary", {})
    unsupported = " ".join(boundary.get("not_supported", [])).lower()
    for required in (
        "trading alpha",
        "real-market validation",
        "statistically significant",
        "full-model dominance",
        "tail-f1 benefit",
        "universal drift-prediction",
        "dcc",
        "covariance-shrinkage",
        "random-matrix",
    ):
        if required not in unsupported:
            fail(f"unsupported-claim boundary lost: {required}")


def validate_manuscript(text: str) -> None:
    required_fragments = (
        "The resulting evidence is mixed rather than a full-model superiority result",
        "all component comparisons are descriptive",
        "does not claim trading alpha or real-market validation",
        "Tail F1 was exactly identical between full and \\texttt{no\\_memory} at every frozen seed",
        "The gate-disabled variant achieved lower mean Drift MSE than the full model",
        "the best mean Eig NMSE among the four frozen variants",
        "do not establish real-market forecasting performance",
        "does not include matched DCC, shrinkage, or random-matrix forecasting baselines",
        "\\input{final_rigor_v2_table.tex}",
        "\\input{final_rigor_v2_paired_table.tex}",
    )
    for fragment in required_fragments:
        if fragment not in text:
            fail(f"required bounded manuscript statement missing: {fragment}")

    forbidden_inputs = ("results_table.tex", "benchmark_ablation.tex")
    for forbidden in forbidden_inputs:
        if forbidden in text:
            fail(f"historical output reintroduced into final-v2 manuscript: {forbidden}")


def validate_retained_summary(text: str) -> None:
    required = (
        "0.305095 ± 0.076342",
        "0.345191 ± 0.035890",
        "0.020178 ± 0.008655",
        "0.285184 ± 0.088981",
        "Tail F1 is exactly tied at every seed",
        "do **not** establish full-model dominance",
    )
    for fragment in required:
        if fragment not in text:
            fail(f"retained summary drifted: {fragment}")


def validate_blob_bindings(root: Path, manifest: dict[str, Any]) -> None:
    entries = manifest.get("evidence_files")
    if not isinstance(entries, list) or not entries:
        fail("evidence file binding list missing")
    seen: set[str] = set()
    for entry in entries:
        path = entry.get("path")
        expected_sha = entry.get("git_blob_sha1")
        if not isinstance(path, str) or not path or path in seen:
            fail("invalid or duplicate evidence path")
        seen.add(path)
        file_path = root / path
        if not file_path.is_file():
            fail(f"bound evidence file missing: {path}")
        actual_sha = git_blob_sha1(file_path.read_bytes())
        if actual_sha != expected_sha:
            fail(f"bound evidence blob drifted: {path}: {actual_sha} != {expected_sha}")


def verify(root: Path = ROOT) -> None:
    manifest = load_manifest(root / "paper" / "SUBMISSION_READINESS_20260906.json")
    validate_manifest(manifest)
    validate_blob_bindings(root, manifest)
    validate_manuscript((root / "paper" / "conference_v2.tex").read_text(encoding="utf-8"))
    validate_retained_summary(
        (root / "results" / "final_rigor_v2" / "RESULT_SUMMARY_20260906.md").read_text(encoding="utf-8")
    )
    main = (root / "paper" / "main.tex").read_text(encoding="utf-8")
    if "\\input{conference_v2.tex}" not in main:
        fail("canonical paper entry point no longer delegates to conference_v2.tex")


def main() -> int:
    verify()
    print(
        "EIGEN_SUBMISSION_CLAIM_GATE_PASS: frozen synthetic-v2 identities, mixed/negative claim boundary, "
        "canonical manuscript surface, and no-outcome/no-self-authorization controls verified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
