from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from eigen_jepa.classical_covariance import (
    ClassicalCovarianceSpec,
    default_candidate_grid,
    forecast_covariance,
    sample_covariance,
    select_by_validation,
    topk_eigenvalues,
)
from eigen_jepa.data import MarketConfig
from eigen_jepa.real_market_folds import build_purged_fold_datasets

from scripts.run_real_market_confirmation import (
    EXPECTED_FOLDS,
    ConfirmationRunnerError,
    _load_json,
    _sha256_bytes,
    _write_json_once,
    build_fold_plan,
    read_csv_dates,
    validate_protocol_shape,
    verify_frozen_inputs,
)


BASELINE_STATUS = "FROZEN_PRE_OUTCOME_AUTHORIZED"
PROTOCOL_ID = "eigen-jepa-real-market-classical-baseline-ladder-v1-candidate-20260926"
EXPECTED_FAMILIES = (
    "sample_persistence",
    "oas",
    "ewma",
    "factor",
    "sample_shrinkage",
    "ewma_shrinkage",
)


def validate_baseline_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ConfirmationRunnerError("baseline protocol schema_version must equal 1")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise ConfirmationRunnerError("unexpected baseline protocol_id")
    families = tuple(protocol.get("reporting_families_exact", ()))
    if families != EXPECTED_FAMILIES:
        raise ConfirmationRunnerError(f"baseline family drift: expected {EXPECTED_FAMILIES}")
    expected_count = int(protocol.get("candidate_grid", {}).get("total_candidates", -1))
    actual_count = len(default_candidate_grid())
    if expected_count != actual_count:
        raise ConfirmationRunnerError(
            f"candidate count drift: protocol={expected_count}, implementation={actual_count}"
        )
    if protocol.get("selection", {}).get("surface") != "validation_only":
        raise ConfirmationRunnerError("baseline hyperparameters must be validation-only selected")
    if protocol.get("test_access_policy") != "single_pass_after_freeze_and_authorization":
        raise ConfirmationRunnerError("unexpected classical baseline test-access policy")


def assert_baseline_execution_authorized(protocol: Mapping[str, Any]) -> None:
    validate_baseline_protocol(protocol)
    if protocol.get("status") != BASELINE_STATUS:
        raise ConfirmationRunnerError(
            f"baseline protocol status must be {BASELINE_STATUS!r} before test evaluation"
        )
    if protocol.get("execution_authorized") is not True:
        raise ConfirmationRunnerError("baseline execution_authorized must be true")
    source_commit = protocol.get("implementation", {}).get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ConfirmationRunnerError("baseline source_commit must be bound before execution")


def _families() -> dict[str, tuple[ClassicalCovarianceSpec, ...]]:
    grid = default_candidate_grid()
    out = {
        name: tuple(spec for spec in grid if spec.name == name)
        for name in EXPECTED_FAMILIES
    }
    for name, specs in out.items():
        if not specs:
            raise ConfirmationRunnerError(f"empty candidate family: {name}")
    return out


def _dataset_windows(dataset) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    contexts: list[np.ndarray] = []
    futures: list[np.ndarray] = []
    indices: list[int] = []
    for i in range(len(dataset)):
        item = dataset[i]
        contexts.append(
            item["returns_ctx"].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        futures.append(
            item["returns_fut"].detach().cpu().numpy().astype(np.float64, copy=False)
        )
        indices.append(int(item["window_index"].item()))
    if not indices or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ConfirmationRunnerError(
            "baseline windows must be non-empty and strictly increasing"
        )
    return contexts, futures, indices


def _evaluate_spec(
    spec: ClassicalCovarianceSpec,
    contexts: list[np.ndarray],
    futures: list[np.ndarray],
    indices: list[int],
    *,
    k: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for ctx, fut, idx in zip(contexts, futures, indices):
        pred_cov = forecast_covariance(ctx, spec)
        true_cov = sample_covariance(fut)
        pred_eig = topk_eigenvalues(pred_cov, k)
        true_eig = topk_eigenvalues(true_cov, k)
        eig_sq_error = float(np.mean((pred_eig - true_eig) ** 2))
        target_energy = float(np.mean(true_eig**2))
        cov_sq_error = float(np.mean((pred_cov - true_cov) ** 2))
        cov_target_energy = float(np.mean(true_cov**2))
        values = (eig_sq_error, target_energy, cov_sq_error, cov_target_energy)
        if (
            not all(np.isfinite(v) and v >= 0.0 for v in values)
            or target_energy <= 0.0
            or cov_target_energy <= 0.0
        ):
            raise ConfirmationRunnerError(
                f"non-finite/invalid classical baseline metric at window {idx}"
            )
        rows.append(
            {
                "window_index": idx,
                "eig_sq_error": eig_sq_error,
                "target_energy": target_energy,
                "cov_sq_error": cov_sq_error,
                "cov_target_energy": cov_target_energy,
            }
        )
    return rows


def run_baseline_ladder(
    parent_protocol: Mapping[str, Any],
    baseline_protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    csv_path: Path,
    parent_protocol_sha256: str,
    baseline_protocol_sha256: str,
    csv_sha256: str,
    plan: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    date_col: str,
) -> None:
    assert_baseline_execution_authorized(baseline_protocol)
    assets = tuple(parent_protocol["asset_universe"]["symbols_exact"])
    data = reference["data"]
    k = int(data["k"])
    families = _families()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "RETAINED_CLASSICAL_BASELINE_EVIDENCE",
        "parent_protocol_sha256": parent_protocol_sha256,
        "baseline_protocol_sha256": baseline_protocol_sha256,
        "normalized_return_csv_sha256": csv_sha256,
        "selection_surface": "validation_only",
        "test_access": "single_pass_after_freeze_and_authorization",
        "folds": {},
    }

    for fold_id in EXPECTED_FOLDS:
        split = plan[fold_id]["indices"]
        cfg = MarketConfig(
            num_assets=len(assets),
            total_steps=1,
            context_len=int(data["context_len"]),
            horizon=int(data["horizon"]),
            num_train=len(split["train"]),
            num_val=len(split["validation"]),
            num_test=len(split["test"]),
            seed=0,
            event_quantile=float(data["event_quantile"]),
            mask_ratio=float(data["mask_ratio"]),
            block_time=int(data["block_time"]),
            data_source="csv",
            csv_path=str(csv_path),
            return_cols=assets,
            date_col=date_col,
        )
        datasets = build_purged_fold_datasets(
            cfg,
            train_indices=np.asarray(split["train"], dtype=np.int64),
            validation_indices=np.asarray(split["validation"], dtype=np.int64),
            test_indices=np.asarray(split["test"], dtype=np.int64),
            k=k,
        )
        val_ctx, val_fut, _ = _dataset_windows(datasets["val"])
        test_ctx, test_fut, test_idx = _dataset_windows(datasets["test"])

        fold_payload: dict[str, Any] = {
            "regime_thresholds_train_only": [
                float(x) for x in datasets["regime_thresholds_train_only"]
            ],
            "regime_fit_last_row": int(datasets["regime_fit_last_row"]),
            "selected": {},
        }
        for family, specs in families.items():
            selected, validation_scores = select_by_validation(
                val_ctx, val_fut, specs, k=k
            )
            fold_payload["selected"][family] = {
                "spec": asdict(selected),
                "validation_scores": validation_scores,
                "test_window_rows": _evaluate_spec(
                    selected, test_ctx, test_fut, test_idx, k=k
                ),
            }
        payload["folds"][fold_id] = fold_payload

    _write_json_once(out_dir / "classical_baseline_evidence.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prospective classical covariance baseline ladder for the Eigen-JEPA "
            "real-market confirmation. Planning is pre-outcome; test evaluation "
            "fails closed until both protocols are explicitly authorized."
        )
    )
    parser.add_argument(
        "--parent-protocol",
        type=Path,
        default=Path(
            "protocols/real_market_confirmation_v1_candidate_20260906.json"
        ),
    )
    parser.add_argument(
        "--baseline-protocol",
        type=Path,
        default=Path(
            "protocols/real_market_classical_baseline_ladder_v1_candidate_20260926.json"
        ),
    )
    parser.add_argument(
        "--reference-protocol",
        type=Path,
        default=Path("protocols/final_rigor_v2_20260905.json"),
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--input-receipt", type=Path)
    parser.add_argument("--date-col", default="date")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/real_market_classical_baselines_v1"),
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    parent = _load_json(args.parent_protocol)
    baseline = _load_json(args.baseline_protocol)
    reference = _load_json(args.reference_protocol)
    validate_protocol_shape(parent)
    validate_baseline_protocol(baseline)

    data = reference.get("data", {})
    context_len = int(data.get("context_len", 0))
    horizon = int(data.get("horizon", 0))
    if context_len <= 0 or horizon <= 0:
        raise ConfirmationRunnerError(
            "reference protocol has invalid context_len/horizon"
        )

    dates = read_csv_dates(args.csv, date_col=args.date_col)
    plan = build_fold_plan(
        parent,
        dates,
        context_len=context_len,
        horizon=horizon,
    )
    plan_payload = {
        "schema_version": 1,
        "status": "PREOUTCOME_CLASSICAL_BASELINE_PLAN_ONLY",
        "parent_protocol_sha256": _sha256_bytes(args.parent_protocol),
        "baseline_protocol_sha256": _sha256_bytes(args.baseline_protocol),
        "normalized_return_csv_sha256": _sha256_bytes(args.csv),
        "families": list(EXPECTED_FAMILIES),
        "candidate_count": len(default_candidate_grid()),
        "folds": {
            fold: plan[fold]["summary"]
            for fold in EXPECTED_FOLDS
        },
    }
    _write_json_once(
        args.out_dir / "classical_baseline_plan.json",
        plan_payload,
    )
    if args.plan_only:
        print("REAL_MARKET_CLASSICAL_BASELINE_PLAN_PREOUTCOME")
        return

    if args.input_receipt is None:
        raise ConfirmationRunnerError(
            "--input-receipt is required for baseline outcome execution"
        )
    verify_frozen_inputs(
        parent,
        csv_path=args.csv,
        input_receipt_path=args.input_receipt,
    )
    assert_baseline_execution_authorized(baseline)
    run_baseline_ladder(
        parent,
        baseline,
        reference,
        csv_path=args.csv,
        parent_protocol_sha256=_sha256_bytes(args.parent_protocol),
        baseline_protocol_sha256=_sha256_bytes(args.baseline_protocol),
        csv_sha256=_sha256_bytes(args.csv),
        plan=plan,
        out_dir=args.out_dir,
        date_col=args.date_col,
    )
    evidence_path = args.out_dir / "classical_baseline_evidence.json"
    print("REAL_MARKET_CLASSICAL_BASELINE_EXECUTION_COMPLETE")
    print(f"EVIDENCE_SHA256={_sha256_bytes(evidence_path)}")


if __name__ == "__main__":
    main()
