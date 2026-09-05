#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from argparse import Namespace
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from eigen_jepa.data import batch_collate
from eigen_jepa.real_market_folds import (
    build_purged_fold_datasets,
    build_purged_fold_indices,
    parse_fold_boundary,
    summarize_fold_indices,
)
from eigen_jepa.train import run as train_run

EXPECTED_FOLDS = ("F1", "F2", "F3")
EXPECTED_SEEDS = (7, 19, 31, 43, 59)
EXPECTED_VARIANTS = ("full", "no_memory", "no_gate", "no_regime")
PRIMARY_VARIANTS = ("full", "no_memory")
AUTHORIZED_STATUS = "FROZEN_PRE_OUTCOME_AUTHORIZED"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ConfirmationRunnerError(ValueError):
    pass


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfirmationRunnerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfirmationRunnerError(f"{path}: expected a JSON object")
    return data


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite retained evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_hex(value: Any, length: int, label: str) -> str:
    pattern = HEX64 if length == 64 else HEX40
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ConfirmationRunnerError(f"{label} must be an exact lowercase {length}-hex identity")
    return value


def validate_protocol_shape(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema_version") != 1:
        raise ConfirmationRunnerError("protocol schema_version must equal 1")
    assets = protocol.get("asset_universe", {}).get("symbols_exact")
    if not isinstance(assets, list) or len(assets) != 12 or len(set(assets)) != 12:
        raise ConfirmationRunnerError("protocol must freeze exactly 12 unique asset symbols")
    policy = protocol.get("model_and_training_policy", {})
    if tuple(policy.get("seeds_exact", ())) != EXPECTED_SEEDS:
        raise ConfirmationRunnerError(f"seed drift: expected {EXPECTED_SEEDS}")
    if tuple(policy.get("variants_exact", ())) != EXPECTED_VARIANTS:
        raise ConfirmationRunnerError(f"variant drift: expected {EXPECTED_VARIANTS}")
    folds = protocol.get("evaluation_design", {}).get("folds")
    if not isinstance(folds, list) or tuple(f.get("id") for f in folds) != EXPECTED_FOLDS:
        raise ConfirmationRunnerError(f"fold drift: expected ordered folds {EXPECTED_FOLDS}")
    for fold in folds:
        parse_fold_boundary(fold)


def assert_execution_authorized(protocol: Mapping[str, Any]) -> None:
    """Fail closed until a separately reviewed protocol explicitly authorizes outcome access."""

    validate_protocol_shape(protocol)
    if protocol.get("status") != AUTHORIZED_STATUS:
        raise ConfirmationRunnerError(
            f"protocol status must be {AUTHORIZED_STATUS!r} before any model/test outcome access"
        )

    freeze = protocol.get("data_freeze_requirements")
    if not isinstance(freeze, Mapping) or freeze.get("execution_authorized") is not True:
        raise ConfirmationRunnerError("data_freeze_requirements.execution_authorized must be true")
    for key in (
        "provider_identity",
        "provider_snapshot_or_retrieval_id",
        "raw_source_sha256",
        "normalized_return_csv_sha256",
        "input_receipt_sha256",
    ):
        value = freeze.get(key)
        if value in (None, "", "TBD", "TODO"):
            raise ConfirmationRunnerError(f"unresolved data freeze field: {key}")
    _require_hex(freeze.get("raw_source_sha256"), 64, "raw_source_sha256")
    _require_hex(freeze.get("normalized_return_csv_sha256"), 64, "normalized_return_csv_sha256")
    _require_hex(freeze.get("input_receipt_sha256"), 64, "input_receipt_sha256")

    authorization = protocol.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ConfirmationRunnerError("missing authorization object")
    if authorization.get("independent_pre_outcome_review_approved") is not True:
        raise ConfirmationRunnerError("independent pre-outcome review has not approved execution")
    _require_hex(authorization.get("source_commit"), 40, "authorization.source_commit")
    _require_hex(
        authorization.get("dependency_lock_sha256"),
        64,
        "authorization.dependency_lock_sha256",
    )
    _require_hex(
        authorization.get("fold_runner_sha256"),
        64,
        "authorization.fold_runner_sha256",
    )


def read_csv_dates(path: Path, *, date_col: str) -> list[str]:
    if not path.is_file():
        raise ConfirmationRunnerError(f"missing normalized CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or date_col not in reader.fieldnames:
            raise ConfirmationRunnerError(f"CSV is missing date column {date_col!r}")
        dates: list[str] = []
        previous = None
        for row_number, row in enumerate(reader, start=2):
            raw = (row.get(date_col) or "").strip()
            try:
                parsed = __import__("datetime").date.fromisoformat(raw)
            except ValueError as exc:
                raise ConfirmationRunnerError(
                    f"row {row_number}: invalid ISO date {raw!r}"
                ) from exc
            if parsed.isoformat() != raw:
                raise ConfirmationRunnerError(
                    f"row {row_number}: date must use canonical ISO YYYY-MM-DD"
                )
            if previous is not None and parsed <= previous:
                raise ConfirmationRunnerError(
                    f"row {row_number}: dates must be unique and strictly increasing"
                )
            previous = parsed
            dates.append(raw)
    if not dates:
        raise ConfirmationRunnerError("normalized CSV has no data rows")
    return dates


def build_fold_plan(
    protocol: Mapping[str, Any],
    dates: Sequence[str],
    *,
    context_len: int,
    horizon: int,
) -> dict[str, dict[str, Any]]:
    validate_protocol_shape(protocol)
    plan: dict[str, dict[str, Any]] = {}
    seen_test: set[int] = set()
    for raw_fold in protocol["evaluation_design"]["folds"]:
        boundary = parse_fold_boundary(raw_fold)
        indices = build_purged_fold_indices(
            dates,
            context_len=context_len,
            horizon=horizon,
            boundary=boundary,
        )
        overlap = seen_test.intersection(int(x) for x in indices["test"])
        if overlap:
            raise ConfirmationRunnerError(
                f"{boundary.fold_id}: test windows overlap a previous fold"
            )
        seen_test.update(int(x) for x in indices["test"])
        plan[boundary.fold_id] = {
            "boundary": raw_fold,
            "indices": {
                name: [int(x) for x in values]
                for name, values in indices.items()
            },
            "summary": summarize_fold_indices(
                dates,
                context_len=context_len,
                horizon=horizon,
                boundary=boundary,
                indices=indices,
            ),
        }
    return plan


def verify_frozen_inputs(
    protocol: Mapping[str, Any],
    *,
    csv_path: Path,
    input_receipt_path: Path,
) -> dict[str, Any]:
    assert_execution_authorized(protocol)
    freeze = protocol["data_freeze_requirements"]
    csv_sha = _sha256_bytes(csv_path)
    receipt_sha = _sha256_bytes(input_receipt_path)
    if csv_sha != freeze["normalized_return_csv_sha256"]:
        raise ConfirmationRunnerError(
            "normalized CSV bytes do not match the frozen protocol identity"
        )
    if receipt_sha != freeze["input_receipt_sha256"]:
        raise ConfirmationRunnerError(
            "input receipt bytes do not match the frozen protocol identity"
        )
    receipt = _load_json(input_receipt_path)
    if receipt.get("status") != "PREPARED_INPUT_ONLY_NOT_AUTHORIZED":
        raise ConfirmationRunnerError("unexpected input receipt status")
    dataset = receipt.get("dataset")
    if not isinstance(dataset, Mapping) or dataset.get("sha256") != csv_sha:
        raise ConfirmationRunnerError("input receipt does not bind the supplied normalized CSV")
    return receipt


def _namespace_from_frozen_reference(
    reference: Mapping[str, Any],
    *,
    csv_path: Path,
    return_cols: Sequence[str],
    date_col: str,
    fold_counts: Mapping[str, int],
    seed: int,
    variant: str,
    out_dir: Path,
) -> Namespace:
    data = reference["data"]
    model = reference["model"]
    training = reference["training"]
    loss = reference["loss_and_masking"]
    return Namespace(
        out_dir=str(out_dir),
        device=torch.device("cpu"),
        seed=int(seed),
        deterministic=True,
        num_assets=len(return_cols),
        total_steps=1,
        context_len=int(data["context_len"]),
        horizon=int(data["horizon"]),
        num_train=int(fold_counts["train"]),
        num_val=int(fold_counts["validation"]),
        num_test=int(fold_counts["test"]),
        k=int(data["k"]),
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        lr=float(training["learning_rate"]),
        wd=float(training["weight_decay"]),
        d_model=int(model["d_model"]),
        n_heads=int(model["n_heads"]),
        n_layers=int(model["n_layers"]),
        dropout=float(model["dropout"]),
        latent_dim=int(model["latent_dim"]),
        memory_size=int(model["memory_size"]),
        memory_top_k=int(model["memory_top_k"]),
        memory_temperature=float(model["memory_temperature"]),
        merge_radius=float(model["merge_radius"]),
        min_salience=float(model["min_salience"]),
        variant=variant,
        market_style="equity",
        regime_weight=float(loss["regime_weight"]),
        gate_weight=float(loss["gate_weight"]),
        tail_boost=float(loss["tail_boost"]),
        event_quantile=float(data["event_quantile"]),
        mask_ratio=float(data["mask_ratio"]),
        block_time=int(data["block_time"]),
        data_source="csv",
        csv_path=str(csv_path),
        return_cols=list(return_cols),
        aux_cols=[],
        date_col=date_col,
        eig_weight=float(loss["eig_weight"]),
        subspace_weight=float(loss["subspace_weight"]),
        drift_weight=float(loss["drift_weight"]),
        risk_weight=float(loss["risk_weight"]),
        entropy_weight=float(loss["entropy_weight"]),
        rank_weight=float(loss["rank_weight"]),
    )


@torch.no_grad()
def collect_window_evidence(
    *,
    model,
    memory,
    dataset,
    device: torch.device,
    variant: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=batch_collate,
    )
    evidence: list[dict[str, Any]] = []
    for batch in loader:
        pred = model(
            batch["x"].to(device),
            memory=None if variant == "no_memory" else memory,
            gate_override=1.0 if variant == "no_gate" else None,
        )
        pred_eig = pred["eig"].detach().cpu()
        true_eig = batch["evals_true"].detach().cpu()
        squared_error = torch.mean((pred_eig - true_eig) ** 2, dim=1)
        target_energy = torch.mean(true_eig**2, dim=1)
        for i in range(len(squared_error)):
            error = float(squared_error[i].item())
            energy = float(target_energy[i].item())
            if not math.isfinite(error) or error < 0:
                raise ConfirmationRunnerError("non-finite or negative per-window eig error")
            if not math.isfinite(energy) or energy <= 0:
                raise ConfirmationRunnerError("non-finite or non-positive target energy")
            evidence.append(
                {
                    "window_index": int(batch["window_index"][i].item()),
                    "eig_sq_error": error,
                    "target_energy": energy,
                }
            )
    indices = [row["window_index"] for row in evidence]
    if not indices or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ConfirmationRunnerError("window evidence must be non-empty and strictly increasing")
    return evidence


def _run_evidence_path(root: Path, fold: str, seed: int, variant: str) -> Path:
    return root / fold / f"seed_{seed}" / variant / "window_evidence.json"


def run_confirmation(
    protocol: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    protocol_sha256: str,
    csv_path: Path,
    csv_sha256: str,
    plan: Mapping[str, Mapping[str, Any]],
    out_dir: Path,
    date_col: str,
) -> None:
    return_cols = list(protocol["asset_universe"]["symbols_exact"])
    reference_seeds = tuple(reference.get("seeds", ()))
    reference_variants = tuple(reference.get("paper_facing_variants", ()))
    if reference_seeds != EXPECTED_SEEDS:
        raise ConfirmationRunnerError("synthetic v2 reference seed set has drifted")
    if reference_variants != EXPECTED_VARIANTS:
        raise ConfirmationRunnerError("synthetic v2 reference variant set has drifted")

    for fold_id in EXPECTED_FOLDS:
        fold = plan[fold_id]
        idx = fold["indices"]
        np_idx = {
            name: np.asarray(values, dtype=np.int64)
            for name, values in idx.items()
        }
        counts = {name: len(values) for name, values in idx.items()}
        for seed in EXPECTED_SEEDS:
            for variant in EXPECTED_VARIANTS:
                run_root = out_dir / fold_id / f"seed_{seed}" / variant
                args = _namespace_from_frozen_reference(
                    reference,
                    csv_path=csv_path,
                    return_cols=return_cols,
                    date_col=date_col,
                    fold_counts=counts,
                    seed=seed,
                    variant=variant,
                    out_dir=run_root,
                )

                def dataset_builder(cfg, *, k):
                    return build_purged_fold_datasets(
                        cfg,
                        train_indices=np_idx["train"],
                        validation_indices=np_idx["validation"],
                        test_indices=np_idx["test"],
                        k=k,
                    )

                _, model, memory, datasets, _ = train_run(
                    deepcopy(args), dataset_builder=dataset_builder
                )
                window_rows = collect_window_evidence(
                    model=model,
                    memory=memory,
                    dataset=datasets["test"],
                    device=args.device,
                    variant=variant,
                    batch_size=args.batch_size,
                )
                payload = {
                    "schema_version": 1,
                    "status": "RETAINED_TEST_WINDOW_EVIDENCE",
                    "protocol_sha256": protocol_sha256,
                    "normalized_return_csv_sha256": csv_sha256,
                    "fold_id": fold_id,
                    "seed": seed,
                    "variant": variant,
                    "regime_thresholds_train_only": [
                        float(x) for x in datasets["regime_thresholds_train_only"]
                    ],
                    "regime_fit_last_row": int(datasets["regime_fit_last_row"]),
                    "event_threshold_train_only": float(datasets["event_threshold"]),
                    "window_rows": window_rows,
                }
                _write_json_once(
                    _run_evidence_path(out_dir, fold_id, seed, variant),
                    payload,
                )


def _load_window_rows(path: Path, *, fold: str, seed: int, variant: str) -> list[dict[str, Any]]:
    data = _load_json(path)
    if data.get("schema_version") != 1:
        raise ConfirmationRunnerError(f"{path}: schema_version drift")
    if data.get("fold_id") != fold or data.get("seed") != seed or data.get("variant") != variant:
        raise ConfirmationRunnerError(f"{path}: run identity drift")
    rows = data.get("window_rows")
    if not isinstance(rows, list) or not rows:
        raise ConfirmationRunnerError(f"{path}: missing window_rows")
    previous = -1
    for row in rows:
        if not isinstance(row, Mapping):
            raise ConfirmationRunnerError(f"{path}: malformed window row")
        idx = row.get("window_index")
        err = row.get("eig_sq_error")
        energy = row.get("target_energy")
        if not isinstance(idx, int) or idx <= previous:
            raise ConfirmationRunnerError(f"{path}: window indices must be strictly increasing")
        if not isinstance(err, (int, float)) or not math.isfinite(float(err)) or float(err) < 0:
            raise ConfirmationRunnerError(f"{path}: invalid eig_sq_error")
        if not isinstance(energy, (int, float)) or not math.isfinite(float(energy)) or float(energy) <= 0:
            raise ConfirmationRunnerError(f"{path}: invalid target_energy")
        previous = idx
    return rows


def assemble_primary_analysis_input(root: Path) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    seen_windows: set[int] = set()
    for fold in EXPECTED_FOLDS:
        baseline_rows = _load_window_rows(
            _run_evidence_path(root, fold, EXPECTED_SEEDS[0], PRIMARY_VARIANTS[0]),
            fold=fold,
            seed=EXPECTED_SEEDS[0],
            variant=PRIMARY_VARIANTS[0],
        )
        window_indices = [int(row["window_index"]) for row in baseline_rows]
        overlap = seen_windows.intersection(window_indices)
        if overlap:
            raise ConfirmationRunnerError(f"{fold}: window indices overlap a previous fold")
        seen_windows.update(window_indices)
        target_energy = [float(row["target_energy"]) for row in baseline_rows]

        fold_payload: dict[str, Any] = {
            "window_indices": window_indices,
            "target_energy": target_energy,
        }
        for variant in PRIMARY_VARIANTS:
            by_seed: dict[str, list[float]] = {}
            for seed in EXPECTED_SEEDS:
                rows = _load_window_rows(
                    _run_evidence_path(root, fold, seed, variant),
                    fold=fold,
                    seed=seed,
                    variant=variant,
                )
                indices = [int(row["window_index"]) for row in rows]
                energies = [float(row["target_energy"]) for row in rows]
                if indices != window_indices:
                    raise ConfirmationRunnerError(
                        f"{fold}/{variant}/{seed}: test window identity drift"
                    )
                if not np.array_equal(
                    np.asarray(energies, dtype=np.float64),
                    np.asarray(target_energy, dtype=np.float64),
                ):
                    raise ConfirmationRunnerError(
                        f"{fold}/{variant}/{seed}: target energy drift"
                    )
                by_seed[str(seed)] = [float(row["eig_sq_error"]) for row in rows]
            fold_payload[variant] = by_seed
        folds[fold] = fold_payload
    return {"schema_version": 1, "folds": folds}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prospective Eigen-JEPA real-market fold runner. Planning is allowed pre-outcome; "
            "model/test execution fails closed until the protocol is explicitly authorized."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("protocols/real_market_confirmation_v1_candidate_20260906.json"),
    )
    parser.add_argument(
        "--reference-protocol",
        type=Path,
        default=Path("protocols/final_rigor_v2_20260905.json"),
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--input-receipt", type=Path)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--out-dir", type=Path, default=Path("results/real_market_confirmation_v1"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    protocol = _load_json(args.protocol)
    reference = _load_json(args.reference_protocol)
    validate_protocol_shape(protocol)

    data = reference.get("data", {})
    context_len = int(data.get("context_len", 0))
    horizon = int(data.get("horizon", 0))
    if context_len <= 0 or horizon <= 0:
        raise ConfirmationRunnerError("reference protocol has invalid context_len/horizon")

    dates = read_csv_dates(args.csv, date_col=args.date_col)
    plan = build_fold_plan(
        protocol,
        dates,
        context_len=context_len,
        horizon=horizon,
    )
    plan_payload = {
        "schema_version": 1,
        "status": "PREOUTCOME_FOLD_PLAN_ONLY",
        "protocol_sha256": _sha256_bytes(args.protocol),
        "normalized_return_csv_sha256": _sha256_bytes(args.csv),
        "context_len": context_len,
        "horizon": horizon,
        "folds": {
            fold: plan[fold]["summary"]
            for fold in EXPECTED_FOLDS
        },
    }
    plan_path = args.out_dir / "fold_plan.json"
    _write_json_once(plan_path, plan_payload)

    if args.plan_only:
        print("REAL_MARKET_FOLD_PLAN_PREOUTCOME")
        print(f"PLAN_SHA256={_sha256_bytes(plan_path)}")
        return

    if args.input_receipt is None:
        raise ConfirmationRunnerError("--input-receipt is required for outcome execution")
    verify_frozen_inputs(
        protocol,
        csv_path=args.csv,
        input_receipt_path=args.input_receipt,
    )
    protocol_sha = _sha256_bytes(args.protocol)
    csv_sha = _sha256_bytes(args.csv)
    run_confirmation(
        protocol,
        reference,
        protocol_sha256=protocol_sha,
        csv_path=args.csv,
        csv_sha256=csv_sha,
        plan=plan,
        out_dir=args.out_dir,
        date_col=args.date_col,
    )
    analysis_input = assemble_primary_analysis_input(args.out_dir)
    analysis_input_path = args.out_dir / "primary_analysis_input.json"
    _write_json_once(analysis_input_path, analysis_input)
    print("REAL_MARKET_CONFIRMATION_EXECUTION_COMPLETE")
    print(f"ANALYSIS_INPUT_SHA256={_sha256_bytes(analysis_input_path)}")


if __name__ == "__main__":
    main()
