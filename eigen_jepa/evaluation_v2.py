"""Prospective cohort evaluation with exact per-window sufficient statistics.

Historical ``eval.py`` remains unchanged. A successor protocol must explicitly
select and source-bind this v2 API before inspecting its protected outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import torch

from .spectral import low_rank_covariance

SCHEMA = "eigen_jepa.evaluation.v2"
EPS = 1e-12
MSE_FIELDS = ("proj_mse", "drift_mse", "cov_mse", "risk_mse", "entropy_mse", "rank_mse")


def _array(value: torch.Tensor, name: str) -> np.ndarray:
    if not torch.is_tensor(value) or value.is_complex() or value.dtype == torch.bool:
        raise ValueError(f"{name} must be a real numeric tensor")
    array = value.detach().cpu().double().numpy()
    if array.ndim == 0 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain a nonempty finite batch")
    return array


def _nonnegative_mean(array: np.ndarray, name: str) -> float:
    """Average finite nonnegative statistics without overflowing their sum."""
    scale = float(array.max())
    if scale == 0.0:
        return 0.0
    result = float((array / scale).mean()) * scale
    if result == 0.0:
        raise ValueError(f"nonzero {name} mean underflow; no finite report")
    if not math.isfinite(result):
        raise ValueError(f"{name} mean exceeds the finite float64 domain")
    return result


def _nmse(squared_error: np.ndarray, energy: np.ndarray) -> float:
    error_mean = _nonnegative_mean(squared_error, "eigenvalue squared error")
    energy_mean = _nonnegative_mean(energy, "eigenvalue target energy")
    value = error_mean / (energy_mean + EPS)
    if not math.isfinite(value) or (error_mean > 0.0 and value == 0.0):
        raise ValueError("NMSE exceeds the representable finite nonzero domain")
    return value


def window_records(
    batch: Mapping, prediction: Mapping, *, latent_target=None
) -> list[dict]:
    """Extract each window's own error, target energy, and classification output."""
    truth_eig = _array(batch["evals_true"], "evals_true")
    pred_eig = _array(prediction["eig"], "eig")
    if truth_eig.ndim != 2 or truth_eig.shape != pred_eig.shape:
        raise ValueError(
            "eigenvalue predictions and targets must have equal [batch, k] shape"
        )
    n, k = truth_eig.shape
    if k < 1:
        raise ValueError("at least one eigenvalue is required")
    columns = {
        "eig_squared_error": np.mean((pred_eig - truth_eig) ** 2, axis=1),
        "eig_target_energy": np.mean(truth_eig**2, axis=1),
    }
    if k > 1:
        columns["gap_mse"] = np.mean(
            (np.diff(pred_eig, axis=1) - np.diff(truth_eig, axis=1)) ** 2, axis=1
        )
    pairs = {
        "proj_mse": (prediction["proj"], batch["proj_true"]),
        "drift_mse": (prediction["drift"], batch["drift_true"]),
        "cov_mse": (
            low_rank_covariance(
                prediction["eig"], prediction.get("sub", prediction.get("evecs"))
            ),
            batch["cov_true"],
        ),
        "risk_mse": (prediction["risk"], batch["risk_true"]),
        "entropy_mse": (prediction["entropy"], batch["entropy_true"]),
        "rank_mse": (prediction["rank"], batch["rank_true"]),
    }
    if latent_target is not None:
        pairs["lat_align"] = (prediction["z_hat"], latent_target)
    for name, (predicted, expected) in pairs.items():
        predicted, expected = (
            _array(predicted, name),
            _array(expected, name + " target"),
        )
        if predicted.shape != expected.shape or len(expected) != n:
            raise ValueError(
                f"{name} predictions and targets must have identical batch shapes"
            )
        columns[name] = np.mean(((predicted - expected) ** 2).reshape(n, -1), axis=1)
    for name, value in (
        ("window_index", batch["window_index"]),
        ("event_true", batch["event_true"]),
        ("regime_true", batch["regime_true"]),
        ("gate", prediction["gate"]),
    ):
        array = _array(value, name)
        if array.shape != (n,):
            raise ValueError(f"{name} must have shape [batch]")
        columns[name] = array
    if torch.is_floating_point(batch["window_index"]):
        raise ValueError("window_index must use an integer tensor dtype")
    if not np.isin(columns["event_true"], [0, 1]).all():
        raise ValueError("event_true must be binary")
    if not np.isin(columns["regime_true"], [0, 1, 2]).all():
        raise ValueError("regime_true must be one of 0, 1, 2")
    if np.any((columns["gate"] < 0) | (columns["gate"] > 1)):
        raise ValueError("gate must lie in [0, 1]")
    logits = _array(prediction["regime_logits"], "regime_logits")
    if logits.shape != (n, 3):
        raise ValueError("regime_logits must have shape [batch, 3]")
    columns["pred_regime"] = logits.argmax(axis=1)
    if any(not np.isfinite(column).all() for column in columns.values()):
        raise ValueError("squared errors exceed the finite float64 domain")
    records = [
        {name: float(values[i]) for name, values in columns.items()} for i in range(n)
    ]
    # Preserve integer identities directly; a float roundtrip loses large IDs.
    for i, record in enumerate(records):
        record["window_index"] = int(batch["window_index"][i].item())
        record["event_true"] = int(record["event_true"])
        record["regime_true"] = int(record["regime_true"])
        record["pred_regime"] = int(record["pred_regime"])
        record["eigenvalue_count"] = k
        digest = hashlib.sha256()
        for name in ("x", "returns_ctx", "target_vec"):
            tensor = batch[name][i].detach().cpu().contiguous()
            digest.update(
                json.dumps(
                    [name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")
                ).encode()
            )
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        record["input_sha256"] = digest.hexdigest()
    return records


def reduce_records(
    records: Sequence[Mapping], *, event_threshold: float = 0.45
) -> dict:
    """Reduce once over the cohort, never over averages of batch ratios or F1s."""
    if not records:
        raise ValueError("evaluation requires at least one window")
    if not math.isfinite(event_threshold) or not 0 <= event_threshold <= 1:
        raise ValueError("event_threshold must lie in [0, 1]")
    indices = [r["window_index"] for r in records]
    if any(
        isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in indices
    ):
        raise ValueError("window identities must be integers")
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate window identities cannot be counted twice")
    records = sorted(records, key=lambda r: r["window_index"])
    if len({r["eigenvalue_count"] for r in records}) != 1:
        raise ValueError("all windows must use the same eigenvalue count")
    keys = set(records[0])
    required = {
        "window_index",
        "input_sha256",
        "eigenvalue_count",
        "eig_squared_error",
        "eig_target_energy",
        "event_true",
        "gate",
        "regime_true",
        "pred_regime",
        *MSE_FIELDS,
    }
    if not required.issubset(keys):
        raise ValueError(
            f"missing required per-window fields: {sorted(required - keys)}"
        )
    count = records[0]["eigenvalue_count"]
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("eigenvalue_count must be a positive integer")
    if (count > 1) != ("gap_mse" in keys):
        raise ValueError(
            "gap_mse is required exactly when eigenvalue_count exceeds one"
        )
    if any(set(row) != keys for row in records):
        raise ValueError("window statistic schemas must match")

    def values(name):
        if any(
            isinstance(row[name], bool)
            or not isinstance(row[name], (int, float, np.number))
            for row in records
        ):
            raise ValueError(f"per-window statistic must be numeric: {name}")
        array = np.asarray([row[name] for row in records], dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite per-window statistic: {name}")
        return array

    # Verify all retained fields, including ones not used by this reduction.
    for name in keys:
        if name == "input_sha256":
            if any(
                not isinstance(row[name], str)
                or re.fullmatch(r"[a-f0-9]{64}", row[name]) is None
                for row in records
            ):
                raise ValueError(
                    "input_sha256 must bind the exact supplied window tensors"
                )
        else:
            values(name)
    energy = values("eig_target_energy")
    squared_error = values("eig_squared_error")
    if np.any(energy < 0) or np.any(squared_error < 0):
        raise ValueError("eigenvalue energies and squared errors must be nonnegative")
    if any(
        np.any(values(name) < 0)
        for name in (*MSE_FIELDS, "gap_mse", "lat_align")
        if name in keys
    ):
        raise ValueError("squared-error statistics must be nonnegative")
    out = {name: _nonnegative_mean(values(name), name) for name in MSE_FIELDS}
    out["eig_nmse"] = _nmse(squared_error, energy)
    out["gap_rmse"] = (
        math.sqrt(_nonnegative_mean(values("gap_mse"), "gap MSE") + EPS)
        if "gap_mse" in keys
        else None
    )
    if "lat_align" in keys:
        out["lat_align"] = _nonnegative_mean(values("lat_align"), "latent alignment")
    event, gate = values("event_true"), values("gate")
    regime, prediction = values("regime_true"), values("pred_regime")
    if (
        not np.isin(event, [0, 1]).all()
        or not np.isin(regime, [0, 1, 2]).all()
        or not np.isin(prediction, [0, 1, 2]).all()
    ):
        raise ValueError("invalid event or regime labels")
    if np.any((gate < 0) | (gate > 1)):
        raise ValueError("gate must lie in [0, 1]")
    positive = gate >= event_threshold
    actual = event.astype(bool)
    tp = int(np.sum(positive & actual))
    fp = int(np.sum(positive & ~actual))
    fn = int(np.sum(~positive & actual))
    out.update(
        {
            "window_count": len(records),
            "gate_cal": float(np.abs(gate - event).mean()),
            "gate_mean": float(gate.mean()),
            "event_rate": float(event.mean()),
            "regime_acc": float((prediction == regime).mean()),
            "regime_bal_acc": float(
                np.mean(
                    [
                        (prediction[regime == c] == c).mean()
                        for c in (0, 1, 2)
                        if np.any(regime == c)
                    ]
                )
            ),
            "tail_f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
            "tail_confusion": {"tp": tp, "fp": fp, "fn": fn},
            "rare_window_count": int(actual.sum()),
            "rare_eig_nmse": _nmse(squared_error[actual], energy[actual])
            if actual.any()
            else None,
            "rare_drift_mse": _nonnegative_mean(values("drift_mse")[actual], "rare drift MSE")
            if actual.any()
            else None,
            "rare_cov_mse": _nonnegative_mean(values("cov_mse")[actual], "rare covariance MSE")
            if actual.any()
            else None,
        }
    )
    return out


def evaluation_report(
    records: Sequence[Mapping], *, event_threshold: float = 0.45
) -> dict:
    # Admitted row values are scalars. Own the row dictionaries so a caller's
    # later edits cannot silently invalidate this report's metrics and digest.
    records = [dict(row) for row in records]
    metrics = reduce_records(records, event_threshold=event_threshold)
    payload = {
        "schema": SCHEMA,
        "event_threshold": event_threshold,
        "reduction": "cohort_ratio_of_means_and_pooled_confusion; equal_window_weight",
        "metrics": metrics,
        "records": sorted(records, key=lambda r: r["window_index"]),
    }
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {**payload, "result_sha256": digest}


@torch.no_grad()
def evaluate_model_v2(
    model,
    loader: Iterable,
    device="cpu",
    memory=None,
    *,
    gate_override=None,
    memory_scale=1.0,
    event_threshold=0.45,
) -> dict:
    """Evaluate supplied windows with one vectorized forward call per batch.

    Comparisons must reuse frozen input tensors: the legacy dataset can sample a
    new mask on every read. Each row hashes the tensors actually consumed so
    such a changed input cannot silently retain the same evidence identity.
    """
    model.to(device).eval()
    records = []
    for batch in loader:
        prediction = model(
            batch["x"].to(device),
            memory=memory,
            gate_override=gate_override,
            memory_scale=memory_scale,
        )
        target = model.target_latent(batch["target_vec"].to(device))
        records.extend(window_records(batch, prediction, latent_target=target))
    return evaluation_report(records, event_threshold=event_threshold)


@torch.no_grad()
def evaluate_baselines_v2(loader: Iterable, *, event_threshold=0.45) -> dict:
    """Run each declared context-only baseline over the entire supplied cohort.

    Use the baseline's own per-window gate. The historical evaluator replaced
    it with a gate normalized by the evaluation batch's mean stress.
    """
    from .eval import _baseline_predictions

    records = {}
    for batch in loader:
        for name, prediction in _baseline_predictions(batch).items():
            records.setdefault(name, []).extend(window_records(batch, prediction))
    if not records:
        raise ValueError("baseline evaluation requires at least one window")
    return {
        name: evaluation_report(rows, event_threshold=event_threshold)
        for name, rows in records.items()
    }
