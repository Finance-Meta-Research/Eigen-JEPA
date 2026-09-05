from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from .data import (
    MarketConfig,
    MarketWindowDataset,
    _window_salience_score,
    generate_market_series,
)


@dataclass(frozen=True)
class FoldBoundary:
    fold_id: str
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date


def _as_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def parse_fold_boundary(raw: Mapping[str, str]) -> FoldBoundary:
    boundary = FoldBoundary(
        fold_id=str(raw["id"]),
        train_end=_as_date(raw["train_end"]),
        validation_start=_as_date(raw["validation_start"]),
        validation_end=_as_date(raw["validation_end"]),
        test_start=_as_date(raw["test_start"]),
        test_end=_as_date(raw["test_end"]),
    )
    if not (
        boundary.train_end
        < boundary.validation_start
        <= boundary.validation_end
        < boundary.test_start
        <= boundary.test_end
    ):
        raise ValueError(
            f"fold {boundary.fold_id}: boundaries must satisfy "
            "train_end < validation_start <= validation_end < test_start <= test_end"
        )
    return boundary


def _canonical_dates(values: Iterable[str | date]) -> list[date]:
    parsed = [_as_date(v) for v in values]
    if not parsed:
        raise ValueError("dates must be non-empty")
    if len(parsed) != len(set(parsed)):
        raise ValueError("dates must be unique")
    if any(b <= a for a, b in zip(parsed, parsed[1:])):
        raise ValueError("dates must be strictly increasing")
    return parsed


def build_purged_fold_indices(
    dates: Sequence[str | date],
    *,
    context_len: int,
    horizon: int,
    boundary: FoldBoundary | Mapping[str, str],
) -> dict[str, np.ndarray]:
    """Build split indices whose full context+target windows stay inside split boundaries.

    The historical count-based splitter places validation immediately after training and
    test immediately after validation. With a positive forecast horizon, that allows a
    training target window to extend into dates later used by validation. This helper
    intentionally purges such boundary-crossing windows.

    Returned indices are the context-end row indices expected by MarketWindowDataset.
    """

    if context_len <= 0 or horizon <= 0:
        raise ValueError("context_len and horizon must be positive")
    parsed_dates = _canonical_dates(dates)
    b = boundary if isinstance(boundary, FoldBoundary) else parse_fold_boundary(boundary)

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []

    first_context_end = context_len - 1
    last_context_end = len(parsed_dates) - horizon - 1
    if last_context_end < first_context_end:
        raise ValueError("not enough rows for one context/target window")

    for idx in range(first_context_end, last_context_end + 1):
        context_start = parsed_dates[idx - context_len + 1]
        target_start = parsed_dates[idx + 1]
        target_end = parsed_dates[idx + horizon]

        if target_end <= b.train_end:
            train.append(idx)
            continue

        if (
            context_start >= b.validation_start
            and target_start >= b.validation_start
            and target_end <= b.validation_end
        ):
            validation.append(idx)
            continue

        if (
            context_start >= b.test_start
            and target_start >= b.test_start
            and target_end <= b.test_end
        ):
            test.append(idx)

    arrays = {
        "train": np.asarray(train, dtype=np.int64),
        "validation": np.asarray(validation, dtype=np.int64),
        "test": np.asarray(test, dtype=np.int64),
    }
    for name, values in arrays.items():
        if values.size == 0:
            raise ValueError(f"fold {b.fold_id}: split {name!r} has no valid purged windows")
    return arrays


def summarize_fold_indices(
    dates: Sequence[str | date],
    *,
    context_len: int,
    horizon: int,
    boundary: FoldBoundary | Mapping[str, str],
    indices: Mapping[str, np.ndarray] | None = None,
) -> dict:
    parsed_dates = _canonical_dates(dates)
    b = boundary if isinstance(boundary, FoldBoundary) else parse_fold_boundary(boundary)
    idx = indices or build_purged_fold_indices(
        parsed_dates, context_len=context_len, horizon=horizon, boundary=b
    )

    def describe(values: np.ndarray) -> dict:
        first = int(values[0])
        last = int(values[-1])
        return {
            "count": int(len(values)),
            "first_context_start": parsed_dates[first - context_len + 1].isoformat(),
            "first_context_end": parsed_dates[first].isoformat(),
            "first_target_start": parsed_dates[first + 1].isoformat(),
            "first_target_end": parsed_dates[first + horizon].isoformat(),
            "last_context_start": parsed_dates[last - context_len + 1].isoformat(),
            "last_context_end": parsed_dates[last].isoformat(),
            "last_target_start": parsed_dates[last + 1].isoformat(),
            "last_target_end": parsed_dates[last + horizon].isoformat(),
            "context_end_indices": [int(x) for x in values],
        }

    return {
        "fold_id": b.fold_id,
        "context_len": context_len,
        "horizon": horizon,
        "boundary": {
            "train_end": b.train_end.isoformat(),
            "validation_start": b.validation_start.isoformat(),
            "validation_end": b.validation_end.isoformat(),
            "test_start": b.test_start.isoformat(),
            "test_end": b.test_end.isoformat(),
        },
        "purge_policy": (
            "Every retained validation/test window has its full context and target inside "
            "that split's declared date interval; every retained training target ends on or "
            "before train_end. Boundary-crossing windows are omitted."
        ),
        "splits": {name: describe(values) for name, values in idx.items()},
    }


def _fit_train_only_regime_labels(
    returns: torch.Tensor,
    *,
    train_last_row: int,
) -> tuple[torch.Tensor, tuple[float, float]]:
    """Fit the CSV volatility/spread regime proxy using training rows only.

    The general CSV loader computes regime quantiles over the whole CSV. That is safe for
    descriptive use but inappropriate for prospective confirmation because future
    validation/test returns would influence labels used by the training loss. This
    prospective helper freezes the thresholds on rows available to the training fold and
    then applies those thresholds unchanged to every later row.
    """

    if returns.ndim != 2 or returns.shape[0] == 0 or returns.shape[1] < 2:
        raise ValueError("returns must be a non-empty [time, assets] tensor with at least 2 assets")
    if train_last_row < 0 or train_last_row >= int(returns.shape[0]):
        raise ValueError("train_last_row must index an observed training row")

    vol = returns.abs().mean(dim=1)
    spread = returns.std(dim=1)
    stress = 0.5 * vol + 0.5 * spread
    train_stress = stress[: train_last_row + 1]
    if train_stress.numel() < 2:
        raise ValueError("at least two training rows are required to fit regime thresholds")
    q1, q2 = torch.quantile(
        train_stress,
        torch.tensor([0.55, 0.82], dtype=train_stress.dtype, device=train_stress.device),
    )
    regime = torch.zeros(len(returns), dtype=torch.long, device=returns.device)
    regime[stress >= q1] = 1
    regime[stress >= q2] = 2
    return regime.cpu(), (float(q1.item()), float(q2.item()))


def build_purged_fold_datasets(
    cfg: MarketConfig,
    *,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    k: int = 3,
):
    """Build Eigen-JEPA datasets from explicit purged context-end indices.

    This is intentionally separate from build_datasets so frozen historical evidence is
    not silently reinterpreted. It is meant for prospective real-market confirmation.
    All label-generating thresholds used by the prospective path are fit on training
    rows only; validation/test returns cannot move the regime or event thresholds.
    """

    if cfg.data_source != "csv":
        raise ValueError("purged real-market folds require data_source='csv'")
    series = generate_market_series(cfg)
    returns = series["returns"]
    aux = series["aux"]

    actual_steps = int(returns.shape[0])
    cfg.total_steps = actual_steps
    actual_assets = int(returns.shape[1])
    cfg.num_assets = actual_assets

    def checked(name: str, values: Sequence[int]) -> np.ndarray:
        arr = np.asarray(values, dtype=np.int64)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"{name} indices must be a non-empty 1D sequence")
        if np.any(arr[1:] <= arr[:-1]):
            raise ValueError(f"{name} indices must be strictly increasing and unique")
        if int(arr[0]) < cfg.context_len - 1:
            raise ValueError(f"{name} contains an index before a full context is available")
        if int(arr[-1]) + cfg.horizon >= actual_steps:
            raise ValueError(f"{name} contains an index whose target exceeds the CSV")
        return arr

    train_idx = checked("train", train_indices)
    val_idx = checked("validation", validation_indices)
    test_idx = checked("test", test_indices)
    all_values = np.concatenate([train_idx, val_idx, test_idx])
    if len(all_values) != len(np.unique(all_values)):
        raise ValueError("train/validation/test context-end indices must be disjoint")

    train_last_row = int(train_idx[-1]) + cfg.horizon
    regime, regime_thresholds = _fit_train_only_regime_labels(
        returns, train_last_row=train_last_row
    )
    # Retain the prospectively fitted labels/thresholds in the returned series evidence.
    series = dict(series)
    series["regime"] = regime
    series["regime_thresholds_train_only"] = regime_thresholds
    series["regime_fit_last_row"] = train_last_row

    train_scores = []
    for idx in train_idx:
        c0 = int(idx) - cfg.context_len + 1
        c1 = int(idx) + 1
        f0 = int(idx) + 1
        f1 = int(idx) + 1 + cfg.horizon
        score, _, _, _ = _window_salience_score(
            returns, regime, c0, c1, f0, f1, k
        )
        train_scores.append(score)
    event_threshold = float(
        np.quantile(np.asarray(train_scores, dtype=np.float32), cfg.event_quantile)
    )

    common = dict(
        returns=returns,
        aux=aux,
        regime=regime,
        context_len=cfg.context_len,
        horizon=cfg.horizon,
        k=k,
        mask_ratio=cfg.mask_ratio,
        block_time=cfg.block_time,
        event_threshold=event_threshold,
    )
    return {
        "train": MarketWindowDataset(indices=train_idx, seed=cfg.seed, **common),
        "val": MarketWindowDataset(indices=val_idx, seed=cfg.seed + 1, **common),
        "test": MarketWindowDataset(indices=test_idx, seed=cfg.seed + 2, **common),
        "series": series,
        "event_threshold": event_threshold,
        "regime_thresholds_train_only": regime_thresholds,
        "regime_fit_last_row": train_last_row,
        "split_indices": {
            "train": train_idx,
            "validation": val_idx,
            "test": test_idx,
        },
    }
