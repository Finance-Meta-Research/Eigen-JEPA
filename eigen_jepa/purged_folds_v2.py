"""Opt-in development API that validates complete context/target footprints.

The original real_market_folds module is source-bound by a frozen prospective
protocol and remains byte-identical. A new study must explicitly select this
module and bind its source identity; existing frozen runners are not redirected.
The underlying feature/label/dataset implementation is unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from . import real_market_folds as legacy
from .data import MarketConfig


def _window_dimensions(context_len: int, horizon: int) -> None:
    if any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)) or value <= 0
        for value in (context_len, horizon)
    ):
        raise ValueError("context_len and horizon must be positive integers")


def _indices(name: str, values: Sequence[int], context_len: int) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} indices must be a non-empty 1D sequence")
    if (
        raw.dtype.kind not in "iu"
        or any(not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
               for value in values)
        or any(int(value) > np.iinfo(np.int64).max for value in raw)
    ):
        raise ValueError(f"{name} indices must contain representable integers")
    arr = raw.astype(np.int64)
    if np.any(arr[1:] <= arr[:-1]):
        raise ValueError(f"{name} indices must be strictly increasing and unique")
    if int(arr[0]) < context_len - 1:
        raise ValueError(f"{name} contains an index before a full context is available")
    return arr


def build_purged_fold_indices(
    dates: Sequence, *, context_len: int, horizon: int,
    boundary: legacy.FoldBoundary | Mapping[str, str],
) -> dict[str, np.ndarray]:
    """Build date-bound windows with strict integer dimension admission."""
    _window_dimensions(context_len, horizon)
    return legacy.build_purged_fold_indices(
        dates, context_len=context_len, horizon=horizon, boundary=boundary
    )


def build_purged_fold_datasets(
    cfg: MarketConfig, *, train_indices: Sequence[int],
    validation_indices: Sequence[int], test_indices: Sequence[int], k: int = 3,
):
    """Construct datasets only after full inclusive footprints are separated.

    A context ending at i uses [i-context_len+1, i+horizon]. The final target
    row in an earlier split must be strictly before the first context row in
    the following split. Within-split rolling windows are intentionally allowed.
    Returns and train-only threshold fitting are delegated to the retained
    implementation after admission, without any protected outcome generation.
    """
    if cfg.data_source != "csv":
        raise ValueError("purged real-market folds require data_source='csv'")
    _window_dimensions(cfg.context_len, cfg.horizon)
    train = _indices("train", train_indices, cfg.context_len)
    validation = _indices("validation", validation_indices, cfg.context_len)
    test = _indices("test", test_indices, cfg.context_len)
    for earlier_name, earlier, later_name, later in (
        ("train", train, "validation", validation),
        ("validation", validation, "test", test),
    ):
        target_end = int(earlier[-1]) + cfg.horizon
        context_start = int(later[0]) - cfg.context_len + 1
        if target_end >= context_start:
            raise ValueError(
                f"{earlier_name}/{later_name}: full context and target windows "
                "must be chronological and non-overlapping; "
                f"earlier target ends at {target_end}, later context starts at {context_start}"
            )
    # The CSV loader changes shape metadata. Failed admission or CSV validation
    # must not mutate the user's configuration; publish shape only on success.
    candidate = replace(cfg)
    datasets = legacy.build_purged_fold_datasets(
        candidate, train_indices=train, validation_indices=validation,
        test_indices=test, k=k,
    )
    cfg.total_steps, cfg.num_assets = candidate.total_steps, candidate.num_assets
    return datasets
