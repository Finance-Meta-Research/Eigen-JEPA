"""Opt-in development API that validates complete context/target footprints.

The original real_market_folds module is source-bound by a frozen prospective
protocol and remains byte-identical. A new study must explicitly select this
module and bind its source identity; existing frozen runners are not redirected.
The underlying feature/label/dataset implementation is unchanged.
"""
from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date
from numbers import Integral
from pathlib import Path
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


def _validate_csv_timeline(cfg: MarketConfig) -> int:
    """Check the observations' actual calendar before positional fold admission.

    Disjoint row indices establish chronology only when the CSV itself is in
    chronological order. The legacy loader ignores its date column, so v2 must
    validate these labels before that loader fits any thresholds.
    """
    if cfg.csv_path is None:
        raise ValueError("csv_path must be provided for purged real-market folds")
    if not isinstance(cfg.date_col, str) or not cfg.date_col:
        raise ValueError("an explicit date_col is required to verify CSV chronology")
    with Path(cfg.csv_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or any(not column for column in header) or len(set(header)) != len(header):
            raise ValueError("CSV header must contain nonempty unique column names")
        if cfg.date_col not in header:
            raise ValueError(f"date_col {cfg.date_col!r} is missing from the CSV header")
        date_index = header.index(cfg.date_col)
        previous = None
        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"CSV row {line_number} does not match its header width")
            label = row[date_index]
            try:
                parsed = date.fromisoformat(label)
            except ValueError as exc:
                raise ValueError(f"CSV row {line_number} must contain a canonical ISO date") from exc
            if parsed.isoformat() != label:
                raise ValueError(f"CSV row {line_number} must contain a canonical ISO date")
            if previous is not None and parsed <= previous:
                raise ValueError("CSV dates must be unique and strictly increasing")
            previous = parsed
            row_count += 1
    if row_count == 0:
        raise ValueError("CSV must contain at least one dated observation")
    return row_count


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
    An explicit CSV date column must contain strictly increasing, unique ISO
    calendar dates, so positional separation also means chronological separation.
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
    observed_rows = _validate_csv_timeline(cfg)
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        if int(indices[-1]) + cfg.horizon >= observed_rows:
            raise ValueError(f"{name} contains an index whose target exceeds the CSV")
    # The CSV loader changes shape metadata. Failed admission or CSV validation
    # must not mutate the user's configuration; publish shape only on success.
    candidate = replace(cfg)
    datasets = legacy.build_purged_fold_datasets(
        candidate, train_indices=train, validation_indices=validation,
        test_indices=test, k=k,
    )
    cfg.total_steps, cfg.num_assets = candidate.total_steps, candidate.num_assets
    return datasets
