"""Generated-data tests for the opt-in v2 purging API; v1 remains frozen."""
import csv
from datetime import date, timedelta

import numpy as np
import pytest

from eigen_jepa.data import MarketConfig
from eigen_jepa.purged_folds_v2 import build_purged_fold_datasets, build_purged_fold_indices
from eigen_jepa.real_market_folds import parse_fold_boundary


def _dates(n: int, start=date(2026, 1, 1)):
    return [(start + timedelta(days=i)).isoformat() for i in range(n)]


def _boundary():
    return parse_fold_boundary(
        {
            "id": "F1",
            "train_end": "2026-01-12",
            "validation_start": "2026-01-13",
            "validation_end": "2026-01-20",
            "test_start": "2026-01-21",
            "test_end": "2026-01-30",
        }
    )


def _write_returns(path, *, future_scale=1.0):
    dates = _dates(30)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "A", "B", "C"])
        for i, d in enumerate(dates):
            scale = future_scale if i >= 20 else 1.0
            writer.writerow(
                [
                    d,
                    scale * 0.001 * (i + 1),
                    scale * -0.0007 * (i + 1),
                    scale * 0.0004 * (i + 1),
                ]
            )
    return dates



def _explicit_cfg(path):
    return MarketConfig(
        context_len=4, horizon=2, data_source="csv", csv_path=str(path),
        return_cols=("A", "B", "C"), date_col="date",
    )


@pytest.mark.parametrize("train,validation,test", [
    ([8], [9], [24]),       # Distinct endpoints; training targets overlap validation.
    ([8], [13], [24]),      # One row of context/target contact is still leakage.
    ([3], [15], [20]),      # Validation targets overlap test context.
    ([15], [3], [24]),      # Disjoint windows in reversed calendar order.
    ([3], [24], [15]),      # Test precedes validation.
])
def test_dataset_builder_rejects_disjoint_endpoints_with_unpurged_footprints(
    tmp_path, train, validation, test
):
    path = tmp_path / "returns.csv"
    _write_returns(path)
    cfg = _explicit_cfg(path)
    original_shape = (cfg.total_steps, cfg.num_assets)
    with pytest.raises(ValueError, match="full context.*target"):
        build_purged_fold_datasets(
            cfg, train_indices=train, validation_indices=validation,
            test_indices=test, k=2,
        )
    assert (cfg.total_steps, cfg.num_assets) == original_shape


@pytest.mark.parametrize("bad", [[3.9], [True], ["3"], [np.nan], [np.inf], [2**70]])
def test_dataset_builder_rejects_indices_that_require_lossy_integer_coercion(tmp_path, bad):
    path = tmp_path / "returns.csv"
    _write_returns(path)
    with pytest.raises(ValueError, match="integer"):
        build_purged_fold_datasets(
            _explicit_cfg(path), train_indices=bad,
            validation_indices=[14], test_indices=[24], k=2,
        )


def test_adjacent_nonoverlapping_full_windows_remain_admissible(tmp_path):
    path = tmp_path / "returns.csv"
    _write_returns(path)
    datasets = build_purged_fold_datasets(
        _explicit_cfg(path), train_indices=[8], validation_indices=[14],
        test_indices=[20], k=2,
    )
    # Train footprint [5,10], validation [11,16], test [17,22].
    assert datasets["regime_fit_last_row"] == 10
    for name in ["train", "val", "test"]:
        assert len(datasets[name]) == 1


@pytest.mark.parametrize("context_len,horizon", [(True, 2), (4, False), (4.5, 2), (4, 1.2)])
def test_window_dimensions_are_positive_integers(context_len, horizon):
    with pytest.raises(ValueError, match="positive integer"):
        build_purged_fold_indices(
            _dates(30), context_len=context_len, horizon=horizon, boundary=_boundary()
        )


def test_admitted_v2_windows_preserve_existing_dataset_values_and_train_thresholds(tmp_path):
    from eigen_jepa.real_market_folds import build_purged_fold_datasets as original_builder

    path = tmp_path / "returns.csv"
    _write_returns(path)
    kwargs = dict(train_indices=[3, 4, 8], validation_indices=[14], test_indices=[20, 24], k=2)
    original = original_builder(_explicit_cfg(path), **kwargs)
    corrected = build_purged_fold_datasets(_explicit_cfg(path), **kwargs)
    assert original["event_threshold"] == corrected["event_threshold"]
    assert original["regime_thresholds_train_only"] == corrected["regime_thresholds_train_only"]
    for key in ["returns", "aux", "regime"]:
        np.testing.assert_array_equal(original["series"][key].numpy(), corrected["series"][key].numpy())
    for key in ["train", "validation", "test"]:
        np.testing.assert_array_equal(original["split_indices"][key], corrected["split_indices"][key])
