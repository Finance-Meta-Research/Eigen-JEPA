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


@pytest.mark.parametrize("case", [
    "reversed", "duplicate", "invalid", "noncanonical", "missing_date_header",
    "duplicate_header", "short_row", "long_row", "empty_observations",
])
def test_csv_calendar_and_shape_are_admitted_before_threshold_fitting(tmp_path, monkeypatch, case):
    from eigen_jepa import purged_folds_v2

    path = tmp_path / "returns.csv"
    _write_returns(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if case == "reversed":
        # Swap across the train/validation footprints: positional purging alone
        # would admit a January 16 training row and January 9 validation row.
        rows[9], rows[16] = rows[16], rows[9]
    elif case == "duplicate":
        rows[10][0] = rows[9][0]
    elif case == "invalid":
        rows[10][0] = "2026-02-30"
    elif case == "noncanonical":
        rows[10][0] = "20260110"
    elif case == "missing_date_header":
        rows[0][0] = "unknown"
    elif case == "duplicate_header":
        rows[0][2] = rows[0][1]
    elif case == "short_row":
        rows[10].pop()
    elif case == "long_row":
        rows[10].append("unexpected")
    else:
        rows = rows[:1]
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    cfg = _explicit_cfg(path)
    before = vars(cfg).copy()

    def unexpected_load(*args, **kwargs):
        pytest.fail("invalid CSV reached the historical label/threshold builder")

    monkeypatch.setattr(purged_folds_v2.legacy, "build_purged_fold_datasets", unexpected_load)
    with pytest.raises(ValueError, match="CSV|date_col"):
        build_purged_fold_datasets(
            cfg, train_indices=[8], validation_indices=[14], test_indices=[20], k=2,
        )
    assert vars(cfg) == before


def test_undated_csv_cannot_claim_chronological_fold_admission(tmp_path):
    path = tmp_path / "returns.csv"
    _write_returns(path)
    cfg = _explicit_cfg(path)
    cfg.date_col = None
    with pytest.raises(ValueError, match="explicit date_col"):
        build_purged_fold_datasets(
            cfg, train_indices=[8], validation_indices=[14], test_indices=[20], k=2,
        )


def test_out_of_bounds_target_is_rejected_before_loading_observations(tmp_path, monkeypatch):
    from eigen_jepa import purged_folds_v2

    path = tmp_path / "returns.csv"
    _write_returns(path)

    def unexpected_load(*args, **kwargs):
        pytest.fail("out-of-bounds indices reached the historical threshold builder")

    monkeypatch.setattr(purged_folds_v2.legacy, "build_purged_fold_datasets", unexpected_load)
    with pytest.raises(ValueError, match="target exceeds the CSV"):
        build_purged_fold_datasets(
            _explicit_cfg(path), train_indices=[8], validation_indices=[14],
            test_indices=[28], k=2,
        )
