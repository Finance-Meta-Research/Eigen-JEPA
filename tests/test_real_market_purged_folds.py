import csv
from datetime import date, timedelta

import numpy as np
import pytest

from eigen_jepa.data import MarketConfig
from eigen_jepa.real_market_folds import (
    build_purged_fold_datasets,
    build_purged_fold_indices,
    parse_fold_boundary,
    summarize_fold_indices,
)


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


def _build_from_csv(path):
    dates = _dates(30)
    indices = build_purged_fold_indices(
        dates, context_len=4, horizon=2, boundary=_boundary()
    )
    cfg = MarketConfig(
        num_assets=99,
        total_steps=999,
        context_len=4,
        horizon=2,
        num_train=1,
        num_val=1,
        num_test=1,
        seed=7,
        data_source="csv",
        csv_path=str(path),
        return_cols=("A", "B", "C"),
        date_col="date",
    )
    datasets = build_purged_fold_datasets(
        cfg,
        train_indices=indices["train"],
        validation_indices=indices["validation"],
        test_indices=indices["test"],
        k=2,
    )
    return cfg, indices, datasets


def test_purged_indices_keep_full_windows_inside_declared_splits():
    dates = _dates(30)
    context_len = 4
    horizon = 2
    indices = build_purged_fold_indices(
        dates, context_len=context_len, horizon=horizon, boundary=_boundary()
    )

    train_last = int(indices["train"][-1])
    val_first = int(indices["validation"][0])
    val_last = int(indices["validation"][-1])
    test_first = int(indices["test"][0])

    assert dates[train_last + horizon] == "2026-01-12"
    assert dates[val_first - context_len + 1] == "2026-01-13"
    assert dates[val_last + horizon] == "2026-01-20"
    assert dates[test_first - context_len + 1] == "2026-01-21"

    # A boundary-crossing window is deliberately purged between train and validation.
    assert val_first - train_last > 1


def test_summary_records_exact_window_date_extents():
    dates = _dates(30)
    summary = summarize_fold_indices(
        dates, context_len=4, horizon=2, boundary=_boundary()
    )
    assert summary["fold_id"] == "F1"
    assert summary["splits"]["train"]["last_target_end"] == "2026-01-12"
    assert summary["splits"]["validation"]["first_context_start"] == "2026-01-13"
    assert summary["splits"]["validation"]["last_target_end"] == "2026-01-20"
    assert summary["splits"]["test"]["first_context_start"] == "2026-01-21"
    assert summary["splits"]["test"]["last_target_end"] == "2026-01-30"


def test_rejects_overlapping_or_reversed_boundaries():
    with pytest.raises(ValueError, match="boundaries must satisfy"):
        parse_fold_boundary(
            {
                "id": "bad",
                "train_end": "2026-01-12",
                "validation_start": "2026-01-12",
                "validation_end": "2026-01-20",
                "test_start": "2026-01-21",
                "test_end": "2026-01-30",
            }
        )


def test_rejects_duplicate_or_unsorted_dates():
    dates = _dates(30)
    duplicate = dates.copy()
    duplicate[5] = duplicate[4]
    with pytest.raises(ValueError, match="unique"):
        build_purged_fold_indices(
            duplicate, context_len=4, horizon=2, boundary=_boundary()
        )

    unsorted = dates.copy()
    unsorted[5], unsorted[6] = unsorted[6], unsorted[5]
    with pytest.raises(ValueError, match="strictly increasing"):
        build_purged_fold_indices(
            unsorted, context_len=4, horizon=2, boundary=_boundary()
        )


def test_dataset_builder_uses_explicit_indices_and_actual_csv_shape(tmp_path):
    path = tmp_path / "returns.csv"
    _write_returns(path)
    cfg, indices, datasets = _build_from_csv(path)

    assert cfg.total_steps == 30
    assert cfg.num_assets == 3
    assert np.array_equal(datasets["split_indices"]["train"], indices["train"])
    assert np.array_equal(datasets["split_indices"]["validation"], indices["validation"])
    assert np.array_equal(datasets["split_indices"]["test"], indices["test"])
    assert len(datasets["train"]) == len(indices["train"])
    assert len(datasets["val"]) == len(indices["validation"])
    assert len(datasets["test"]) == len(indices["test"])
    assert datasets["regime_fit_last_row"] == int(indices["train"][-1]) + cfg.horizon


def test_future_test_return_shock_cannot_move_train_fitted_regime_or_event_threshold(tmp_path):
    base_path = tmp_path / "base.csv"
    shocked_path = tmp_path / "shocked.csv"
    _write_returns(base_path, future_scale=1.0)
    # A huge shock is confined to rows starting in the held-out test interval.
    _write_returns(shocked_path, future_scale=10000.0)

    _, base_indices, base = _build_from_csv(base_path)
    _, shocked_indices, shocked = _build_from_csv(shocked_path)

    assert np.array_equal(base_indices["train"], shocked_indices["train"])
    assert base["regime_fit_last_row"] == shocked["regime_fit_last_row"]
    assert base["regime_thresholds_train_only"] == pytest.approx(
        shocked["regime_thresholds_train_only"]
    )
    assert base["event_threshold"] == pytest.approx(shocked["event_threshold"])

    last_train_row = base["regime_fit_last_row"]
    base_train_regime = base["series"]["regime"][: last_train_row + 1]
    shocked_train_regime = shocked["series"]["regime"][: last_train_row + 1]
    assert np.array_equal(base_train_regime.numpy(), shocked_train_regime.numpy())

    # The perturbation is real and affects only the held-out future; this makes the
    # invariance above a meaningful leakage regression rather than an identical-input test.
    assert not np.array_equal(
        base["series"]["returns"][20:].numpy(),
        shocked["series"]["returns"][20:].numpy(),
    )


def test_dataset_builder_rejects_non_csv_and_overlapping_indices():
    cfg = MarketConfig(data_source="synthetic")
    with pytest.raises(ValueError, match="data_source='csv'"):
        build_purged_fold_datasets(
            cfg,
            train_indices=[10],
            validation_indices=[20],
            test_indices=[30],
        )


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
