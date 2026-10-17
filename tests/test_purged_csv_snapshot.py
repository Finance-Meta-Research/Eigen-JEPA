"""Prospective dataset construction must consume the admitted CSV snapshot."""
import csv
import hashlib

import numpy as np
import pytest

from eigen_jepa.data import MarketConfig
from eigen_jepa import purged_folds_v2 as module
from test_real_market_purged_folds_v2 import _write_returns


def config(path):
    return MarketConfig(context_len=4, horizon=2, data_source='csv',
                        csv_path=str(path), return_cols=('A', 'B', 'C'), date_col='date')


def build(cfg, k=2):
    return module.build_purged_fold_datasets(cfg, train_indices=[8], validation_indices=[14],
                                            test_indices=[20], k=k)


def test_file_replacement_after_admission_cannot_change_consumed_values(tmp_path, monkeypatch):
    path = tmp_path / 'input.csv'
    _write_returns(path)
    original = path.read_bytes()
    cfg = config(path)
    admitted = build(cfg)
    legacy_builder = module.legacy.build_purged_fold_datasets

    def replace_original_then_load(candidate, **kwargs):
        # Reproduces a concurrently replaced data path after admission.
        _write_returns(path, future_scale=1000)
        return legacy_builder(candidate, **kwargs)

    monkeypatch.setattr(module.legacy, 'build_purged_fold_datasets', replace_original_then_load)
    result = build(config(path))
    np.testing.assert_array_equal(result['series']['returns'].numpy(), admitted['series']['returns'].numpy())
    assert result['input_csv_sha256'] == hashlib.sha256(original).hexdigest()
    assert cfg.csv_path == str(path)


@pytest.mark.parametrize('columns', [('A', 'A'), ('A',), ('A', 'missing'), ('date', 'A')])
def test_ambiguous_return_columns_are_rejected_before_threshold_construction(tmp_path, monkeypatch, columns):
    path = tmp_path / 'input.csv'
    _write_returns(path)
    cfg = config(path)
    cfg.return_cols = columns
    def unexpected(*args, **kwargs):
        pytest.fail('invalid columns reached label and threshold construction')
    monkeypatch.setattr(module.legacy, 'build_purged_fold_datasets', unexpected)
    with pytest.raises(ValueError):
        build(cfg)


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '1e100'])
def test_nonfinite_float32_csv_values_are_rejected_before_thresholds(tmp_path, monkeypatch, value):
    path = tmp_path / 'input.csv'
    _write_returns(path)
    with path.open(newline='') as handle:
        rows = list(csv.reader(handle))
    rows[3][1] = value
    with path.open('w', newline='') as handle:
        csv.writer(handle).writerows(rows)
    def unexpected(*args, **kwargs):
        pytest.fail('invalid values reached label and threshold construction')
    monkeypatch.setattr(module.legacy, 'build_purged_fold_datasets', unexpected)
    with pytest.raises(ValueError):
        build(config(path))


@pytest.mark.parametrize('k', [0, True, 1.2, 4])
def test_spectral_rank_is_a_positive_integer_within_observed_asset_dimension(tmp_path, monkeypatch, k):
    path = tmp_path / 'input.csv'
    _write_returns(path)
    def unexpected(*args, **kwargs):
        pytest.fail('invalid spectral rank reached label and threshold construction')
    monkeypatch.setattr(module.legacy, 'build_purged_fold_datasets', unexpected)
    with pytest.raises(ValueError):
        build(config(path), k=k)
