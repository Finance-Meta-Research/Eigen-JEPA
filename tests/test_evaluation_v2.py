"""Generated fixtures only; no retained/protected outcomes are re-evaluated."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from eigen_jepa.data import MarketConfig, batch_collate, build_datasets
from eigen_jepa.evaluation_v2 import (
    MSE_FIELDS,
    evaluate_baselines_v2,
    evaluate_model_v2,
    evaluation_report,
    reduce_records,
)
from eigen_jepa.model import EigenJEPA, EigenJEPAConfig


def _row(index, error=0.0, energy=1.0, event=0, gate=0.0, regime=0, predicted=0):
    return {
        "window_index": index,
        "eigenvalue_count": 2,
        "input_sha256": "0" * 64,
        "eig_squared_error": error,
        "eig_target_energy": energy,
        "gap_mse": error,
        **{name: error for name in MSE_FIELDS},
        "event_true": event,
        "gate": gate,
        "regime_true": regime,
        "pred_regime": predicted,
    }


def test_nmse_is_ratio_of_cohort_errors_and_energy():
    rows = [_row(1, error=1, energy=1), _row(2, error=0, energy=100)]
    report = reduce_records(rows)
    assert report["eig_nmse"] == pytest.approx(1.0 / 101)
    assert report["eig_nmse"] != pytest.approx(0.5)
    assert report["gap_rmse"] == pytest.approx(np.sqrt(0.5))


def test_f1_and_balanced_accuracy_pool_confusion_before_reducing():
    rows = [
        _row(1, event=1, gate=0.9, regime=0, predicted=0),
        _row(2, event=0, gate=0.9, regime=1, predicted=0),
        _row(3, event=1, gate=0.0, regime=1, predicted=1),
    ]
    result = reduce_records(rows)
    assert result["tail_confusion"] == {"tp": 1, "fp": 1, "fn": 1}
    assert result["tail_f1"] == 0.5
    assert result["regime_bal_acc"] == 0.75


def test_subgroups_only_use_their_own_numerators_and_denominators():
    rows = [_row(1, error=1, energy=2, event=1), _row(2, error=1000, energy=1)]
    result = reduce_records(rows)
    assert result["rare_eig_nmse"] == pytest.approx(0.5)
    assert result["rare_cov_mse"] == 1.0
    assert reduce_records(rows[:1])["cov_mse"] == 1.0
    assert reduce_records(rows[1:])["cov_mse"] == 1000.0


def test_unavailable_rare_metrics_are_explicit_null_and_json_serializable():
    result = evaluation_report([_row(1)])
    assert result["metrics"]["rare_eig_nmse"] is None
    assert result["metrics"]["rare_window_count"] == 0
    assert len(result["result_sha256"]) == 64


def test_order_does_not_change_result_identity():
    rows = [_row(2, error=1), _row(1, error=2)]
    assert evaluation_report(rows) == evaluation_report(list(reversed(rows)))


def test_missing_input_identity_is_rejected():
    row = _row(1)
    del row["input_sha256"]
    with pytest.raises(ValueError, match="missing required"):
        evaluation_report([row])


def test_single_eigenvalue_has_no_fabricated_gap_metric():
    row = _row(1)
    row["eigenvalue_count"] = 1
    del row["gap_mse"]
    assert evaluation_report([row])["metrics"]["gap_rmse"] is None


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "nan", "bad_event", "bad_gate", "negative_error", "mismatched_k"],
)
def test_invalid_statistics_cannot_produce_a_report(mutation):
    rows = [_row(1), _row(2)]
    if mutation == "duplicate":
        rows[1]["window_index"] = 1
    elif mutation == "nan":
        rows[1]["cov_mse"] = float("nan")
    elif mutation == "bad_event":
        rows[1]["event_true"] = 0.5
    elif mutation == "bad_gate":
        rows[1]["gate"] = 1.1
    elif mutation == "negative_error":
        rows[1]["eig_squared_error"] = -1
    else:
        rows[1]["eigenvalue_count"] = 3
    with pytest.raises(ValueError):
        evaluation_report(rows)


@pytest.fixture
def model_and_dataset():
    torch.manual_seed(3)
    cfg = MarketConfig(
        num_assets=4,
        total_steps=80,
        context_len=10,
        horizon=4,
        num_train=12,
        num_val=5,
        num_test=5,
        seed=3,
        market_style="equity",
    )
    # Freeze the concrete input tensors once. The legacy dataset deliberately
    # samples a new feature mask on each read; that is a changed input, not a
    # different batching of the same input.
    source = build_datasets(cfg, k=2)["test"]
    dataset = [source[i] for i in range(len(source))]
    item = dataset[0]
    model = EigenJEPA(
        EigenJEPAConfig(
            input_dim=item["x"].shape[-1] - 1,
            num_assets=4,
            context_len=10,
            k=2,
            d_model=16,
            latent_dim=16,
            memory_dim=16,
        ),
        target_dim=item["target_vec"].numel(),
    )
    return model, dataset


def _loader(dataset, size):
    return DataLoader(dataset, batch_size=size, shuffle=False, collate_fn=batch_collate)


def test_real_model_evaluation_is_invariant_to_uneven_batch_partition(
    model_and_dataset,
):
    model, dataset = model_and_dataset
    by_one = evaluate_model_v2(model, _loader(dataset, 1))
    uneven = evaluate_model_v2(model, _loader(dataset, 3))
    assert len(uneven["records"]) == len(dataset)
    for key, value in by_one["metrics"].items():
        if isinstance(value, float):
            assert uneven["metrics"][key] == pytest.approx(value, rel=1e-5, abs=1e-7)
        else:
            assert uneven["metrics"][key] == value
    # Reconstruct arbitrary slices solely from their retained per-window rows.
    first = reduce_records(uneven["records"][:1])
    assert first["eig_nmse"] == pytest.approx(
        by_one["records"][0]["eig_squared_error"]
        / (by_one["records"][0]["eig_target_energy"] + 1e-12),
        rel=1e-5,
    )


def test_baselines_cover_the_complete_same_cohort_and_ignore_batch_composition(
    model_and_dataset,
):
    _, dataset = model_and_dataset
    single = evaluate_baselines_v2(_loader(dataset, 1))
    uneven = evaluate_baselines_v2(_loader(dataset, 3))
    assert set(single) == {"persistence", "trend"}
    for baseline in single:
        assert uneven[baseline]["metrics"]["window_count"] == len(dataset)
        assert len(uneven[baseline]["records"]) == len(dataset)
        for key, value in single[baseline]["metrics"].items():
            if isinstance(value, float):
                assert uneven[baseline]["metrics"][key] == pytest.approx(
                    value, rel=1e-5, abs=1e-7
                )
            else:
                assert uneven[baseline]["metrics"][key] == value


def test_empty_loader_is_rejected(model_and_dataset):
    model, _ = model_and_dataset
    with pytest.raises(ValueError, match="at least one window"):
        evaluate_model_v2(model, [])
    with pytest.raises(ValueError, match="at least one window"):
        evaluate_baselines_v2([])
