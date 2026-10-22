"""Generated numerical and real-model examples; no financial outcome access."""

import copy
import hashlib
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from eigen_jepa.covariance_geometry_v3 import (
    SCHEMA,
    covariance_factor_geometry,
    evaluate_model_geometry_v3,
)
from eigen_jepa.data import MarketConfig, batch_collate, build_datasets
from eigen_jepa.evaluation_v2 import evaluate_model_v2
from eigen_jepa.model import EigenJEPA, EigenJEPAConfig


def _geometry(weights, vectors, **kwargs):
    return covariance_factor_geometry(
        torch.tensor(weights, dtype=torch.float64),
        torch.tensor(vectors, dtype=torch.float64),
        **kwargs,
    )


def test_duplicate_unit_columns_are_rank_one_with_one_nonzero_eigenvalue():
    actual = _geometry([[1, 1]], [[[1, 1], [0, 0], [0, 0]]])
    torch.testing.assert_close(
        actual.eigenvalues, torch.tensor([[2.0, 0.0]], dtype=torch.float64)
    )
    expected_projector = torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))[
        None
    ]
    torch.testing.assert_close(actual.support_projector, expected_projector)
    torch.testing.assert_close(actual.covariance, 2 * expected_projector)
    assert actual.support_rank.tolist() == [1]
    torch.testing.assert_close(
        actual.support_projector @ actual.support_projector, expected_projector
    )


def test_unsorted_orthogonal_factors_keep_covariance_and_sort_the_spectrum():
    actual = _geometry([[1, 4]], [[[1, 0], [0, 1], [0, 0]]])
    torch.testing.assert_close(
        actual.eigenvalues, torch.tensor([[4.0, 1.0]], dtype=torch.float64)
    )
    expected = torch.diag(torch.tensor([1.0, 4.0, 0.0], dtype=torch.float64))[None]
    reconstructed = (
        actual.eigenvectors * actual.eigenvalues[:, None]
    ) @ actual.eigenvectors.mT
    torch.testing.assert_close(reconstructed, expected)


def test_independent_dense_covariance_eigendecomposition_and_projector_oracle():
    rng = np.random.default_rng(212)
    weights = rng.uniform(0.2, 3.0, (4, 3))
    vectors = rng.normal(size=(4, 6, 3))
    actual = _geometry(weights, vectors)
    for i in range(4):
        covariance = vectors[i] @ np.diag(weights[i]) @ vectors[i].T
        eig, vec = np.linalg.eigh(covariance)
        np.testing.assert_allclose(
            actual.covariance[i], covariance, rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            actual.eigenvalues[i], eig[-3:][::-1], rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            actual.support_projector[i], vec[:, -3:] @ vec[:, -3:].T, atol=1e-13
        )
        projector = actual.support_projector[i].numpy()
        np.testing.assert_allclose(projector @ projector, projector, atol=1e-13)


def test_factor_signs_and_paired_permutation_do_not_change_geometry():
    weights = torch.tensor([[1.0, 5.0, 2.0]], dtype=torch.float64)
    vectors = torch.tensor(
        [[[1.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 1.0]]], dtype=torch.float64
    )
    expected = covariance_factor_geometry(weights, vectors)
    permutation = [2, 0, 1]
    actual = covariance_factor_geometry(
        weights[:, permutation], vectors[:, :, permutation] * torch.tensor([-1, 1, -1])
    )
    for name in ("eigenvalues", "covariance", "support_projector"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("scale", [1e-150, 1e-75, 1.0, 1e75, 1e150])
def test_representable_extreme_factor_scales_preserve_normalized_geometry(scale):
    vectors = np.array([[[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])
    reference = _geometry([[1.0, 2.0]], vectors)
    actual = _geometry([[1.0, 2.0]], vectors * scale)
    np.testing.assert_allclose(
        actual.eigenvalues.numpy() / scale / scale,
        reference.eigenvalues.numpy(),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        actual.covariance.numpy() / scale / scale,
        reference.covariance.numpy(),
        rtol=1e-13,
    )
    torch.testing.assert_close(actual.support_projector, reference.support_projector)


def test_zero_and_partially_zero_weights_do_not_create_phantom_support():
    actual = _geometry([[0, 0], [1, 0]], [[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
    assert actual.support_rank.tolist() == [0, 1]
    torch.testing.assert_close(
        actual.support_projector[0], torch.zeros(2, 2, dtype=torch.float64)
    )
    torch.testing.assert_close(
        actual.eigenvalues[0], torch.zeros(2, dtype=torch.float64)
    )


def test_relative_rank_threshold_does_not_truncate_the_covariance_eigenvalues():
    actual = _geometry([[1, 1]], [[[1, 0], [0, 0.01]]], rank_rtol=0.01)
    assert actual.support_rank.tolist() == [1]  # Equality is excluded by contract.
    torch.testing.assert_close(
        actual.eigenvalues, torch.tensor([[1.0, 1e-4]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        actual.covariance,
        torch.diag(torch.tensor([1.0, 1e-4], dtype=torch.float64))[None],
    )
    assert actual.rank_rtol == 0.01


@pytest.mark.parametrize("rtol", [True, "0.01", -0.1, 1.0, float("inf"), float("nan")])
def test_invalid_rank_tolerance_is_rejected(rtol):
    with pytest.raises(ValueError, match="rank_rtol"):
        _geometry([[1]], [[[1]]], rank_rtol=rtol)


@pytest.mark.parametrize(
    "mutation",
    [
        "integer",
        "boolean",
        "complex",
        "negative",
        "nan",
        "empty",
        "too_many_columns",
        "mismatched_batch",
        "overflow",
    ],
)
def test_invalid_factors_fail_without_a_repair_or_fallback(mutation):
    weights = torch.ones(1, 2, dtype=torch.float64)
    vectors = torch.eye(2, dtype=torch.float64)[None]
    if mutation == "integer":
        weights = weights.long()
    elif mutation == "boolean":
        vectors = vectors.bool()
    elif mutation == "complex":
        vectors = vectors.cdouble()
    elif mutation == "negative":
        weights[0, 0] = -1
    elif mutation == "nan":
        vectors[0, 0, 0] = float("nan")
    elif mutation == "empty":
        weights, vectors = weights[:0], vectors[:0]
    elif mutation == "too_many_columns":
        vectors = vectors[:, :1]
    elif mutation == "mismatched_batch":
        vectors = vectors.repeat(2, 1, 1)
    else:
        vectors *= 1e308
    with pytest.raises(ValueError):
        covariance_factor_geometry(weights, vectors)


def test_svd_failure_is_not_replaced_with_jitter_or_a_new_basis(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("constructed failure")

    monkeypatch.setattr(torch.linalg, "svd", fail)
    with pytest.raises(ValueError, match="no fallback"):
        _geometry([[1]], [[[1]]])


def test_unrepresentable_rank_tolerance_is_a_contract_error():
    with pytest.raises(ValueError, match="rank_rtol"):
        _geometry([[1]], [[[1]]], rank_rtol=10**400)


def test_supported_eigenvalue_underflow_cannot_create_phantom_zero_covariance_support():
    with pytest.raises(ValueError, match="underflow"):
        _geometry([[1]], [[[1e-200]]])


def test_finite_covariance_with_unrepresentable_eigenvalue_is_rejected():
    with pytest.raises(ValueError, match="eigensystem exceeds"):
        _geometry([[1]], [[[1e154], [1e154]]])


def test_inference_only_operator_detaches_without_mutating_inputs():
    weights = torch.tensor([[1.0, 2.0]], requires_grad=True)
    vectors = torch.eye(2)[None].requires_grad_()
    original = (weights.clone(), vectors.clone())
    actual = covariance_factor_geometry(weights, vectors)
    assert not actual.eigenvalues.requires_grad
    assert not actual.covariance.requires_grad
    torch.testing.assert_close(weights, original[0])
    torch.testing.assert_close(vectors, original[1])


@pytest.fixture
def model_and_windows():
    torch.manual_seed(212)
    config = MarketConfig(
        num_assets=3,
        total_steps=60,
        context_len=8,
        horizon=3,
        num_train=8,
        num_val=5,
        num_test=5,
        seed=212,
    )
    source = build_datasets(config, k=2)["train"]
    windows = [source[i] for i in range(5)]  # Materialize fresh masks exactly once.
    model = EigenJEPA(
        EigenJEPAConfig(
            input_dim=windows[0]["x"].shape[-1] - 1,
            num_assets=3,
            context_len=8,
            k=2,
            d_model=16,
            latent_dim=16,
            memory_dim=16,
        ),
        target_dim=windows[0]["target_vec"].numel(),
    ).eval()
    return model, windows


def _loader(windows, batch_size):
    return DataLoader(
        windows, batch_size=batch_size, shuffle=False, collate_fn=batch_collate
    )


def test_actual_model_duplicate_column_witness_repairs_geometry_without_changing_covariance(
    model_and_windows,
):
    model, windows = model_and_windows
    with torch.no_grad():
        model.subspace_head[-1].weight.zero_()
        model.subspace_head[-1].bias.copy_(torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
        model.eig_head[-2].weight.zero_()
        model.eig_head[-2].bias.fill_(float(np.log(np.expm1(1.0))))
    for window in windows:
        window["evals_true"] = torch.tensor([2.0, 0.0])
        window["proj_true"] = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
        window["cov_true"] = torch.diag(torch.tensor([2.0, 0.0, 0.0]))
    state_before = copy.deepcopy(model.state_dict())
    legacy = evaluate_model_v2(model, _loader(windows, 2))
    result = evaluate_model_geometry_v3(model, _loader(windows, 2))
    assert legacy["metrics"]["eig_nmse"] == pytest.approx(0.5)
    assert legacy["metrics"]["proj_mse"] == pytest.approx(1 / 9)
    assert result["metrics"]["eig_nmse"] == pytest.approx(0, abs=1e-13)
    assert result["metrics"]["proj_mse"] == pytest.approx(0, abs=1e-13)
    assert result["metrics"]["cov_mse"] == pytest.approx(
        legacy["metrics"]["cov_mse"], abs=1e-13
    )
    assert {row["support_rank"] for row in result["geometry_records"]} == {1}
    for name, value in state_before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_actual_model_uneven_batches_share_input_ids_and_numerical_results(
    model_and_windows,
):
    model, windows = model_and_windows
    single = evaluate_model_geometry_v3(model, _loader(windows, 1))
    uneven = evaluate_model_geometry_v3(model, _loader(windows, 3))
    assert single["geometry"] == uneven["geometry"]
    assert [row["input_sha256"] for row in single["records"]] == [
        row["input_sha256"] for row in uneven["records"]
    ]
    for name, value in single["metrics"].items():
        if isinstance(value, float):
            assert uneven["metrics"][name] == pytest.approx(value, rel=1e-5, abs=1e-7)
        else:
            assert uneven["metrics"][name] == value


def test_report_identity_binds_raw_factor_provenance_and_rank_policy(model_and_windows):
    model, windows = model_and_windows
    report = evaluate_model_geometry_v3(model, _loader(windows, 3))
    assert report["schema"] == SCHEMA
    assert len(report["geometry_records"]) == len(windows)
    for row in report["geometry_records"]:
        assert len(row["factor_sha256"]) == len(row["covariance_sha256"]) == 64
    supplied_digest = report.pop("result_sha256")
    assert (
        hashlib.sha256(
            json.dumps(
                report, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        == supplied_digest
    )
    changed = evaluate_model_geometry_v3(model, _loader(windows, 3), rank_rtol=0.1)
    assert changed["result_sha256"] != supplied_digest


def test_empty_cohort_fails_before_emitting_an_evaluation(model_and_windows):
    model, _ = model_and_windows
    with pytest.raises(ValueError, match="at least one window"):
        evaluate_model_geometry_v3(model, [])


def test_equivalent_covariance_keeps_metrics_but_binds_changed_factor_bytes(
    model_and_windows,
):
    model, windows = model_and_windows
    before = evaluate_model_geometry_v3(model, _loader(windows, 3))
    with torch.no_grad():
        model.subspace_head[-1].weight.mul_(-1)
        model.subspace_head[-1].bias.mul_(-1)
    after = evaluate_model_geometry_v3(model, _loader(windows, 3))
    assert before["result_sha256"] != after["result_sha256"]
    for old, new in zip(before["geometry_records"], after["geometry_records"]):
        assert old["factor_sha256"] != new["factor_sha256"]
    for name in ("eig_nmse", "proj_mse", "cov_mse", "gap_rmse"):
        assert after["metrics"][name] == pytest.approx(
            before["metrics"][name], rel=1e-12, abs=1e-12
        )
