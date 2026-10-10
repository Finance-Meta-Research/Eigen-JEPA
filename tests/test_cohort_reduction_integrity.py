"""Finite arithmetic and receipt ownership for prospective evaluation only."""

import hashlib
import json

import numpy as np
import pytest

from eigen_jepa.evaluation_v2 import MSE_FIELDS, evaluation_report, reduce_records


def row(index, value, energy=None, event=1):
    return {
        "window_index": index,
        "eigenvalue_count": 2,
        "input_sha256": "0" * 64,
        "eig_squared_error": value,
        "eig_target_energy": value if energy is None else energy,
        "gap_mse": value,
        "lat_align": value,
        **{name: value for name in MSE_FIELDS},
        "event_true": event,
        "gate": float(event),
        "regime_true": 0,
        "pred_regime": 0,
    }


@pytest.mark.parametrize("count", [2, 3, 17])
def test_repeated_large_finite_statistics_have_the_same_representable_mean(count):
    report = evaluation_report([row(i, 1e308) for i in range(count)])
    metrics = report["metrics"]
    for name in (*MSE_FIELDS, "lat_align", "rare_cov_mse", "rare_drift_mse"):
        assert metrics[name] == pytest.approx(1e308)
    assert metrics["eig_nmse"] == pytest.approx(1.)
    assert metrics["rare_eig_nmse"] == pytest.approx(1.)
    assert metrics["gap_rmse"] == pytest.approx(1e154)
    json.dumps(report, allow_nan=False)


def test_large_unequal_windows_and_rare_slice_keep_their_own_means():
    metrics = reduce_records([row(1, 1e308, energy=5e307), row(2, 5e307, energy=1e308, event=0)])
    assert metrics["cov_mse"] == pytest.approx(7.5e307)
    assert metrics["eig_nmse"] == pytest.approx(1.)
    assert metrics["rare_cov_mse"] == pytest.approx(1e308)
    assert metrics["rare_eig_nmse"] == pytest.approx(2.)


def test_existing_epsilon_and_zero_energy_policy_are_preserved():
    metrics = reduce_records([row(1, 0., energy=0.)])
    assert metrics["eig_nmse"] == 0.
    assert metrics["gap_rmse"] == pytest.approx(1e-6)
    assert reduce_records([row(1, 1., energy=0.)])["eig_nmse"] == pytest.approx(1e12)


@pytest.mark.parametrize("error,energy", [(1e308, 0.), (np.nextafter(0., 1.), 1e308)])
def test_unrepresentable_nonzero_nmse_is_an_explicit_error(error, energy):
    with pytest.raises(ValueError, match="NMSE"):
        reduce_records([row(1, error, energy=energy)])


def test_nonzero_mean_underflow_is_an_explicit_error():
    with pytest.raises(ValueError, match="underflow"):
        reduce_records([row(1, np.nextafter(0., 1.), energy=1.), row(2, 0., energy=1.)])


def test_report_retains_its_own_validated_records_and_digest():
    supplied = [row(2, 2.), row(1, 1.)]
    report = evaluation_report(supplied)
    retained = json.loads(json.dumps(report))
    supplied[0]["cov_mse"] = 999.
    supplied[1].clear()
    supplied.reverse()
    assert report == retained
    payload = {key: value for key, value in report.items() if key != "result_sha256"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert report["result_sha256"] == actual


def test_reports_built_from_shared_rows_do_not_alias_each_other():
    supplied = [row(1, 1.)]
    first = evaluation_report(supplied)
    second = evaluation_report(supplied)
    first["records"][0]["cov_mse"] = 99.
    assert second["records"][0]["cov_mse"] == 1.
    assert supplied[0]["cov_mse"] == 1.
