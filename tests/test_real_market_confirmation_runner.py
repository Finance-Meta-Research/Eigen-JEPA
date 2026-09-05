import copy
import json
from pathlib import Path

import pytest

from scripts.run_real_market_confirmation import (
    AUTHORIZED_STATUS,
    EXPECTED_FOLDS,
    EXPECTED_SEEDS,
    PRIMARY_VARIANTS,
    ConfirmationRunnerError,
    _run_evidence_path,
    assemble_primary_analysis_input,
    assert_execution_authorized,
    read_csv_dates,
)


def _candidate_protocol():
    return json.loads(
        Path("protocols/real_market_confirmation_v1_candidate_20260906.json").read_text(
            encoding="utf-8"
        )
    )


def _write_evidence(root, *, mutate=None):
    offset = 0
    for fold_no, fold in enumerate(EXPECTED_FOLDS):
        indices = list(range(offset, offset + 4))
        offset += 4
        for variant_no, variant in enumerate(PRIMARY_VARIANTS):
            for seed_no, seed in enumerate(EXPECTED_SEEDS):
                rows = [
                    {
                        "window_index": idx,
                        "eig_sq_error": 0.1 + 0.01 * fold_no + 0.001 * seed_no + 0.002 * variant_no,
                        "target_energy": 1.0 + 0.1 * j,
                    }
                    for j, idx in enumerate(indices)
                ]
                payload = {
                    "schema_version": 1,
                    "status": "RETAINED_TEST_WINDOW_EVIDENCE",
                    "fold_id": fold,
                    "seed": seed,
                    "variant": variant,
                    "window_rows": rows,
                }
                if mutate is not None:
                    mutate(payload)
                path = _run_evidence_path(root, fold, seed, variant)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")


def test_current_candidate_protocol_cannot_authorize_outcome_access():
    protocol = _candidate_protocol()
    assert protocol["status"] != AUTHORIZED_STATUS
    with pytest.raises(ConfirmationRunnerError, match="protocol status must be"):
        assert_execution_authorized(protocol)


def test_authorization_still_fails_when_status_is_flipped_without_review_and_hashes():
    protocol = copy.deepcopy(_candidate_protocol())
    protocol["status"] = AUTHORIZED_STATUS
    protocol["data_freeze_requirements"]["execution_authorized"] = True
    with pytest.raises(ConfirmationRunnerError, match="unresolved data freeze field"):
        assert_execution_authorized(protocol)


def test_analysis_input_assembly_requires_exact_seed_and_window_alignment(tmp_path):
    _write_evidence(tmp_path)
    payload = assemble_primary_analysis_input(tmp_path)
    assert payload["schema_version"] == 1
    assert tuple(payload["folds"]) == EXPECTED_FOLDS
    assert tuple(payload["folds"]["F1"]["full"]) == tuple(str(s) for s in EXPECTED_SEEDS)
    assert payload["folds"]["F3"]["window_indices"] == [8, 9, 10, 11]
    assert len(payload["folds"]["F2"]["no_memory"]["31"]) == 4


def test_analysis_input_assembly_rejects_target_energy_drift(tmp_path):
    _write_evidence(tmp_path)
    path = _run_evidence_path(tmp_path, "F2", 31, "no_memory")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["window_rows"][2]["target_energy"] += 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfirmationRunnerError, match="target energy drift"):
        assemble_primary_analysis_input(tmp_path)


def test_analysis_input_assembly_rejects_seed_window_drift(tmp_path):
    _write_evidence(tmp_path)
    path = _run_evidence_path(tmp_path, "F3", 59, "full")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["window_rows"][2]["window_index"] += 100
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ConfirmationRunnerError, match="window indices must be strictly increasing|test window identity drift"):
        assemble_primary_analysis_input(tmp_path)


def test_csv_date_reader_rejects_future_order_and_duplicates(tmp_path):
    good = tmp_path / "good.csv"
    good.write_text(
        "date,A,B\n2020-01-02,0.1,0.2\n2020-01-03,0.2,0.3\n",
        encoding="utf-8",
    )
    assert read_csv_dates(good, date_col="date") == ["2020-01-02", "2020-01-03"]

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "date,A,B\n2020-01-02,0.1,0.2\n2020-01-02,0.2,0.3\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfirmationRunnerError, match="strictly increasing"):
        read_csv_dates(duplicate, date_col="date")
