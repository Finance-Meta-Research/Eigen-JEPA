import csv
import hashlib
from pathlib import Path

import pytest

from scripts.prepare_real_market_input import inspect_csv


RETURN_COLS = ["A", "B", "C"]


def _write_csv(path: Path, rows, header=("date", "A", "B", "C")) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _rows(n: int):
    rows = []
    for i in range(n):
        day = i + 1
        rows.append(
            {
                "date": f"2026-01-{day:02d}",
                "A": f"{0.001 * (i + 1):.6f}",
                "B": f"{-0.002 * (i + 1):.6f}",
                "C": f"{0.0005 * (i + 1):.6f}",
            }
        )
    return rows


def _inspect(path: Path, **extra):
    return inspect_csv(
        path,
        date_col="date",
        return_cols=RETURN_COLS,
        expected_assets=3,
        context_len=4,
        horizon=2,
        num_train=5,
        num_val=2,
        num_test=2,
        **extra,
    )


def test_valid_csv_emits_exact_row_bound_benchmark_args(tmp_path):
    path = tmp_path / "market.csv"
    _write_csv(path, _rows(14))
    receipt = _inspect(path)
    assert receipt["status"] == "PREPARED_INPUT_ONLY_NOT_AUTHORIZED"
    assert receipt["dataset"]["rows"] == 14
    assert receipt["dataset"]["date_start"] == "2026-01-01"
    assert receipt["dataset"]["date_end"] == "2026-01-14"
    assert receipt["dataset"]["num_assets"] == 3
    assert receipt["windowing"]["usable_windows"] == 9
    assert receipt["windowing"]["unused_tail_windows"] == 0
    args = receipt["required_benchmark_args"]
    assert args[args.index("--total_steps") + 1] == "14"
    assert args[args.index("--num_assets") + 1] == "3"
    assert args[args.index("--return_cols") + 1 : args.index("--num_assets")] == RETURN_COLS


def test_source_provenance_binds_raw_bytes_and_provider_identity(tmp_path):
    path = tmp_path / "market.csv"
    raw = tmp_path / "provider-snapshot.bin"
    _write_csv(path, _rows(14))
    raw.write_bytes(b"provider-snapshot-v1")
    receipt = _inspect(
        path,
        raw_source_path=raw,
        provider_identity="provider-a",
        provider_snapshot_or_retrieval_id="snapshot-1",
    )
    source = receipt["source_provenance"]
    assert source["provider_identity"] == "provider-a"
    assert source["provider_snapshot_or_retrieval_id"] == "snapshot-1"
    assert source["raw_source_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert source["raw_source_bytes"] == len(raw.read_bytes())


def test_source_provenance_is_all_or_nothing(tmp_path):
    path = tmp_path / "market.csv"
    raw = tmp_path / "provider-snapshot.bin"
    _write_csv(path, _rows(14))
    raw.write_bytes(b"provider-snapshot-v1")
    with pytest.raises(ValueError, match="must be supplied together"):
        _inspect(path, raw_source_path=raw, provider_identity="provider-a")


def test_hash_changes_when_dataset_bytes_change(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    rows = _rows(14)
    _write_csv(a, rows)
    rows[7]["B"] = "0.123456"
    _write_csv(b, rows)
    assert _inspect(a)["dataset"]["sha256"] != _inspect(b)["dataset"]["sha256"]


def test_rejects_non_monotone_dates(tmp_path):
    path = tmp_path / "market.csv"
    rows = _rows(14)
    rows[5]["date"], rows[6]["date"] = rows[6]["date"], rows[5]["date"]
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="strictly increasing"):
        _inspect(path)


def test_rejects_duplicate_dates(tmp_path):
    path = tmp_path / "market.csv"
    rows = _rows(14)
    rows[6]["date"] = rows[5]["date"]
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="duplicate date"):
        _inspect(path)


def test_rejects_missing_or_nonfinite_returns(tmp_path):
    missing = tmp_path / "missing.csv"
    rows = _rows(14)
    rows[3]["A"] = ""
    _write_csv(missing, rows)
    with pytest.raises(ValueError, match="missing value"):
        _inspect(missing)

    nonfinite = tmp_path / "nonfinite.csv"
    rows = _rows(14)
    rows[3]["A"] = "nan"
    _write_csv(nonfinite, rows)
    with pytest.raises(ValueError, match="non-finite"):
        _inspect(nonfinite)


def test_rejects_split_request_larger_than_available_windows(tmp_path):
    path = tmp_path / "market.csv"
    _write_csv(path, _rows(10))
    with pytest.raises(ValueError, match="insufficient chronological windows"):
        _inspect(path)


def test_rejects_asset_count_mismatch(tmp_path):
    path = tmp_path / "market.csv"
    _write_csv(path, _rows(14))
    with pytest.raises(ValueError, match="expected exactly 4 return columns"):
        inspect_csv(
            path,
            date_col="date",
            return_cols=RETURN_COLS,
            expected_assets=4,
            context_len=4,
            horizon=2,
            num_train=5,
            num_val=2,
            num_test=2,
        )
