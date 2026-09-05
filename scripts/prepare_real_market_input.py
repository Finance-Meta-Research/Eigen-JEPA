from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Iterable


def _sha256_bytes(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_iso_date(value: str, row_number: int) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"row {row_number}: date must be ISO YYYY-MM-DD, got {value!r}"
        ) from exc
    if parsed.isoformat() != value:
        raise ValueError(
            f"row {row_number}: date must use canonical ISO YYYY-MM-DD, got {value!r}"
        )
    return parsed


def _finite_float(value: str, column: str, row_number: int) -> float:
    if value is None or value.strip() == "":
        raise ValueError(f"row {row_number}: missing value in return column {column!r}")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"row {row_number}: non-numeric value in return column {column!r}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"row {row_number}: non-finite value in return column {column!r}: {value!r}"
        )
    return number


def _source_receipt(
    *,
    raw_source_path: Path | None,
    provider_identity: str | None,
    provider_snapshot_or_retrieval_id: str | None,
) -> dict | None:
    supplied = (
        raw_source_path is not None,
        provider_identity not in (None, ""),
        provider_snapshot_or_retrieval_id not in (None, ""),
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "raw_source_path, provider_identity, and provider_snapshot_or_retrieval_id "
            "must be supplied together"
        )
    if not any(supplied):
        return None
    assert raw_source_path is not None
    if not raw_source_path.is_file():
        raise FileNotFoundError(raw_source_path)
    return {
        "provider_identity": str(provider_identity).strip(),
        "provider_snapshot_or_retrieval_id": str(provider_snapshot_or_retrieval_id).strip(),
        "raw_source_path": raw_source_path.as_posix(),
        "raw_source_sha256": _sha256_bytes(raw_source_path),
        "raw_source_bytes": raw_source_path.stat().st_size,
        "lineage_attestation": (
            "The normalized CSV supplied to this preflight was produced from this exact raw-source "
            "snapshot before model/test outcome access. This receipt binds identities and bytes; it "
            "does not independently certify provider adjustment semantics."
        ),
    }


def inspect_csv(
    path: Path,
    *,
    date_col: str,
    return_cols: Iterable[str],
    expected_assets: int,
    context_len: int,
    horizon: int,
    num_train: int,
    num_val: int,
    num_test: int,
    raw_source_path: Path | None = None,
    provider_identity: str | None = None,
    provider_snapshot_or_retrieval_id: str | None = None,
) -> dict:
    return_cols = list(return_cols)
    if expected_assets <= 1:
        raise ValueError("expected_assets must be at least 2")
    if len(return_cols) != expected_assets:
        raise ValueError(
            f"expected exactly {expected_assets} return columns, got {len(return_cols)}"
        )
    if len(set(return_cols)) != len(return_cols):
        raise ValueError("return_cols must be unique")
    if date_col in return_cols:
        raise ValueError("date_col must not also be a return column")
    if min(context_len, horizon, num_train, num_val, num_test) <= 0:
        raise ValueError("context_len, horizon, and split sizes must all be positive")

    if not path.is_file():
        raise FileNotFoundError(path)

    source = _source_receipt(
        raw_source_path=raw_source_path,
        provider_identity=provider_identity,
        provider_snapshot_or_retrieval_id=provider_snapshot_or_retrieval_id,
    )

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("CSV header contains duplicate column names")
        required = [date_col, *return_cols]
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        row_count = 0
        first_date: date | None = None
        last_date: date | None = None
        previous_date: date | None = None
        seen_dates: set[date] = set()

        for row_number, row in enumerate(reader, start=2):
            row_count += 1
            raw_date = row.get(date_col)
            if raw_date is None or raw_date.strip() == "":
                raise ValueError(f"row {row_number}: missing date in column {date_col!r}")
            parsed_date = _parse_iso_date(raw_date.strip(), row_number)
            if parsed_date in seen_dates:
                raise ValueError(f"row {row_number}: duplicate date {parsed_date.isoformat()}")
            if previous_date is not None and parsed_date <= previous_date:
                raise ValueError(
                    f"row {row_number}: dates must be strictly increasing; "
                    f"{parsed_date.isoformat()} follows {previous_date.isoformat()}"
                )
            seen_dates.add(parsed_date)
            previous_date = parsed_date
            first_date = parsed_date if first_date is None else first_date
            last_date = parsed_date

            for column in return_cols:
                _finite_float(row.get(column), column, row_number)

    if row_count == 0:
        raise ValueError("CSV contains no data rows")

    usable_windows = row_count - context_len - horizon + 1
    requested_windows = num_train + num_val + num_test
    if usable_windows < requested_windows:
        raise ValueError(
            "insufficient chronological windows: "
            f"rows={row_count}, context_len={context_len}, horizon={horizon}, "
            f"usable={usable_windows}, requested={requested_windows}"
        )

    receipt = {
        "schema_version": 1,
        "status": "PREPARED_INPUT_ONLY_NOT_AUTHORIZED",
        "dataset": {
            "path": path.as_posix(),
            "sha256": _sha256_bytes(path),
            "bytes": path.stat().st_size,
            "rows": row_count,
            "date_col": date_col,
            "date_start": first_date.isoformat() if first_date else None,
            "date_end": last_date.isoformat() if last_date else None,
            "return_cols": return_cols,
            "num_assets": expected_assets,
        },
        "windowing": {
            "context_len": context_len,
            "horizon": horizon,
            "usable_windows": usable_windows,
            "num_train": num_train,
            "num_val": num_val,
            "num_test": num_test,
            "unused_tail_windows": usable_windows - requested_windows,
            "split_policy": "chronological contiguous windows; no shuffling across train/validation/test",
        },
        "required_benchmark_args": [
            "--data_source", "csv",
            "--csv_path", path.as_posix(),
            "--date_col", date_col,
            "--return_cols", *return_cols,
            "--num_assets", str(expected_assets),
            "--total_steps", str(row_count),
            "--context_len", str(context_len),
            "--horizon", str(horizon),
            "--num_train", str(num_train),
            "--num_val", str(num_val),
            "--num_test", str(num_test),
        ],
        "integrity_notes": [
            "The receipt validates input shape, finite returns, canonical unique increasing dates, split capacity, and exact normalized-file bytes.",
            "When source_provenance is present, it also binds the exact raw-source bytes and provider/snapshot identity used to prepare the normalized CSV.",
            "It does not authorize outcome access, establish a claim threshold, or independently certify provider adjustment semantics or absence of economic/data-vendor bias.",
            "Eigen-JEPA's CSV loader derives regime labels from the supplied returns; those labels are not independently observed market regimes.",
        ],
    }
    if source is not None:
        receipt["source_provenance"] = source
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed preflight for immutable real-market Eigen-JEPA CSV inputs."
    )
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--date-col", required=True)
    parser.add_argument("--return-cols", required=True, nargs="+")
    parser.add_argument("--expected-assets", required=True, type=int)
    parser.add_argument("--context-len", required=True, type=int)
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--num-train", required=True, type=int)
    parser.add_argument("--num-val", required=True, type=int)
    parser.add_argument("--num-test", required=True, type=int)
    parser.add_argument("--raw-source", type=Path)
    parser.add_argument("--provider-identity")
    parser.add_argument("--provider-snapshot-or-retrieval-id")
    parser.add_argument("--receipt-out", required=True, type=Path)
    args = parser.parse_args()

    receipt = inspect_csv(
        args.csv,
        date_col=args.date_col,
        return_cols=args.return_cols,
        expected_assets=args.expected_assets,
        context_len=args.context_len,
        horizon=args.horizon,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        raw_source_path=args.raw_source,
        provider_identity=args.provider_identity,
        provider_snapshot_or_retrieval_id=args.provider_snapshot_or_retrieval_id,
    )
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt_out.exists():
        raise FileExistsError(
            f"refusing to overwrite existing receipt: {args.receipt_out}"
        )
    args.receipt_out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("REAL_MARKET_INPUT_PREPARED_NOT_AUTHORIZED")
    print(f"DATASET_SHA256={receipt['dataset']['sha256']}")
    if "source_provenance" in receipt:
        print(f"RAW_SOURCE_SHA256={receipt['source_provenance']['raw_source_sha256']}")
    print(f"ROWS={receipt['dataset']['rows']}")
    print(f"USABLE_WINDOWS={receipt['windowing']['usable_windows']}")


if __name__ == "__main__":
    main()
