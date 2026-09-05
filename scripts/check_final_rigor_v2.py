#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

EXPECTED_SEEDS = [7, 19, 31, 43, 59]
EXPECTED_VARIANTS = ["full", "no_memory", "no_gate", "no_regime"]
EXPECTED_MARKET_STYLE = "equity"


def fail(message: str) -> None:
    raise SystemExit(f"FINAL_RIGOR_V2_FAIL: {message}")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail closed unless the exact preregistered Eigen-JEPA final-rigor v2 package is complete."
    )
    parser.add_argument("--metrics", default="results/final_rigor_v2/metrics.json")
    args = parser.parse_args()

    metrics_path = Path(args.metrics)
    if not metrics_path.is_file():
        fail(f"missing aggregate metrics: {metrics_path}")

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    if data.get("seed_list") != EXPECTED_SEEDS:
        fail(f"seed_list drift: expected {EXPECTED_SEEDS}, observed {data.get('seed_list')!r}")
    if data.get("num_seeds") != len(EXPECTED_SEEDS):
        fail(f"num_seeds drift: expected {len(EXPECTED_SEEDS)}, observed {data.get('num_seeds')!r}")
    if data.get("variants") != EXPECTED_VARIANTS:
        fail(f"variant/order drift: expected {EXPECTED_VARIANTS}, observed {data.get('variants')!r}")
    if data.get("market_style") != EXPECTED_MARKET_STYLE:
        fail(f"market_style drift: expected {EXPECTED_MARKET_STYLE!r}, observed {data.get('market_style')!r}")

    aggregate = data.get("aggregate")
    if not isinstance(aggregate, dict) or list(aggregate) != EXPECTED_VARIANTS:
        fail(f"aggregate variant drift: expected {EXPECTED_VARIANTS}, observed {list(aggregate) if isinstance(aggregate, dict) else aggregate!r}")

    aggregate_summaries = 0
    for variant in EXPECTED_VARIANTS:
        metrics = aggregate.get(variant)
        if not isinstance(metrics, dict) or not metrics:
            fail(f"missing aggregate metrics for {variant}")
        for metric_name, summary in metrics.items():
            if not isinstance(summary, dict):
                fail(f"{variant}/{metric_name} summary is not a mapping")
            mean = summary.get("mean")
            std = summary.get("std")
            if not _finite_number(mean):
                fail(f"{variant}/{metric_name} has invalid mean {mean!r}")
            if not _finite_number(std) or float(std) < 0:
                fail(f"{variant}/{metric_name} has invalid std {std!r}")
            aggregate_summaries += 1

    root = metrics_path.parent
    missing_runs: list[str] = []
    malformed_runs: list[str] = []
    for seed in EXPECTED_SEEDS:
        for variant in EXPECTED_VARIANTS:
            run_metrics = root / f"seed_{seed}" / variant / "metrics.json"
            if not run_metrics.is_file():
                missing_runs.append(str(run_metrics))
                continue
            try:
                run_data = json.loads(run_metrics.read_text(encoding="utf-8"))
            except Exception as exc:  # fail closed on any unreadable retained run
                malformed_runs.append(f"{run_metrics}: {exc}")
                continue
            test = run_data.get("test")
            history = run_data.get("history")
            if not isinstance(test, dict) or not test:
                malformed_runs.append(f"{run_metrics}: missing non-empty test mapping")
            if not isinstance(history, list) or len(history) != 8:
                malformed_runs.append(f"{run_metrics}: expected exactly 8 retained training epochs")

    if missing_runs:
        fail("missing preregistered run metrics: " + "; ".join(missing_runs))
    if malformed_runs:
        fail("malformed preregistered run evidence: " + "; ".join(malformed_runs))

    gate = Path(__file__).with_name("check_rigor_gate.py")
    proc = subprocess.run(
        [sys.executable, str(gate), "--metrics", str(metrics_path), "--min-seeds", "5"],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = (proc.stdout + proc.stderr).strip()
        fail(f"generic rigor gate rejected v2 package: {detail}")
    if "RIGOR_GATE_PASS:" not in proc.stdout:
        fail("generic rigor gate did not emit literal RIGOR_GATE_PASS")

    print(
        "FINAL_RIGOR_V2_PASS: exact seeds, variants, 20 retained run-metric files, "
        f"and {aggregate_summaries} finite aggregate summaries verified."
    )


if __name__ == "__main__":
    main()
