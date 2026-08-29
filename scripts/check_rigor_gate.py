#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"RIGOR_GATE_FAIL: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed when paper-facing Eigen-JEPA rigor evidence is under-seeded or malformed.")
    parser.add_argument("--metrics", default="results/final_rigor/metrics.json")
    parser.add_argument("--min-seeds", type=int, default=3)
    args = parser.parse_args()

    path = Path(args.metrics)
    if not path.is_file():
        fail(f"missing metrics file: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    seeds = data.get("seed_list")
    num_seeds = data.get("num_seeds")
    aggregate = data.get("aggregate")

    if not isinstance(seeds, list) or not seeds:
        fail("seed_list must be a non-empty list")
    if len(set(seeds)) != len(seeds):
        fail("seed_list contains duplicates")
    if num_seeds != len(seeds):
        fail(f"num_seeds={num_seeds!r} does not match len(seed_list)={len(seeds)}")
    if len(seeds) < args.min_seeds:
        fail(f"only {len(seeds)} seed(s) present; at least {args.min_seeds} are required for paper-facing rigor claims")
    if not isinstance(aggregate, dict) or not aggregate:
        fail("aggregate results are missing")

    checked = 0
    for variant, metrics in aggregate.items():
        if not isinstance(metrics, dict) or not metrics:
            fail(f"variant {variant!r} has no aggregate metrics")
        for metric_name, summary in metrics.items():
            if not isinstance(summary, dict):
                fail(f"{variant}/{metric_name} summary is not a mapping")
            mean = summary.get("mean")
            std = summary.get("std")
            if not isinstance(mean, (int, float)) or not math.isfinite(float(mean)):
                fail(f"{variant}/{metric_name} has invalid mean {mean!r}")
            if not isinstance(std, (int, float)) or not math.isfinite(float(std)) or float(std) < 0:
                fail(f"{variant}/{metric_name} has invalid std {std!r}")
            checked += 1

    print(
        f"RIGOR_GATE_PASS: {len(seeds)} seeds {seeds}; "
        f"validated {checked} aggregate metric summaries across {len(aggregate)} variants."
    )


if __name__ == "__main__":
    main()
