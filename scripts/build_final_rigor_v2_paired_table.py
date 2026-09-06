#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SOURCE = Path("results/final_rigor_v2/SEED_LEVEL_METRICS_20260906.json")
TEX_OUT = Path("paper/final_rigor_v2_paired_table.tex")

COMPARISONS = (
    ("no_memory", "No memory"),
    ("no_gate", "No gate"),
    ("no_regime", "No regime"),
)
METRICS = (
    ("eig_nmse", "Eig NMSE $\\downarrow$", "lower"),
    ("drift_mse", "Drift MSE $\\downarrow$", "lower"),
    ("gate_cal", "Gate Cal $\\downarrow$", "lower"),
    ("tail_f1", "Tail F1 $\\uparrow$", "higher"),
    ("regime_acc", "Regime Acc $\\uparrow$", "higher"),
)


def load_source(path: Path = SOURCE) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("seed-level evidence must be a JSON object")
    boundary = payload.get("interpretation_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("missing interpretation_boundary")
    if boundary.get("all_seed_rows_retained") is not True:
        raise ValueError("all seed rows must remain retained")
    if boundary.get("post_outcome_significance_test_added") is not False:
        raise ValueError("paired table cannot add a post-outcome significance test")
    if boundary.get("practical_effect_threshold_added") is not False:
        raise ValueError("paired table cannot add a post-outcome practical-effect threshold")
    return payload


def build_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    paired = payload.get("paired_descriptive_differences")
    if not isinstance(paired, dict):
        raise ValueError("missing paired_descriptive_differences")
    rows: list[dict[str, Any]] = []
    for comparison, comparison_label in COMPARISONS:
        comparison_data = paired.get(comparison)
        if not isinstance(comparison_data, dict):
            raise ValueError(f"missing paired comparison: {comparison}")
        for metric, metric_label, direction in METRICS:
            metric_data = comparison_data.get(metric)
            if not isinstance(metric_data, dict):
                raise ValueError(f"missing paired metric: {comparison}/{metric}")
            deltas = metric_data.get("paired_full_minus_ablation")
            if not isinstance(deltas, list) or len(deltas) != 5:
                raise ValueError(f"{comparison}/{metric} must retain exactly five paired seed deltas")
            if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in deltas):
                raise ValueError(f"{comparison}/{metric} contains a non-finite paired delta")
            deltas = [float(value) for value in deltas]
            recomputed_mean = math.fsum(deltas) / len(deltas)
            recorded_mean = metric_data.get("mean_full_minus_ablation")
            if not isinstance(recorded_mean, (int, float)) or not math.isclose(
                recomputed_mean, float(recorded_mean), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{comparison}/{metric} recorded paired mean disagrees with retained deltas")
            full_better = 0
            ablation_better = 0
            ties = 0
            for delta in deltas:
                if delta == 0.0:
                    ties += 1
                elif (direction == "lower" and delta < 0.0) or (direction == "higher" and delta > 0.0):
                    full_better += 1
                else:
                    ablation_better += 1
            rows.append({
                "comparison": comparison,
                "comparison_label": comparison_label,
                "metric": metric,
                "metric_label": metric_label,
                "mean_full_minus_ablation": recomputed_mean,
                "full_better_seeds": full_better,
                "ablation_better_seeds": ablation_better,
                "ties": ties,
            })
    return rows


def render_tex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Ablation & Metric & Mean $\\Delta$ & Full better & Ablation better & Ties \\\\",
        "\\midrule",
    ]
    previous = None
    for row in rows:
        comparison = row["comparison"]
        if previous is not None and comparison != previous:
            lines.append("\\addlinespace[2pt]")
        lines.append(
            f"{row['comparison_label']} & {row['metric_label']} & "
            f"{row['mean_full_minus_ablation']:+.6f} & "
            f"{row['full_better_seeds']} & {row['ablation_better_seeds']} & {row['ties']} \\\\")
        previous = comparison
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> None:
    rows = build_rows(load_source())
    TEX_OUT.write_text(render_tex(rows), encoding="utf-8")
    print(f"WROTE {TEX_OUT}")
    print("STATUS=DESCRIPTIVE_PAIRED_EVIDENCE_ONLY")


if __name__ == "__main__":
    main()
