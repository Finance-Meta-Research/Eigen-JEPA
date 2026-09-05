from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SEEDS = [7, 19, 31, 43, 59]
VARIANTS = ["full", "no_memory", "no_gate", "no_regime"]
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_final_rigor_v2.py"


def _write_complete_package(tmp_path: Path, *, seeds=SEEDS) -> Path:
    out = tmp_path / "final_rigor_v2"
    aggregate = {}
    for vi, variant in enumerate(VARIANTS):
        aggregate[variant] = {
            "eig_nmse": {"mean": 1.0 + vi * 0.1, "std": 0.01},
            "proj_mse": {"mean": 0.2 + vi * 0.01, "std": 0.02},
        }
    summary = {
        "market_style": "equity",
        "num_seeds": len(seeds),
        "seed_list": list(seeds),
        "variants": VARIANTS,
        "aggregate": aggregate,
        "benchmark": {},
        "representative_run": str(out / "seed_7" / "full"),
    }
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps(summary), encoding="utf-8")

    for seed in SEEDS:
        for variant in VARIANTS:
            run_dir = out / f"seed_{seed}" / variant
            run_dir.mkdir(parents=True)
            payload = {
                "test": {"eig_nmse": 1.0, "proj_mse": 0.2},
                "history": [{"epoch": epoch} for epoch in range(1, 9)],
            }
            (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return out / "metrics.json"


def _run(metrics: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--metrics", str(metrics)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_final_rigor_v2_gate_accepts_exact_complete_package(tmp_path: Path) -> None:
    metrics = _write_complete_package(tmp_path)
    proc = _run(metrics)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "FINAL_RIGOR_V2_PASS:" in proc.stdout


def test_final_rigor_v2_gate_rejects_seed_drift(tmp_path: Path) -> None:
    metrics = _write_complete_package(tmp_path, seeds=[7, 19, 31, 43, 61])
    proc = _run(metrics)
    assert proc.returncode != 0
    assert "FINAL_RIGOR_V2_FAIL: seed_list drift" in proc.stdout + proc.stderr


def test_final_rigor_v2_gate_rejects_missing_preregistered_run(tmp_path: Path) -> None:
    metrics = _write_complete_package(tmp_path)
    missing = metrics.parent / "seed_59" / "no_regime" / "metrics.json"
    missing.unlink()
    proc = _run(metrics)
    assert proc.returncode != 0
    assert "FINAL_RIGOR_V2_FAIL: missing preregistered run metrics" in proc.stdout + proc.stderr
