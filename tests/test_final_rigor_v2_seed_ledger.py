import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "final_rigor_v2" / "SEED_LEVEL_METRICS_20260906.json"
VERIFY = ROOT / "scripts" / "verify_final_rigor_v2_seed_ledger.py"


def run_verify(path: Path):
    return subprocess.run(
        [sys.executable, str(VERIFY), str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_retained_seed_ledger_passes():
    result = run_verify(LEDGER)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FINAL_RIGOR_V2_SEED_LEDGER_PASS:" in result.stdout


def test_seed_removal_fails_closed(tmp_path):
    data = json.loads(LEDGER.read_text())
    data["rows"] = data["rows"][:-1]
    mutated = tmp_path / "missing-row.json"
    mutated.write_text(json.dumps(data))
    result = run_verify(mutated)
    assert result.returncode != 0
    assert "expected exactly 20 seed-by-variant rows" in result.stdout + result.stderr


def test_metric_tamper_fails_closed(tmp_path):
    data = json.loads(LEDGER.read_text())
    target = next(
        row
        for row in data["rows"]
        if row["seed"] == 59 and row["variant"] == "full"
    )
    target["eig_nmse"] += 0.01
    mutated = tmp_path / "metric-tamper.json"
    mutated.write_text(json.dumps(data))
    result = run_verify(mutated)
    assert result.returncode != 0
    assert "aggregate mean drift: full/eig_nmse" in result.stdout + result.stderr


def test_post_outcome_threshold_fails_closed(tmp_path):
    data = json.loads(LEDGER.read_text())
    data["interpretation_boundary"]["practical_effect_threshold_added"] = True
    mutated = tmp_path / "threshold-tamper.json"
    mutated.write_text(json.dumps(data))
    result = run_verify(mutated)
    assert result.returncode != 0
    assert "post-outcome practical-effect threshold introduced" in result.stdout + result.stderr
