"""Generated-fixture tests for the additive saved-evidence reader; stdlib only."""
import copy
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts import verify_retained_v2_package as reader

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    runs = {}
    for i, seed in enumerate(reader.SEEDS):
        for j, variant in enumerate(reader.VARIANTS):
            runs[seed, variant] = {
                "test": {metric: (i + j + 1) / 20 for metric in reader.METRICS},
                "history": [{"epoch": epoch, "train_loss": 0.5} for epoch in range(1, 9)],
                "benchmark": {base: {metric: (i + 1) / 10 for metric in reader.BASELINE_METRICS}
                              for base in reader.BASELINES},
            }
    def summary(values):
        return {"mean": statistics.mean(values), "std": statistics.pstdev(values)}
    data = {"market_style": "equity", "num_seeds": 5, "seed_list": list(reader.SEEDS),
            "variants": list(reader.VARIANTS), "representative_run": "results/final_rigor_v2/seed_7/full",
            "aggregate": {v: {m: summary([runs[s, v]["test"][m] for s in reader.SEEDS])
                              for m in reader.METRICS} for v in reader.VARIANTS},
            "benchmark": {b: {m: summary([runs[s, v]["benchmark"][b][m]
                                         for s in reader.SEEDS for v in reader.VARIANTS])
                              for m in reader.BASELINE_METRICS} for b in reader.BASELINES}}
    return data, runs


class RetainedPackageReaderTests(unittest.TestCase):
    def setUp(self):
        self.data, self.runs = fixture()

    def test_complete_fixture_recomputes_all_100_summaries(self):
        before = copy.deepcopy((self.data, self.runs))
        checks = reader.validate_payload(self.data, self.runs)
        self.assertEqual(len(checks), 100)
        self.assertEqual(sum(c["values_aggregated"] == 5 for c in checks), 72)
        self.assertEqual(sum(c["values_aggregated"] == 20 for c in checks), 28)
        self.assertTrue(all(c["independent_seeds"] == 5 for c in checks))
        self.assertEqual(before, (self.data, self.runs))

    def test_each_mean_and_sd_is_recomputed(self):
        for group in ("aggregate", "benchmark"):
            for name, metrics in self.data[group].items():
                for metric in metrics:
                    for field in ("mean", "std"):
                        with self.subTest(group=group, name=name, metric=metric, field=field):
                            changed = copy.deepcopy(self.data)
                            changed[group][name][metric][field] += 0.01
                            with self.assertRaisesRegex(reader.EvidenceError, "recomputed"):
                                reader.validate_payload(changed, self.runs)

    def test_numeric_and_finite_types_fail_closed(self):
        for value in (True, False, None, "0.2", [], {}, float("nan"), float("inf"), -0.1, 10**1000):
            with self.subTest(value=str(value)[:20]):
                runs = copy.deepcopy(self.runs)
                runs[7, "full"]["test"]["eig_nmse"] = value
                with self.assertRaises(reader.EvidenceError):
                    reader.validate_payload(self.data, runs)
        for value in (True, "0.2", float("nan")):
            changed = copy.deepcopy(self.data)
            changed["aggregate"]["full"]["eig_nmse"]["mean"] = value
            with self.assertRaises(reader.EvidenceError):
                reader.validate_payload(changed, self.runs)

    def test_missing_extra_and_duplicate_surface_rejected(self):
        mutations = [lambda d, r: d.update(num_seeds=5.0),
                     lambda d, r: d.update(seed_list=[7, 19, 31, 43, 43]),
                     lambda d, r: d["aggregate"]["full"].pop("eig_nmse"),
                     lambda d, r: d["aggregate"]["full"].update(extra={"mean": 0, "std": 0}),
                     lambda d, r: r.pop((7, "full")),
                     lambda d, r: r[7, "full"]["test"].pop("eig_nmse"),
                     lambda d, r: r[7, "full"]["history"].pop(),
                     lambda d, r: r[7, "full"]["history"][0].update(epoch=True),
                     lambda d, r: r[7, "full"]["history"][1].update(epoch=1),
                     lambda d, r: r[7, "full"]["test"].update(tail_f1=1.1)]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                data, runs = copy.deepcopy((self.data, self.runs))
                mutate(data, runs)
                with self.assertRaises(reader.EvidenceError):
                    reader.validate_payload(data, runs)

    def test_baseline_variants_cannot_be_claimed_as_extra_seeds(self):
        self.runs[7, "no_gate"]["benchmark"]["persistence"]["eig_nmse"] += 0.1
        with self.assertRaisesRegex(reader.EvidenceError, "variant-dependent baseline"):
            reader.validate_payload(self.data, self.runs)

    def test_json_rejects_duplicates_nonfinite_and_nonobject(self):
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'[]', b'{', b'\xff'):
            with self.subTest(raw=raw), self.assertRaises(reader.EvidenceError):
                reader.read_json(raw)

    def test_auxiliary_nan_exception_requires_exact_explicit_paths(self):
        raw = b'{"walkforward":{"late":{"rare_cov_mse":NaN}}}'
        allowed = frozenset({"walkforward/late/rare_cov_mse"})
        self.assertIn("walkforward", reader.read_json(raw, retained_nan_paths=allowed))
        for bad in (raw.replace(b'NaN', b'Infinity'), raw.replace(b'NaN', b'0.1'),
                    raw.replace(b'rare_cov_mse', b'eig_nmse')):
            with self.assertRaises(reader.EvidenceError):
                reader.read_json(bad, retained_nan_paths=allowed)

    def test_ledger_digest_cannot_be_rebound_to_generated_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / "ledger.json"
            ledger.write_text('{}')
            with self.assertRaisesRegex(reader.EvidenceError, "ledger SHA-256 drift"):
                reader.verify_package(Path(tmp), ledger)

    def test_generated_package_binds_raw_bytes_before_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for (seed, variant), run in self.runs.items():
                if seed == 43:
                    run["walkforward"] = {
                        window: {metric: float("nan") for metric in
                                 ("rare_cov_mse", "rare_drift_mse", "rare_eig_nmse")}
                        for window in ("mid", "late")}
                path = root / f"seed_{seed}" / variant / "metrics.json"
                path.parent.mkdir(parents=True)
                raw = json.dumps(run).encode()
                path.write_bytes(raw)
                rows.append({"seed": seed, "variant": variant,
                             "source_sha256": hashlib.sha256(raw).hexdigest()})
            ledger = root / "ledger.json"
            ledger.write_text(json.dumps({"rows": rows, "source_protocol": "GENERATED_FIXTURE",
                                         "execution_head": "fixture", "workflow_run_id": 0,
                                         "artifact_id": 0, "artifact_digest": "fixture"}))
            (root / "metrics.json").write_text(json.dumps(self.data))
            # The trust root is replaced only inside this generated-fixture test.
            with mock.patch.object(reader, "LEDGER_SHA256", hashlib.sha256(ledger.read_bytes()).hexdigest()):
                receipt = reader.verify_package(root, ledger)
                self.assertEqual(len(receipt["aggregate_checks"]), 100)
                self.assertEqual(len(receipt["nonfinite_auxiliary_diagnostics"]), 24)
                path = root / "seed_7/full/metrics.json"
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaisesRegex(reader.EvidenceError, "raw SHA-256 drift"):
                    reader.verify_package(root, ledger)
                path.write_bytes(original)
                self.data["aggregate"]["full"]["eig_nmse"]["mean"] = 0.0
                (root / "metrics.json").write_text(json.dumps(self.data))
                with self.assertRaisesRegex(reader.EvidenceError, "recomputed"):
                    reader.verify_package(root, ledger)

    def test_old_gate_accepts_arithmetic_drift_new_reader_rejects(self):
        # Witness uses generated data, never changes the frozen archive.
        self.data["aggregate"]["full"]["eig_nmse"]["mean"] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for (seed, variant), run in self.runs.items():
                path = root / f"seed_{seed}" / variant / "metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(run))
            aggregate = root / "metrics.json"
            aggregate.write_text(json.dumps(self.data))
            proc = subprocess.run([sys.executable, str(ROOT / "scripts/check_final_rigor_v2.py"),
                                   "--metrics", str(aggregate)], capture_output=True, text=True, timeout=10)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("FINAL_RIGOR_V2_PASS", proc.stdout)
            with self.assertRaisesRegex(reader.EvidenceError, "recomputed"):
                reader.validate_payload(self.data, self.runs)


if __name__ == "__main__":
    unittest.main()
