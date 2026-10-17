"""Regression tests for the evidence receipt validator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "verify_research_evidence.py"
spec = importlib.util.spec_from_file_location("verify_research_evidence", MODULE)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)

class EvidenceReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "result.txt").write_text("data", encoding="utf-8")
        self.manifest = {
            "artifacts": [{"path": "result.txt", "sha256": hashlib.sha256(b"data").hexdigest()}],
            "splits": {"train": ["a"], "test": ["b"]},
            "claims": [{"text": "A retained result", "artifact_paths": ["result.txt"]}],
            "phase": "development",
        }

    def check(self):
        path = self.root / "evidence.json"
        path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return validator.validate(path)

    def test_valid_receipt(self):
        self.assertEqual(self.check(), [])

    def test_tampered_evidence(self):
        (self.root / "result.txt").write_text("tampered", encoding="utf-8")
        self.assertTrue(any("Hash mismatch" in x for x in self.check()))

    def test_split_leakage(self):
        self.manifest["splits"]["test"].append("a")
        self.assertTrue(any("Split overlap" in x for x in self.check()))

    def test_unverified_claim(self):
        self.manifest["claims"][0]["artifact_paths"] = ["nonexistent.csv"]
        self.assertTrue(any("claims[0]" in x for x in self.check()))

    def test_missing_freeze(self):
        self.manifest["phase"] = "confirmatory"
        self.assertTrue(any("frozen_commit" in x for x in self.check()))

    def test_path_escape(self):
        self.manifest["artifacts"][0]["path"] = "../result.txt"
        self.assertTrue(any("unsafe" in x for x in self.check()))

if __name__ == "__main__":
    unittest.main()
