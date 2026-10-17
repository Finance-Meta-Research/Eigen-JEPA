#!/usr/bin/env python3
"""Validate immutable research evidence receipts. Run: python scripts/verify_research_evidence.py evidence.json"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def validate(manifest_path):
    try:
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"Cannot read manifest: {exc}"]
    if not isinstance(m, dict):
        return ["Manifest must be an object"]
    root = Path(manifest_path).resolve().parent
    errors, known, paths = [], set(), set()
    artifacts = m.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("artifacts must be a nonempty list")
        artifacts = []
    for i, a in enumerate(artifacts):
        if not isinstance(a, dict):
            errors.append(f"artifacts[{i}] must be an object")
            continue
        name, digest = a.get("path"), a.get("sha256")
        if not isinstance(name, str) or not name or Path(name).is_absolute():
            errors.append(f"artifacts[{i}] invalid relative path")
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(f"artifacts[{i}] missing or unsafe: {name}")
            continue
        if name in paths:
            errors.append(f"Duplicate artifact: {name}")
        paths.add(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            errors.append(f"artifacts[{i}] invalid sha256")
            continue
        with path.open("rb") as fh:
            actual = hashlib.file_digest(fh, "sha256").hexdigest()
        if actual != digest:
            errors.append(f"Hash mismatch: {name}")
        else:
            known.add(name)
    seen = {}
    splits = m.get("splits", {})
    if not isinstance(splits, dict):
        errors.append("splits must be an object")
    else:
        for label, ids in splits.items():
            if not isinstance(ids, list) or any(not isinstance(x, (str, int)) or isinstance(x, bool) for x in ids):
                errors.append(f"Invalid split: {label}")
                continue
            for sid in ids:
                key = str(sid)
                if key in seen:
                    errors.append(f"Split overlap: {key} ({seen[key]}, {label})")
                seen[key] = label
    claims = m.get("claims", [])
    if not isinstance(claims, list):
        errors.append("claims must be a list")
    else:
        for i, c in enumerate(claims):
            if not isinstance(c, dict) or not isinstance(c.get("text"), str) or not c["text"].strip():
                errors.append(f"claims[{i}] requires text")
                continue
            refs = c.get("artifact_paths")
            if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in known for r in refs):
                errors.append(f"claims[{i}] lacks valid hashed evidence")
    if m.get("phase") == "confirmatory" and not isinstance(m.get("frozen_commit"), str):
        errors.append("Confirmatory evidence requires frozen_commit")
    return errors

def main():
    parser = argparse.ArgumentParser(description="Verify hashes, leakage-free split IDs, evidence-backed claims, frozen commit provenance")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = validate(args.manifest)
    for error in errors:
        print("ERROR:", error, file=sys.stderr)
    if errors:
        return 1
    print("Evidence manifest verified:", args.manifest)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
