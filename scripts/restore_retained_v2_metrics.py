#!/usr/bin/env python3
"""Restore existing frozen metric files from the digest-bound retained artifact."""
import argparse
import hashlib
from pathlib import Path
import zipfile

EXPECTED_DIGEST = "5de19317b079570db8a97f9cbd5b01da5c1c06878325e3f42c9c3a78db63f6b4"
METRIC_PATHS = ["metrics.json"] + [
    f"seed_{seed}/{variant}/metrics.json"
    for seed in [7, 19, 31, 43, 59]
    for variant in ["full", "no_memory", "no_gate", "no_regime"]
]


def restore(archive, destination):
    archive, destination = Path(archive), Path(destination)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != EXPECTED_DIGEST:
        raise ValueError("retained artifact SHA256 mismatch; no files restored")
    pending = []
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        for name in METRIC_PATHS:
            if names.count(name) != 1:
                raise ValueError(f"expected exactly one retained member: {name}")
            if bundle.getinfo(name).file_size > 10_000_000:
                raise ValueError(f"oversized metric file: {name}")
            content = bundle.read(name)
            target = destination / name
            # Reject symlinks at every existing component and any conflicting file.
            for component in [target, *target.parents]:
                if component.is_symlink():
                    raise ValueError(f"symlink destination rejected: {component}")
            if target.exists():
                if not target.is_file() or target.read_bytes() != content:
                    raise ValueError(f"existing destination differs: {target}; no files restored")
            pending.append((target, content))
    # Preflight every file before writing; preserve any already-identical copy.
    for target, content in pending:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(content)
    return len(pending)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="downloaded GitHub artifact 9975833698 ZIP")
    parser.add_argument("--destination", default="results/final_rigor_v2")
    args = parser.parse_args()
    try:
        count = restore(args.archive, args.destination)
        print(f"RESTORED_RETAINED_V2: {count} metric files verified/restored; no training or new outcomes.")
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(2, f"Restore failed: {exc}\n")
