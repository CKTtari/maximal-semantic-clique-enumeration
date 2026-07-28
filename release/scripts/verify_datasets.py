#!/usr/bin/env python3
"""Verify prepared dataset artifacts against the hashes used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "dataset" / "checksums.json").read_text(encoding="utf-8"))
    failures = 0
    checked = 0
    for dataset, metadata in catalog["datasets"].items():
        for artifact in metadata["artifacts"]:
            path = ROOT / artifact["path"]
            if not path.is_file():
                state = "MISSING"
                if not args.allow_missing:
                    failures += 1
            elif path.stat().st_size != artifact["bytes"]:
                state = "SIZE-MISMATCH"
                failures += 1
            elif sha256(path) != artifact["sha256"]:
                state = "HASH-MISMATCH"
                failures += 1
            else:
                state = "OK"
                checked += 1
            print(f"{state:13} {dataset:13} {artifact['path']}")
    print(f"Verified {checked} artifacts; failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()