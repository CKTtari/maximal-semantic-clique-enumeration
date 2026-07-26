#!/usr/bin/env python3
"""Verify ignored dataset files and create a Git-trackable provenance manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "experiments" / "dataset-archive"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_project_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Dataset manifest path leaves the project: {raw}") from error
    return resolved


def portable_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def verify_file(raw_path: str, expected_bytes: int, expected_hash: str) -> dict[str, Any]:
    path = resolve_project_path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    observed_bytes = path.stat().st_size
    observed_hash = sha256(path)
    if observed_bytes != int(expected_bytes) or observed_hash != expected_hash.lower():
        raise ValueError(
            f"Dataset file mismatch: {path} "
            f"bytes {observed_bytes}/{expected_bytes}, hash {observed_hash}/{expected_hash}"
        )
    return {"path": portable_path(path), "bytes": observed_bytes, "sha256": observed_hash}


def verify_import_provenance(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    source = provenance["source"]
    output = provenance["output"]
    return [
        verify_file(source["archive_path"], source["archive_bytes"], source["archive_sha256"]),
        verify_file(
            output["graph"]["path"],
            output["graph"]["bytes"],
            output["graph"]["sha256"],
        ),
        verify_file(
            output["vectors"]["path"],
            output["vectors"]["bytes"],
            output["vectors"]["sha256"],
        ),
    ]


def verify_derived_provenance(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    verified = []
    for source in provenance["source"]["files"]:
        verified.append(verify_file(source["path"], source["bytes"], source["sha256"]))
    for level in provenance["levels"]:
        graph_path = resolve_project_path(level["graph"])
        vector_path = resolve_project_path(level["vectors"])
        verified.append(
            verify_file(level["graph"], graph_path.stat().st_size, level["graph_sha256"])
        )
        verified.append(
            verify_file(
                level["vectors"], vector_path.stat().st_size, level["vectors_sha256"]
            )
        )
    return verified


def verify_generated_attribute_provenance(
    provenance: dict[str, Any],
) -> list[dict[str, Any]]:
    verified = []
    for dataset in provenance["datasets"]:
        graph = dataset["graph"]
        vectors = dataset["vectors"]
        verified.append(verify_file(graph["path"], graph["bytes"], graph["sha256"]))
        verified.append(
            verify_file(vectors["path"], vectors["bytes"], vectors["sha256"])
        )
    return verified


def archive_dataset(name: str, provenance_path: Path, audit_path: Path | None) -> Path:
    provenance_path = provenance_path.resolve()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("kind") == "generated_attributes_on_public_graphs":
        kind = "generated_attributes_on_public_graphs"
        files = verify_generated_attribute_provenance(provenance)
    elif "levels" in provenance and "construction" in provenance:
        kind = "derived_nested_subgraphs"
        files = verify_derived_provenance(provenance)
    elif "dataset" in provenance and "output" in provenance:
        kind = "imported_public_dataset"
        files = verify_import_provenance(provenance)
    else:
        raise ValueError(f"Unsupported provenance schema: {provenance_path}")

    audit = None
    if audit_path is not None:
        audit_path = audit_path.resolve()
        json.loads(audit_path.read_text(encoding="utf-8"))
        audit = {
            "path": portable_path(audit_path),
            "bytes": audit_path.stat().st_size,
            "sha256": sha256(audit_path),
        }

    manifest = {
        "schema_version": 1,
        "name": name,
        "kind": kind,
        "archived_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_file": {
            "path": portable_path(provenance_path),
            "bytes": provenance_path.stat().st_size,
            "sha256": sha256(provenance_path),
        },
        "verified_files": files,
        "audit": audit,
        "provenance": provenance,
    }
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = ARCHIVE_ROOT / f"{name}.manifest.json"
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    archived = archive_dataset(arguments.name, arguments.provenance, arguments.audit)
    print(f"Archived dataset: {archived}")
