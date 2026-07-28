#!/usr/bin/env python3
"""Normalize a public edge list or Matrix Market pattern into the artifact graph format."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, TextIO


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def edge_list_rows(path: Path, skip_first_data_row: bool) -> Iterable[tuple[int, int]]:
    skipped = not skip_first_data_row
    with open_text(path) as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "%")):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed edge at {path}:{line_number}")
            if not skipped:
                skipped = True
                continue
            yield int(fields[0]), int(fields[1])


def matrix_market_rows(path: Path) -> Iterable[tuple[int, int]]:
    with open_text(path) as stream:
        header = stream.readline().strip().lower()
        if not header.startswith("%%matrixmarket matrix coordinate"):
            raise ValueError(f"Expected Matrix Market coordinate input: {path}")
        dimensions_seen = False
        for line_number, line in enumerate(stream, 2):
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            fields = stripped.split()
            if not dimensions_seen:
                if len(fields) < 3:
                    raise ValueError(f"Malformed Matrix Market dimensions at {path}:{line_number}")
                dimensions_seen = True
                continue
            if len(fields) < 2:
                raise ValueError(f"Malformed coordinate at {path}:{line_number}")
            yield int(fields[0]) - 1, int(fields[1]) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument(
        "--format",
        choices=("edge-list", "matrix-market"),
        default="edge-list",
    )
    parser.add_argument(
        "--skip-first-data-row",
        action="store_true",
        help="use only when an edge-list file begins with a V E row",
    )
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    provenance = (args.provenance or output.with_suffix(".provenance.json")).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if (output.exists() or provenance.exists()) and not args.force:
        raise FileExistsError("Refusing to overwrite output without --force")

    iterator = (
        matrix_market_rows(source)
        if args.format == "matrix-market"
        else edge_list_rows(source, args.skip_first_data_row)
    )
    edges: set[tuple[int, int]] = set()
    source_rows = 0
    self_loops = 0
    active: set[int] = set()
    for u, v in iterator:
        source_rows += 1
        if u < 0 or v < 0:
            raise ValueError(f"Negative vertex identifier: {u} {v}")
        if u == v:
            self_loops += 1
            continue
        edge = (u, v) if u < v else (v, u)
        edges.add(edge)
        active.update(edge)

    old_ids = sorted(active)
    remap = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
    normalized = sorted((remap[u], remap[v]) for u, v in edges)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline=chr(10)) as stream:
        stream.write(f"{len(old_ids)} {len(normalized)}" + chr(10))
        for u, v in normalized:
            stream.write(f"{u} {v}" + chr(10))
    temporary.replace(output)

    id_digest = hashlib.sha256()
    for old_id in old_ids:
        id_digest.update(str(old_id).encode("ascii") + bytes((10,)))
    record = {
        "schema_version": 1,
        "name": args.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "url": args.source_url,
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
            "format": args.format,
        },
        "conversion": {
            "undirected": True,
            "self_loops_removed": self_loops,
            "duplicate_or_reciprocal_rows_removed": source_rows - self_loops - len(edges),
            "isolated_vertices_removed": True,
            "vertex_mapping": "ascending active source identifier to contiguous zero-based identifier",
            "ordered_source_ids_sha256": id_digest.hexdigest(),
        },
        "output": {
            "path": str(output),
            "vertices": len(old_ids),
            "edges": len(normalized),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
    }
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(record, indent=2) + chr(10), encoding="utf-8")
    print(f"Wrote {output}: V={len(old_ids)}, E={len(normalized)}")
    print(f"Wrote {provenance}")


if __name__ == "__main__":
    main()