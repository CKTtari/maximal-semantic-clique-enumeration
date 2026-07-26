#!/usr/bin/env python3
"""Convert a SNAP edge list into the repository's deterministic graph format."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def inside_repo(path: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(ROOT)
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-url", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = inside_repo(args.input)
    output = inside_repo(args.output)
    provenance_path = inside_repo(args.provenance)

    vertices: set[int] = set()
    edges: set[tuple[int, int]] = set()
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(f"Invalid edge at {source}:{line_number}")
            u, v = int(fields[0]), int(fields[1])
            if u == v:
                continue
            vertices.update((u, v))
            edges.add((u, v) if u < v else (v, u))

    old_ids = sorted(vertices)
    remap = {old_id: new_id for new_id, old_id in enumerate(old_ids)}
    normalized_edges = sorted((remap[u], remap[v]) for u, v in edges)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{len(old_ids)} {len(normalized_edges)}\n")
        for u, v in normalized_edges:
            stream.write(f"{u} {v}\n")

    id_digest = hashlib.sha256()
    for old_id in old_ids:
        id_digest.update(f"{old_id}\n".encode("ascii"))
    provenance = {
        "schema_version": 1,
        "name": args.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": args.source_url,
        "source": {
            "path": source.relative_to(ROOT).as_posix(),
            "bytes": source.stat().st_size,
            "sha256": sha256(source),
        },
        "conversion": {
            "undirected": True,
            "self_loops_removed": True,
            "duplicate_edges_removed": True,
            "vertex_mapping": "ascending source identifier to contiguous zero-based identifier",
            "ordered_source_ids_sha256": id_digest.hexdigest(),
        },
        "graph": {
            "path": output.relative_to(ROOT).as_posix(),
            "vertices": len(old_ids),
            "edges": len(normalized_edges),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
        },
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}: V={len(old_ids)}, E={len(normalized_edges)}")
    print(f"Wrote provenance: {provenance_path}")


if __name__ == "__main__":
    main()
