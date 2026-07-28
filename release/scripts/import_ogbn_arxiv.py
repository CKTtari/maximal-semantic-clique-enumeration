#!/usr/bin/env python3
"""Convert the official ogbn-arxiv archive to the ACMSC graph/vector format."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
SOURCE_SHA256 = "49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276"
ARCHIVE = ROOT / "dataset" / "downloads" / "ogbn-arxiv" / "arxiv.zip"
GRAPH_OUTPUT = ROOT / "dataset" / "dataset" / "ogbn-arxiv.txt"
VECTOR_OUTPUT = ROOT / "dataset" / "vectors" / "ogbn-arxiv_vectors.bin"
PROVENANCE_OUTPUT = ROOT / "dataset" / "provenance" / "ogbn-arxiv.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gzip_text(archive: ZipFile, member: str) -> io.TextIOWrapper:
    compressed = gzip.GzipFile(fileobj=archive.open(member))
    return io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def read_vertex_count(archive: ZipFile) -> int:
    with gzip_text(archive, "arxiv/raw/num-node-list.csv.gz") as stream:
        values = [field for row in csv.reader(stream) for field in row if field]
    if len(values) != 1:
        raise ValueError(f"Unexpected num-node-list contents: {values}")
    return int(values[0])


def read_edges(archive: ZipFile, vertex_count: int) -> tuple[list[tuple[int, int]], dict[str, int]]:
    undirected: set[tuple[int, int]] = set()
    directed_rows = 0
    self_loops = 0
    with gzip_text(archive, "arxiv/raw/edge.csv.gz") as stream:
        for row in csv.reader(stream):
            if len(row) != 2:
                raise ValueError(f"Malformed edge row: {row}")
            source, target = map(int, row)
            directed_rows += 1
            if not (0 <= source < vertex_count and 0 <= target < vertex_count):
                raise ValueError(f"Edge endpoint out of range: {(source, target)}")
            if source == target:
                self_loops += 1
                continue
            undirected.add((source, target) if source < target else (target, source))
    edges = sorted(undirected)
    return edges, {
        "directed_input_rows": directed_rows,
        "removed_self_loops": self_loops,
        "collapsed_duplicate_or_reciprocal_rows": directed_rows - self_loops - len(edges),
        "undirected_simple_edges": len(edges),
    }


def read_features(archive: ZipFile, vertex_count: int) -> np.ndarray:
    with archive.open("arxiv/raw/node-feat.csv.gz") as member:
        with gzip.GzipFile(fileobj=member) as stream:
            features = np.loadtxt(stream, delimiter=",", dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != vertex_count:
        raise ValueError(
            f"Feature shape {features.shape} does not match {vertex_count} vertices"
        )
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains NaN or infinity")
    return features


def write_graph(path: Path, vertex_count: int, edges: list[tuple[int, int]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{vertex_count} {len(edges)}\n")
        for source, target in edges:
            stream.write(f"{source} {target}\n")
    temporary.replace(path)


def write_vectors(path: Path, features: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    vertex_count, dimension = features.shape
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<ii", vertex_count, dimension))
        np.asarray(features, dtype="<f8", order="C").tofile(stream)
    expected_bytes = 8 + vertex_count * dimension * 8
    if temporary.stat().st_size != expected_bytes:
        raise ValueError("Incomplete vector output")
    temporary.replace(path)


def ensure_outputs_absent(force: bool) -> None:
    existing = [path for path in (GRAPH_OUTPUT, VECTOR_OUTPUT, PROVENANCE_OUTPUT) if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing outputs without --force: "
            + ", ".join(map(str, existing))
        )


def convert(force: bool) -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(f"Download {SOURCE_URL} to {ARCHIVE}")
    observed_archive_hash = sha256(ARCHIVE)
    if observed_archive_hash != SOURCE_SHA256:
        raise ValueError(
            f"Archive SHA-256 mismatch: {observed_archive_hash} != {SOURCE_SHA256}"
        )
    ensure_outputs_absent(force)
    GRAPH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    VECTOR_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(ARCHIVE) as archive:
        vertex_count = read_vertex_count(archive)
        edges, edge_stats = read_edges(archive, vertex_count)
        features = read_features(archive, vertex_count)

    write_graph(GRAPH_OUTPUT, vertex_count, edges)
    write_vectors(VECTOR_OUTPUT, features)
    provenance = {
        "schema_version": 1,
        "dataset": "ogbn-arxiv",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "url": SOURCE_URL,
            "documentation": "https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv",
            "license": "ODC-BY",
            "archive_path": ARCHIVE.relative_to(ROOT).as_posix(),
            "archive_bytes": ARCHIVE.stat().st_size,
            "archive_sha256": observed_archive_hash,
        },
        "transformation": {
            "graph": "drop self-loops, ignore citation direction, deduplicate, sort by endpoint IDs",
            "vectors": "preserve official float32 values and store as little-endian float64 rows",
            "normalization": "none during conversion; algorithms apply their shared L2 normalization",
            **edge_stats,
        },
        "output": {
            "vertices": vertex_count,
            "edges": len(edges),
            "dimension": int(features.shape[1]),
            "graph": {
                "path": GRAPH_OUTPUT.relative_to(ROOT).as_posix(),
                "bytes": GRAPH_OUTPUT.stat().st_size,
                "sha256": sha256(GRAPH_OUTPUT),
            },
            "vectors": {
                "path": VECTOR_OUTPUT.relative_to(ROOT).as_posix(),
                "bytes": VECTOR_OUTPUT.stat().st_size,
                "sha256": sha256(VECTOR_OUTPUT),
            },
        },
    }
    PROVENANCE_OUTPUT.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(
        f"Converted ogbn-arxiv: V={vertex_count}, E={len(edges)}, "
        f"D={features.shape[1]}"
    )
    print(f"Provenance: {PROVENANCE_OUTPUT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace existing converted outputs")
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args().force)
