#!/usr/bin/env python3
"""Import the SNAP GitHub Social graph into the project's native formats."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAGE = "https://snap.stanford.edu/data/github-social.html"
SOURCE_URL = "https://snap.stanford.edu/data/git_web_ml.zip"
SOURCE_SHA256 = "56c0c1c2c460dc4c7bf89357b1fec020f45e2ecebab81d131d073b10e28ec031"
ARCHIVE = ROOT / "dataset" / "downloads" / "github-social.zip"
GRAPH_OUT = ROOT / "dataset" / "dataset" / "github-social.txt"
VECTOR_OUT = ROOT / "dataset" / "vectors" / "github-social_vectors.bin"
LABEL_OUT = ROOT / "dataset" / "metadata" / "github-social_labels.csv"
PROVENANCE_OUT = ROOT / "dataset" / "metadata" / "github-social_provenance.json"

EDGE_MEMBER = "git_web_ml/musae_git_edges.csv"
FEATURE_MEMBER = "git_web_ml/musae_git_features.json"
TARGET_MEMBER = "git_web_ml/musae_git_target.csv"

EXPECTED_VERTICES = 37_700
EXPECTED_EDGES = 289_003
EXPECTED_DIMENSION = 4_005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_writable(outputs: list[Path], force: bool) -> None:
    existing = [str(path.relative_to(ROOT)) for path in outputs if path.exists()]
    if existing and not force:
        joined = ", ".join(existing)
        raise FileExistsError(f"Refusing to overwrite {joined}; pass --force to regenerate")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)


def read_targets(archive: zipfile.ZipFile) -> tuple[list[tuple[int, str, int]], int]:
    rows: list[tuple[int, str, int]] = []
    with archive.open(TARGET_MEMBER) as raw:
        reader = csv.DictReader(line.decode("utf-8") for line in raw)
        for record in reader:
            rows.append((int(record["id"]), record["name"], int(record["ml_target"])))
    rows.sort(key=lambda row: row[0])
    vertex_count = len(rows)
    if [row[0] for row in rows] != list(range(vertex_count)):
        raise ValueError("GitHub-Social vertex IDs are not contiguous from zero")
    if vertex_count != EXPECTED_VERTICES:
        raise ValueError(f"Expected {EXPECTED_VERTICES} vertices, found {vertex_count}")
    return rows, vertex_count


def read_edges(archive: zipfile.ZipFile, vertex_count: int) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    with archive.open(EDGE_MEMBER) as raw:
        reader = csv.DictReader(line.decode("utf-8") for line in raw)
        for record in reader:
            u, v = int(record["id_1"]), int(record["id_2"])
            if not (0 <= u < vertex_count and 0 <= v < vertex_count):
                raise ValueError(f"Edge endpoint outside [0,{vertex_count}): {(u, v)}")
            if u == v:
                continue
            edges.add((u, v) if u < v else (v, u))
    ordered = sorted(edges)
    if len(ordered) != EXPECTED_EDGES:
        raise ValueError(f"Expected {EXPECTED_EDGES} undirected edges, found {len(ordered)}")
    return ordered


def read_features(
    archive: zipfile.ZipFile, vertex_count: int
) -> tuple[list[list[int]], int]:
    with archive.open(FEATURE_MEMBER) as raw:
        encoded = json.load(raw)
    features: list[list[int]] = [[] for _ in range(vertex_count)]
    maximum = -1
    for raw_id, raw_indices in encoded.items():
        vertex = int(raw_id)
        if not 0 <= vertex < vertex_count:
            raise ValueError(f"Feature row has invalid vertex ID {vertex}")
        indices = sorted(set(int(index) for index in raw_indices))
        if not indices:
            raise ValueError(f"Vertex {vertex} has no source features")
        if indices[0] < 0:
            raise ValueError(f"Vertex {vertex} has a negative feature index")
        features[vertex] = indices
        maximum = max(maximum, indices[-1])
    missing = [vertex for vertex, indices in enumerate(features) if not indices]
    if missing:
        raise ValueError(f"Missing source features for {len(missing)} vertices")
    dimension = maximum + 1
    if dimension != EXPECTED_DIMENSION:
        raise ValueError(f"Expected dimension {EXPECTED_DIMENSION}, found {dimension}")
    return features, dimension


def write_graph(path: Path, vertex_count: int, edges: list[tuple[int, int]]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{vertex_count} {len(edges)}\n")
        handle.writelines(f"{u} {v}\n" for u, v in edges)


def write_vectors(path: Path, features: list[list[int]], dimension: int) -> None:
    row = np.zeros(dimension, dtype="<f8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<ii", len(features), dimension))
        for indices in features:
            row.fill(0.0)
            row[indices] = 1.0
            handle.write(row.tobytes(order="C"))


def write_labels(path: Path, targets: list[tuple[int, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["vertex", "label", "name"])
        for vertex, name, label in targets:
            writer.writerow([vertex, label, name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace generated files")
    args = parser.parse_args()

    outputs = [GRAPH_OUT, VECTOR_OUT, LABEL_OUT, PROVENANCE_OUT]
    require_writable(outputs, args.force)
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Download {SOURCE_URL} to {ARCHIVE.relative_to(ROOT)}")
    archive_hash = sha256(ARCHIVE)
    if archive_hash != SOURCE_SHA256:
        raise ValueError(f"Source SHA-256 mismatch: {archive_hash}")

    with zipfile.ZipFile(ARCHIVE) as archive:
        targets, vertex_count = read_targets(archive)
        edges = read_edges(archive, vertex_count)
        features, dimension = read_features(archive, vertex_count)

    write_graph(GRAPH_OUT, vertex_count, edges)
    write_vectors(VECTOR_OUT, features, dimension)
    write_labels(LABEL_OUT, targets)
    provenance = {
        "dataset": "GitHub-Social",
        "source_page": SOURCE_PAGE,
        "source_url": SOURCE_URL,
        "source_sha256": archive_hash,
        "vertices": vertex_count,
        "undirected_edges": len(edges),
        "feature_dimension": dimension,
        "feature_semantics": ["location", "starred repositories", "employer", "email"],
        "label": "web developer (0) or machine-learning developer (1)",
        "transformations": [
            "removed self-loops",
            "canonicalized and deduplicated undirected edges",
            "preserved the original sparse binary features as dense float64 rows",
        ],
        "outputs": {
            "graph": str(GRAPH_OUT.relative_to(ROOT)).replace("\\", "/"),
            "vectors": str(VECTOR_OUT.relative_to(ROOT)).replace("\\", "/"),
            "labels": str(LABEL_OUT.relative_to(ROOT)).replace("\\", "/"),
        },
    }
    PROVENANCE_OUT.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
