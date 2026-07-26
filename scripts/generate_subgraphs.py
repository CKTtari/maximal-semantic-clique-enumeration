#!/usr/bin/env python3
"""Build nested, fully induced datasets from a real graph and its real vectors.

Vertices are ranked by decreasing source-graph degree, with original vertex id
as a deterministic tie break.  Every level uses a prefix of that ranking.  The
script then performs a separate edge pass, so every emitted graph contains all
and only source edges whose two endpoints are selected.  Vector rows are copied
from the source store in exactly the same new-id order.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO


HEADER = struct.Struct("<ii")
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def require_project_output(path: Path) -> None:
    try:
        path.resolve().relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Output directory must stay inside {ROOT}: {path}") from error


def resolve_listed_path(index_path: Path, listed: str) -> Path:
    candidate = Path(listed.replace("\\", os.sep))
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    cwd_candidate = (Path.cwd() / candidate).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    index_candidate = (index_path.parent / candidate).resolve()
    if index_candidate.exists():
        return index_candidate
    raise FileNotFoundError(f"Vector chunk does not exist: {listed}")


def read_graph_header(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8") as stream:
        fields = stream.readline().split()
    if len(fields) < 2:
        raise ValueError(f"Invalid graph header in {path}")
    vertices, edges = map(int, fields[:2])
    if vertices <= 0 or edges < 0:
        raise ValueError(f"Invalid graph dimensions in {path}: {vertices} {edges}")
    return vertices, edges


def iter_edges(path: Path, vertex_count: int) -> Iterable[tuple[int, int]]:
    with path.open("r", encoding="utf-8") as stream:
        next(stream)
        for line_number, line in enumerate(stream, 2):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed edge at {path}:{line_number}")
            u, v = int(fields[0]), int(fields[1])
            if not (0 <= u < vertex_count and 0 <= v < vertex_count):
                raise ValueError(
                    f"Out-of-range endpoint at {path}:{line_number}: {u} {v}"
                )
            yield u, v


@dataclass(frozen=True)
class VectorSource:
    vertex_count: int
    dimension: int
    chunks: tuple[Path, ...]
    chunk_starts: tuple[int, ...]
    chunk_counts: tuple[int, ...]

    @classmethod
    def open(cls, path: Path) -> "VectorSource":
        if path.suffix.lower() != ".index":
            with path.open("rb") as stream:
                vertex_count, dimension = HEADER.unpack(stream.read(HEADER.size))
            expected = HEADER.size + vertex_count * dimension * 8
            if path.stat().st_size != expected:
                raise ValueError(
                    f"Binary vector size mismatch: expected {expected}, "
                    f"found {path.stat().st_size}"
                )
            return cls(vertex_count, dimension, (path,), (0,), (vertex_count,))

        tokens = path.read_text(encoding="utf-8").split()
        if len(tokens) < 3:
            raise ValueError(f"Invalid vector index: {path}")
        vertex_count, dimension = int(tokens[0]), int(tokens[1])
        chunks: list[Path] = []
        starts: list[int] = []
        counts: list[int] = []
        loaded = 0
        for listed in tokens[2:]:
            chunk = resolve_listed_path(path, listed)
            with chunk.open("rb") as stream:
                count, chunk_dimension = HEADER.unpack(stream.read(HEADER.size))
            if chunk_dimension != dimension:
                raise ValueError(f"Dimension mismatch in vector chunk {chunk}")
            expected = HEADER.size + count * dimension * 8
            if chunk.stat().st_size != expected:
                raise ValueError(f"Truncated or oversized vector chunk: {chunk}")
            chunks.append(chunk)
            starts.append(loaded)
            counts.append(count)
            loaded += count
        if loaded != vertex_count:
            raise ValueError(
                f"Vector index declares {vertex_count} rows but chunks contain {loaded}"
            )
        return cls(
            vertex_count,
            dimension,
            tuple(chunks),
            tuple(starts),
            tuple(counts),
        )

    def copy_rows(self, old_ids: list[int], output: BinaryIO) -> None:
        handles = [chunk.open("rb") for chunk in self.chunks]
        row_bytes = self.dimension * 8
        try:
            for old_id in old_ids:
                chunk_index = bisect.bisect_right(self.chunk_starts, old_id) - 1
                if chunk_index < 0:
                    raise ValueError(f"No vector chunk covers vertex {old_id}")
                local_id = old_id - self.chunk_starts[chunk_index]
                if local_id >= self.chunk_counts[chunk_index]:
                    raise ValueError(f"No vector row for vertex {old_id}")
                stream = handles[chunk_index]
                stream.seek(HEADER.size + local_id * row_bytes)
                row = stream.read(row_bytes)
                if len(row) != row_bytes:
                    raise ValueError(f"Short vector row for vertex {old_id}")
                output.write(row)
        finally:
            for stream in handles:
                stream.close()


def choose_vertices(graph_path: Path, vertex_count: int, maximum: int) -> list[int]:
    degrees = [0] * vertex_count
    for u, v in iter_edges(graph_path, vertex_count):
        if u == v:
            continue
        degrees[u] += 1
        degrees[v] += 1
    ranked = sorted(range(vertex_count), key=lambda vertex: (-degrees[vertex], vertex))
    return ranked[:maximum]


def write_graph_levels(
    graph_path: Path,
    vertex_count: int,
    selected: list[int],
    sizes: list[int],
    output_dir: Path,
) -> tuple[list[Path], list[int], int]:
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(selected)}
    paths = [output_dir / f"level-{size}.graph.txt" for size in sizes]
    streams: list[TextIO] = []
    counts = [0] * len(sizes)
    duplicate_count = 0
    seen: set[int] = set()
    maximum = sizes[-1]
    try:
        for size, path in zip(sizes, paths):
            stream = path.open("w+", encoding="ascii", newline="\n")
            stream.write(f"{size:20d} {0:20d}\n")
            streams.append(stream)
        for old_u, old_v in iter_edges(graph_path, vertex_count):
            if old_u == old_v:
                continue
            new_u = old_to_new.get(old_u)
            new_v = old_to_new.get(old_v)
            if new_u is None or new_v is None:
                continue
            if new_u > new_v:
                new_u, new_v = new_v, new_u
            edge_key = new_u * maximum + new_v
            if edge_key in seen:
                duplicate_count += 1
                continue
            seen.add(edge_key)
            first_level = bisect.bisect_left(sizes, max(new_u, new_v) + 1)
            for level in range(first_level, len(sizes)):
                streams[level].write(f"{new_u} {new_v}\n")
                counts[level] += 1
        for stream, size, count in zip(streams, sizes, counts):
            stream.seek(0)
            stream.write(f"{size:20d} {count:20d}\n")
    finally:
        for stream in streams:
            stream.close()
    return paths, counts, duplicate_count


def write_vector_levels(
    source: VectorSource,
    selected: list[int],
    sizes: list[int],
    output_dir: Path,
) -> list[Path]:
    paths = [output_dir / f"level-{size}.vectors.bin" for size in sizes]
    maximum_path = paths[-1]
    with maximum_path.open("wb") as output:
        output.write(HEADER.pack(sizes[-1], source.dimension))
        source.copy_rows(selected, output)

    row_bytes = source.dimension * 8
    for size, path in zip(sizes[:-1], paths[:-1]):
        with maximum_path.open("rb") as source_stream, path.open("wb") as output:
            source_stream.seek(HEADER.size)
            output.write(HEADER.pack(size, source.dimension))
            remaining = size * row_bytes
            while remaining:
                block = source_stream.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("Unexpected end of maximum-level vector file")
                output.write(block)
                remaining -= len(block)
    return paths


def validate_level(graph_path: Path, vector_path: Path, size: int, edges: int) -> None:
    graph_vertices, header_edges = read_graph_header(graph_path)
    if (graph_vertices, header_edges) != (size, edges):
        raise ValueError(f"Graph validation failed for {graph_path}")
    observed_edges = sum(1 for _ in iter_edges(graph_path, size))
    if observed_edges != edges:
        raise ValueError(
            f"Graph edge count mismatch in {graph_path}: {observed_edges} != {edges}"
        )
    with vector_path.open("rb") as stream:
        vector_vertices, dimension = HEADER.unpack(stream.read(HEADER.size))
    expected_bytes = HEADER.size + size * dimension * 8
    if vector_vertices != size or vector_path.stat().st_size != expected_bytes:
        raise ValueError(f"Vector validation failed for {vector_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_path = args.graph.resolve()
    vector_path = args.vectors.resolve()
    output_dir = args.output_dir.resolve()
    require_project_output(output_dir)
    sizes = sorted(set(args.sizes))
    vertex_count, source_header_edges = read_graph_header(graph_path)
    if not sizes or sizes[0] < 2 or sizes[-1] > vertex_count:
        raise ValueError(f"Sizes must be between 2 and {vertex_count}")

    vectors = VectorSource.open(vector_path)
    if vectors.vertex_count != vertex_count:
        raise ValueError(
            f"Graph/vector vertex mismatch: {vertex_count} != {vectors.vertex_count}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = choose_vertices(graph_path, vertex_count, sizes[-1])
    graph_paths, edge_counts, duplicates = write_graph_levels(
        graph_path, vertex_count, selected, sizes, output_dir
    )
    vector_paths = write_vector_levels(vectors, selected, sizes, output_dir)

    levels = []
    for size, graph_output, vector_output, edge_count in zip(
        sizes, graph_paths, vector_paths, edge_counts
    ):
        validate_level(graph_output, vector_output, size, edge_count)
        levels.append(
            {
                "vertices": size,
                "edges": edge_count,
                "graph": portable_path(graph_output),
                "vectors": portable_path(vector_output),
                "graph_sha256": sha256(graph_output),
                "vectors_sha256": sha256(vector_output),
            }
        )

    source_files = [graph_path, vector_path, *vectors.chunks]
    unique_source_files = list(dict.fromkeys(source_files))
    manifest = {
        "schema_version": 1,
        "construction": {
            "type": "nested_full_induced_subgraphs",
            "vertex_order": "source degree descending; original id ascending tie-break",
            "random_seed": None,
            "edge_rule": "all unique non-self source edges with both endpoints selected",
            "vector_rule": "unaltered source vector row, reordered with vertex ids",
        },
        "source": {
            "graph_vertices": vertex_count,
            "graph_header_edges": source_header_edges,
            "vector_dimension": vectors.dimension,
            "files": [
                {
                    "path": portable_path(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in unique_source_files
            ],
        },
        "selected_original_ids": selected,
        "discarded_duplicate_edges_within_largest_level": duplicates,
        "levels": levels,
    }
    manifest_path = output_dir / "provenance.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(levels)} validated levels in {output_dir}")
    print(f"Provenance: {manifest_path}")


if __name__ == "__main__":
    main()
