#!/usr/bin/env python3
"""Convert the project edge-list format to FaPlex binary without renumbering vertices."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.input.open("r", encoding="ascii") as handle:
        vertex_count, expected_edges = map(int, handle.readline().split())
        adjacency = [[] for _ in range(vertex_count)]
        seen: set[tuple[int, int]] = set()
        for line_number, line in enumerate(handle, 2):
            if not line.strip():
                continue
            u, v = map(int, line.split())
            if not (0 <= u < vertex_count and 0 <= v < vertex_count):
                raise ValueError(f"Endpoint outside vertex domain at line {line_number}")
            if u == v:
                raise ValueError(f"Self-loop at line {line_number}")
            edge = (u, v) if u < v else (v, u)
            if edge in seen:
                raise ValueError(f"Duplicate edge at line {line_number}")
            seen.add(edge)
            adjacency[u].append(v)
            adjacency[v].append(u)
    if len(seen) != expected_edges:
        raise ValueError(f"Header declares {expected_edges} edges, found {len(seen)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(struct.pack("<III", 4, vertex_count, 2 * expected_edges))
        handle.write(struct.pack(f"<{vertex_count}I", *(len(row) for row in adjacency)))
        for row in adjacency:
            row.sort()
            if row:
                handle.write(struct.pack(f"<{len(row)}I", *row))
    print(f"wrote {vertex_count} vertices and {expected_edges} undirected edges")


if __name__ == "__main__":
    main()
