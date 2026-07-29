#!/usr/bin/env python3
"""Convert a project graph into HBBMC's headerless edge-list format."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.input.resolve()
    target = args.output.resolve()
    if target.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {target}; pass --force")
    target.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="ascii") as reader:
        header = reader.readline().split()
        if len(header) != 2:
            raise ValueError("Expected graph header: V E")
        vertex_count, expected_edges = map(int, header)
        seen: set[tuple[int, int]] = set()
        with target.open("w", encoding="ascii", newline="\n") as writer:
            for line_number, line in enumerate(reader, 2):
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f"Malformed edge at line {line_number}")
                u, v = map(int, fields)
                if not (0 <= u < vertex_count and 0 <= v < vertex_count):
                    raise ValueError(f"Endpoint out of range at line {line_number}")
                if u == v:
                    raise ValueError(f"Self-loop at line {line_number}")
                edge = (u, v) if u < v else (v, u)
                if edge in seen:
                    raise ValueError(f"Duplicate undirected edge at line {line_number}")
                seen.add(edge)
                writer.write(f"{edge[0]} {edge[1]}\n")

    if len(seen) != expected_edges:
        raise ValueError(f"Header declares {expected_edges} edges, found {len(seen)}")
    print(f"wrote {len(seen)} edges over declared vertex domain [0,{vertex_count})")


if __name__ == "__main__":
    main()
