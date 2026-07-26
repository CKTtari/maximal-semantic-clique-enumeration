#!/usr/bin/env python3
"""Generate reproducible attributes for public structure-only graph datasets.

The binary format is ``[int32 V][int32 dim][V * dim float64 values]``.  Each
coordinate is sampled independently from U[-1, 1).  The shared C++ graph
loader performs the L2 normalization used by every mining algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "dataset" / "vectors-256"
DEFAULT_PROVENANCE = DEFAULT_OUTPUT_DIR / "provenance.json"
DIMENSION = 256
CHUNK_ROWS = 8192

# Explicit per-dataset seeds make every binary independent of invocation order.
DATASETS: dict[str, dict[str, Any]] = {
    "sc-ldoor": {
        "graph": "dataset/dataset/sc-ldoor.txt",
        "seed": 2026072301,
    },
    "sc-nasasrb": {
        "graph": "dataset/dataset/sc-nasasrb.txt",
        "seed": 2026072302,
    },
    "sc-pkustk11": {
        "graph": "dataset/dataset/sc-pkustk11.txt",
        "seed": 2026072303,
    },
    "soc-buzznet": {
        "graph": "dataset/dataset/soc-buzznet.txt",
        "seed": 2026072304,
    },
    "soc-digg": {
        "graph": "dataset/dataset/soc-digg.txt",
        "seed": 2026072305,
    },
    "tech-as-skitter": {
        "graph": "dataset/dataset/tech-as-skitter.txt",
        "seed": 2026072306,
    },
    "sc-pwtk": {
        "graph": "dataset/dataset/sc-pwtk.txt",
        "seed": 2026072307,
    },
    "com-amazon": {
        "graph": "dataset/dataset/com-amazon.txt",
        "seed": 2026072312,
    },
    "ca-dblp-2012": {
        "graph": "dataset/dataset/ca-dblp-2012.txt",
        "seed": 2026072313,
    },
    "email-Enron": {
        "graph": "dataset/Processed/email-Enron/graph.txt",
        "seed": 2026072315,
    },
}


def ensure_inside_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Path leaves the project: {path}") from error
    return resolved


def portable_path(path: Path) -> str:
    return ensure_inside_project(path).relative_to(ROOT).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_vertex_count(graph_path: Path) -> int:
    with graph_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(f"Invalid graph header in {graph_path}: {stripped}")
            vertices = int(fields[0])
            if vertices <= 0:
                raise ValueError(f"Non-positive vertex count in {graph_path}")
            return vertices
    raise ValueError(f"Missing graph header: {graph_path}")


def generate_one(
    name: str,
    graph_path: Path,
    output_path: Path,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    graph_path = ensure_inside_project(graph_path)
    output_path = ensure_inside_project(output_path)
    if not graph_path.is_file():
        raise FileNotFoundError(graph_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite without --force: {output_path}")

    vertices = read_vertex_count(graph_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.Generator(np.random.PCG64(seed))
    output_digest = hashlib.sha256()
    header = struct.pack("<ii", vertices, DIMENSION)

    with output_path.open("wb") as stream:
        stream.write(header)
        output_digest.update(header)
        for start in range(0, vertices, CHUNK_ROWS):
            rows = min(CHUNK_ROWS, vertices - start)
            values = rng.random((rows, DIMENSION), dtype=np.float64)
            values *= 2.0
            values -= 1.0
            payload = values.astype("<f8", copy=False).tobytes(order="C")
            stream.write(payload)
            output_digest.update(payload)

    expected_bytes = 8 + vertices * DIMENSION * 8
    observed_bytes = output_path.stat().st_size
    if observed_bytes != expected_bytes:
        raise ValueError(
            f"Generated size mismatch for {name}: {observed_bytes} != {expected_bytes}"
        )

    return {
        "name": name,
        "vertices": vertices,
        "dimension": DIMENSION,
        "seed": seed,
        "graph": {
            "path": portable_path(graph_path),
            "bytes": graph_path.stat().st_size,
            "sha256": sha256(graph_path),
        },
        "vectors": {
            "path": portable_path(output_path),
            "bytes": observed_bytes,
            "sha256": output_digest.hexdigest(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASETS),
        help="Generate only this dataset; repeat for multiple datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_inside_project(args.output_dir)
    provenance_path = ensure_inside_project(args.provenance)
    selected = args.dataset or list(DATASETS)

    generated = []
    for name in selected:
        config = DATASETS[name]
        graph_path = ROOT / config["graph"]
        output_path = output_dir / f"{name}_vectors.bin"
        print(f"Generating {name}: seed={config['seed']}, dim={DIMENSION}")
        generated.append(
            generate_one(name, graph_path, output_path, config["seed"], args.force)
        )

    provenance = {
        "schema_version": 1,
        "kind": "generated_attributes_on_public_graphs",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": portable_path(Path(__file__)),
            "script_sha256": sha256(Path(__file__)),
            "numpy_version": np.__version__,
            "bit_generator": "PCG64",
            "distribution": "independent Uniform[-1, 1) coordinates",
            "normalization": "L2 normalization by the shared C++ graph loader",
            "dimension": DIMENSION,
            "chunk_rows": CHUNK_ROWS,
            "binary_format": "little-endian int32 V, int32 dimension, row-major float64 values",
        },
        "datasets": generated,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote provenance: {provenance_path}")


if __name__ == "__main__":
    main()
