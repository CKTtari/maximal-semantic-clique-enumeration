#!/usr/bin/env python3
"""Download and convert the public Planetoid citation graphs.

The converter keeps the published node order, makes citation edges undirected,
removes self-loops/duplicates, and writes the binary format consumed by the
shared C++ loader: int32 vertex count, int32 dimension, then row-major float64
features.  Raw downloads and converted data stay under ignored dataset paths.
"""

from __future__ import annotations

import gzip
import pickle
import urllib.request
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix, vstack


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "dataset" / "downloads" / "planetoid"
OUT_ROOT = ROOT / "dataset" / "Processed"
DATASETS = {"Cora": "cora", "CiteSeer": "citeseer", "PubMed": "pubmed"}
BASE = "https://github.com/kimiyoung/planetoid/raw/master/data"
FILES = ("x", "tx", "allx", "y", "ty", "ally", "graph", "test.index")


def fetch(name: str, short: str) -> Path:
    target = RAW_ROOT / short / f"ind.{short}.{name}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        urllib.request.urlretrieve(f"{BASE}/ind.{short}.{name}", target)
    return target


def read_pickle(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream, encoding="latin1")


def parse_index(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def convert(name: str, short: str) -> dict[str, int | str]:
    files = {item: fetch(item, short) for item in FILES}
    x = read_pickle(files["x"])
    tx = read_pickle(files["tx"])
    allx = read_pickle(files["allx"])
    test_index = parse_index(files["test.index"])
    test_index_sorted = sorted(test_index)

    features = vstack((allx, tx)).tolil()
    required_rows = max(len(test_index_sorted), max(test_index) + 1)
    if required_rows > features.shape[0]:
        padded = lil_matrix((required_rows - features.shape[0], features.shape[1]))
        features = vstack((features, padded)).tolil()
    features[test_index, :] = features[test_index_sorted, :]
    dense = np.asarray(features.toarray(), dtype=np.float64)

    graph = read_pickle(files["graph"])
    edges = {(min(u, v), max(u, v)) for u, nbrs in graph.items() for v in nbrs if u != v}
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    graph_path = out / "graph.txt"
    vector_path = out / "vectors.bin"
    with graph_path.open("w", encoding="utf-8") as stream:
        stream.write(f"{dense.shape[0]} {len(edges)}\n")
        for u, v in sorted(edges):
            stream.write(f"{u} {v}\n")
    with vector_path.open("wb") as stream:
        np.asarray([dense.shape[0], dense.shape[1]], dtype="<i4").tofile(stream)
        dense.astype("<f8", copy=False).tofile(stream)
    return {"name": name, "vertices": int(dense.shape[0]), "edges": len(edges), "dimension": int(dense.shape[1])}


if __name__ == "__main__":
    for dataset_name, short_name in DATASETS.items():
        print(convert(dataset_name, short_name))
