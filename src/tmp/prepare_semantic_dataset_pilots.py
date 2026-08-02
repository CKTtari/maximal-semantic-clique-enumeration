"""Convert official Wiki-CS and Flickr files into the project's pilot format."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def write_graph(path: Path, n: int, edges: set[int]) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.fromiter(sorted(edges), dtype=np.int64, count=len(edges))
    src = ordered // n
    dst = ordered % n
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{n} {len(ordered)}\n")
        for u, v in zip(src, dst):
            handle.write(f"{int(u)} {int(v)}\n")
    return src, dst


def write_vectors(path: Path, features: np.ndarray) -> None:
    n, dim = features.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(struct.pack("ii", n, dim))
        for start in range(0, n, 2048):
            block = np.asarray(features[start : start + 2048], dtype=np.float64, order="C")
            handle.write(block.tobytes(order="C"))


def summarize(features: np.ndarray, src: np.ndarray, dst: np.ndarray) -> dict[str, float | int]:
    norms = np.linalg.norm(features, axis=1)
    if np.any(norms == 0):
        raise ValueError(f"found {int(np.count_nonzero(norms == 0))} zero feature vectors")
    similarities = np.empty(len(src), dtype=np.float32)
    for start in range(0, len(src), 100_000):
        stop = min(start + 100_000, len(src))
        left = np.asarray(features[src[start:stop]], dtype=np.float32)
        right = np.asarray(features[dst[start:stop]], dtype=np.float32)
        dots = np.einsum("ij,ij->i", left, right)
        similarities[start:stop] = dots / (norms[src[start:stop]] * norms[dst[start:stop]])
    rounded = np.round(similarities, 6)
    return {
        "vertices": int(features.shape[0]),
        "edges": int(len(src)),
        "dimension": int(features.shape[1]),
        "q20": float(np.quantile(similarities, 0.20)),
        "q80": float(np.quantile(similarities, 0.80)),
        "q99": float(np.quantile(similarities, 0.99)),
        "min": float(similarities.min()),
        "median": float(np.median(similarities)),
        "max": float(similarities.max()),
        "zero_fraction": float(np.mean(np.abs(similarities) < 1e-8)),
        "unique_6dp_fraction": float(np.unique(rounded).size / len(rounded)),
    }


def prepare_wikics(root: Path) -> dict[str, float | int]:
    with (root / "dataset/Raw/WikiCS/dataset/data.json").open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    features = np.asarray(raw["features"], dtype=np.float32)
    valid = np.linalg.norm(features, axis=1) > 0
    remap = np.full(features.shape[0], -1, dtype=np.int64)
    remap[valid] = np.arange(int(valid.sum()))
    features = features[valid]
    n = features.shape[0]
    edges: set[int] = set()
    for u, neighbors in enumerate(raw["links"]):
        if remap[u] < 0:
            continue
        for value in neighbors:
            v = int(value)
            if remap[v] >= 0 and u != v:
                ru, rv = int(remap[u]), int(remap[v])
                a, b = (ru, rv) if ru < rv else (rv, ru)
                edges.add(a * n + b)
    out = root / "dataset/Processed/WikiCS"
    src, dst = write_graph(out / "graph.txt", n, edges)
    write_vectors(out / "vectors.bin", features)
    return summarize(features, src, dst)


def prepare_flickr(root: Path) -> dict[str, float | int]:
    raw_dir = root / "dataset/Raw/Flickr/extracted"
    archive = np.load(raw_dir / "adj_full.npz")
    indices = archive["indices"]
    indptr = archive["indptr"]
    shape = archive["shape"]
    old_n = int(shape[0])
    source_features = np.load(raw_dir / "feats.npy", mmap_mode="r")
    if source_features.shape[0] != old_n:
        raise ValueError("Flickr adjacency and feature counts differ")
    valid = np.linalg.norm(source_features, axis=1) > 0
    remap = np.full(old_n, -1, dtype=np.int64)
    remap[valid] = np.arange(int(valid.sum()))
    features = np.asarray(source_features[valid], dtype=np.float32)
    n = features.shape[0]
    edges: set[int] = set()
    for u in range(old_n):
        if remap[u] < 0:
            continue
        for value in indices[indptr[u] : indptr[u + 1]]:
            v = int(value)
            if remap[v] >= 0 and u != v:
                ru, rv = int(remap[u]), int(remap[v])
                a, b = (ru, rv) if ru < rv else (rv, ru)
                edges.add(a * n + b)
    out = root / "dataset/Processed/Flickr"
    src, dst = write_graph(out / "graph.txt", n, edges)
    write_vectors(out / "vectors.bin", features)
    return summarize(features, src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("wikics", "flickr"), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    stats = prepare_wikics(root) if args.dataset == "wikics" else prepare_flickr(root)
    output = root / "dataset/Processed" / ("WikiCS" if args.dataset == "wikics" else "Flickr") / "pilot_stats.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
        handle.write("\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
