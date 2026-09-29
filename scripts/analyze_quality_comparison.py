#!/usr/bin/env python3
"""Compute size-aligned output quality for the RQ5 comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

import numpy as np

import analyze_github_quality as github
import analyze_ogbn25k_quality as ogbn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tex-data" / "data" / "quality-comparison.csv"
PROVENANCE = ROOT / "experiments" / "quality" / "multi-baseline-quality.json"
MIN_SIZE = 10
PURITY_CUTOFFS = (0.90, 0.95)
FASTQC_GAMMA = 0.9
FAPLEX_K = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(path: Path) -> list[set[int]]:
    with path.open("r", encoding="ascii") as handle:
        vertex_count, expected_edges = map(int, handle.readline().split())
        adjacency = [set() for _ in range(vertex_count)]
        edges = 0
        for line in handle:
            if not line.strip():
                continue
            u, v = map(int, line.split())
            adjacency[u].add(v)
            adjacency[v].add(u)
            edges += 1
    if edges != expected_edges:
        raise ValueError(f"Header declares {expected_edges} edges, found {edges}")
    return adjacency


def load_vectors(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        vertex_count, dimension = struct.unpack("<ii", handle.read(8))
    matrix = np.memmap(
        path,
        dtype="<f8",
        mode="r",
        offset=8,
        shape=(vertex_count, dimension),
    )
    vectors = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError(f"Zero vector in {path}")
    vectors /= norms
    return vectors


def load_outputs(module, dataset_slug: str) -> dict[str, tuple[Path, str]]:
    captured = module.captured_outputs()
    outputs = {
        name: (path, "plain") for name, (path, _result) in captured.items()
    }
    outputs["FastQC"] = (
        ROOT / "experiments" / "output" / "external-baselines" / dataset_slug
        / "fastqc-g090-u10-mapped" / "maximal-output.txt",
        "size-prefix",
    )
    outputs["FaPlex"] = (
        ROOT / "experiments" / "output" / "external-baselines" / dataset_slug
        / "faplex-k2-u10" / "run.stdout.txt",
        "solution-prefix",
    )
    return outputs


def analyze(
    path: Path,
    labels: list[int],
    adjacency: list[set[int]],
    row_format: str,
    method: str,
    vectors: np.ndarray,
) -> dict[str, float | int]:
    retained = 0
    purity_sum = 0.0
    density_sum = 0.0
    similarity_sum = 0.0
    pair_count_sum = 0
    induced_edge_sum = 0
    same_label_pair_sum = 0
    similarity_pair_sum = 0.0
    exact_cliques = 0
    covered: set[int] = set()
    pure_outputs = {cutoff: 0 for cutoff in PURITY_CUTOFFS}
    pure_covered = {cutoff: set() for cutoff in PURITY_CUTOFFS}
    seen_outputs: set[tuple[int, ...]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            if row_format == "solution-prefix":
                if not line.startswith("Sol:"):
                    continue
                line = line.removeprefix("Sol:")
            values = [int(value) for value in line.split()]
            if row_format == "size-prefix":
                declared_size, vertices = values[0], values[1:]
                if declared_size != len(vertices):
                    raise ValueError(f"Invalid FastQC row at {path}:{line_number}")
            else:
                vertices = values
            if len(vertices) < MIN_SIZE:
                continue
            if len(vertices) != len(set(vertices)):
                raise ValueError(f"Repeated vertex at {path}:{line_number}")
            if min(vertices) < 0 or max(vertices) >= len(labels):
                raise ValueError(f"Vertex outside label domain at {path}:{line_number}")
            canonical = tuple(sorted(vertices))
            if canonical in seen_outputs:
                raise ValueError(f"Duplicate output at {path}:{line_number}")
            seen_outputs.add(canonical)

            counts = Counter(labels[vertex] for vertex in vertices)
            purity = max(counts.values()) / len(vertices)
            same_label_pairs = sum(
                count * (count - 1) // 2 for count in counts.values()
            )
            possible_edges = len(vertices) * (len(vertices) - 1) // 2
            induced_degrees = [0] * len(vertices)
            induced_edges = 0
            for index, u in enumerate(vertices):
                for other_index in range(index + 1, len(vertices)):
                    if vertices[other_index] in adjacency[u]:
                        induced_edges += 1
                        induced_degrees[index] += 1
                        induced_degrees[other_index] += 1

            if method == "FastQC":
                required_degree = math.ceil(
                    FASTQC_GAMMA * (len(vertices) - 1)
                )
                if min(induced_degrees) < required_degree:
                    raise ValueError(
                        f"FastQC degree constraint failed at {path}:{line_number}"
                    )
            elif method == "FaPlex":
                if min(induced_degrees) < len(vertices) - FAPLEX_K:
                    raise ValueError(
                        f"FaPlex {FAPLEX_K}-plex constraint failed at "
                        f"{path}:{line_number}"
                    )
            elif induced_edges != possible_edges:
                raise ValueError(
                    f"Non-clique output for {method} at {path}:{line_number}"
                )

            density = induced_edges / possible_edges
            vector_sum = vectors[
                np.asarray(vertices, dtype=np.int64)
            ].sum(axis=0, dtype=np.float64)
            similarity = (
                float(vector_sum @ vector_sum) - len(vertices)
            ) / (2.0 * possible_edges)
            retained += 1
            purity_sum += purity
            density_sum += density
            similarity_sum += similarity
            pair_count_sum += possible_edges
            induced_edge_sum += induced_edges
            same_label_pair_sum += same_label_pairs
            similarity_pair_sum += similarity * possible_edges
            exact_cliques += induced_edges == possible_edges
            covered.update(vertices)
            for cutoff in PURITY_CUTOFFS:
                if purity >= cutoff:
                    pure_outputs[cutoff] += 1
                    pure_covered[cutoff].update(vertices)
    if retained == 0:
        raise ValueError(f"No output of size at least {MIN_SIZE}: {path}")
    result: dict[str, float | int] = {
        "outputs": retained,
        "mean_purity": purity_sum / retained,
        "mean_density": density_sum / retained,
        "mean_similarity": similarity_sum / retained,
        "pair_weighted_density": induced_edge_sum / pair_count_sum,
        "pair_weighted_similarity": similarity_pair_sum / pair_count_sum,
        "pair_weighted_label_agreement": same_label_pair_sum / pair_count_sum,
        "exact_clique_fraction": exact_cliques / retained,
        "covered_vertices": len(covered),
        "coverage_fraction": len(covered) / len(labels),
    }
    for cutoff in PURITY_CUTOFFS:
        suffix = str(int(cutoff * 100))
        result[f"pure{suffix}_outputs"] = pure_outputs[cutoff]
        result[f"pure{suffix}_covered_vertices"] = len(pure_covered[cutoff])
        result[f"pure{suffix}_coverage_fraction"] = (
            len(pure_covered[cutoff]) / len(labels)
        )
    return result


def main() -> None:
    datasets = [
        (
            "GitHub-Social",
            github.load_labels(),
            load_graph(ROOT / "dataset" / "dataset" / "github-social.txt"),
            ROOT / "dataset" / "vectors" / "github-social_vectors.bin",
            load_outputs(github, "github-social"),
        ),
        (
            "ogbn-arxiv-25K",
            ogbn.load_labels(),
            load_graph(
                ROOT / "dataset" / "scale_datasets" / "ogbn-arxiv-degree-prefix"
                / "level-25000.graph.txt"
            ),
            ROOT
            / "dataset"
            / "scale_datasets"
            / "ogbn-arxiv-degree-prefix"
            / "level-25000.vectors.bin",
            load_outputs(ogbn, "ogbn-arxiv-25k"),
        ),
    ]
    rows: list[dict[str, object]] = []
    provenance_inputs: dict[str, dict[str, object]] = {}
    for dataset, labels, adjacency, vector_path, outputs in datasets:
        provenance_inputs[dataset] = {}
        vectors = load_vectors(vector_path)
        if len(vectors) != len(labels):
            raise ValueError(f"Vector/label mismatch for {dataset}")
        for method in ("StructBK", "PairSim-MCE", "FastQC", "FaPlex", "MSC"):
            path, row_format = outputs[method]
            row: dict[str, object] = {"dataset": dataset, "method": method}
            row.update(
                analyze(
                    path, labels, adjacency, row_format, method, vectors
                )
            )
            rows.append(row)
            provenance_inputs[dataset][method] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
            }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "schema_version": 1,
        "comparison": {
            "datasets": ["GitHub-Social", "ogbn-arxiv-25K"],
            "msc_and_pairsim_quantile": 80,
            "minimum_output_size": MIN_SIZE,
            "fastqc_gamma": FASTQC_GAMMA,
            "fastqc_source_commit": "488af494dcc9fa23093b99d36c639453e7c24ced",
            "fastqc_output_mapping_patch": (
                "experiments/provenance/fastqc-original-id-output.patch"
            ),
            "faplex_k": FAPLEX_K,
            "faplex_source_commit": "1268d83fdf5ed21e3a31afb90f91ff70657fc22c",
        },
        "validation": {
            "all_methods": [
                "unique vertex IDs within each output",
                "unique output vertex sets",
                "vertex IDs within the original graph domain",
            ],
            "StructBK_PairSim_MSC": "every retained output is a clique",
            "FastQC": "minimum induced degree is ceil(0.9 * (size - 1))",
            "FaPlex": "minimum induced degree is at least size - 2",
        },
        "metrics": {
            "pair_weighted_density": (
                "induced edge incidences divided by possible pair incidences"
            ),
            "pair_weighted_similarity": (
                "sum of pairwise cosine similarities divided by pair incidences"
            ),
            "pair_weighted_label_agreement": (
                "same-label pair incidences divided by pair incidences"
            ),
        },
        "inputs": provenance_inputs,
        "output_csv": OUTPUT.relative_to(ROOT).as_posix(),
        "output_csv_sha256": sha256(OUTPUT),
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(OUTPUT.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
