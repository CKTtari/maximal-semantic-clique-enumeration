#!/usr/bin/env python3
"""Analyze ogbn-arxiv output quality on a fixed threshold grid."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
BASE_RUN = ROOT / "experiments" / "output" / "batch18-ogbn-quality-capture"
ACMSC_SWEEP_RUN = ROOT / "experiments" / "output" / "batch20-ogbn-quality-sweep-acmsc"
PAIRSIM_Q80_RUN = ROOT / "experiments" / "output" / "batch19-ogbn-pairsim-quality-capture"
PAIRSIM_SWEEP_RUN = ROOT / "experiments" / "output" / "batch21-ogbn-pairsim-quality-sweep"
ARCHIVE = ROOT / "dataset" / "downloads" / "ogbn-arxiv" / "arxiv.zip"
VECTOR_FILE = ROOT / "dataset" / "vectors" / "ogbn-arxiv_vectors.bin"
DATA_DIR = ROOT / "tex-data" / "data"
QUALITY_DIR = ROOT / "experiments" / "quality"
QUANTILES = {
    20: 0.773547,
    50: 0.840429,
    80: 0.888756,
    95: 0.924347,
    99: 0.949708,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels() -> tuple[list[int], dict[int, str]]:
    with ZipFile(ARCHIVE) as archive:
        with archive.open("arxiv/raw/node-label.csv.gz") as member:
            with gzip.GzipFile(fileobj=member) as compressed:
                text = io.TextIOWrapper(compressed, encoding="utf-8")
                labels = [int(row[0]) for row in csv.reader(text)]
        with archive.open("arxiv/mapping/labelidx2arxivcategeory.csv.gz") as member:
            with gzip.GzipFile(fileobj=member) as compressed:
                text = io.TextIOWrapper(compressed, encoding="utf-8")
                rows = csv.reader(text)
                next(rows)
                categories = {int(row[0]): row[1] for row in rows}
    return labels, categories


def result_records(run_dir: Path, algorithm: int) -> list[tuple[Path, dict[str, Any]]]:
    records = []
    for result_path in sorted(run_dir.glob("*.attempt-*.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if int(result["task"]["algorithm"]) != algorithm or result["status"] != "completed":
            continue
        clique_path = Path(result["artifacts"]["cliques"])
        if not clique_path.is_absolute():
            clique_path = ROOT / clique_path
        records.append((clique_path.resolve(), result))
    return records


def quantile_for_tau(tau: float) -> int:
    quantile, expected = min(QUANTILES.items(), key=lambda item: abs(item[1] - tau))
    if abs(expected - tau) > 1e-6:
        raise ValueError(f"Unexpected threshold {tau}")
    return quantile


def collect_outputs() -> dict[tuple[str, int | None], tuple[Path, dict[str, Any]]]:
    outputs: dict[tuple[str, int | None], tuple[Path, dict[str, Any]]] = {}
    base_struct = result_records(BASE_RUN, 1)
    if len(base_struct) != 1:
        raise ValueError("Expected exactly one StructBK capture")
    outputs[("StructBK", None)] = base_struct[0]

    for run_dir in (BASE_RUN, ACMSC_SWEEP_RUN):
        for path, result in result_records(run_dir, 3):
            quantile = quantile_for_tau(float(result["task"]["tau"]))
            key = ("ACMSC", quantile)
            if key in outputs:
                raise ValueError(f"Duplicate output for {key}")
            outputs[key] = (path, result)

    graph_pattern = re.compile(r"pairsim-q(\d+)\.txt$")
    for run_dir in (PAIRSIM_Q80_RUN, PAIRSIM_SWEEP_RUN):
        for path, result in result_records(run_dir, 1):
            match = graph_pattern.search(str(result["task"].get("graph", "")))
            if not match:
                raise ValueError(f"Cannot identify PairSim quantile in {result['task']}")
            quantile = int(match.group(1))
            key = ("PairSim-MCE", quantile)
            if key in outputs:
                raise ValueError(f"Duplicate output for {key}")
            outputs[key] = (path, result)

    expected = {
        *(("ACMSC", quantile) for quantile in QUANTILES),
        *(("PairSim-MCE", quantile) for quantile in QUANTILES),
        ("StructBK", None),
    }
    if set(outputs) != expected:
        raise ValueError(f"Missing outputs: {sorted(expected - set(outputs))}")
    return outputs


def size_bin(size: int) -> str:
    if size == 2:
        return "2"
    if size == 3:
        return "3"
    if size <= 5:
        return "4-5"
    return "6+"


def clique_metrics(vertices: list[int], labels: list[int], label_count: int) -> dict[str, float]:
    counts = Counter(labels[vertex] for vertex in vertices)
    size = len(vertices)
    pair_count = size * (size - 1) // 2
    same_label_pairs = sum(count * (count - 1) // 2 for count in counts.values())
    entropy = -sum((count / size) * math.log(count / size) for count in counts.values())
    maximum_entropy = math.log(min(size, label_count))
    return {
        "purity": max(counts.values()) / size,
        "normalized_entropy": entropy / maximum_entropy if maximum_entropy else 0.0,
        "pair_agreement": same_label_pairs / pair_count,
        "pair_count": float(pair_count),
        "same_label_pairs": float(same_label_pairs),
    }


def analyze(path: Path, labels: list[int], label_count: int) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[list[int]]
]:
    by_bin: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_size: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    singleton_cliques = 0
    covered: set[int] = set()
    case_candidates: list[list[int]] = []

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            vertices = [int(value) for value in line.split()]
            if not vertices or len(vertices) != len(set(vertices)):
                raise ValueError(f"Malformed clique at {path}:{line_number}")
            if min(vertices) < 0 or max(vertices) >= len(labels):
                raise ValueError(f"Vertex out of range at {path}:{line_number}")
            if len(vertices) == 1:
                singleton_cliques += 1
                continue

            measured = clique_metrics(vertices, labels, label_count)
            covered.update(vertices)
            if measured["purity"] == 1.0 and len(vertices) >= 4:
                case_candidates.append(sorted(vertices))
            for bucket in (by_bin[size_bin(len(vertices))], by_size[len(vertices)]):
                bucket["cliques"] += 1
                bucket["size_sum"] += len(vertices)
                bucket["purity_sum"] += measured["purity"]
                bucket["entropy_sum"] += measured["normalized_entropy"]
                bucket["agreement_sum"] += measured["pair_agreement"]
                bucket["pair_count"] += measured["pair_count"]
                bucket["same_label_pairs"] += measured["same_label_pairs"]

    def finish(key: str | int, values: dict[str, float]) -> dict[str, Any]:
        count = int(values["cliques"])
        return {
            "group": key,
            "cliques": count,
            "mean_size": values["size_sum"] / count,
            "mean_purity": values["purity_sum"] / count,
            "mean_normalized_entropy": values["entropy_sum"] / count,
            "mean_pair_agreement": values["agreement_sum"] / count,
            "pooled_pair_agreement": values["same_label_pairs"] / values["pair_count"],
        }

    bins = [finish(key, by_bin[key]) for key in ("2", "3", "4-5", "6+") if by_bin[key]]
    sizes = [finish(key, by_size[key]) for key in sorted(by_size)]
    summary_values: dict[str, float] = defaultdict(float)
    for values in by_size.values():
        for key, value in values.items():
            summary_values[key] += value
    summary = finish("all", summary_values)
    summary.update(
        {
            "singleton_cliques": singleton_cliques,
            "covered_vertices": len(covered),
            "coverage_fraction": len(covered) / len(labels),
            "max_size": max(by_size),
        }
    )
    case_candidates.sort(key=lambda clique: (-len(clique), clique))
    return bins, sizes, summary, case_candidates


def load_normalized_vectors(vertices: set[int]) -> dict[int, Any]:
    import numpy as np

    with VECTOR_FILE.open("rb") as stream:
        vertex_count, dimension = struct.unpack("<ii", stream.read(8))
    matrix = np.memmap(VECTOR_FILE, dtype="<f8", mode="r", offset=8, shape=(vertex_count, dimension))
    result: dict[int, Any] = {}
    for vertex in sorted(vertices):
        row = np.asarray(matrix[vertex], dtype=np.float64)
        norm = float(np.linalg.norm(row))
        if norm == 0.0:
            raise ValueError(f"Zero vector for vertex {vertex}")
        result[vertex] = np.asarray(row / norm, dtype=np.float32)
    return result


def select_cases(candidates: list[list[int]], labels: list[int], tau: float, limit: int = 3) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidate_pool = candidates[:2000]
    vectors = load_normalized_vectors({vertex for clique in candidate_pool for vertex in clique})
    for clique in candidate_pool:
        similarities = [
            float(vectors[left].dot(vectors[right]))
            for index, left in enumerate(clique)
            for right in clique[index + 1 :]
        ]
        below = sum(value < tau for value in similarities)
        if below == 0:
            continue
        selected.append(
            {
                "rank": len(selected) + 1,
                "size": len(clique),
                "category_index": labels[clique[0]],
                "vertices": clique,
                "average_similarity": sum(similarities) / len(similarities),
                "minimum_similarity": min(similarities),
                "below_threshold_pairs": below,
                "total_pairs": len(similarities),
            }
        )
        if len(selected) == limit:
            break
    if len(selected) != limit:
        raise ValueError(f"Only found {len(selected)} deterministic ACMSC cases")
    return selected


def write_group_csv(path: Path, rows: list[dict[str, Any]], group_name: str) -> None:
    fields = [
        "algorithm", group_name, "cliques", "mean_size", "mean_purity",
        "mean_normalized_entropy", "mean_pair_agreement", "pooled_pair_agreement",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output[group_name] = output.pop("group")
            for field in fields[3:]:
                output[field] = f"{float(output[field]):.8f}"
            writer.writerow(output)


def write_case_similarities(case: dict[str, Any], tau: float) -> None:
    vertices = case["vertices"]
    vectors = load_normalized_vectors(set(vertices))
    path = DATA_DIR / "ogbn-arxiv-case-similarity.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["left", "right", "similarity", "below_threshold"])
        writer.writeheader()
        for index, left in enumerate(vertices):
            for right in vertices[index + 1 :]:
                similarity = float(vectors[left].dot(vectors[right]))
                writer.writerow(
                    {
                        "left": left,
                        "right": right,
                        "similarity": f"{similarity:.8f}",
                        "below_threshold": int(similarity < tau),
                    }
                )


def main() -> None:
    labels, categories = load_labels()
    if len(labels) != 169343 or set(labels) - set(categories):
        raise ValueError("Official label rows do not match ogbn-arxiv vertices")
    outputs = collect_outputs()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    q80_bins: list[dict[str, Any]] = []
    q80_sizes: list[dict[str, Any]] = []
    provenance_outputs: dict[str, Any] = {}
    acmsc_q80_candidates: list[list[int]] = []

    ordered_keys = [("StructBK", None)] + [
        (algorithm, quantile)
        for algorithm in ("PairSim-MCE", "ACMSC")
        for quantile in QUANTILES
    ]
    for algorithm, quantile in ordered_keys:
        clique_path, result = outputs[(algorithm, quantile)]
        bins, sizes, summary, candidates = analyze(clique_path, labels, len(categories))
        summary.update(
            {
                "algorithm": algorithm,
                "quantile": "" if quantile is None else quantile,
                "tau": "" if quantile is None else QUANTILES[quantile],
            }
        )
        summaries.append(summary)
        if quantile == 80 or quantile is None:
            for row in bins:
                row["algorithm"] = algorithm
            for row in sizes:
                row["algorithm"] = algorithm
            q80_bins.extend(bins)
            q80_sizes.extend(sizes)
        if algorithm == "ACMSC" and quantile == 80:
            acmsc_q80_candidates = candidates

        expected = int(result["parsed"]["clique_count"]) - int(summary["singleton_cliques"])
        if int(summary["cliques"]) != expected:
            raise ValueError(f"{algorithm} q{quantile}: captured {summary['cliques']}, expected {expected}")
        key = algorithm if quantile is None else f"{algorithm}-q{quantile}"
        provenance_outputs[key] = {
            "cliques": int(summary["cliques"]),
            "path": clique_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(clique_path),
            "task_id": result["task"]["task_id"],
            "build_fingerprint": result["build"]["fingerprint"],
        }

    write_group_csv(DATA_DIR / "result_quality.csv", q80_bins, "size_bin")
    write_group_csv(DATA_DIR / "result_quality_by_size.csv", q80_sizes, "size")
    summary_fields = [
        "algorithm", "quantile", "tau", "cliques", "singleton_cliques",
        "covered_vertices", "coverage_fraction", "mean_size", "max_size",
        "mean_purity", "mean_normalized_entropy", "mean_pair_agreement",
        "pooled_pair_agreement",
    ]
    with (DATA_DIR / "result_quality_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for row in summaries:
            output = {field: row[field] for field in summary_fields}
            for field in (
                "coverage_fraction", "mean_size", "mean_purity",
                "mean_normalized_entropy", "mean_pair_agreement", "pooled_pair_agreement",
            ):
                output[field] = f"{float(output[field]):.8f}"
            writer.writerow(output)

    q80_rows = [row for row in summaries if row["quantile"] in ("", 80)]
    with (DATA_DIR / "result_quality_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for row in q80_rows:
            writer.writerow({field: row[field] for field in summary_fields})

    cases = select_cases(acmsc_q80_candidates, labels, QUANTILES[80])
    for case in cases:
        case["category"] = categories[case["category_index"]]
    (QUALITY_DIR / "ogbn-arxiv-cases.json").write_text(
        json.dumps(cases, indent=2) + "\n", encoding="utf-8"
    )
    write_case_similarities(cases[0], QUANTILES[80])

    pairsim_graphs = {}
    for quantile, tau in QUANTILES.items():
        path = ROOT / "dataset" / "derived" / f"ogbn-arxiv-pairsim-q{quantile}.txt"
        vertices, retained_edges = map(int, path.open("r", encoding="utf-8").readline().split())
        pairsim_graphs[f"q{quantile}"] = {
            "tau": tau,
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "vertices": vertices,
            "retained_edges": retained_edges,
        }
    provenance = {
        "schema_version": 2,
        "dataset": "ogbn-arxiv",
        "threshold_grid": [{"quantile": q, "tau": tau} for q, tau in QUANTILES.items()],
        "labels": {
            "vertices": len(labels),
            "categories": len(categories),
            "source_archive": ARCHIVE.relative_to(ROOT).as_posix(),
            "source_archive_sha256": sha256(ARCHIVE),
        },
        "metrics": {
            "coverage_fraction": "vertices in at least one non-singleton output divided by graph vertices",
            "pooled_pair_agreement": "same-label pair incidences divided by all pair incidences over non-singleton outputs",
        },
        "outputs": provenance_outputs,
        "pairsim_graphs": pairsim_graphs,
        "case_selection": "largest label-pure q80 ACMSCs, lexicographic tie break, requiring at least one pair below q80",
    }
    provenance_path = QUALITY_DIR / "ogbn-arxiv-result-quality.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DATA_DIR / 'result_quality_sweep.csv'}")
    print(f"Wrote {DATA_DIR / 'ogbn-arxiv-case-similarity.csv'}")
    print(f"Wrote {QUALITY_DIR / 'ogbn-arxiv-cases.json'}")
    print(f"Wrote {provenance_path}")


if __name__ == "__main__":
    main()
