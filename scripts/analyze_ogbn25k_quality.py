#!/usr/bin/env python3
"""Compare four mining objectives on the ogbn-arxiv 25K induced subgraph."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "experiments" / "output" / "ogbn-arxiv-25k-quality-capture"
FASTQC_DIR = ROOT / "experiments" / "output" / "external-baselines" / "ogbn-arxiv-25k" / "fastqc-g090-u10-mapped"
SOURCE = ROOT / "dataset" / "downloads" / "ogbn-arxiv" / "arxiv.zip"
MAPPING = ROOT / "dataset" / "scale_datasets" / "ogbn-arxiv-degree-prefix" / "provenance.json"
OUTPUT_CSV = ROOT / "tex-data" / "data" / "ogbn-arxiv-25k-quality.csv"
PROVENANCE = ROOT / "experiments" / "quality" / "ogbn-arxiv-25k-quality.json"
MIN_SIZE = 10
VERTICES = 25_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels() -> list[int]:
    with zipfile.ZipFile(SOURCE) as archive:
        compressed = archive.read("arxiv/raw/node-label.csv.gz")
    source_labels = [int(value) for value in gzip.decompress(compressed).decode("ascii").splitlines()]
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))["selected_original_ids"]
    if len(source_labels) != 169_343 or len(mapping) != 169_343:
        raise ValueError("Unexpected ogbn-arxiv label or mapping cardinality")
    labels = [source_labels[original] for original in mapping[:VERTICES]]
    if len(labels) != VERTICES or not set(labels).issubset(set(range(40))):
        raise ValueError("The 25K induced graph does not map to official ogbn-arxiv labels")
    return labels


def captured_outputs() -> dict[str, tuple[Path, dict[str, Any]]]:
    outputs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for result_path in RUN_DIR.glob("*.attempt-*.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "completed":
            continue
        task = result["task"]
        graph = task.get("graph") or ""
        if task["algorithm"] == 3:
            name = "MSC"
        elif "pairsim" in graph:
            name = "PairSim-MCE"
        elif task["algorithm"] == 1:
            name = "StructBK"
        else:
            continue
        if name in outputs:
            raise ValueError(f"Duplicate capture for {name}")
        outputs[name] = (Path(result["artifacts"]["cliques"]), result)
    if set(outputs) != {"StructBK", "PairSim-MCE", "MSC"}:
        raise ValueError(f"Incomplete captures: {sorted(outputs)}")
    return outputs


def analyze(path: Path, labels: list[int], size_prefix: bool) -> dict[str, Any]:
    cliques = 0
    all_rows = 0
    size_sum = 0
    purity_sum = 0.0
    agreement_sum = 0.0
    entropy_sum = 0.0
    maximum_size = 0
    covered: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            values = [int(value) for value in line.split()]
            all_rows += 1
            if size_prefix:
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
            counts = Counter(labels[vertex] for vertex in vertices)
            size = len(vertices)
            pair_count = size * (size - 1) // 2
            same_pairs = sum(count * (count - 1) // 2 for count in counts.values())
            probabilities = [count / size for count in counts.values()]
            entropy = -sum(value * math.log(value) for value in probabilities)
            cliques += 1
            size_sum += size
            purity_sum += max(counts.values()) / size
            agreement_sum += same_pairs / pair_count
            entropy_sum += entropy / math.log(40)
            maximum_size = max(maximum_size, size)
            covered.update(vertices)
    if not cliques:
        raise ValueError(f"No output of size at least {MIN_SIZE}: {path}")
    return {
        "all_output_rows": all_rows,
        "cliques": cliques,
        "covered_vertices": len(covered),
        "coverage_fraction": len(covered) / len(labels),
        "mean_size": size_sum / cliques,
        "max_size": maximum_size,
        "mean_purity": purity_sum / cliques,
        "mean_pair_agreement": agreement_sum / cliques,
        "mean_normalized_entropy": entropy_sum / cliques,
    }


def main() -> None:
    labels = load_labels()
    captured = captured_outputs()
    rows: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, Any]] = {}
    for name in ("StructBK", "PairSim-MCE", "FastQC", "MSC"):
        if name == "FastQC":
            path = FASTQC_DIR / "maximal-output.txt"
            result = None
            size_prefix = True
        else:
            path, result = captured[name]
            size_prefix = False
        metrics = analyze(path, labels, size_prefix)
        metrics["method"] = name
        rows.append(metrics)
        inputs[name] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256(path),
            "all_output_rows": metrics["all_output_rows"],
        }
        if result is not None:
            inputs[name]["task_id"] = result["task"]["task_id"]
            inputs[name]["build_fingerprint"] = result["build"]["fingerprint"]

    fields = [
        "method", "cliques", "covered_vertices", "coverage_fraction", "mean_size",
        "max_size", "mean_purity", "mean_pair_agreement", "mean_normalized_entropy",
    ]
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row[field] for field in fields}
            for field in (
                "coverage_fraction", "mean_size", "mean_purity",
                "mean_pair_agreement", "mean_normalized_entropy",
            ):
                output[field] = f"{float(output[field]):.8f}"
            writer.writerow(output)

    fastqc_run = json.loads((FASTQC_DIR / "run.json").read_text(encoding="utf-8"))
    fastqc_post = json.loads((FASTQC_DIR / "postprocess.json").read_text(encoding="utf-8"))
    provenance = {
        "schema_version": 1,
        "dataset": "ogbn-arxiv-25K",
        "source_archive": SOURCE.relative_to(ROOT).as_posix(),
        "source_archive_sha256": sha256(SOURCE),
        "mapping_sha256": sha256(MAPPING),
        "comparison": {
            "quantile": 80,
            "tau": 0.888756,
            "minimum_output_size": MIN_SIZE,
            "fastqc_gamma": 0.9,
            "fastqc_commit": "488af494dcc9fa23093b99d36c639453e7c24ced",
            "fastqc_enumeration_seconds": fastqc_run["wall_seconds"],
            "fastqc_postprocessing_seconds": fastqc_post["wall_seconds"],
        },
        "metrics": {
            "mean_purity": "mean majority-label fraction over outputs",
            "mean_pair_agreement": "mean same-label pair fraction over outputs",
            "coverage_fraction": "vertices covered by retained outputs divided by 25,000",
        },
        "inputs": inputs,
        "output_csv": OUTPUT_CSV.relative_to(ROOT).as_posix(),
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_CSV.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
