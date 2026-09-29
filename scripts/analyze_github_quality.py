#!/usr/bin/env python3
"""Compare output quality on GitHub-Social under a shared size threshold."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "experiments" / "output" / "github-social-quality-capture"
FASTQC_DIR = ROOT / "experiments" / "output" / "external-baselines" / "github-social" / "fastqc-g090-u10-mapped"
LABELS = ROOT / "dataset" / "metadata" / "github-social_labels.csv"
SOURCE = ROOT / "dataset" / "downloads" / "github-social.zip"
OUTPUT_CSV = ROOT / "tex-data" / "data" / "github-social-quality.csv"
PROVENANCE = ROOT / "experiments" / "quality" / "github-social-quality.json"
MIN_SIZE = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_labels() -> list[int]:
    rows = list(csv.DictReader(LABELS.open(encoding="utf-8")))
    rows.sort(key=lambda row: int(row["vertex"]))
    labels = [int(row["label"]) for row in rows]
    if len(labels) != 37_700 or set(labels) != {0, 1}:
        raise ValueError("GitHub-Social labels do not match the source dataset")
    return labels


def captured_outputs() -> dict[str, tuple[Path, dict[str, Any]]]:
    outputs: dict[str, tuple[Path, dict[str, Any]]] = {}
    for result_path in RUN_DIR.glob("*.attempt-*.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["status"] != "completed":
            continue
        task = result["task"]
        if task["algorithm"] == 3:
            name = "MSC"
        elif task["algorithm"] == 1 and task.get("graph"):
            name = "PairSim-MCE"
        elif task["algorithm"] == 1:
            name = "StructBK"
        else:
            continue
        if name in outputs:
            raise ValueError(f"Duplicate capture for {name}")
        clique_path = Path(result["artifacts"]["cliques"])
        outputs[name] = (clique_path, result)
    if set(outputs) != {"StructBK", "PairSim-MCE", "MSC"}:
        raise ValueError(f"Incomplete captures: {sorted(outputs)}")
    return outputs


def analyze(path: Path, labels: list[int], size_prefix: bool) -> dict[str, Any]:
    clique_count = 0
    all_rows = 0
    size_sum = 0
    purity_sum = 0.0
    agreement_sum = 0.0
    entropy_sum = 0.0
    covered: set[int] = set()
    maximum_size = 0
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
            if len(vertices) != len(set(vertices)):
                raise ValueError(f"Repeated vertex at {path}:{line_number}")
            if len(vertices) < MIN_SIZE:
                continue
            if min(vertices) < 0 or max(vertices) >= len(labels):
                raise ValueError(f"Vertex outside label domain at {path}:{line_number}")
            counts = Counter(labels[vertex] for vertex in vertices)
            size = len(vertices)
            pair_count = size * (size - 1) // 2
            same_pairs = sum(count * (count - 1) // 2 for count in counts.values())
            probabilities = [count / size for count in counts.values()]
            entropy = -sum(value * math.log(value) for value in probabilities)
            clique_count += 1
            size_sum += size
            purity_sum += max(counts.values()) / size
            agreement_sum += same_pairs / pair_count
            entropy_sum += entropy / math.log(2)
            covered.update(vertices)
            maximum_size = max(maximum_size, size)
    if not clique_count:
        raise ValueError(f"No output of size at least {MIN_SIZE}: {path}")
    return {
        "all_output_rows": all_rows,
        "cliques": clique_count,
        "mean_size": size_sum / clique_count,
        "max_size": maximum_size,
        "mean_purity": purity_sum / clique_count,
        "mean_pair_agreement": agreement_sum / clique_count,
        "mean_normalized_entropy": entropy_sum / clique_count,
        "covered_vertices": len(covered),
        "coverage_fraction": len(covered) / len(labels),
    }


def main() -> None:
    labels = load_labels()
    captured = captured_outputs()
    inputs: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
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

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "cliques", "covered_vertices", "coverage_fraction", "mean_size",
        "max_size", "mean_purity", "mean_pair_agreement", "mean_normalized_entropy",
    ]
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
        "dataset": "GitHub-Social",
        "source_archive": SOURCE.relative_to(ROOT).as_posix(),
        "source_archive_sha256": sha256(SOURCE),
        "labels_sha256": sha256(LABELS),
        "comparison": {
            "quantile": 80,
            "tau": 0.417029,
            "minimum_output_size": MIN_SIZE,
            "fastqc_gamma": 0.9,
            "fastqc_commit": "488af494dcc9fa23093b99d36c639453e7c24ced",
            "fastqc_enumeration_seconds": fastqc_run["wall_seconds"],
            "fastqc_postprocessing_seconds": fastqc_post["wall_seconds"],
        },
        "metrics": {
            "mean_purity": "mean majority-label fraction over outputs",
            "mean_pair_agreement": "mean same-label pair fraction over outputs",
            "coverage_fraction": "vertices covered by retained outputs divided by 37,700",
        },
        "inputs": inputs,
        "output_csv": OUTPUT_CSV.relative_to(ROOT).as_posix(),
        "output_csv_sha256": sha256(OUTPUT_CSV),
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_CSV.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
