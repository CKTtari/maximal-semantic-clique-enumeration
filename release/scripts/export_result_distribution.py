#!/usr/bin/env python3
"""Export normalized ACMSC output distributions from archived exact runs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU_FILE = ROOT / "scripts" / "tau_values.json"
DATA_FILE = ROOT / "tex-data" / "data" / "result_distribution.csv"
PROVENANCE_FILE = ROOT / "experiments" / "quality" / "result-distribution.json"
QUANTILES = (20, 50, 80, 95, 99)
DATASETS = {
    7: "FB15K-237",
    11: "ogbn-arxiv",
    2: "sc-nasasrb",
    1: "sc-ldoor",
}
ACMSC_MANIFESTS = [
    "batch11-main-core-current.manifest.json",
    "batch12-threshold-fb15k-current.manifest.json",
    "batch12-threshold-ogbn-current.manifest.json",
    "batch14-threshold-ldoor-formal-repeats.manifest.json",
    "batch11-main-alg3-ldoor-q80-repeat.manifest.json",
    "batch2-main-structural-uniform-ds2.manifest.json",
    "batch22-nasasrb-result-distribution-q95.manifest.json",
]
STRUCTURAL_MANIFESTS = [
    "batch13-main-structbk-formal-current.manifest.json",
    "batch18-ogbn-quality-capture.manifest.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean_size(histogram: dict[str, int]) -> float:
    count = sum(histogram.values())
    return sum(int(size) * value for size, value in histogram.items()) / count


def median_size(histogram: dict[str, int]) -> int:
    target = (sum(histogram.values()) + 1) // 2
    cumulative = 0
    for size in sorted(map(int, histogram)):
        cumulative += histogram[str(size)]
        if cumulative >= target:
            return size
    raise AssertionError("Empty histogram")


def load_records(names: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    hashes = {}
    for name in names:
        path = ROOT / "experiments" / "archive" / name
        hashes[path.relative_to(ROOT).as_posix()] = sha256(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for record in manifest["records"]:
            if record["status"] == "completed" and record["task"]["mode"] == "raw":
                records.append(record)
    return records, hashes


def matching_records(
    records: list[dict[str, Any]], algorithm: int, dataset: int, tau: float | None
) -> list[dict[str, Any]]:
    matched = []
    for record in records:
        task = record["task"]
        if int(task["algorithm"]) != algorithm or int(task["dataset"]) != dataset:
            continue
        if task.get("graph") is not None or task.get("vectors") is not None:
            continue
        if tau is not None and abs(float(task["tau"]) - tau) > 5e-7:
            continue
        matched.append(record)
    return matched


def select_consistent(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not records:
        raise ValueError(f"No archived result for {label}")
    signatures = {
        (
            int(record["parsed"]["clique_count"]),
            json.dumps(record["parsed"]["size_histogram"], sort_keys=True),
        )
        for record in records
    }
    if len(signatures) != 1:
        raise ValueError(f"Inconsistent archived results for {label}: {signatures}")
    return records[0]


def record_source(record: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / record["result_json"]["path"]
    return {
        "result_json": record["result_json"]["path"],
        "result_json_sha256": sha256(path),
        "task_id": record["task"]["task_id"],
        "build_fingerprint": record["build"]["fingerprint"],
        "source_verification": record["build"]["source_verification"]["status"],
    }


def main() -> None:
    tau_values = json.loads(TAU_FILE.read_text(encoding="utf-8"))
    acmsc_records, acmsc_hashes = load_records(ACMSC_MANIFESTS)
    structural_records, structural_hashes = load_records(STRUCTURAL_MANIFESTS)
    rows = []
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "datasets": list(DATASETS.values()),
        "quantiles": list(QUANTILES),
        "manifests": {**acmsc_hashes, **structural_hashes},
        "sources": {},
        "normalization": {
            "count_ratio": "ACMSC count divided by structural maximal-clique count on the same graph",
            "mean_size_ratio": "mean ACMSC size divided by mean structural maximal-clique size on the same graph",
        },
    }

    for dataset, name in DATASETS.items():
        structural = select_consistent(
            matching_records(structural_records, 1, dataset, None),
            f"StructBK {name}",
        )
        structural_hist = {
            str(key): int(value) for key, value in structural["parsed"]["size_histogram"].items()
        }
        structural_count = int(structural["parsed"]["clique_count"])
        structural_mean = mean_size(structural_hist)
        structural_median = median_size(structural_hist)
        provenance["sources"][f"{name}-StructBK"] = record_source(structural)

        for quantile in QUANTILES:
            tau = float(tau_values[str(dataset)][f"tau_{quantile}"])
            candidates = matching_records(acmsc_records, 3, dataset, tau)
            record = select_consistent(candidates, f"StrSub {name} q{quantile}")
            histogram = {
                str(key): int(value) for key, value in record["parsed"]["size_histogram"].items()
            }
            acmsc_count = int(record["parsed"]["clique_count"])
            acmsc_mean = mean_size(histogram)
            rows.append(
                {
                    "dataset": name,
                    "dataset_id": dataset,
                    "quantile": quantile,
                    "tau": tau,
                    "acmsc_count": acmsc_count,
                    "structural_count": structural_count,
                    "count_ratio": acmsc_count / structural_count,
                    "mean_size": acmsc_mean,
                    "structural_mean_size": structural_mean,
                    "mean_size_ratio": acmsc_mean / structural_mean,
                    "median_size": median_size(histogram),
                    "structural_median_size": structural_median,
                    "max_size": max(map(int, histogram)),
                }
            )
            provenance["sources"][f"{name}-q{quantile}"] = record_source(record)

    for quantile in QUANTILES:
        quantile_rows = [row for row in rows if row["quantile"] == quantile]
        rows.append(
            {
                "dataset": "Geometric mean",
                "dataset_id": "",
                "quantile": quantile,
                "tau": "",
                "acmsc_count": "",
                "structural_count": "",
                "count_ratio": math.prod(row["count_ratio"] for row in quantile_rows) ** (1 / len(quantile_rows)),
                "mean_size": "",
                "structural_mean_size": "",
                "mean_size_ratio": math.prod(row["mean_size_ratio"] for row in quantile_rows) ** (1 / len(quantile_rows)),
                "median_size": "",
                "structural_median_size": "",
                "max_size": "",
            }
        )

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with DATA_FILE.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for field in ("count_ratio", "mean_size", "structural_mean_size", "mean_size_ratio"):
                if output[field] != "":
                    output[field] = f"{float(output[field]):.8f}"
            writer.writerow(output)
    PROVENANCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE_FILE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DATA_FILE}")
    print(f"Wrote {PROVENANCE_FILE}")


if __name__ == "__main__":
    main()
