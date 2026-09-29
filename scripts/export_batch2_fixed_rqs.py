#!/usr/bin/env python3
"""Export batch-2 threshold and output-space data for the fixed four graphs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TAUS = json.loads((ROOT / "scripts" / "tau_values.json").read_text(encoding="utf-8"))
THRESHOLD_CSV = ROOT / "tex-data" / "data" / "threshold_sensitivity.csv"
OUTPUT_CSV = ROOT / "tex-data" / "data" / "result_distribution.csv"
MAIN_CSV = ROOT / "tex-data" / "data" / "main_runtime.csv"
ADMISSION = ROOT / "experiments" / "output" / "batch-rq2-github-nasasrb-admission-once"
LONG = ROOT / "experiments" / "output" / "batch-rq2-longpoints-once"
GITHUB_FORMAL = ROOT / "experiments" / "output" / "github-social-formal"
PROVENANCE = ROOT / "experiments" / "quality" / "batch2-fixed-rq-data.json"
QUANTILES = (5, 20, 50, 70, 80, 90, 95, 97, 99)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def results(path: Path) -> list[tuple[Path, dict]]:
    return [(item, json.loads(item.read_text(encoding="utf-8"))) for item in path.glob("*.attempt-*.json")]


def find(records: list[tuple[Path, dict]], algorithm: int, dataset: int, tau: float, status: str | None = None) -> tuple[Path, dict]:
    matched = [
        item for item in records
        if int(item[1]["task"]["algorithm"]) == algorithm
        and int(item[1]["task"]["dataset"]) == dataset
        and abs(float(item[1]["task"]["tau"]) - tau) < 5e-7
        and (status is None or item[1]["status"] == status)
    ]
    if not matched:
        raise ValueError((algorithm, dataset, tau, status))
    return matched[0]


def main_row(dataset: str, quantile: int) -> dict[str, str]:
    return next(row for row in read_csv(MAIN_CSV) if row["dataset"] == dataset and int(row["quantile"]) == quantile)


def mean_size(histogram: dict[str, int]) -> float:
    total = sum(int(value) for value in histogram.values())
    return sum(int(size) * int(value) for size, value in histogram.items()) / total


def median_size(histogram: dict[str, int]) -> int:
    target = (sum(int(value) for value in histogram.values()) + 1) // 2
    cumulative = 0
    for size in sorted(map(int, histogram)):
        cumulative += int(histogram[str(size)])
        if cumulative >= target:
            return size
    raise ValueError("empty histogram")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export_thresholds() -> list[Path]:
    existing = [row for row in read_csv(THRESHOLD_CSV) if row["dataset"] not in {"GitHub-Social", "sc-nasasrb"}]
    admission = results(ADMISSION)
    long_runs = results(LONG)
    sources: list[Path] = []
    rows: list[dict[str, str | int | float]] = []
    datasets = ((17, "GitHub-Social"), (2, "sc-nasasrb"))
    for dataset_id, dataset_name in datasets:
        for quantile in QUANTILES:
            tau = float(TAUS[str(dataset_id)][f"tau_{quantile}"])
            for algorithm, algorithm_name in ((3, "StrSub"), (4, "MonoSemMCE")):
                if quantile in {20, 80, 99}:
                    base = main_row(dataset_name, quantile)
                    status = base[f"alg{algorithm}_status"]
                    seconds = base[f"alg{algorithm}_seconds"]
                    count = base["acmsc_count"] if status == "completed" else ""
                elif dataset_id == 2 and algorithm == 4 and quantile == 5:
                    status, seconds, count = "TLE", "", ""
                else:
                    pool = long_runs if (
                        (dataset_id == 17 and algorithm == 4 and quantile in {5, 50, 70})
                        or (dataset_id == 2 and quantile in {50, 70} and algorithm == 3)
                        or (dataset_id == 2 and quantile == 50 and algorithm == 4)
                    ) else admission
                    path, record = find(pool, algorithm, dataset_id, tau)
                    sources.append(path)
                    status = "TLE" if record["status"] == "tle" else record["status"]
                    seconds = f"{record['parsed']['algorithm_ms'] / 1000:.3f}" if record["status"] == "completed" else ""
                    count = str(record["parsed"]["clique_count"]) if record["status"] == "completed" else ""
                rows.append({
                    "dataset": dataset_name,
                    "dataset_id": dataset_id,
                    "quantile": quantile,
                    "tau": tau,
                    "algorithm": algorithm_name,
                    "algorithm_id": algorithm,
                    "status": status,
                    "seconds": seconds,
                    "acmsc_count": count,
                })
    combined = existing + [{key: str(value) for key, value in row.items()} for row in rows]
    order = {"GitHub-Social": 0, "ogbn-arxiv": 1, "sc-nasasrb": 2, "sc-ldoor": 3}
    combined.sort(key=lambda row: (order.get(row["dataset"], 99), int(row["quantile"]), int(row["algorithm_id"])))
    with THRESHOLD_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    return sources


def export_output_space() -> list[Path]:
    old = [row for row in read_csv(OUTPUT_CSV) if row["dataset"] not in {"FB15K-237", "GitHub-Social", "Geometric mean"}]
    formal = results(GITHUB_FORMAL)
    admission = results(ADMISSION)
    struct_path, structural = next(item for item in formal if int(item[1]["task"]["algorithm"]) == 1)
    structural_hist = {str(key): int(value) for key, value in structural["parsed"]["size_histogram"].items()}
    structural_count = int(structural["parsed"]["clique_count"])
    structural_mean = mean_size(structural_hist)
    rows: list[dict[str, str | int | float]] = []
    sources = [struct_path]
    for quantile in (20, 50, 80, 95, 99):
        tau = float(TAUS["17"][f"tau_{quantile}"])
        pool = formal if quantile in {20, 80, 99} else admission
        path, record = find(pool, 3, 17, tau, "completed")
        sources.append(path)
        histogram = {str(key): int(value) for key, value in record["parsed"]["size_histogram"].items()}
        count = int(record["parsed"]["clique_count"])
        avg = mean_size(histogram)
        rows.append({
            "dataset": "GitHub-Social", "dataset_id": 17, "quantile": quantile, "tau": tau,
            "acmsc_count": count, "structural_count": structural_count,
            "count_ratio": count / structural_count, "mean_size": avg,
            "structural_mean_size": structural_mean, "mean_size_ratio": avg / structural_mean,
            "median_size": median_size(histogram), "structural_median_size": median_size(structural_hist),
            "max_size": max(map(int, histogram)),
        })
    all_rows: list[dict] = [*old, *rows]
    for quantile in (20, 50, 80, 95, 99):
        selected = [row for row in all_rows if int(row["quantile"]) == quantile]
        all_rows.append({
            "dataset": "Geometric mean", "dataset_id": "", "quantile": quantile, "tau": "",
            "acmsc_count": "", "structural_count": "",
            "count_ratio": math.prod(float(row["count_ratio"]) for row in selected) ** 0.25,
            "mean_size": "", "structural_mean_size": "",
            "mean_size_ratio": math.prod(float(row["mean_size_ratio"]) for row in selected) ** 0.25,
            "median_size": "", "structural_median_size": "", "max_size": "",
        })
    order = {"GitHub-Social": 0, "ogbn-arxiv": 1, "sc-nasasrb": 2, "sc-ldoor": 3, "Geometric mean": 4}
    all_rows.sort(key=lambda row: (order[row["dataset"]], int(row["quantile"])))
    fields = list(all_rows[0])
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for raw in all_rows:
            row = dict(raw)
            for field in ("count_ratio", "mean_size", "structural_mean_size", "mean_size_ratio"):
                if row[field] != "":
                    row[field] = f"{float(row[field]):.8f}"
            writer.writerow(row)
    return sources


if __name__ == "__main__":
    sources = export_thresholds() + export_output_space()
    unique = sorted(set(sources))
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps({
        "schema_version": 1,
        "purpose": "Fixed-four-dataset RQ2 and RQ6 export",
        "datasets": ["GitHub-Social", "ogbn-arxiv", "sc-nasasrb", "sc-ldoor"],
        "sources": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)} for path in unique],
        "derived_files": [THRESHOLD_CSV.relative_to(ROOT).as_posix(), OUTPUT_CSV.relative_to(ROOT).as_posix()],
        "note": "One completed run is used for newly added batch-2 configurations; existing paper rows retain their archived values.",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {THRESHOLD_CSV}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {PROVENANCE}")
