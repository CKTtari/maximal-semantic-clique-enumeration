#!/usr/bin/env python3
"""Export GitHub-Social formal trials and replace WN18RR in the paper grid."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "experiment-output" / "github-social-formal"
MAIN_CSV = ROOT / "tex-data" / "data" / "main_runtime.csv"
TRIAL_CSV = ROOT / "tex-data" / "data" / "github_social_runtime_trials.csv"
PROVENANCE = ROOT / "experiments" / "provenance" / "github-social-runtime.json"
QUANTILES = {20: 0.25049, 80: 0.417029, 99: 0.571662}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_results() -> dict[tuple[int, float | None], list[tuple[Path, dict[str, Any]]]]:
    grouped: dict[tuple[int, float | None], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in RUN_DIR.glob("*.attempt-*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "completed":
            raise ValueError(f"Formal run is not complete: {path}")
        task = result["task"]
        grouped[(int(task["algorithm"]), task["tau"])].append((path, result))
    expected = {(1, None)} | {
        (algorithm, tau) for algorithm in (2, 3, 4) for tau in QUANTILES.values()
    }
    if set(grouped) != expected:
        raise ValueError(f"Formal grid mismatch: {sorted(grouped)}")
    for key, records in grouped.items():
        if len(records) != 3:
            raise ValueError(f"Expected three repetitions for {key}, found {len(records)}")
        if len({record[1]["parsed"]["clique_count"] for record in records}) != 1:
            raise ValueError(f"Count mismatch across repetitions for {key}")
    return grouped


def summary(records: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    seconds = sorted(result["parsed"]["algorithm_ms"] / 1000 for _, result in records)
    return {
        "seconds": statistics.median(seconds),
        "min_seconds": min(seconds),
        "max_seconds": max(seconds),
        "memory_gib": max(result["parsed"]["peak_memory_gib"] for _, result in records),
        "count": records[0][1]["parsed"]["clique_count"],
    }


def format_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def main() -> None:
    grouped = load_results()
    trial_fields = [
        "algorithm", "quantile", "tau", "repetition", "seconds", "wall_seconds",
        "peak_memory_gib", "clique_count", "task_id", "build_fingerprint", "result_sha256",
    ]
    TRIAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    provenance_runs = []
    with TRIAL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_fields)
        writer.writeheader()
        for (algorithm, tau), records in sorted(
            grouped.items(), key=lambda item: (item[0][0], -1 if item[0][1] is None else item[0][1])
        ):
            quantile = "" if tau is None else min(QUANTILES, key=lambda q: abs(QUANTILES[q] - tau))
            for path, result in sorted(records, key=lambda record: record[1]["task"]["repetition"]):
                row = {
                    "algorithm": algorithm,
                    "quantile": quantile,
                    "tau": "" if tau is None else tau,
                    "repetition": result["task"]["repetition"],
                    "seconds": f"{result['parsed']['algorithm_ms'] / 1000:.6f}",
                    "wall_seconds": f"{result['wall_seconds']:.6f}",
                    "peak_memory_gib": result["parsed"]["peak_memory_gib"],
                    "clique_count": result["parsed"]["clique_count"],
                    "task_id": result["task"]["task_id"],
                    "build_fingerprint": result["build"]["fingerprint"],
                    "result_sha256": sha256(path),
                }
                writer.writerow(row)
                provenance_runs.append({"path": path.relative_to(ROOT).as_posix(), **row})

    with MAIN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        existing = [row for row in reader if row["dataset"] != "WN18RR"]
    struct = summary(grouped[(1, None)])
    inserted: list[dict[str, str]] = []
    for quantile, tau in QUANTILES.items():
        row = {field: "" for field in fields}
        row.update(
            {
                "dataset": "GitHub-Social",
                "dataset_id": "17",
                "quantile": str(quantile),
                "tau": str(tau),
                "acmsc_count": str(summary(grouped[(3, tau)])["count"]),
                "structural_count": str(struct["count"]),
            }
        )
        for algorithm in (1, 2, 3, 4):
            values = struct if algorithm == 1 else summary(grouped[(algorithm, tau)])
            prefix = f"alg{algorithm}"
            row[f"{prefix}_status"] = "completed"
            row[f"{prefix}_seconds"] = format_number(values["seconds"])
            row[f"{prefix}_min_seconds"] = format_number(values["min_seconds"])
            row[f"{prefix}_max_seconds"] = format_number(values["max_seconds"])
            row[f"{prefix}_memory_gib"] = format_number(values["memory_gib"])
        inserted.append(row)

    order = ["FB15K-237", "GitHub-Social", "ogbn-arxiv", "email-Enron", "sc-nasasrb", "sc-pkustk11", "sc-pwtk", "sc-ldoor"]
    all_rows = existing + inserted
    all_rows.sort(key=lambda row: (order.index(row["dataset"]), int(row["quantile"])))
    with MAIN_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    provenance = {
        "schema_version": 1,
        "dataset": "GitHub-Social",
        "formal_repetitions": 3,
        "thresholds": QUANTILES,
        "runs": provenance_runs,
        "trial_csv": TRIAL_CSV.relative_to(ROOT).as_posix(),
        "trial_csv_sha256": sha256(TRIAL_CSV),
        "main_csv": MAIN_CSV.relative_to(ROOT).as_posix(),
        "main_csv_sha256": sha256(MAIN_CSV),
        "replaced_dataset": "WN18RR",
    }
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Exported GitHub-Social formal medians and updated main_runtime.csv")


if __name__ == "__main__":
    main()
