#!/usr/bin/env python3
"""Evaluate RQ5 output-family preservation with published cover metrics."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "quality" / "multi-baseline-quality.json"
QUALITY_CSV = ROOT / "tex-data" / "data" / "quality-comparison.csv"
SUMMARY_CSV = ROOT / "tex-data" / "data" / "group-recovery.csv"
DETAIL_CSV = ROOT / "tex-data" / "data" / "group-recovery-detail.csv"
PROVENANCE = ROOT / "experiments" / "quality" / "group-recovery.json"
REFERENCE_MIN_SIZE = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_groups(path: Path, row_format: str) -> Iterator[tuple[int, ...]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if row_format == "solution-prefix":
                if not line.startswith("Sol:"):
                    continue
                line = line.removeprefix("Sol:").strip()
            values = tuple(int(value) for value in line.split())
            if row_format == "size-prefix":
                if not values or values[0] != len(values) - 1:
                    raise ValueError(f"Invalid size prefix at {path}:{line_number}")
                values = values[1:]
            if len(values) != len(set(values)):
                raise ValueError(f"Repeated vertex at {path}:{line_number}")
            yield values


def pair_key(left: int, right: int) -> int:
    if left > right:
        left, right = right, left
    return (left << 32) | right


def add_pair_counts(counts: dict[int, int], vertices: Iterable[int]) -> None:
    ordered = sorted(vertices)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index:]:
            key = pair_key(left, right)
            counts[key] = counts.get(key, 0) + 1


def extended_bcubed(
    universe: set[int], gold_pairs: dict[int, int], predicted_pairs: dict[int, int]
) -> tuple[float, float, float]:
    recall_sum = {vertex: 0.0 for vertex in universe}
    recall_n = {vertex: 0 for vertex in universe}
    precision_sum = {vertex: 0.0 for vertex in universe}
    precision_n = {vertex: 0 for vertex in universe}
    for key, gold_count in gold_pairs.items():
        left, right = key >> 32, key & 0xFFFFFFFF
        value = min(predicted_pairs.get(key, 0), gold_count) / gold_count
        recall_sum[left] += value
        recall_n[left] += 1
        if right != left:
            recall_sum[right] += value
            recall_n[right] += 1
    for key, predicted_count in predicted_pairs.items():
        left, right = key >> 32, key & 0xFFFFFFFF
        value = min(predicted_count, gold_pairs.get(key, 0)) / predicted_count
        precision_sum[left] += value
        precision_n[left] += 1
        if right != left:
            precision_sum[right] += value
            precision_n[right] += 1
    recall = sum(recall_sum[v] / recall_n[v] for v in universe) / len(universe)
    precision = sum(
        precision_sum[v] / precision_n[v] if precision_n[v] else 0.0
        for v in universe
    ) / len(universe)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_cover(
    reference_path: Path,
    candidate_path: Path,
    candidate_format: str,
    candidate_min_size: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    references = [
        group for group in iter_groups(reference_path, "plain")
        if len(group) >= REFERENCE_MIN_SIZE
    ]
    if not references:
        raise ValueError(f"No reference groups of size {REFERENCE_MIN_SIZE}: {reference_path}")
    universe = set().union(*(set(group) for group in references))
    memberships: dict[int, list[int]] = {vertex: [] for vertex in universe}
    gold_pairs: dict[int, int] = {}
    for reference_id, group in enumerate(references):
        add_pair_counts(gold_pairs, group)
        for vertex in group:
            memberships[vertex].append(reference_id)
    predicted_pairs: dict[int, int] = {}
    best_recall = [0.0] * len(references)
    best_jaccard = [0.0] * len(references)
    best_f1 = [0.0] * len(references)
    candidate_groups = 0
    candidate_best_f1_sum = 0.0
    for candidate in iter_groups(candidate_path, candidate_format):
        if len(candidate) < candidate_min_size:
            continue
        candidate_groups += 1
        projected = [vertex for vertex in candidate if vertex in universe]
        if projected:
            add_pair_counts(predicted_pairs, projected)
        overlaps: Counter[int] = Counter()
        for vertex in projected:
            overlaps.update(memberships[vertex])
        candidate_best = 0.0
        for reference_id, intersection in overlaps.items():
            reference_size = len(references[reference_id])
            recall = intersection / reference_size
            jaccard = intersection / (reference_size + len(candidate) - intersection)
            f1 = 2 * intersection / (reference_size + len(candidate))
            best_recall[reference_id] = max(best_recall[reference_id], recall)
            best_jaccard[reference_id] = max(best_jaccard[reference_id], jaccard)
            best_f1[reference_id] = max(best_f1[reference_id], f1)
            candidate_best = max(candidate_best, f1)
        candidate_best_f1_sum += candidate_best
    if candidate_groups == 0:
        raise ValueError(f"No candidate groups of size {candidate_min_size}: {candidate_path}")
    for vertex in universe:
        key = pair_key(vertex, vertex)
        if key not in predicted_pairs:
            predicted_pairs[key] = 1
    precision, recall, f1 = extended_bcubed(universe, gold_pairs, predicted_pairs)
    macro_best_f1 = sum(best_f1) / len(best_f1)
    candidate_best_f1 = candidate_best_f1_sum / candidate_groups
    summary: dict[str, float | int] = {
        "reference_groups": len(references),
        "candidate_groups": candidate_groups,
        "reference_vertices": len(universe),
        "bcubed_precision": precision,
        "bcubed_recall": recall,
        "bcubed_f1": f1,
        "best_match_recall": sum(best_recall) / len(best_recall),
        "best_match_jaccard": sum(best_jaccard) / len(best_jaccard),
        "best_match_f1": macro_best_f1,
        "candidate_best_f1": candidate_best_f1,
        "symmetric_best_f1": 0.5 * (macro_best_f1 + candidate_best_f1),
        "intact_reference_fraction": sum(value == 1.0 for value in best_recall) / len(best_recall),
    }
    details = [
        {
            "reference_id": reference_id,
            "reference_size": len(references[reference_id]),
            "best_match_recall": best_recall[reference_id],
            "best_match_jaccard": best_jaccard[reference_id],
            "best_match_f1": best_f1[reference_id],
        }
        for reference_id in range(len(references))
    ]
    return summary, details


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    quality_rows = {
        (row["dataset"], row["method"]): row
        for row in csv.DictReader(QUALITY_CSV.open(encoding="utf-8"))
    }
    formats = {
        "PairSim-MCE": "plain",
        "StructBK": "plain",
        "FastQC": "size-prefix",
        "FaPlex": "solution-prefix",
    }
    comparisons = {
        "definition": (["PairSim-MCE"], 2),
        "external": (["StructBK", "FastQC", "FaPlex"], REFERENCE_MIN_SIZE),
    }
    summaries: list[dict[str, object]] = []
    details: list[dict[str, object]] = []
    input_hashes: dict[str, dict[str, str]] = {}
    for dataset, paths in manifest["inputs"].items():
        reference_path = ROOT / paths["MSC"]["path"]
        input_hashes[dataset] = {"MSC": sha256(reference_path)}
        for comparison, (methods, minimum_size) in comparisons.items():
            for method in methods:
                candidate_path = ROOT / paths[method]["path"]
                summary, per_reference = evaluate_cover(
                    reference_path, candidate_path, formats[method], minimum_size
                )
                row: dict[str, object] = {
                    "comparison": comparison,
                    "dataset": dataset,
                    "method": method,
                    "reference_method": "MSC",
                    "reference_min_size": REFERENCE_MIN_SIZE,
                    "candidate_min_size": minimum_size,
                    **summary,
                }
                quality = quality_rows.get((dataset, method))
                if quality is not None:
                    row["pair_weighted_density"] = quality["pair_weighted_density"]
                    row["pair_weighted_similarity"] = quality["pair_weighted_similarity"]
                    row["coverage_fraction"] = quality["coverage_fraction"]
                summaries.append(row)
                for detail in per_reference:
                    details.append({"comparison": comparison, "dataset": dataset, "method": method, **detail})
                input_hashes[dataset][method] = sha256(candidate_path)
    summary_fields = list(summaries[0])
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    detail_fields = list(details[0])
    with DETAIL_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(details)
    provenance = {
        "schema_version": 1,
        "reference_family": "MSC outputs with size at least 10",
        "definition_candidates": "all non-singleton PairSim-MCE outputs",
        "external_candidates": (
            "StructBK, FastQC, and FaPlex outputs with size at least 10"
        ),
        "metrics": {
            "extended_bcubed": "Amigo et al., Information Retrieval 12(4), 2009",
            "symmetric_best_f1": "Yang and Leskovec, WSDM 2013",
            "best_match_jaccard": "Hric et al., Physical Review E 90, 2014",
        },
        "inputs_sha256": input_hashes,
        "summary_csv": SUMMARY_CSV.relative_to(ROOT).as_posix(),
        "summary_csv_sha256": sha256(SUMMARY_CSV),
        "detail_csv": DETAIL_CSV.relative_to(ROOT).as_posix(),
        "detail_csv_sha256": sha256(DETAIL_CSV),
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for row in summaries:
        print(
            row["comparison"], row["dataset"], row["method"],
            f"B3-R={float(row['bcubed_recall']):.4f}",
            f"best-R={float(row['best_match_recall']):.4f}",
            f"sym-F1={float(row['symmetric_best_f1']):.4f}",
        )


if __name__ == "__main__":
    main()
