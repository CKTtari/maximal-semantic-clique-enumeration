"""Export the paper's canonical CSV files from archived experiment records.

Every row is selected from a versioned summary or a hash-verified artifact
named by its source manifest. Plotting scripts consume only the CSV files
emitted here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
SUMMARY_DIR = ROOT / "experiments" / "summaries"
OUT_DIR = ROOT / "tex-data" / "data"
TAU_FILE = ROOT / "scripts" / "tau_values.json"

FORMAL_SUMMARIES = [
    "batch11-main-core-current",
    "batch11-main-alg3-ldoor-q80-repeat",
    "batch11-main-alg3-q80-long",
    "batch11-main-alg3-q80-pwtk-ldoor",
    "batch11-main-alg4-open-cells",
    "batch11-main-sembk-new-datasets",
    "batch13-main-sembk-missing-current",
    "batch13-main-structbk-formal-current",
    "batch13-threshold-ldoor-missing-current",
    "batch14-threshold-ldoor-formal-repeats",
    "batch12-thread-fb15k-alg3-current",
    "batch12-thread-ogbn-alg4-current",
    "batch12-fb15k-alg3-ablation-current",
    "batch12-alg3-phase-attribution-current",
    "batch12-mechanism-stats-current",
    "batch12-scale-ogbn-regime-current",
    "batch12-threshold-ogbn-current",
    "batch16-alg4-generation-ablation-formal",
    "batch17-nasasrb-alg3-ablation-formal",
    "batch2-main-ogbn-arxiv",
    "batch2-natural-sensitivity-supplement",
    "batch2-main-fb15k-current",
    "batch2-main-wn18rr",
    "batch7-main-wn18rr-q20",
    "batch2-main-structural-uniform-ds2",
    "batch2-main-structural-uniform-ds3",
    "batch7-main-email-enron",
]

PILOT_SUMMARIES: list[str] = []

ALGORITHM_NAMES = {1: "StructBK", 2: "SemBK", 3: "StrSub", 4: "MonoSemMCE"}


def load_summary(name: str) -> dict[str, Any]:
    path = SUMMARY_DIR / f"{name}.summary.json"
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not data.get("source_manifest"):
        raise ValueError(f"summary lacks source manifest: {path}")
    return data


def load_manifest(summary: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / summary["source_manifest"]
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(name: str, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path = OUT_DIR / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def graph_level(group: dict[str, Any]) -> int | None:
    graph = group["configuration"].get("graph") or ""
    marker = "level-"
    if marker not in graph:
        return None
    return int(graph.split(marker, 1)[1].split(".", 1)[0])


def effective_dataset(group: dict[str, Any]) -> int:
    """Resolve plans that use a dataset slot together with graph overrides."""
    graph = (group["configuration"].get("graph") or "").replace("\\", "/").lower()
    if "com-amazon" in graph:
        return 12
    if "email-enron" in graph:
        return 16
    return int(group["configuration"]["dataset"])


def completed_repetitions(group: dict[str, Any]) -> int:
    timing = group.get("timing") or {}
    return int(timing.get("completed_repetitions") or 0)


def status_of(group: dict[str, Any]) -> str:
    counts = group.get("status_counts") or {}
    if counts.get("completed"):
        return "completed"
    if counts.get("oom"):
        return "OOM"
    if counts.get("tle"):
        return "TLE"
    return "NR"


def defines_key(group: dict[str, Any]) -> str:
    defines = group["configuration"].get("defines") or {}
    return json.dumps(defines, sort_keys=True, separators=(",", ":"))


with TAU_FILE.open("r", encoding="utf-8") as handle:
    TAUS: dict[str, dict[str, Any]] = json.load(handle)


def quantile_for(dataset: int, tau: float | None) -> int | None:
    if tau is None:
        return None
    candidates: list[tuple[float, int]] = []
    for key, value in TAUS[str(dataset)].items():
        if key.startswith("tau_"):
            candidates.append((abs(float(value) - float(tau)), int(key.split("_", 1)[1])))
    distance, quantile = min(candidates)
    if distance > 2e-6:
        return None
    return quantile


def flatten(names: list[str]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for source_rank, name in enumerate(names):
        summary = load_summary(name)
        for group in summary["groups"]:
            copy = dict(group)
            copy["_source"] = name
            copy["_source_rank"] = source_rank
            groups.append(copy)
    return groups


FORMAL_GROUPS = flatten(FORMAL_SUMMARIES)
PILOT_GROUPS = flatten(PILOT_SUMMARIES)


def select_group(
    groups: list[dict[str, Any]],
    *,
    dataset: int,
    algorithm: int,
    quantile: int | None,
    level: int | None = None,
    threads: int | None = 32,
    defines: str | None = "{}",
    stats: bool | None = None,
    source: str | None = None,
) -> dict[str, Any] | None:
    matches = []
    for group in groups:
        cfg = group["configuration"]
        if source is not None and group["_source"] != source:
            continue
        if effective_dataset(group) != dataset or int(cfg["algorithm"]) != algorithm:
            continue
        if quantile is None:
            if cfg.get("tau") is not None:
                continue
        elif quantile_for(effective_dataset(group), cfg.get("tau")) != quantile:
            continue
        if graph_level(group) != level:
            continue
        if threads is not None and int(cfg.get("threads") or 0) != threads:
            continue
        if defines is not None and defines_key(group) != defines:
            continue
        is_stats = bool((group.get("result") or {}).get("stats"))
        if stats is not None and is_stats != stats:
            continue
        matches.append(group)
    if not matches:
        return None
    return max(matches, key=lambda item: (completed_repetitions(item), -int(item["_source_rank"])))


def seconds(group: dict[str, Any] | None) -> float | None:
    if group is None or status_of(group) != "completed":
        return None
    value = (group.get("timing") or {}).get("algorithm_ms_median")
    return None if value is None else float(value) / 1000.0


def merge_ldoor_repetitions(algorithm: int, quantile: int) -> dict[str, Any] | None:
    original = select_group(
        FORMAL_GROUPS,
        dataset=1,
        algorithm=algorithm,
        quantile=quantile,
        stats=False,
        source="batch13-threshold-ldoor-missing-current",
    )
    repeats = select_group(
        FORMAL_GROUPS,
        dataset=1,
        algorithm=algorithm,
        quantile=quantile,
        stats=False,
        source="batch14-threshold-ldoor-formal-repeats",
    )
    if repeats is None:
        return original
    if original is None or status_of(original) != "completed" or status_of(repeats) != "completed":
        raise ValueError(f"cannot merge sc-ldoor repetitions for algorithm {algorithm}, q{quantile}")
    if completed_repetitions(original) != 1 or completed_repetitions(repeats) != 2:
        raise ValueError(f"unexpected sc-ldoor repetition counts for algorithm {algorithm}, q{quantile}")
    if original["result"]["clique_count"] != repeats["result"]["clique_count"]:
        raise ValueError(f"sc-ldoor output mismatch for algorithm {algorithm}, q{quantile}")

    samples = [
        float(original["timing"]["algorithm_ms_median"]),
        float(repeats["timing"]["algorithm_ms_min"]),
        float(repeats["timing"]["algorithm_ms_max"]),
    ]
    samples.sort()
    merged = dict(repeats)
    merged["timing"] = dict(repeats["timing"])
    merged["timing"].update(
        {
            "completed_repetitions": 3,
            "algorithm_ms_median": samples[1],
            "algorithm_ms_min": samples[0],
            "algorithm_ms_max": samples[2],
        }
    )
    merged["status_counts"] = {"completed": 3}
    merged["_source"] = "batch13+batch14-threshold-ldoor-three-repetitions"
    return merged


def emit_dataset_inventory() -> None:
    roles = {
        1: ("generated", "common grid", 256),
        2: ("generated", "common grid", 256),
        3: ("generated", "common grid", 256),
        14: ("generated", "common grid", 256),
        7: ("natural", "common grid", 384),
        8: ("natural", "common grid", 384),
        11: ("natural", "common grid", 128),
        16: ("generated", "common grid", 256),
    }
    rows = []
    for dataset in [7, 8, 11, 16, 2, 3, 14, 1]:
        meta = TAUS[str(dataset)]
        attribute, role, dimension = roles[dataset]
        rows.append(
            {
                "dataset_id": dataset,
                "dataset": meta["name"],
                "vertices": meta["vertices"],
                "edges": meta["edges"],
                "dimension": dimension,
                "attribute_type": attribute,
                "paper_role": role,
            }
        )
    write_csv(
        "dataset_inventory.csv",
        ["dataset_id", "dataset", "vertices", "edges", "dimension", "attribute_type", "paper_role"],
        rows,
    )


def emit_main_runtime() -> None:
    rows = []
    for dataset in [7, 8, 11, 16, 2, 3, 14, 1]:
        structural = select_group(
            FORMAL_GROUPS,
            dataset=dataset,
            algorithm=1,
            quantile=None,
            stats=False,
        )
        if structural is None:
            raise ValueError(f"missing StructBK result for dataset {dataset}")
        for quantile in [20, 80, 99]:
            row: dict[str, Any] = {
                "dataset": TAUS[str(dataset)]["name"],
                "dataset_id": dataset,
                "quantile": quantile,
                "tau": TAUS[str(dataset)][f"tau_{quantile}"],
                "structural_count": ((structural.get("result") or {}).get("clique_count")),
                "alg1_status": status_of(structural),
                "alg1_seconds": seconds(structural),
                "alg1_min_seconds": float(structural["timing"]["algorithm_ms_min"]) / 1000.0,
                "alg1_max_seconds": float(structural["timing"]["algorithm_ms_max"]) / 1000.0,
                "alg1_memory_gib": (structural.get("timing") or {}).get("peak_memory_gib_median"),
            }
            result_count = None
            for algorithm in [2, 3, 4]:
                group = select_group(
                    FORMAL_GROUPS,
                    dataset=dataset,
                    algorithm=algorithm,
                    quantile=quantile,
                    stats=False,
                )
                prefix = f"alg{algorithm}"
                row[f"{prefix}_status"] = status_of(group) if group else "NR"
                row[f"{prefix}_seconds"] = seconds(group)
                row[f"{prefix}_min_seconds"] = (
                    float(group["timing"]["algorithm_ms_min"]) / 1000.0
                    if group and status_of(group) == "completed" and group.get("timing", {}).get("algorithm_ms_min") is not None
                    else None
                )
                row[f"{prefix}_max_seconds"] = (
                    float(group["timing"]["algorithm_ms_max"]) / 1000.0
                    if group and status_of(group) == "completed" and group.get("timing", {}).get("algorithm_ms_max") is not None
                    else None
                )
                row[f"{prefix}_memory_gib"] = (
                    (group.get("timing") or {}).get("peak_memory_gib_median")
                    if group and status_of(group) == "completed"
                    else None
                )
                count = ((group or {}).get("result") or {}).get("clique_count")
                if count is not None:
                    if result_count is not None and result_count != count:
                        raise ValueError(f"count mismatch for dataset {dataset}, q{quantile}")
                    result_count = count
            row["acmsc_count"] = result_count
            rows.append(row)
    fields = [
        "dataset",
        "dataset_id",
        "quantile",
        "tau",
        "acmsc_count",
        "structural_count",
        "alg1_status",
        "alg1_seconds",
        "alg1_min_seconds",
        "alg1_max_seconds",
        "alg1_memory_gib",
    ]
    for algorithm in [2, 3, 4]:
        fields.extend(
            [
                f"alg{algorithm}_status",
                f"alg{algorithm}_seconds",
                f"alg{algorithm}_min_seconds",
                f"alg{algorithm}_max_seconds",
                f"alg{algorithm}_memory_gib",
            ]
        )
    write_csv("main_runtime.csv", fields, rows)


def emit_threshold_sensitivity() -> None:
    rows = []
    grids = {
        11: [5, 20, 50, 70, 80, 90, 95, 97, 99],
        1: [5, 20, 50, 70, 80, 90, 95, 97, 99],
    }
    for dataset, quantiles in grids.items():
        for quantile in quantiles:
            for algorithm in [2, 3, 4]:
                source = None
                if dataset == 11 and algorithm in (3, 4):
                    source = (
                        "batch12-threshold-ogbn-current"
                        if quantile in (5, 50, 70, 90, 95, 97)
                        else "batch11-main-core-current"
                    )
                group = select_group(
                    FORMAL_GROUPS,
                    dataset=dataset,
                    algorithm=algorithm,
                    quantile=quantile,
                    stats=False,
                    source=source,
                )
                if dataset == 1 and algorithm in (3, 4) and quantile in (5, 50, 70, 90, 95, 97):
                    group = merge_ldoor_repetitions(algorithm, quantile)
                if dataset == 1 and algorithm == 4 and quantile == 5 and group is None:
                    rows.append(
                        {
                            "dataset": TAUS[str(dataset)]["name"],
                            "dataset_id": dataset,
                            "quantile": quantile,
                            "tau": TAUS[str(dataset)][f"tau_{quantile}"],
                            "algorithm": ALGORITHM_NAMES[algorithm],
                            "algorithm_id": algorithm,
                            "status": "TLE",
                            "seconds": None,
                            "acmsc_count": None,
                        }
                    )
                    continue
                if group is None:
                    continue
                rows.append(
                    {
                        "dataset": TAUS[str(dataset)]["name"],
                        "dataset_id": dataset,
                        "quantile": quantile,
                        "tau": TAUS[str(dataset)][f"tau_{quantile}"],
                        "algorithm": ALGORITHM_NAMES[algorithm],
                        "algorithm_id": algorithm,
                        "status": status_of(group),
                        "seconds": seconds(group),
                        "acmsc_count": ((group.get("result") or {}).get("clique_count")),
                    }
                )
    write_csv(
        "threshold_sensitivity.csv",
        ["dataset", "dataset_id", "quantile", "tau", "algorithm", "algorithm_id", "status", "seconds", "acmsc_count"],
        rows,
    )


def emit_scale_regime() -> None:
    rows = []
    for level in [25000, 50000, 100000, 169343]:
        for quantile in [20, 50, 80, 90, 95, 99]:
            alg3 = select_group(FORMAL_GROUPS, dataset=11, algorithm=3, quantile=quantile, level=level, stats=False)
            alg4 = select_group(FORMAL_GROUPS, dataset=11, algorithm=4, quantile=quantile, level=level, stats=False)
            if alg3 is None or alg4 is None:
                raise ValueError(f"missing scale result for level {level}, q{quantile}")
            t3, t4 = seconds(alg3), seconds(alg4)
            assert t3 is not None and t4 is not None
            count3 = alg3["result"]["clique_count"]
            count4 = alg4["result"]["clique_count"]
            if count3 != count4:
                raise ValueError(f"scale count mismatch for level {level}, q{quantile}")
            rows.append(
                {
                    "vertices": level,
                    "quantile": quantile,
                    "tau": TAUS["11"][f"tau_{quantile}"],
                    "strsub_seconds": t3,
                    "monosemmce_seconds": t4,
                    "log2_t4_over_t3": math.log2(t4 / t3),
                    "acmsc_count": count3,
                }
            )
    write_csv(
        "scale_regime.csv",
        ["vertices", "quantile", "tau", "strsub_seconds", "monosemmce_seconds", "log2_t4_over_t3", "acmsc_count"],
        rows,
    )


def emit_parallel_scaling() -> None:
    rows = []
    experiments = [(3, 7, 80), (4, 11, 50)]
    for algorithm, dataset, quantile in experiments:
        source = (
            "batch12-thread-fb15k-alg3-current"
            if algorithm == 3
            else "batch12-thread-ogbn-alg4-current"
        )
        thread_groups = []
        for threads in [1, 2, 4, 8, 16, 32]:
            group = select_group(
                FORMAL_GROUPS,
                dataset=dataset,
                algorithm=algorithm,
                quantile=quantile,
                threads=threads,
                stats=False,
                source=source,
            )
            if group is None:
                raise ValueError(f"missing thread result for alg{algorithm}, {threads}")
            thread_groups.append((threads, group))
        baseline = seconds(thread_groups[0][1])
        assert baseline is not None
        for threads, group in thread_groups:
            runtime = seconds(group)
            assert runtime is not None
            rows.append(
                {
                    "algorithm": ALGORITHM_NAMES[algorithm],
                    "dataset": TAUS[str(dataset)]["name"],
                    "quantile": quantile,
                    "threads": threads,
                    "seconds": runtime,
                    "speedup": baseline / runtime,
                    "memory_gib": group["timing"]["peak_memory_gib_median"],
                }
            )
    write_csv(
        "parallel_scaling.csv",
        ["algorithm", "dataset", "quantile", "threads", "seconds", "speedup", "memory_gib"],
        rows,
    )


def emit_alg3_phase_attribution() -> None:
    """Structure the two archived phase-attribution runs without rerunning them."""
    summary = load_summary("batch12-alg3-phase-attribution-current")
    manifest = load_manifest(summary)
    rows = []
    for record in manifest["records"]:
        task = record["task"]
        if int(task["algorithm"]) != 3 or task["mode"] != "stats" or int(task["dataset"]) != 7:
            continue
        artifact = record["artifacts"]["stdout"]
        path = ROOT / artifact["path"]
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"phase-attribution artifact hash mismatch: {path}")
        text = path.read_text(encoding="utf-8")

        def extract(pattern: str, label: str) -> str:
            match = re.search(pattern, text)
            if match is None:
                raise ValueError(f"missing {label} in {path}")
            return match.group(1)

        algorithm_ms = int(extract(r"=== TOTAL:\s*(\d+) ms,", "total time"))
        output = int(extract(r"=== TOTAL:\s*\d+ ms,\s*(\d+) maximal cliques", "output count"))
        if algorithm_ms != int(record["parsed"]["algorithm_ms"]) or output != int(record["parsed"]["clique_count"]):
            raise ValueError(f"phase-attribution parsed-value mismatch: {path}")
        rows.append(
            {
                "threads": int(task["threads"]),
                "algorithm_ms": algorithm_ms,
                "bk_ratio_percent": float(extract(r"BK ratio in Phase1\+2:\s*([0-9.]+)%", "BK ratio")),
                "subset_ratio_percent": float(
                    extract(r"Subset ratio in Phase1\+2:\s*([0-9.]+)%", "subset ratio")
                ),
                "global_filter_ms": int(extract(r"Phase3 dedup:\s*(\d+) ms", "global-filter time")),
                "acmsc_count": output,
                "source_task_id": record["task_id"],
            }
        )
    if sorted(row["threads"] for row in rows) != [1, 32]:
        raise ValueError("phase-attribution archive must contain exactly the 1- and 32-thread runs")
    rows.sort(key=lambda row: row["threads"])
    write_csv(
        "alg3_phase_attribution.csv",
        [
            "threads",
            "algorithm_ms",
            "bk_ratio_percent",
            "subset_ratio_percent",
            "global_filter_ms",
            "acmsc_count",
            "source_task_id",
        ],
        rows,
    )


def variant_name(group: dict[str, Any]) -> str:
    defines = group["configuration"].get("defines") or {}
    if not defines:
        return "Full"
    key = next(iter(defines))
    return {
        "ACMSC_STRSUB_TOP_EDGE": "No Top-Edge",
        "ACMSC_STRSUB_VERTEX_BOUND": "No Vertex-Bound",
        "ACMSC_STRSUB_LOCAL_MAX": "No Local-Max",
    }[key]


def emit_mechanisms() -> None:
    ablation_rows = []
    for group in FORMAL_GROUPS:
        cfg = group["configuration"]
        if group["_source"] not in {
            "batch12-fb15k-alg3-ablation-current",
            "batch17-nasasrb-alg3-ablation-formal",
        }:
            continue
        dataset = int(cfg["dataset"])
        quantile = quantile_for(dataset, cfg.get("tau"))
        ablation_rows.append(
            {
                "dataset": {7: "FB15K-237", 2: "sc-nasasrb"}[dataset],
                "quantile": quantile,
                "variant": variant_name(group),
                "seconds": seconds(group),
                "min_seconds": float(group["timing"]["algorithm_ms_min"]) / 1000.0,
                "max_seconds": float(group["timing"]["algorithm_ms_max"]) / 1000.0,
                "memory_gib": group["timing"]["peak_memory_gib_median"],
                "acmsc_count": group["result"]["clique_count"],
            }
        )
    order = {"Full": 0, "No Top-Edge": 1, "No Vertex-Bound": 2, "No Local-Max": 3}
    dataset_order = {"FB15K-237": 0, "sc-nasasrb": 1}
    ablation_rows.sort(key=lambda row: (dataset_order[row["dataset"]], order[row["variant"]]))
    write_csv(
        "alg3_ablation_runtime.csv",
        [
            "dataset",
            "quantile",
            "variant",
            "seconds",
            "min_seconds",
            "max_seconds",
            "memory_gib",
            "acmsc_count",
        ],
        ablation_rows,
    )

    counter_rows = []
    for group in FORMAL_GROUPS:
        if group["_source"] != "batch12-mechanism-stats-current":
            continue
        cfg = group["configuration"]
        if int(cfg["algorithm"]) != 3 or int(cfg["dataset"]) != 2:
            continue
        stats = group["result"]["stats"]
        counter_rows.append(
            {
                "variant": variant_name(group),
                "subset_tests": stats["subset_tests"],
                "covered_subsets": stats["covered_subsets"],
                "feasible_local": stats["feasible_local"],
                "exact_duplicates": stats["exact_duplicates"],
                "output": group["result"]["clique_count"],
            }
        )
    counter_rows.sort(key=lambda row: order[row["variant"]])
    write_csv(
        "alg3_mechanism_counters.csv",
        ["variant", "subset_tests", "covered_subsets", "feasible_local", "exact_duplicates", "output"],
        counter_rows,
    )

    state_rows = []
    for quantile in [5, 20, 50, 70, 80, 90, 95, 97, 99]:
        group = select_group(FORMAL_GROUPS, dataset=11, algorithm=4, quantile=quantile, stats=True)
        if group is None:
            raise ValueError(f"missing Alg4 stats at q{quantile}")
        stats = group["result"]["stats"]
        state_rows.append(
            {
                "quantile": quantile,
                "tau": TAUS["11"][f"tau_{quantile}"],
                "states": stats["states"],
                "candidate_tests": stats["candidate_tests"],
                "canonical_rejects": stats["canonical_rejects"],
                "w_prunes": stats["w_prunes"],
                "raw_candidates": stats["raw_candidates"],
                "output": group["result"]["clique_count"],
            }
        )
    write_csv(
        "alg4_state_contraction.csv",
        ["quantile", "tau", "states", "candidate_tests", "canonical_rejects", "w_prunes", "raw_candidates", "output"],
        state_rows,
    )


def emit_alg4_generation_ablation() -> None:
    source = "batch16-alg4-generation-ablation-formal"
    strategy_by_defines = {
        "{}": "Canonical",
        '{"ACMSC_ALG4_GENERATION":1}': "Visited DFS",
        '{"ACMSC_ALG4_GENERATION":2}': "Layered",
    }
    dataset_names = {11: "ogbn-arxiv", 2: "sc-nasasrb"}
    rows = []
    for group in FORMAL_GROUPS:
        if group["_source"] != source:
            continue
        cfg = group["configuration"]
        dataset = int(cfg["dataset"])
        key = defines_key(group)
        if dataset not in dataset_names or key not in strategy_by_defines:
            continue
        if status_of(group) != "completed" or completed_repetitions(group) != 3:
            continue
        rows.append(
            {
                "dataset": dataset_names[dataset],
                "quantile": quantile_for(dataset, cfg.get("tau")),
                "strategy": strategy_by_defines[key],
                "seconds": seconds(group),
                "min_seconds": float(group["timing"]["algorithm_ms_min"]) / 1000.0,
                "max_seconds": float(group["timing"]["algorithm_ms_max"]) / 1000.0,
                "memory_gib": group["timing"]["peak_memory_gib_median"],
                "acmsc_count": group["result"]["clique_count"],
            }
        )
    strategy_order = {"Canonical": 0, "Visited DFS": 1, "Layered": 2}
    dataset_order = {"ogbn-arxiv": 0, "sc-nasasrb": 1}
    rows.sort(key=lambda row: (dataset_order[row["dataset"]], strategy_order[row["strategy"]]))
    if len(rows) != 6:
        raise ValueError(f"expected six completed Alg4 strategy rows, found {len(rows)}")
    for dataset in dataset_names.values():
        selected = [row for row in rows if row["dataset"] == dataset]
        if len({row["acmsc_count"] for row in selected}) != 1:
            raise ValueError(f"Alg4 generation strategies disagree on {dataset}")
    write_csv(
        "alg4_generation_ablation.csv",
        [
            "dataset",
            "quantile",
            "strategy",
            "seconds",
            "min_seconds",
            "max_seconds",
            "memory_gib",
            "acmsc_count",
        ],
        rows,
    )


def emit_provenance() -> None:
    quality_path = ROOT / "experiments" / "quality" / "ogbn-arxiv-result-quality.json"
    payload = {
        "schema_version": 1,
        "exporter": "src/experiments/standard/paper_results_export.py",
        "formal_summaries": [f"experiments/summaries/{name}.summary.json" for name in FORMAL_SUMMARIES],
        "pilot_status_summaries": [f"experiments/summaries/{name}.summary.json" for name in PILOT_SUMMARIES],
        "quality_analysis": {
            "path": quality_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(quality_path),
        },
        "note": "CSV values are copied or derived deterministically from archived formal summaries and hash-verified artifacts referenced by their manifests; no pilot timing enters paper data.",
    }
    with (OUT_DIR / "provenance.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    emit_dataset_inventory()
    emit_main_runtime()
    emit_threshold_sensitivity()
    emit_scale_regime()
    emit_parallel_scaling()
    emit_alg3_phase_attribution()
    emit_mechanisms()
    emit_alg4_generation_ablation()
    emit_provenance()
    print(f"Exported canonical paper data to {OUT_DIR}")


if __name__ == "__main__":
    main()
