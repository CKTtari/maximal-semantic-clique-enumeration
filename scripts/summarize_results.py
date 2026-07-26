#!/usr/bin/env python3
"""Aggregate archived experiment attempts without using stats-build timing."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_ROOT = ROOT / "experiments" / "summaries"


def group_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task["algorithm"],
        task["mode"],
        task["dataset"],
        task.get("tau"),
        task["threads"],
        task.get("graph"),
        task.get("vectors"),
        json.dumps(task.get("defines", {}), sort_keys=True),
        task.get("warmup", False),
    )


def summarize(manifest_path: Path, allow_incomplete: bool = False) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plan_complete = manifest.get("summary", {}).get("plan_complete")
    if plan_complete is False and not allow_incomplete:
        raise ValueError(
            f"Refusing to summarize incomplete plan {manifest['run_name']}; "
            "use --allow-incomplete only for diagnostic summaries"
        )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in manifest["records"]:
        groups[group_key(record["task"])].append(record)

    summaries = []
    for key, records in sorted(groups.items(), key=lambda item: str(item[0])):
        task = dict(records[0]["task"])
        task.pop("task_id", None)
        task.pop("repetition", None)
        task.pop("repetitions", None)
        statuses: dict[str, int] = defaultdict(int)
        for record in records:
            statuses[record["status"]] += 1
        completed = [record for record in records if record["status"] == "completed"]

        timing = None
        result_metrics = None
        if completed:
            if task["mode"] != "raw":
                algorithm_times: list[int] = []
            else:
                algorithm_times = [record["parsed"]["algorithm_ms"] for record in completed]
            counts = {record["parsed"]["clique_count"] for record in completed}
            histograms = {
                json.dumps(record["parsed"]["size_histogram"], sort_keys=True)
                for record in completed
            }
            if len(counts) != 1 or len(histograms) != 1:
                raise ValueError(f"Output changed across repetitions: {task}")
            result_metrics = {
                "clique_count": counts.pop(),
                "size_histogram": json.loads(histograms.pop()),
            }
            if task["mode"] == "stats":
                stat_records = {
                    json.dumps(record["parsed"].get("stats", {}), sort_keys=True)
                    for record in completed
                }
                if len(stat_records) != 1:
                    raise ValueError(f"Stats changed across repetitions: {task}")
                result_metrics["stats"] = json.loads(stat_records.pop())
            if algorithm_times:
                peak_values = [
                    record["parsed"]["peak_memory_gib"]
                    for record in completed
                    if record["parsed"]["peak_memory_gib"] is not None
                ]
                wall_values = [record["wall_seconds"] for record in completed]
                timing = {
                    "completed_repetitions": len(completed),
                    "algorithm_ms_median": statistics.median(algorithm_times),
                    "algorithm_ms_min": min(algorithm_times),
                    "algorithm_ms_max": max(algorithm_times),
                    "wall_seconds_median": statistics.median(wall_values),
                    "peak_memory_gib_median": (
                        statistics.median(peak_values) if peak_values else None
                    ),
                    "peak_memory_gib_max": max(peak_values) if peak_values else None,
                }

        summaries.append(
            {
                "configuration": task,
                "status_counts": dict(sorted(statuses.items())),
                "timing": timing,
                "result": result_metrics,
            }
        )

    return {
        "schema_version": 1,
        "source_manifest": manifest_path.relative_to(ROOT).as_posix(),
        "source_run": manifest["run_name"],
        "source_plan_summary": manifest.get("summary"),
        "groups": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="write a diagnostic summary even when planned tasks are unattempted",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    SUMMARY_ROOT.mkdir(parents=True, exist_ok=True)
    for raw_path in args.manifests:
        manifest_path = raw_path.resolve()
        summary = summarize(manifest_path, args.allow_incomplete)
        output = SUMMARY_ROOT / f"{summary['source_run']}.summary.json"
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Summary: {output}")


if __name__ == "__main__":
    main()
