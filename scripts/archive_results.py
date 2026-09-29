#!/usr/bin/env python3
"""Verify local experiment artifacts and write a Git-trackable manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_experiments import expand_plan, fingerprint_inputs, resolve_inputs


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "experiments" / "output"
ARCHIVE_ROOT = ROOT / "experiments" / "archive"
SOURCE_DIR = ROOT / "src"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def portable_metadata(value: Any) -> Any:
    """Make repository-local paths portable without changing raw artifacts."""
    if isinstance(value, dict):
        return {key: portable_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_metadata(item) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            return portable_path(candidate)
    return value


def verify_build_sources(
    result: dict[str, Any], allow_source_mismatch: str | None
) -> dict[str, Any]:
    """Reconstruct the runner fingerprint and record the exact source hashes."""
    task = result["task"]
    algorithm = int(task["algorithm"])
    mode = task["mode"]
    source = SOURCE_DIR / f"alg{algorithm}-{mode}.cpp"
    source_paths = (
        source,
        SOURCE_DIR / f"alg{algorithm}-raw.cpp",
        SOURCE_DIR / "semantic_graph.cpp",
        SOURCE_DIR / "semantic_graph.h",
        SOURCE_DIR / "experiment_runtime.h",
    )
    source_hashes = {
        path.name: sha256(path) for path in source_paths if path.exists()
    }
    defines_tuple = tuple(sorted(task.get("defines", {}).items()))
    fingerprint_input = json.dumps(
        {
            "algorithm": algorithm,
            "mode": mode,
            "defines": defines_tuple,
            "source_hashes": source_hashes,
        },
        sort_keys=True,
    )
    observed = hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]
    expected = result["build"].get("fingerprint")
    if observed != expected and not allow_source_mismatch:
        raise ValueError(
            "Current source no longer matches the executable used by "
            f"task {task['task_id']}: {expected} != {observed}"
        )
    build = {
        "fingerprint": expected,
        "executable_sha256": result["build"].get("executable_sha256"),
        "command": portable_metadata(result["build"].get("command")),
        "source_verification": {
            "status": "verified_current" if observed == expected else "legacy_mismatch",
            "observed_current_fingerprint": observed,
            "reason": None if observed == expected else allow_source_mismatch,
        },
    }
    if observed == expected:
        build["source_hashes"] = source_hashes
    return build


def verify_inputs(
    result: dict[str, Any],
    cache: dict[tuple[str, str | None], dict[str, Any]],
    allow_legacy_inputs: str | None,
) -> dict[str, Any]:
    task = result["task"]
    graph_path, vector_path = resolve_inputs(task)
    key = (str(graph_path), str(vector_path) if vector_path else None)
    if key not in cache:
        cache[key] = fingerprint_inputs(graph_path, vector_path)
    current = cache[key]
    recorded = result.get("inputs")
    if recorded is None:
        if not allow_legacy_inputs:
            raise ValueError(
                f"Task {task['task_id']} has no run-time input fingerprint"
            )
        return {
            "status": "legacy_post_run_snapshot",
            "reason": allow_legacy_inputs,
            "snapshot": current,
        }
    if recorded != current:
        raise ValueError(
            f"Input files changed after task {task['task_id']}; refusing to archive"
        )
    return {"status": "verified_run_time", "snapshot": current}


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": portable_path(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def validate_result(result: dict[str, Any], result_path: Path) -> None:
    status = result.get("status")
    if status not in {"completed", "tle", "oom", "crash", "parse_error", "mismatch"}:
        raise ValueError(f"Unknown status in {result_path}: {status}")
    if status == "completed" and result.get("parsed") is None:
        raise ValueError(f"Completed result has no parsed metrics: {result_path}")
    if status == "mismatch":
        raise ValueError(f"Refusing to archive an output mismatch: {result_path}")
    task = result.get("task", {})
    validation = result.get("validation")
    if status == "completed" and task.get("capture_cliques"):
        if not validation or not validation.get("exact_match"):
            raise ValueError(f"Captured output was not validated: {result_path}")
        parsed_count = result["parsed"].get("clique_count")
        if validation.get("unique_cliques") != parsed_count:
            raise ValueError(
                f"Captured unique count differs from program count: {result_path}"
            )


def archive_run(
    run_name: str,
    allow_source_mismatch: str | None = None,
    allow_legacy_inputs: str | None = None,
) -> Path:
    result_dir = (OUTPUT_ROOT / run_name).resolve()
    if not result_dir.is_dir():
        raise FileNotFoundError(result_dir)
    result_paths = sorted(result_dir.glob("*.attempt-*.json"))
    if not result_paths:
        raise ValueError(f"No result JSON files in {result_dir}")

    records = []
    seen_attempts: set[tuple[str, int]] = set()
    statuses: Counter[str] = Counter()
    plan_files: dict[str, str] = {}
    plan_paths: set[Path] = set()
    input_cache: dict[tuple[str, str | None], dict[str, Any]] = {}
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validate_result(result, result_path)
        task_id = result["task"]["task_id"]
        attempt = int(result["attempt"])
        attempt_key = (task_id, attempt)
        if attempt_key in seen_attempts:
            raise ValueError(f"Duplicate task attempt: {attempt_key}")
        seen_attempts.add(attempt_key)
        statuses[result["status"]] += 1

        artifacts: dict[str, Any] = {}
        for name, raw_path in result.get("artifacts", {}).items():
            if raw_path:
                artifacts[name] = file_record(Path(raw_path))

        plan_path = Path(result["run"]["plan"])
        observed_plan_hash = sha256(plan_path)
        expected_plan_hash = result["run"].get("plan_sha256")
        if observed_plan_hash != expected_plan_hash:
            raise ValueError(
                f"Plan changed after the run: {plan_path} "
                f"({expected_plan_hash} != {observed_plan_hash})"
            )
        plan_files[portable_path(plan_path)] = observed_plan_hash
        plan_paths.add(plan_path.resolve())

        records.append(
            {
                "task_id": task_id,
                "attempt": attempt,
                "status": result["status"],
                "rerun_reason": result.get("rerun_reason"),
                "started_utc": result.get("started_utc"),
                "wall_seconds": result.get("wall_seconds"),
                "task": result["task"],
                "parsed": result.get("parsed"),
                "validation": result.get("validation"),
                "build": verify_build_sources(result, allow_source_mismatch),
                "inputs": verify_inputs(result, input_cache, allow_legacy_inputs),
                "run_environment": portable_metadata(result["run"]),
                "result_json": file_record(result_path),
                "artifacts": artifacts,
            }
        )

    expected_task_ids: set[str] = set()
    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        expected_task_ids.update(task["task_id"] for task in expand_plan(plan))
    attempted_task_ids = {record["task_id"] for record in records}
    completed_task_ids = {
        record["task_id"] for record in records if record["status"] == "completed"
    }
    unattempted_task_ids = sorted(expected_task_ids - attempted_task_ids)

    manifest = {
        "schema_version": 2,
        "run_name": run_name,
        "archived_utc": datetime.now(timezone.utc).isoformat(),
        "local_result_directory": portable_path(result_dir),
        "plans": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(plan_files.items())
        ],
        "summary": {
            "attempts": len(records),
            "planned_tasks": len(expected_task_ids),
            "attempted_tasks": len(attempted_task_ids),
            "completed_tasks": len(completed_task_ids),
            "unattempted_tasks": len(unattempted_task_ids),
            "unattempted_task_ids": unattempted_task_ids,
            "plan_complete": not unattempted_task_ids,
            "status_counts": dict(sorted(statuses.items())),
        },
        "records": records,
    }
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    archive_path = ARCHIVE_ROOT / f"{run_name}.manifest.json"
    archive_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_names", nargs="+", help="Directories under experiments/output")
    parser.add_argument(
        "--allow-source-mismatch",
        metavar="REASON",
        help=(
            "archive a legacy executable whose run-time source snapshot is no longer "
            "available; the mismatch and reason are recorded"
        ),
    )
    parser.add_argument(
        "--allow-legacy-inputs",
        metavar="REASON",
        help=(
            "archive a legacy run that predates run-time input hashing; a current "
            "post-run snapshot and the reason are recorded"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for run_name in args.run_names:
        archive_path = archive_run(
            run_name, args.allow_source_mismatch, args.allow_legacy_inputs
        )
        print(f"Archived {run_name}: {archive_path}")


if __name__ == "__main__":
    main()
