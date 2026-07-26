#!/usr/bin/env python3
"""Build and run a bounded batch of reproducible ACMSC experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "src"
BUILD_DIR = ROOT / "no-use" / "compiled" / "experiment-runner"
OUTPUT_ROOT = ROOT / "experiment-output"
DATASET_CATALOG = ROOT / "experiments" / "datasets.json"
MAX_BATCH_SIZE = 8

TOTAL_RE = re.compile(r"=== TOTAL:\s*(\d+) ms,\s*(\d+) maximal cliques ===")
PEAK_RE = re.compile(r"Peak process memory:\s*([0-9.]+) GiB")
SIZE_RE = re.compile(r"^\s*size\s+(\d+):\s+(\d+)", re.MULTILINE)
STAT_RE = re.compile(r"^STAT\s+(.+)$", re.MULTILINE)


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


def resolve_inputs(task: dict[str, Any]) -> tuple[Path, Path | None]:
    catalog = json.loads(DATASET_CATALOG.read_text(encoding="utf-8"))
    entry = catalog["datasets"].get(str(task["dataset"]))
    if entry is None and not task.get("graph"):
        raise ValueError(f"Dataset {task['dataset']} is absent from {DATASET_CATALOG}")
    graph_raw = task.get("graph") or entry["graph"]
    vector_raw = task.get("vectors") or (entry["vectors"] if entry else None)
    graph_path = Path(graph_raw)
    if not graph_path.is_absolute():
        graph_path = ROOT / graph_path
    vector_path = None
    if task["algorithm"] != 1:
        if not vector_raw:
            raise ValueError("A vector file is required for semantic algorithms")
        vector_path = Path(vector_raw)
        if not vector_path.is_absolute():
            vector_path = ROOT / vector_path
    return graph_path.resolve(), vector_path.resolve() if vector_path else None


def input_files(graph_path: Path, vector_path: Path | None) -> list[tuple[str, Path]]:
    files = [("graph", graph_path)]
    if vector_path is None:
        return files
    if vector_path.suffix != ".index":
        files.append(("vector_binary", vector_path))
        return files
    files.append(("vector_index", vector_path))
    tokens = vector_path.read_text(encoding="utf-8").split()
    if len(tokens) < 3:
        raise ValueError(f"Vector index has no chunks: {vector_path}")
    int(tokens[0])
    int(tokens[1])
    for token in tokens[2:]:
        chunk_path = Path(token)
        if not chunk_path.is_absolute():
            chunk_path = ROOT / chunk_path
        files.append(("vector_chunk", chunk_path.resolve()))
    return files


def fingerprint_inputs(
    graph_path: Path, vector_path: Path | None
) -> dict[str, Any]:
    records = []
    for role, path in input_files(graph_path, vector_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "role": role,
                "path": portable_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "dataset_catalog": {
            "path": portable_path(DATASET_CATALOG),
            "sha256": sha256(DATASET_CATALOG),
        },
        "files": records,
    }


def run_readonly(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (completed.stdout or completed.stderr).strip()


def git_state() -> dict[str, Any]:
    commit = run_readonly(["git", "rev-parse", "HEAD"])
    status = run_readonly(["git", "status", "--porcelain"])
    return {"commit": commit or None, "dirty": bool(status)}


def machine_state() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "compiler": run_readonly(["g++", "--version"]).splitlines()[0],
    }


def canonical_task(task: dict[str, Any], repetition: int) -> dict[str, Any]:
    normalized = {
        "algorithm": int(task["algorithm"]),
        "mode": task.get("mode", "raw"),
        "dataset": int(task.get("dataset", 2)),
        "tau": task.get("tau"),
        "threads": int(task.get("threads", 32)),
        "timeout_seconds": int(task.get("timeout_seconds", 3600)),
        "graph": task.get("graph"),
        "vectors": task.get("vectors"),
        "capture_cliques": bool(task.get("capture_cliques", False)),
        "defines": dict(sorted(task.get("defines", {}).items())),
        "warmup": bool(task.get("warmup", False)),
        "repetition": repetition,
    }
    if normalized["algorithm"] not in (1, 2, 3, 4):
        raise ValueError("algorithm must be one of 1, 2, 3, 4")
    if normalized["mode"] not in ("raw", "stats"):
        raise ValueError("mode must be raw or stats")
    if normalized["algorithm"] != 1 and normalized["tau"] is None:
        raise ValueError("tau is required for algorithms 2-4")
    if not (1 <= normalized["threads"] <= 32):
        raise ValueError("threads must be in [1, 32]")
    if normalized["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be positive; it is the external TLE")
    return normalized


def expand_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for task in plan.get("tasks", []):
        repetitions = int(task.get("repetitions", 1))
        if repetitions < 1:
            raise ValueError("repetitions must be positive")
        for repetition in range(repetitions):
            normalized = canonical_task(task, repetition)
            task_json = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
            normalized["task_id"] = hashlib.sha256(task_json.encode()).hexdigest()[:16]
            expanded.append(normalized)
    if not expanded:
        raise ValueError("The plan contains no tasks")
    return expanded


def build_key(task: dict[str, Any]) -> tuple[int, str, tuple[tuple[str, Any], ...]]:
    return task["algorithm"], task["mode"], tuple(task["defines"].items())


def build_executable(task: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    algorithm, mode, defines_tuple = build_key(task)
    source = SOURCE_DIR / f"alg{algorithm}-{mode}.cpp"
    if not source.exists():
        raise FileNotFoundError(source)
    defines = [f"-D{name}={value}" for name, value in defines_tuple]
    fingerprint_input = json.dumps(
        {
            "algorithm": algorithm,
            "mode": mode,
            "defines": defines_tuple,
            "source_hashes": {
                path.name: sha256(path)
                for path in (
                    source,
                    SOURCE_DIR / f"alg{algorithm}-raw.cpp",
                    SOURCE_DIR / "semantic_graph.cpp",
                    SOURCE_DIR / "semantic_graph.h",
                    SOURCE_DIR / "experiment_runtime.h",
                )
                if path.exists()
            },
        },
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(fingerprint_input.encode()).hexdigest()[:16]
    executable = BUILD_DIR / f"alg{algorithm}-{mode}-{fingerprint}.exe"
    command = [
        "g++",
        "-std=c++17",
        "-fopenmp",
        "-mavx2",
        "-O3",
        *defines,
        str(source),
        str(SOURCE_DIR / "semantic_graph.cpp"),
        "-o",
        str(executable),
    ]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if not executable.exists():
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Build failed for alg{algorithm}-{mode}:\n{completed.stderr}"
            )
    metadata = {
        "command": command,
        "fingerprint": fingerprint,
        "executable": str(executable),
        "executable_sha256": sha256(executable),
    }
    return executable, metadata


def parse_stats(text: str) -> dict[str, int | float | str]:
    values: dict[str, int | float | str] = {}
    for match in STAT_RE.finditer(text):
        for field in match.group(1).split():
            if "=" not in field:
                continue
            key, raw = field.split("=", 1)
            try:
                values[key] = int(raw)
            except ValueError:
                try:
                    values[key] = float(raw)
                except ValueError:
                    values[key] = raw
    return values


def parse_completed_run(stdout: str) -> dict[str, Any] | None:
    totals = TOTAL_RE.findall(stdout)
    if not totals:
        return None
    milliseconds, clique_count = totals[-1]
    peaks = PEAK_RE.findall(stdout)
    histogram = {int(size): int(count) for size, count in SIZE_RE.findall(stdout)}
    return {
        "algorithm_ms": int(milliseconds),
        "clique_count": int(clique_count),
        "peak_memory_gib": float(peaks[-1]) if peaks else None,
        "size_histogram": histogram,
        "stats": parse_stats(stdout),
    }


def next_attempt(task_dir: Path, task_id: str) -> int:
    attempts = list(task_dir.glob(f"{task_id}.attempt-*.json"))
    return len(attempts) + 1


def completed_ids(task_dir: Path) -> set[str]:
    done: set[str] = set()
    for result_path in task_dir.glob("*.attempt-*.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("status") == "completed":
            done.add(result["task"]["task_id"])
    return done


def attempted_ids(task_dir: Path) -> set[str]:
    attempted: set[str] = set()
    for result_path in task_dir.glob("*.attempt-*.json"):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            attempted.add(result["task"]["task_id"])
        except (KeyError, OSError, json.JSONDecodeError):
            continue
    return attempted


def load_clique_set(path: Path) -> set[tuple[int, ...]]:
    cliques: set[tuple[int, ...]] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            clique = tuple(sorted(map(int, line.split())))
            if len(clique) != len(set(clique)):
                raise ValueError(f"Repeated vertex in {path}:{line_number}")
            cliques.add(clique)
    return cliques


def validate_captured_output(
    task_dir: Path, task: dict[str, Any], clique_path: Path
) -> dict[str, Any]:
    current = load_clique_set(clique_path)
    validation: dict[str, Any] = {
        "unique_cliques": len(current),
        "compared_with": [],
        "exact_match": True,
    }
    for result_path in task_dir.glob("*.attempt-*.json"):
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        previous_task = previous.get("task", {})
        if previous.get("status") != "completed":
            continue
        if not previous_task.get("capture_cliques"):
            continue
        if (previous_task.get("algorithm") == 1) != (task.get("algorithm") == 1):
            continue
        comparable_fields = ("dataset", "tau", "graph", "vectors", "repetition")
        if any(previous_task.get(key) != task.get(key) for key in comparable_fields):
            continue
        previous_path_raw = previous.get("artifacts", {}).get("cliques")
        if not previous_path_raw:
            continue
        previous_path = Path(previous_path_raw)
        previous_set = load_clique_set(previous_path)
        same = current == previous_set
        validation["compared_with"].append(
            {
                "task_id": previous_task.get("task_id"),
                "algorithm": previous_task.get("algorithm"),
                "exact_match": same,
                "only_current": len(current - previous_set),
                "only_previous": len(previous_set - current),
            }
        )
        validation["exact_match"] = validation["exact_match"] and same
    return validation


def execute_task(
    task: dict[str, Any],
    executable: Path,
    build_metadata: dict[str, Any],
    task_dir: Path,
    run_metadata: dict[str, Any],
    rerun_reason: str | None,
    input_metadata: dict[str, Any],
    graph_path: Path,
    vector_path: Path | None,
) -> dict[str, Any]:
    task_id = task["task_id"]
    attempt = next_attempt(task_dir, task_id)
    if attempt > 1 and not rerun_reason:
        raise ValueError(
            f"Task {task_id} has a previous attempt; provide --rerun-reason"
        )
    stem = f"{task_id}.attempt-{attempt:03d}"
    program_log = task_dir / f"{stem}.program.log"
    stdout_path = task_dir / f"{stem}.stdout.txt"
    stderr_path = task_dir / f"{stem}.stderr.txt"
    clique_path = task_dir / f"{stem}.cliques.txt"

    command = [
        str(executable),
        "dataset",
        str(task["dataset"]),
        "threads",
        str(task["threads"]),
        "timeout",
        "0",
        "log",
        str(program_log),
    ]
    command.extend(["graph", str(graph_path)])
    if vector_path is not None:
        command.extend(["vectors", str(vector_path)])

    input_lines: list[str] = []
    if task["capture_cliques"]:
        input_lines.append(f"sink {clique_path}")
    if task["algorithm"] == 1:
        input_lines.append("mine")
    else:
        input_lines.append(f"mine {task['tau']:.17g}")
    input_lines.append("quit")
    stdin_text = "\n".join(input_lines) + "\n"

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    status = "crash"
    return_code: int | None = None
    stdout = ""
    stderr = ""
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=task["timeout_seconds"],
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        parsed = parse_completed_run(stdout)
        if return_code == 0 and parsed is not None:
            status = "completed"
        elif "bad_alloc" in (stdout + stderr) or "memory" in stderr.lower():
            status = "oom"
        elif return_code == 0:
            status = "parse_error"
    except subprocess.TimeoutExpired as error:
        status = "tle"
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        parsed = None

    wall_seconds = time.perf_counter() - started
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    if status != "completed":
        parsed = None
    validation = None
    if status == "completed" and task["capture_cliques"]:
        validation = validate_captured_output(task_dir, task, clique_path)
        if not validation["exact_match"]:
            status = "mismatch"
    result = {
        "schema_version": 1,
        "status": status,
        "attempt": attempt,
        "rerun_reason": rerun_reason if attempt > 1 else None,
        "started_utc": started_utc,
        "wall_seconds": wall_seconds,
        "return_code": return_code,
        "task": task,
        "command": command,
        "build": build_metadata,
        "run": run_metadata,
        "inputs": input_metadata,
        "parsed": parsed,
        "validation": validation,
        "artifacts": {
            "program_log": str(program_log),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "cliques": str(clique_path) if task["capture_cliques"] else None,
        },
    }
    result_path = task_dir / f"{stem}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-attempted",
        action="store_true",
        help="With --resume, skip every recorded attempt, including TLE/OOM/crash.",
    )
    parser.add_argument(
        "--rerun-reason",
        help="Required when a failed/TLE/OOM task is attempted again.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue after TLE/OOM/crash/parse failure (default: fail fast).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (1 <= args.batch_size <= MAX_BATCH_SIZE):
        raise ValueError(f"batch-size must be in [1, {MAX_BATCH_SIZE}]")
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = expand_plan(plan)
    run_name = plan.get("name", plan_path.stem)
    task_dir = OUTPUT_ROOT / run_name
    task_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_attempted and not args.resume:
        raise ValueError("--skip-attempted requires --resume")
    if args.resume:
        done = attempted_ids(task_dir) if args.skip_attempted else completed_ids(task_dir)
    else:
        done = set()
    pending = [task for task in tasks if task["task_id"] not in done]
    selected = pending[: args.batch_size]
    if not selected:
        print("No pending tasks.")
        return

    run_metadata = {
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "git": git_state(),
        "machine": machine_state(),
        "memory_limit_gib": 16,
        "external_timeout_policy": "hard per-process TLE from task.timeout_seconds",
    }
    builds: dict[tuple[int, str, tuple[tuple[str, Any], ...]], tuple[Path, dict[str, Any]]] = {}
    inputs: dict[tuple[str, str | None], dict[str, Any]] = {}
    processed = 0
    for index, task in enumerate(selected, 1):
        key = build_key(task)
        if key not in builds:
            builds[key] = build_executable(task)
        executable, build_metadata = builds[key]
        graph_path, vector_path = resolve_inputs(task)
        input_key = (str(graph_path), str(vector_path) if vector_path else None)
        if input_key not in inputs:
            inputs[input_key] = fingerprint_inputs(graph_path, vector_path)
        print(
            f"[{index}/{len(selected)}] alg{task['algorithm']}-{task['mode']} "
            f"ds={task['dataset']} tau={task['tau']} rep={task['repetition']}"
        )
        result = execute_task(
            task,
            executable,
            build_metadata,
            task_dir,
            run_metadata,
            args.rerun_reason,
            inputs[input_key],
            graph_path,
            vector_path,
        )
        processed += 1
        print(f"  {result['status']} in {result['wall_seconds']:.3f}s")
        if result["status"] != "completed" and not args.continue_on_failure:
            print("Stopped after the first failed task; inspect its artifacts.")
            break

    remaining = len(pending) - processed
    if remaining > 0:
        print(f"{remaining} task(s) remain; rerun with --resume for the next batch.")


if __name__ == "__main__":
    main()
