#!/usr/bin/env python3
"""Run one external baseline with bounded time, memory monitoring, and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def process_tree_rss(root: psutil.Process) -> int:
    total = 0
    try:
        processes = [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return total


def terminate_tree(root: psutil.Process) -> None:
    processes = root.children(recursive=True)
    for process in reversed(processes):
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    try:
        root.kill()
    except (psutil.AccessDenied, psutil.NoSuchProcess):
        pass
    psutil.wait_procs([*processes, root], timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("--allow-nonzero", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A command is required after --")

    cwd = args.cwd.resolve()
    prefix = args.output_prefix.resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = prefix.with_suffix(".stdout.txt")
    stderr_path = prefix.with_suffix(".stderr.txt")
    result_path = prefix.with_suffix(".json")
    executable = Path(command[0]).resolve()
    memory_limit = int(args.memory_gib * (1 << 30))

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    peak_rss = 0
    status = "running"
    return_code = None
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        child = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            peak_rss = max(peak_rss, process_tree_rss(process))
            elapsed = time.perf_counter() - started
            if peak_rss > memory_limit:
                status = "oom"
                terminate_tree(process)
                break
            if elapsed > args.timeout_seconds:
                status = "tle"
                terminate_tree(process)
                break
            time.sleep(0.1)
        return_code = child.poll()
        peak_rss = max(peak_rss, process_tree_rss(process))
        if status == "running":
            status = "completed" if return_code == 0 or args.allow_nonzero else "crash"

    result = {
        "schema_version": 1,
        "status": status,
        "started_utc": started_utc,
        "wall_seconds": time.perf_counter() - started,
        "return_code": return_code,
        "peak_memory_gib": peak_rss / (1 << 30),
        "timeout_seconds": args.timeout_seconds,
        "memory_limit_gib": args.memory_gib,
        "allow_nonzero": args.allow_nonzero,
        "cwd": str(cwd),
        "command": command,
        "executable": str(executable),
        "executable_sha256": sha256(executable),
        "artifacts": {"stdout": str(stdout_path), "stderr": str(stderr_path)},
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
