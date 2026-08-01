"""Export the formal engineering profile sweep and the selected paper rows."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUMMARY = ROOT / "experiments" / "summaries" / "batch30-engineering-profile-selection-30s.summary.json"
DATA_ROOT = ROOT / "tex-data" / "data"


DEFAULTS = {
    2: {"bitmap": 1, "reorder": 1, "incremental_acc": 0},
    3: {"bitmap": 1, "reorder": 1, "incremental_acc": None},
    4: {"bitmap": 1, "reorder": 1, "incremental_acc": 1},
}
DEFINE_TO_FIELD = {
    "ACMSC_ENGINE_BITMAP": "bitmap",
    "ACMSC_ENGINE_REORDER": "reorder",
    "ACMSC_ENGINE_INCREMENTAL_ACC": "incremental_acc",
}
METHODS = {2: "SemBK", 3: "StrSub", 4: "MonoSemMCE"}
DATASETS = {7: "FB15K-237", 16: "email-Enron", 1: "sc-ldoor"}


def load_groups() -> list[dict]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))["groups"]


def resolved(group: dict) -> dict[str, int | None]:
    algorithm = int(group["configuration"]["algorithm"])
    profile = dict(DEFAULTS[algorithm])
    for define, value in group["configuration"]["defines"].items():
        field = DEFINE_TO_FIELD.get(define)
        if field is not None:
            profile[field] = int(value)
    return profile


def find(groups: list[dict], algorithm: int, dataset: int, tau: float,
         *, bitmap: int, reorder: int, incremental_acc: int | None) -> dict:
    expected = {
        "bitmap": bitmap,
        "reorder": reorder,
        "incremental_acc": incremental_acc,
    }
    for group in groups:
        config = group["configuration"]
        if (
            int(config["algorithm"]) == algorithm
            and int(config["dataset"]) == dataset
            and abs(float(config["tau"]) - tau) < 1e-9
            and resolved(group) == expected
        ):
            return group
    raise ValueError(f"Missing profile: alg={algorithm}, dataset={dataset}, tau={tau}, {expected}")


def seconds(group: dict) -> float:
    return float(group["timing"]["algorithm_ms_median"]) / 1000.0


def selected_row(method: str, workload: str, choice: str,
                 chosen: dict, alternative: dict) -> list[object]:
    if chosen["result"] != alternative["result"]:
        raise ValueError(f"Output mismatch for {method} / {choice}")
    chosen_s = seconds(chosen)
    alternative_s = seconds(alternative)
    return [
        method,
        workload,
        choice,
        f"{chosen_s:.3f}",
        f"{alternative_s:.3f}",
        f"{alternative_s / chosen_s:.3f}",
        chosen["result"]["clique_count"],
        min(chosen["timing"]["completed_repetitions"],
            alternative["timing"]["completed_repetitions"]),
        "batch30-engineering-profile-selection-30s",
    ]


def main() -> None:
    groups = load_groups()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    profile_path = DATA_ROOT / "engineering_profile_selection.csv"
    with profile_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "method", "dataset", "tau", "bitmap", "reorder",
            "incremental_acc", "median_seconds", "min_seconds", "max_seconds",
            "memory_gib", "output_count", "repetitions", "source_summary",
        ])
        for group in sorted(groups, key=lambda item: (
            item["configuration"]["algorithm"], item["configuration"]["dataset"],
            item["configuration"]["tau"], json.dumps(item["configuration"]["defines"], sort_keys=True),
        )):
            config = group["configuration"]
            timing = group["timing"]
            profile = resolved(group)
            writer.writerow([
                METHODS[int(config["algorithm"])],
                DATASETS[int(config["dataset"])],
                config["tau"],
                profile["bitmap"],
                profile["reorder"],
                "" if profile["incremental_acc"] is None else profile["incremental_acc"],
                f"{timing['algorithm_ms_median'] / 1000:.3f}",
                f"{timing['algorithm_ms_min'] / 1000:.3f}",
                f"{timing['algorithm_ms_max'] / 1000:.3f}",
                timing["peak_memory_gib_median"],
                group["result"]["clique_count"],
                timing["completed_repetitions"],
                "batch30-engineering-profile-selection-30s",
            ])

    sembk_sparse = find(groups, 2, 16, 0.0525957, bitmap=0, reorder=1, incremental_acc=0)
    sembk_bitmap = find(groups, 2, 16, 0.0525957, bitmap=1, reorder=1, incremental_acc=0)
    sembk_legacy_acc = find(groups, 2, 16, 0.0525957, bitmap=1, reorder=1, incremental_acc=1)
    strsub_full = find(groups, 3, 7, 0.234886, bitmap=1, reorder=1, incremental_acc=None)
    strsub_no_bitmap = find(groups, 3, 7, 0.234886, bitmap=0, reorder=1, incremental_acc=None)
    strsub_no_reorder = find(groups, 3, 7, 0.234886, bitmap=1, reorder=0, incremental_acc=None)
    mono_full = find(groups, 4, 7, 0.537601, bitmap=1, reorder=1, incremental_acc=1)
    mono_no_acc = find(groups, 4, 7, 0.537601, bitmap=1, reorder=1, incremental_acc=0)

    selected = [
        selected_row("SemBK", "email-Enron q80", "sparse_rows", sembk_sparse, sembk_bitmap),
        selected_row("SemBK", "email-Enron q80", "direct_sum", sembk_bitmap, sembk_legacy_acc),
        selected_row("StrSub", "FB15K-237 q80", "sparse_blocked_bitmap", strsub_full, strsub_no_bitmap),
        selected_row("StrSub", "FB15K-237 q80", "vector_reorder", strsub_full, strsub_no_reorder),
        selected_row("MonoSemMCE", "FB15K-237 q99", "incremental_accumulator", mono_full, mono_no_acc),
    ]
    selected_path = DATA_ROOT / "engineering_ablation.csv"
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([
            "method", "workload", "design_choice", "chosen_seconds",
            "alternative_seconds", "speedup", "output_count", "repetitions",
            "source_summary",
        ])
        writer.writerows(selected)

    print(f"Wrote {profile_path}")
    print(f"Wrote {selected_path}")


if __name__ == "__main__":
    main()
