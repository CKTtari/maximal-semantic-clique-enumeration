"""Generate all data-driven figures used by main.tex.

Inputs are canonical CSV files in ../data. The CSV files are produced only by
src/experiments/standard/paper_results_export.py from archived summaries.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ROOT = HERE.parents[1]
QUALITY = ROOT / "experiments" / "quality"

BLUE = "#2F6DB0"
RED = "#C83E4D"
GOLD = "#E3A008"
GREEN = "#2E8B57"
GRAY = "#5A5A5A"

COLORS = {"SemBK": GOLD, "StrSub": BLUE, "MonoSemMCE": RED}
MARKERS = {"SemBK": "^", "StrSub": "o", "MonoSemMCE": "s"}
LINESTYLES = {"SemBK": ":", "StrSub": "-", "MonoSemMCE": "--"}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 20,
            "axes.labelsize": 22,
            "axes.titlesize": 22,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "axes.linewidth": 1.4,
            "lines.linewidth": 3.0,
            "lines.markersize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle="--", linewidth=1.0, alpha=0.25)
    ax.tick_params(width=1.2, length=5)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def threshold_runtime() -> None:
    rows = read_csv("threshold_sensitivity.csv")
    panels = [
        ("ogbn-arxiv", ["StrSub", "MonoSemMCE"]),
        ("sc-ldoor", ["StrSub", "MonoSemMCE"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))

    for panel_idx, (ax, (dataset, methods)) in enumerate(zip(axes, panels)):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        quantiles = sorted({int(row["quantile"]) for row in dataset_rows})
        positions = {quantile: index for index, quantile in enumerate(quantiles)}
        for method in methods:
            method_rows = {int(row["quantile"]): row for row in dataset_rows if row["algorithm"] == method}
            completed = [row for row in method_rows.values() if row["status"] == "completed" and row["seconds"]]
            if not completed:
                continue
            x = np.arange(len(quantiles), dtype=float)
            y = np.asarray(
                [
                    float(method_rows[q]["seconds"])
                    if q in method_rows and method_rows[q]["status"] == "completed" and method_rows[q]["seconds"]
                    else np.nan
                    for q in quantiles
                ],
                dtype=float,
            )
            ax.plot(
                x,
                y,
                color=COLORS[method],
                marker=MARKERS[method],
                linestyle=LINESTYLES[method],
                label=method,
                markeredgecolor="black",
                markeredgewidth=0.7,
            )
            tle_x = [positions[q] for q, row in method_rows.items() if row["status"] == "TLE"]
            if tle_x:
                cap = max(float(row["seconds"]) for row in completed) * 1.8
                ax.scatter(tle_x, [cap] * len(tle_x), marker="x", s=120, linewidths=2.4, color=COLORS[method])
        ax.set_yscale("log")
        shown = [q for q in quantiles if q in {5, 50, 80, 95, 99}]
        ax.set_xticks([positions[q] for q in shown])
        ax.set_xticklabels([str(q) for q in shown])
        ax.set_xlabel(f"({chr(97 + panel_idx)}) {dataset}")
        clean_axis(ax)
    axes[0].set_ylabel("Time (s)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10), frameon=False)
    fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.90))
    save(fig, "threshold_runtime")


def scale_regime() -> None:
    rows = read_csv("scale_regime.csv")
    sizes = [25000, 50000, 100000, 169343]
    quantiles = [20, 50, 80, 90, 95, 99]
    by_key = {(int(row["vertices"]), int(row["quantile"])): row for row in rows}
    matrix = np.asarray(
        [[float(by_key[(size, quantile)]["log2_t4_over_t3"]) for quantile in quantiles] for size in sizes]
    )

    fig, ax = plt.subplots(figsize=(13.8, 5.3))
    limit = max(abs(float(matrix.min())), abs(float(matrix.max())))
    image = ax.imshow(matrix, cmap="RdYlBu", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), aspect="auto")
    ax.set_xticks(np.arange(len(quantiles)))
    ax.set_xticklabels([f"q{q}" for q in quantiles])
    ax.set_yticks(np.arange(len(sizes)))
    ax.set_yticklabels(["25K", "50K", "100K", "169K"])
    ax.set_xlabel("Edge-similarity threshold")
    ax.set_ylabel("Vertices")
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            color = "white" if abs(value) > 0.62 * limit else "black"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=20, color=color)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025, fraction=0.045)
    colorbar.set_label(r"$\log_2(T_{\mathrm{Mono}}/T_{\mathrm{StrSub}})$", fontsize=22)
    colorbar.ax.tick_params(labelsize=20)
    fig.tight_layout(pad=0.35)
    save(fig, "scale_regime")


def parallel_scaling() -> None:
    rows = read_csv("parallel_scaling.csv")
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    x = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    for algorithm, dataset, quantile in [
        ("StrSub", "FB15K-237", 80),
        ("MonoSemMCE", "ogbn-arxiv", 50),
    ]:
        selected = sorted(
            [row for row in rows if row["algorithm"] == algorithm], key=lambda row: int(row["threads"])
        )
        y = np.asarray([float(row["speedup"]) for row in selected])
        ax.plot(
            x,
            y,
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            linestyle=LINESTYLES[algorithm],
            markeredgecolor="black",
            markeredgewidth=0.7,
            label=algorithm,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(value)) for value in x])
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.set_ylim(0.8, 6.0)
    clean_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.91))
    save(fig, "parallel_scaling")


def mechanism_summary() -> None:
    alg3_rows = read_csv("alg3_mechanism_counters.csv")
    alg4_rows = read_csv("alg4_state_contraction.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1.10]})

    alg3_rows = [row for row in alg3_rows if row["variant"] != "No Local-Max"]
    variants = [row["variant"] for row in alg3_rows]
    tests = np.asarray([float(row["subset_tests"]) / 1e6 for row in alg3_rows])
    bars = axes[0].bar(
        np.arange(len(variants)),
        tests,
        color=[BLUE, GOLD, RED],
        edgecolor="black",
        linewidth=1.0,
    )
    axes[0].set_xticks(np.arange(len(variants)))
    axes[0].set_xticklabels(
        ["Full", "Top off", "Vertex off"],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    axes[0].set_ylabel("Subset tests (M)")
    axes[0].set_xlabel("(a) StrSub variants")
    axes[0].bar_label(bars, fmt="%.1f", fontsize=20, padding=4)
    axes[0].set_ylim(0, max(tests) * 1.18)
    clean_axis(axes[0])

    quantiles = np.asarray([int(row["quantile"]) for row in alg4_rows])
    positions = np.arange(len(quantiles))
    states = np.asarray([int(row["states"]) for row in alg4_rows])
    outputs = np.asarray([int(row["output"]) for row in alg4_rows])
    axes[1].plot(positions, states, color=RED, marker="s", linestyle="--", label="Canonical states")
    axes[1].plot(positions, outputs, color=BLUE, marker="o", linestyle="-", label="ACMSCs")
    axes[1].set_yscale("log")
    shown = [index for index, value in enumerate(quantiles) if value in {5, 80, 99}]
    axes[1].set_xticks(shown)
    axes[1].set_xticklabels([str(quantiles[index]) for index in shown])
    axes[1].set_xlabel("(b) Threshold quantile (%)")
    axes[1].set_ylabel("Count (log)")
    clean_axis(axes[1])
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.67, 1.03), ncol=2, frameon=False)

    fig.tight_layout(w_pad=1.1, pad=0.35, rect=(0, 0, 1, 0.84))
    save(fig, "mechanism_summary")


def result_quality() -> None:
    rows = read_csv("result_quality_sweep.csv")
    colors = {"StructBK": BLUE, "PairSim-MCE": GOLD, "ACMSC": GREEN}
    markers = {"StructBK": "D", "PairSim-MCE": "s", "ACMSC": "o"}
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for algorithm in ("PairSim-MCE", "ACMSC"):
        selected = sorted(
            (row for row in rows if row["algorithm"] == algorithm),
            key=lambda row: int(row["quantile"]),
        )
        x = [100.0 * float(row["coverage_fraction"]) for row in selected]
        y = [float(row["mean_purity"]) for row in selected]
        ax.plot(
            x,
            y,
            color=colors[algorithm],
            marker=markers[algorithm],
            label=algorithm,
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        if algorithm == "ACMSC":
            offsets = [(-9.5, -0.007), (1.0, 0.004), (1.0, 0.004), (1.0, 0.004), (1.0, -0.007)]
            for row, x_value, y_value, (dx, dy) in zip(selected, x, y, offsets):
                ax.text(x_value + dx, y_value + dy, f"q{row['quantile']}", fontsize=20)

    structural = next(row for row in rows if row["algorithm"] == "StructBK")
    ax.scatter(
        [100.0 * float(structural["coverage_fraction"])],
        [float(structural["mean_purity"])],
        color=colors["StructBK"],
        marker=markers["StructBK"],
        s=120,
        edgecolor="black",
        linewidth=0.7,
        label="StructBK",
        zorder=5,
    )
    ax.set_xlabel("Covered vertices (%)")
    ax.set_ylabel("Mean subject purity")
    ax.set_xlim(4, 104)
    ax.set_ylim(0.81, 0.925)
    clean_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.90))
    save(fig, "result_quality")


def result_distribution() -> None:
    rows = read_csv("result_distribution.csv")
    quantiles = [20, 50, 80, 95, 99]
    datasets = ["FB15K-237", "ogbn-arxiv", "sc-nasasrb", "sc-ldoor"]
    colors = {
        "FB15K-237": RED,
        "ogbn-arxiv": BLUE,
        "sc-nasasrb": GOLD,
        "sc-ldoor": GREEN,
        "Geometric mean": GRAY,
    }
    markers = {
        "FB15K-237": "o",
        "ogbn-arxiv": "s",
        "sc-nasasrb": "^",
        "sc-ldoor": "v",
        "Geometric mean": "D",
    }
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.3), sharex=True)
    positions = np.arange(len(quantiles), dtype=float)
    for dataset in datasets + ["Geometric mean"]:
        selected = {
            int(row["quantile"]): row for row in rows if row["dataset"] == dataset
        }
        linewidth = 4.2 if dataset == "Geometric mean" else 2.3
        alpha = 1.0 if dataset == "Geometric mean" else 0.80
        for ax, field in zip(axes, ("count_ratio", "mean_size_ratio")):
            ax.plot(
                positions,
                [float(selected[q][field]) for q in quantiles],
                color=colors[dataset],
                marker=markers[dataset],
                linewidth=linewidth,
                alpha=alpha,
                label=dataset,
                markeredgecolor="black",
                markeredgewidth=0.6,
            )
    for ax in axes:
        ax.set_yscale("log")
        ax.axhline(1.0, color="black", linewidth=1.2, linestyle=":")
        clean_axis(ax)
    axes[0].set_ylabel("Output-count ratio")
    axes[1].set_ylabel("Mean-size ratio")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels([f"q{quantile}" for quantile in quantiles])
    axes[1].set_xlabel("Threshold quantile")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.tight_layout(h_pad=0.6, pad=0.35, rect=(0, 0, 1, 0.83))
    save(fig, "result_distribution")


def case_network() -> None:
    metadata = json.loads((QUALITY / "ogbn-arxiv-case-metadata.json").read_text(encoding="utf-8"))
    papers = metadata["papers"]
    vertices = [int(paper["vertex"]) for paper in papers]
    index = {vertex: position for position, vertex in enumerate(vertices)}
    similarities = np.eye(len(vertices), dtype=float)
    for row in read_csv("ogbn-arxiv-case-similarity.csv"):
        left = index[int(row["left"])]
        right = index[int(row["right"])]
        similarities[left, right] = similarities[right, left] = float(row["similarity"])

    squared_distances = np.maximum(0.0, 2.0 - 2.0 * similarities)
    centering = np.eye(len(vertices)) - np.ones((len(vertices), len(vertices))) / len(vertices)
    gram = -0.5 * centering @ squared_distances @ centering
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    coordinates = vectors[:, order] * np.sqrt(np.maximum(values[order], 0.0))
    for dimension in range(2):
        pivot = np.argmax(np.abs(coordinates[:, dimension]))
        if coordinates[pivot, dimension] < 0:
            coordinates[:, dimension] *= -1
    coordinates /= max(np.linalg.norm(coordinates, axis=1).max(), 1e-9)
    coordinates *= 0.86
    original_coordinates = coordinates.copy()
    for _ in range(500):
        displacement = 0.012 * (original_coordinates - coordinates)
        for left in range(len(vertices)):
            for right in range(left + 1, len(vertices)):
                delta = coordinates[left] - coordinates[right]
                distance = max(float(np.linalg.norm(delta)), 1e-6)
                if distance >= 0.31:
                    continue
                push = 0.045 * (0.31 - distance) * delta / distance
                displacement[left] += push
                displacement[right] -= push
        coordinates += displacement
        coordinates -= coordinates.mean(axis=0)
    coordinates /= max(np.linalg.norm(coordinates, axis=1).max(), 1e-9)
    coordinates *= 0.96

    author_counts = Counter(
        author.split()[-1]
        for paper in papers
        for author in paper["authors"]
    )
    recurrent_authors = [
        author for author, count in sorted(author_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 3
    ]
    angles = np.linspace(0.15 * np.pi, 2.15 * np.pi, len(recurrent_authors), endpoint=False)
    author_positions = {
        author: np.array([1.34 * np.cos(angle), 1.16 * np.sin(angle)])
        for author, angle in zip(recurrent_authors, angles)
    }

    fig, ax = plt.subplots(figsize=(7.2, 6.8))
    tau = 0.888756
    semantic_backbone = set()
    for left in range(len(vertices)):
        nearest = sorted(
            (right for right in range(len(vertices)) if right != left),
            key=lambda right: (-similarities[left, right], right),
        )[:2]
        semantic_backbone.update(tuple(sorted((left, right))) for right in nearest)
    for left in range(len(vertices)):
        for right in range(left + 1, len(vertices)):
            similarity = similarities[left, right]
            below = similarity < tau
            if not below and (left, right) not in semantic_backbone:
                continue
            ax.plot(
                [coordinates[left, 0], coordinates[right, 0]],
                [coordinates[left, 1], coordinates[right, 1]],
                color=RED if below else BLUE,
                linewidth=1.8 if below else 2.0,
                linestyle="--" if below else "-",
                alpha=0.58 if below else 0.34,
                zorder=1,
            )
    for paper_index, paper in enumerate(papers):
        surnames = {author.split()[-1] for author in paper["authors"]}
        for author in recurrent_authors:
            if author not in surnames:
                continue
            position = author_positions[author]
            ax.plot(
                [coordinates[paper_index, 0], position[0]],
                [coordinates[paper_index, 1], position[1]],
                color=GOLD,
                linewidth=1.4,
                alpha=0.52,
                zorder=2,
            )

    year_colors = {2016: BLUE, 2017: GREEN, 2018: GOLD, 2019: RED}
    for year, color in year_colors.items():
        selected = [idx for idx, paper in enumerate(papers) if int(paper["year"]) == year]
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=320,
            color=color,
            edgecolor="black",
            linewidth=1.1,
            zorder=4,
        )
    for paper_index, paper in enumerate(papers):
        ax.text(
            coordinates[paper_index, 0],
            coordinates[paper_index, 1],
            str(paper["case_index"]),
            ha="center",
            va="center",
            fontsize=20,
            color="white" if int(paper["year"]) in (2016, 2017, 2019) else "black",
            zorder=5,
        )
    for author, position in author_positions.items():
        ax.scatter([position[0]], [position[1]], marker="s", s=380, color="white", edgecolor=GOLD, linewidth=3.0, zorder=5)
        ax.text(position[0], position[1] - 0.13, author, ha="center", va="top", fontsize=20, zorder=6)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="black", markersize=12, label=str(year))
        for year, color in year_colors.items()
    ]
    handles.append(Line2D([0], [0], marker="s", color=GOLD, markerfacecolor="white", markersize=12, label="Author"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=5, frameon=False, handletextpad=0.3, columnspacing=0.8)
    ax.set_xlim(-1.58, 1.58)
    ax.set_ylim(-1.40, 1.40)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.93))
    save(fig, "quality_case_network")


def main() -> None:
    configure_style()
    threshold_runtime()
    scale_regime()
    parallel_scaling()
    mechanism_summary()
    result_distribution()
    result_quality()
    case_network()
    print(f"Generated figures in {HERE}")


if __name__ == "__main__":
    main()
