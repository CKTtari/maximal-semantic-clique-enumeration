"""Generate the paper's data-driven figures from canonical experiment CSVs.

The figures use high-resolution source canvases for PVLDB single-column placement. Multi-panel
figures are exported as independent panels and composed with LaTeX subfigures.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ROOT = HERE.parents[1]
QUALITY = ROOT / "experiments" / "quality"
QA_DIR = ROOT / "tmp" / "figure-qa"

# Repo-local export helpers provide deterministic PDF/SVG/PNG output and layout audits.
SCIPILOT_SCRIPTS = ROOT / "tmp" / "scipilot-figure-skill" / "scripts"
if SCIPILOT_SCRIPTS.is_dir():
    sys.path.insert(0, str(SCIPILOT_SCRIPTS))

try:
    from export_figure import export_figure
    from layout_tools import finalize_figure
    from setup_style import setup_style
    from visual_qa import audit_layout, render_preview
except ImportError as exc:  # pragma: no cover - only used without the local skill
    raise RuntimeError(
        "SciPilot figure helpers are required. Clone scipilot-figure-skill "
        "under tmp/scipilot-figure-skill before regenerating figures."
    ) from exc


COLUMN_WIDTH = 7.10
RQ5_COLUMN_WIDTH = 3.25
RQ5_PANEL_WIDTH = 1.60
FINAL_WIDTH_MM = 89
HALF_COLUMN_WIDTH = 3.42
INSET_COLUMN_WIDTH = 6.40
TIMEOUT_SECONDS = 1200.0
EXPORT_SUFFIXES = (".svg", ".pdf", ".png")

# Okabe-Ito-derived colors with line/marker redundancy for grayscale printing.
BLUE = "#0072B2"
RED = "#D55E00"
CRIMSON = "#C83E4D"
GOLD = "#E69F00"
GREEN = "#009E73"
GRAY = "#5B5B5B"
LIGHT_GRAY = "#B8B8B8"

COLORS = {"SemBK": GOLD, "StrSub": BLUE, "MonoSemMCE": RED}
MARKERS = {"SemBK": "^", "StrSub": "o", "MonoSemMCE": "s"}
LINESTYLES = {"SemBK": ":", "StrSub": "-", "MonoSemMCE": "--"}


def configure_style() -> None:
    setup_style(journal="ieee", lang="en", use_sciplots=False, constrained_layout=True)
    plt.rcParams.update(
        {
            "figure.figsize": (COLUMN_WIDTH, 4.60),
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            "font.size": 20.0,
            "axes.labelsize": 21.0,
            "axes.titlesize": 21.0,
            "xtick.labelsize": 20.0,
            "ytick.labelsize": 20.0,
            "legend.fontsize": 20.0,
            "axes.linewidth": 1.35,
            "lines.linewidth": 2.60,
            "lines.markersize": 9.0,
            "xtick.major.width": 1.25,
            "ytick.major.width": 1.25,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "axes.unicode_minus": False,
        }
    )


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.45, color=LIGHT_GRAY, alpha=0.55)
    ax.set_axisbelow(True)


def legend_above(ax: plt.Axes, handles=None, labels=None, ncol: int = 2) -> None:
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=ncol,
        frameon=False,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.45,
        borderaxespad=0.0,
    )


def save(
    fig: plt.Figure,
    stem: str,
    size: tuple[float, float],
    *,
    adaptive_layout: bool = True,
) -> None:
    fig.set_size_inches(*size)
    if adaptive_layout:
        finalize_figure(fig)
        layout_engine = fig.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(w_pad=0.055, h_pad=0.055)
            fig.canvas.draw()
    else:
        fig.canvas.draw()
    issues = audit_layout(fig)
    failures = [message for severity, message in issues if severity == "FAIL"]
    if failures:
        raise RuntimeError(f"{stem}: SciPilot layout audit failed: {' | '.join(failures)}")
    for severity, message in issues:
        print(f"[{stem}] {severity}: {message}")

    QA_DIR.mkdir(parents=True, exist_ok=True)
    render_preview(fig, str(QA_DIR / f"{stem}.png"), dpi=600)
    export_figure(
        fig,
        str(HERE / stem),
        formats=("svg", "pdf", "png"),
        dpi=600,
        size_inches=size,
        tight=False,
    )
    plt.close(fig)


def threshold_panel(dataset: str, stem: str, show_ylabel: bool) -> None:
    rows = [row for row in read_csv("threshold_sensitivity.csv") if row["dataset"] == dataset]
    quantiles = sorted({int(row["quantile"]) for row in rows})
    positions = {quantile: index for index, quantile in enumerate(quantiles)}
    panel_size = (3.52, 2.65)
    fig, ax = plt.subplots(figsize=panel_size)

    for method in ("StrSub", "MonoSemMCE"):
        by_quantile = {int(row["quantile"]): row for row in rows if row["algorithm"] == method}
        y = [
            float(by_quantile[q]["seconds"])
            if q in by_quantile and by_quantile[q]["status"] == "completed"
            else np.nan
            for q in quantiles
        ]
        ax.plot(
            np.arange(len(quantiles)),
            y,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            label=method,
            markerfacecolor="white",
            markeredgecolor=COLORS[method],
            markeredgewidth=0.85,
        )
        tle_positions = [positions[q] for q, row in by_quantile.items() if row["status"] == "TLE"]
        if tle_positions:
            ax.scatter(
                tle_positions,
                [TIMEOUT_SECONDS] * len(tle_positions),
                marker="x",
                s=24,
                linewidths=1.25,
                color=COLORS[method],
                zorder=5,
            )

    # Every plotted runtime/count is strictly positive; TLE points use TIMEOUT_SECONDS.
    ax.set_yscale("log")
    shown = [q for q in quantiles if q in {5, 50, 80, 99}]
    ax.set_xticks([positions[q] for q in shown], [str(q) for q in shown])
    ax.set_xlabel("Quantile (%)", labelpad=1)
    if show_ylabel:
        ax.set_ylabel("Time (s)", labelpad=2)
    if dataset == "sc-ldoor":
        ax.set_ylim(1.0, 1900.0)
    else:
        ax.set_ylim(0.14, 4.5)
    clean_axis(ax)
    save(fig, stem, panel_size)


def threshold_runtime() -> None:
    handles = [
        Line2D([0], [0], color=COLORS[method], marker=MARKERS[method],
               linestyle=LINESTYLES[method], markerfacecolor="white", label=method)
        for method in ("StrSub", "MonoSemMCE")
    ]
    legend_fig, legend_ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.52))
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", ncol=2, frameon=False,
                     handlelength=1.8, columnspacing=0.9, handletextpad=0.4)
    save(legend_fig, "threshold_runtime_legend", (COLUMN_WIDTH, 0.52))
    threshold_panel("ogbn-arxiv", "threshold_runtime_arxiv", show_ylabel=True)
    threshold_panel("sc-ldoor", "threshold_runtime_ldoor", show_ylabel=False)


def scale_regime() -> None:
    rows = read_csv("scale_regime.csv")
    sizes = [25000, 50000, 100000, 169343]
    quantiles = [20, 50, 80, 90, 95, 99]
    by_key = {(int(row["vertices"]), int(row["quantile"])): row for row in rows}
    matrix = np.asarray(
        [[float(by_key[(size, q)]["log2_t4_over_t3"]) for q in quantiles] for size in sizes]
    )

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 4.35))
    limit = max(abs(float(matrix.min())), abs(float(matrix.max())))
    image = ax.imshow(
        matrix,
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(quantiles)), [f"q{q}" for q in quantiles])
    ax.set_yticks(np.arange(len(sizes)), ["25K", "50K", "100K", "169K"])
    ax.set_xlabel("Threshold quantile")
    ax.set_ylabel("Vertices")
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            color = "white" if abs(value) > 0.58 * limit else "black"
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=20.0, color=color)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", location="top", pad=0.07, fraction=0.10)
    colorbar.set_label(r"$\log_2(T_{\mathrm{Mono}}/T_{\mathrm{StrSub}})$", labelpad=2)
    colorbar.ax.tick_params(labelsize=7.5, length=2)
    save(fig, "scale_regime", (COLUMN_WIDTH, 4.35))


def scale_runtime() -> None:
    """Show absolute runtime growth for each exact search organization."""
    rows = read_csv("scale_regime.csv")
    sizes = [25000, 50000, 100000, 169343]
    positions = np.log2(np.asarray(sizes, dtype=float))
    quantiles = [20, 50, 80, 90, 95, 99]
    by_key = {(int(row["vertices"]), int(row["quantile"])): row for row in rows}
    colors = [BLUE, GOLD, GREEN, "#CC79A7", RED, GRAY]
    markers = ["o", "s", "^", "D", "P", "X"]

    handles = [
        Line2D(
            [0], [0], color=color, marker=marker, linestyle="-",
            markerfacecolor="white", markeredgewidth=0.75,
            label=f"$q_{{{quantile}}}$",
        )
        for quantile, color, marker in zip(quantiles, colors, markers)
    ]
    legend_fig, legend_ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.52))
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=6,
        frameon=False,
        fontsize=24.0,
        handlelength=1.20,
        columnspacing=0.55,
        handletextpad=0.25,
    )
    save(legend_fig, "scale_runtime_legend", (COLUMN_WIDTH, 0.52))

    for method, stem, show_ylabel in (
        ("StrSub", "scale_runtime_strsub", True),
        ("MonoSemMCE", "scale_runtime_monosem", False),
    ):
        fig, axis = plt.subplots(figsize=(HALF_COLUMN_WIDTH, 2.55))
        for quantile, color, marker in zip(quantiles, colors, markers):
            values = []
            for size in sizes:
                row = by_key[(size, quantile)]
                field = "strsub_seconds" if method == "StrSub" else "monosemmce_seconds"
                values.append(float(row[field]))
            axis.plot(
                positions,
                values,
                color=color,
                marker=marker,
                linestyle="-",
                linewidth=2.20,
                markersize=7.2,
                markerfacecolor="white",
                markeredgewidth=0.75,
                label=f"$q_{{{quantile}}}$",
            )
        axis.set_xlim(float(positions[0]) - 0.12, float(positions[-1]) + 0.12)
        axis.set_xticks(positions, ["25", "50", "100", "169"])
        axis.set_yscale("log")
        axis.set_ylim(0.04, 4.1)
        axis.set_xlabel("Vertices (K)", fontsize=23.0)
        axis.tick_params(labelsize=22.0, length=4.0, width=1.0)
        if show_ylabel:
            axis.set_ylabel("Time (s)", fontsize=23.0)
        else:
            axis.tick_params(axis="y", labelleft=False)
        clean_axis(axis)
        save(fig, stem, (HALF_COLUMN_WIDTH, 2.55))


def parallel_scaling() -> None:
    rows = read_csv("parallel_scaling.csv")
    fig, ax = plt.subplots(figsize=(INSET_COLUMN_WIDTH, 3.45))
    threads = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    for algorithm in ("StrSub", "MonoSemMCE"):
        selected = sorted(
            [row for row in rows if row["algorithm"] == algorithm], key=lambda row: int(row["threads"])
        )
        ax.plot(
            threads,
            [float(row["speedup"]) for row in selected],
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            linestyle=LINESTYLES[algorithm],
            markerfacecolor="white",
            markeredgecolor=COLORS[algorithm],
            markeredgewidth=0.85,
            label=algorithm,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads, [str(int(value)) for value in threads])
    ax.set_xlabel("Threads")
    ax.set_ylabel("Speedup")
    ax.set_ylim(0.8, 6.05)
    clean_axis(ax)
    legend_above(ax, ncol=2)
    save(fig, "parallel_scaling", (INSET_COLUMN_WIDTH, 3.45))


def mechanism_summary() -> None:
    alg3_rows = [row for row in read_csv("alg3_mechanism_counters.csv") if row["variant"] != "No Local-Max"]
    labels = ["Full", "No edge", "No vertex"]
    tests = np.asarray([float(row["subset_tests"]) / 1e6 for row in alg3_rows])
    panel_size = (3.52, 2.92)
    fig, ax = plt.subplots(figsize=panel_size)
    fig, ax = plt.subplots(figsize=panel_size)
    bars = ax.bar(
        np.arange(len(labels)),
        tests,
        color=[BLUE, GOLD, RED],
        edgecolor="black",
        linewidth=0.6,
        width=0.64,
    )
    for bar, hatch in zip(bars, ("///", "...", "xxx")):
        bar.set_hatch(hatch)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("Subset tests (M)")
    ax.set_ylim(0, max(tests) * 1.17)
    ax.bar_label(bars, fmt="%.1f", fontsize=13.5, padding=2.0)
    clean_axis(ax)
    fig._layout_engine = None  # lock manual geometry for paired panels
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.23, top=0.65)
    save(fig, "mechanism_strsub", panel_size, adaptive_layout=False)

    rows = read_csv("alg4_state_contraction.csv")
    quantiles = np.asarray([int(row["quantile"]) for row in rows])
    positions = np.arange(len(quantiles))
    mono_panel_size = (RQ5_PANEL_WIDTH, 1.30)
    mono_style = {
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "lines.linewidth": 1.35,
        "lines.markersize": 4.6,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    }

    legend_handles = [
        Line2D([0], [0], color=RED, marker="s", linestyle="--",
               markerfacecolor="white", label="Accepted"),
        Line2D([0], [0], color=BLUE, marker="o", linestyle="-",
               markerfacecolor="white", label="MSCs"),
        Line2D([0], [0], color=CRIMSON, marker="^", linestyle="-",
               markerfacecolor="white", label=r"$W$ prune"),
        Line2D([0], [0], color=GREEN, marker="D", linestyle="--",
               markerfacecolor="white", label="Canonical reject"),
    ]
    with plt.rc_context(mono_style):
        legend_size = (RQ5_COLUMN_WIDTH, 0.30)
        legend_fig, legend_ax = plt.subplots(figsize=legend_size)
        legend_ax.axis("off")
        legend_ax.legend(
            handles=legend_handles, loc="center", ncol=4, frameon=False,
            bbox_to_anchor=(0.08, 0.0, 0.84, 1.0), mode="expand",
            borderaxespad=0.0, fontsize=7.0, markerscale=0.8,
            handlelength=1.45, columnspacing=0.55, handletextpad=0.36,
        )
        save(legend_fig, "mechanism_monosem_legend", legend_size)

    fig, ax = plt.subplots(figsize=mono_panel_size)
    ax.plot(
        positions,
        [int(row["states"]) for row in rows],
        color=RED,
        marker="s",
        linestyle="--",
        markerfacecolor="white",
        markeredgecolor=RED,
        linewidth=1.35,
        markersize=4.6,
        label="Accepted states",
    )
    ax.plot(
        positions,
        [int(row["output"]) for row in rows],
        color=BLUE,
        marker="o",
        linestyle="-",
        markerfacecolor="white",
        markeredgecolor=BLUE,
        linewidth=1.35,
        markersize=4.6,
        label="Emitted MSCs",
    )
    ax.set_yscale("log")
    shown = [i for i, q in enumerate(quantiles) if q in {5, 50, 80, 95, 99}]
    ax.set_xticks(shown, [str(quantiles[i]) for i in shown])
    ax.set_xlabel("Quantile (%)", labelpad=1, fontsize=7.5)
    ax.set_ylabel("Count", labelpad=1, fontsize=7.5)
    ax.tick_params(labelsize=7.0, length=3.0, width=0.8)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(-0.40, len(positions) - 0.60)
    ax.set_ylim(7.0e3, max(int(row["states"]) for row in rows) * 1.9)
    clean_axis(ax)
    fig._layout_engine = None
    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.27, top=0.95)
    save(fig, "mechanism_monosem_states", mono_panel_size, adaptive_layout=False)

    fig, ax = plt.subplots(figsize=mono_panel_size)
    candidate_tests = np.asarray([float(row["candidate_tests"]) for row in rows])
    w_prune_share = 100.0 * np.asarray(
        [float(row["w_prunes"]) for row in rows]
    ) / candidate_tests
    canonical_share = 100.0 * np.asarray(
        [float(row["canonical_rejects"]) for row in rows]
    ) / candidate_tests
    ax.plot(
        positions, w_prune_share, color=CRIMSON, marker="^", linestyle="-",
        markerfacecolor="white", markeredgecolor=CRIMSON,
        linewidth=1.35, markersize=4.6,
    )
    ax.plot(
        positions, canonical_share, color=GREEN, marker="D", linestyle="--",
        markerfacecolor="white", markeredgecolor=GREEN,
        linewidth=1.35, markersize=4.6,
    )
    ax.set_xticks(shown, [str(quantiles[i]) for i in shown])
    ax.set_xlabel("Quantile (%)", labelpad=1, fontsize=7.5)
    ax.set_ylabel("Candidate share (%)", labelpad=1, fontsize=7.5)
    ax.tick_params(labelsize=7.0, length=3.0, width=0.8)
    ax.set_xlim(-0.40, len(positions) - 0.60)
    ax.set_ylim(-3.0, 103.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    clean_axis(ax)
    fig._layout_engine = None
    fig.subplots_adjust(left=0.30, right=0.97, bottom=0.27, top=0.95)
    save(fig, "mechanism_monosem_rejections", mono_panel_size, adaptive_layout=False)


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
    linestyles = {
        "FB15K-237": "-",
        "ogbn-arxiv": "--",
        "sc-nasasrb": "-.",
        "sc-ldoor": ":",
        "Geometric mean": "-",
    }
    positions = np.arange(len(quantiles), dtype=float)

    handles = [
        Line2D(
            [0],
            [0],
            color=colors[dataset],
            marker=markers[dataset],
            linestyle=linestyles[dataset],
            linewidth=2.0 if dataset == "Geometric mean" else 1.05,
            markersize=6.72 if dataset == "Geometric mean" else 5.32,
            markerfacecolor=colors[dataset] if dataset == "Geometric mean" else "white",
            markeredgecolor=colors[dataset],
            label=dataset,
        )
        for dataset in datasets + ["Geometric mean"]
    ]
    handles.append(
        Line2D(
            [0], [0], color="black", linestyle=(0, (1.5, 2.0)),
            linewidth=0.75, label="StructBK = 1",
        )
    )
    legend_fig, legend_ax = plt.subplots(figsize=(COLUMN_WIDTH, 0.88))
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=20.0,
        handlelength=1.9,
        columnspacing=0.85,
        handletextpad=0.38,
    )
    save(legend_fig, "result_distribution_legend", (COLUMN_WIDTH, 0.88))

    for field, ylabel, stem in (
        ("count_ratio", "Output-count ratio", "result_distribution_count"),
        ("mean_size_ratio", "Mean-size ratio", "result_distribution_size"),
    ):
        panel_size = (3.52, 2.65)
        fig, ax = plt.subplots(figsize=panel_size)
        for dataset in datasets + ["Geometric mean"]:
            selected = {int(row["quantile"]): row for row in rows if row["dataset"] == dataset}
            aggregate = dataset == "Geometric mean"
            ax.plot(
                positions,
                [float(selected[q][field]) for q in quantiles],
                color=colors[dataset],
                marker=markers[dataset],
                linestyle=linestyles[dataset],
                linewidth=2.0 if aggregate else 1.05,
                markersize=6.72 if aggregate else 5.32,
                alpha=1.0 if aggregate else 0.82,
                markerfacecolor="white" if not aggregate else colors[dataset],
                markeredgecolor=colors[dataset],
                label=dataset,
            )
        # Every plotted runtime/count is strictly positive; TLE points use TIMEOUT_SECONDS.
        ax.set_yscale("log")
        ax.axhline(1.0, color="black", linewidth=0.75, linestyle=(0, (1.5, 2.0)))
        ax.set_xticks(positions, [f"q{q}" for q in quantiles])
        ax.set_ylabel(ylabel)
        clean_axis(ax)
        save(fig, stem, panel_size)


def rq5_style() -> dict[str, float]:
    return {
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 5.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
    }


def definition_recovery() -> None:
    """Plot published fragmentation metrics for MSC versus PairSim-MCE."""
    summaries = [
        row for row in read_csv("group-recovery.csv")
        if row["comparison"] == "definition"
    ]
    details = [
        row for row in read_csv("group-recovery-detail.csv")
        if row["comparison"] == "definition"
    ]
    datasets = ["GitHub-Social", "ogbn-arxiv-25K"]
    colors = {"GitHub-Social": BLUE, "ogbn-arxiv-25K": CRIMSON}
    markers = {"GitHub-Social": "o", "ogbn-arxiv-25K": "D"}
    size = (RQ5_PANEL_WIDTH, 1.20)

    definition_style = {
        **rq5_style(),
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "lines.markersize": 4.8,
    }
    with plt.rc_context(definition_style):
        handles = [
            Line2D(
                [0], [0], color=colors[dataset], marker=markers[dataset],
                linestyle="none", markeredgecolor="black", markeredgewidth=0.55,
                label=dataset,
            )
            for dataset in datasets
        ]
        handles.append(
            Line2D(
                [0], [0], color=GRAY, linestyle=(0, (2.2, 2.0)),
                linewidth=1.0, label="MSC self-match = 1",
            )
        )
        handles.append(
            Line2D(
                [0], [0], color=GRAY, linestyle=":",
                linewidth=1.0, label="80% retained",
            )
        )
        legend_size = (RQ5_COLUMN_WIDTH, 0.52)
        legend_fig, legend_ax = plt.subplots(figsize=legend_size)
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles, loc="center", ncol=2, frameon=False,
            handlelength=1.8, columnspacing=0.8, handletextpad=0.35,
        )
        save(legend_fig, "definition_recovery_legend", legend_size)

        fig, ax = plt.subplots(figsize=size, layout="none")
        ax.set_position([0.29, 0.29, 0.64, 0.65])
        positions = np.arange(2, dtype=float)
        fields = ["bcubed_recall", "best_match_recall"]
        for offset, dataset in zip((-0.10, 0.10), datasets):
            row = next(item for item in summaries if item["dataset"] == dataset)
            values = [float(row[field]) for field in fields]
            ax.scatter(
                positions + offset,
                values,
                s=52,
                color=colors[dataset],
                marker=markers[dataset],
                edgecolor="black",
                linewidth=0.55,
                label=dataset,
                clip_on=True,
                zorder=4,
            )
        ax.axhline(
            1.0, color=GRAY, linestyle=(0, (2.2, 2.0)), linewidth=1.0,

        )
        ax.set_xticks(
            positions,
            [r"B$^3$", "Best-match"],
        )
        ax.set_ylabel("Recall")
        ax.yaxis.labelpad = 1.0
        ax.set_ylim(0.0, 1.08)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(-0.50, 1.50)
        clean_axis(ax)
        save(
            fig, "definition_recovery_summary", size,
            adaptive_layout=False,
        )

        fig, ax = plt.subplots(figsize=size, layout="none")
        ax.set_position([0.29, 0.29, 0.64, 0.65])
        for dataset in datasets:
            values = sorted(
                float(row["best_match_recall"])
                for row in details if row["dataset"] == dataset
            )
            cumulative = np.arange(1, len(values) + 1, dtype=float) / len(values)
            ax.step(
                values,
                cumulative,
                where="post",
                color=colors[dataset],
                marker=markers[dataset],
                markevery=max(1, len(values) // 7),
                markerfacecolor="white",
                markeredgecolor=colors[dataset],
                label=dataset,
                clip_on=True,
            )
        ax.axvline(0.8, color=GRAY, linestyle=":", linewidth=1.0)
        ax.set_xlabel("Best-match recall")
        ax.set_ylabel("MSC fraction")
        ax.xaxis.labelpad = 1.0
        ax.yaxis.labelpad = 1.0
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        clean_axis(ax, grid_axis="both")
        save(
            fig, "definition_recovery_cdf", size,
            adaptive_layout=False,
        )


def external_recovery() -> None:
    """Compare directional cover agreement for structural and relaxed baselines."""
    recovery_rows = [
        row for row in read_csv("group-recovery.csv")
        if row["comparison"] == "external"
    ]
    methods = ["StructBK", "FastQC", "FaPlex"]
    datasets = ["GitHub-Social", "ogbn-arxiv-25K"]
    colors = {"GitHub-Social": BLUE, "ogbn-arxiv-25K": CRIMSON}
    markers = {"GitHub-Social": "o", "ogbn-arxiv-25K": "D"}
    offsets = {"GitHub-Social": -0.09, "ogbn-arxiv-25K": 0.09}
    panel_specs = [
        ("best_match_f1", "external_recovery_recovery"),
        ("candidate_best_f1", "external_recovery_fidelity"),
    ]
    size = (RQ5_PANEL_WIDTH, 1.26)

    external_style = {
        **rq5_style(),
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "lines.markersize": 4.8,
    }

    with plt.rc_context(external_style):
        handles = [
            Line2D(
                [0], [0], color=colors[dataset], marker=markers[dataset],
                linestyle="none", markeredgecolor="black", markeredgewidth=0.65,
                label=dataset,
            )
            for dataset in datasets
        ]
        handles.append(Line2D(
            [0], [0], color=GRAY, linestyle=(0, (2.2, 2.0)),
            linewidth=1.0, label="MSC self-match",
        ))
        legend_size = (RQ5_COLUMN_WIDTH, 0.42)
        legend_fig, legend_ax = plt.subplots(figsize=legend_size)
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles, loc="center", ncol=3, frameon=False,
            handlelength=1.5, columnspacing=0.75, handletextpad=0.30,
        )
        save(legend_fig, "external_recovery_legend", legend_size)

        for metric, stem in panel_specs:
            fig, ax = plt.subplots(figsize=size, layout="none")
            ax.set_position([0.27, 0.31, 0.69, 0.62])
            positions = np.arange(len(methods), dtype=float)
            for dataset in datasets:
                recovery = {
                    row["method"]: row for row in recovery_rows
                    if row["dataset"] == dataset
                }
                values = [float(recovery[method][metric]) for method in methods]
                ax.scatter(
                    positions + offsets[dataset],
                    values,
                    s=50,
                    color=colors[dataset],
                    marker=markers[dataset],
                    edgecolor="black",
                    linewidth=0.65,
                    label=dataset,
                    clip_on=True,
                    zorder=4,
                )
            ax.axhline(1.0, color=GRAY, linestyle=(0, (2.2, 2.0)), linewidth=1.0)
            ax.set_xticks(positions, methods, rotation=18, ha="right")
            ax.set_ylabel("Best-match F1")
            ax.set_ylim(0.0, 1.10)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlim(-0.48, 2.48)
            clean_axis(ax)
            save(fig, stem, size, adaptive_layout=False)


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

    author_counts = Counter(author.split()[-1] for paper in papers for author in paper["authors"])
    recurrent_authors = [
        author
        for author, count in sorted(author_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 3
    ]
    angles = np.linspace(0.15 * np.pi, 2.15 * np.pi, len(recurrent_authors), endpoint=False)
    author_positions = {
        author: np.array([1.34 * np.cos(angle), 1.16 * np.sin(angle)])
        for author, angle in zip(recurrent_authors, angles)
    }

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 5.80))
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
                linewidth=0.75 if below else 0.85,
                linestyle="--" if below else "-",
                alpha=0.58 if below else 0.34,
                zorder=1,
            )
    for paper_index, paper in enumerate(papers):
        surnames = {author.split()[-1] for author in paper["authors"]}
        for author in recurrent_authors:
            if author in surnames:
                position = author_positions[author]
                ax.plot(
                    [coordinates[paper_index, 0], position[0]],
                    [coordinates[paper_index, 1], position[1]],
                    color=GOLD,
                    linewidth=0.65,
                    alpha=0.50,
                    zorder=2,
                )

    year_colors = {2016: BLUE, 2017: GREEN, 2018: GOLD, 2019: RED}
    for year, color in year_colors.items():
        selected = [idx for idx, paper in enumerate(papers) if int(paper["year"]) == year]
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=290,
            color=color,
            edgecolor="black",
            linewidth=0.55,
            zorder=4,
        )
    for paper_index, paper in enumerate(papers):
        ax.text(
            coordinates[paper_index, 0],
            coordinates[paper_index, 1],
            str(paper["case_index"]),
            ha="center",
            va="center",
            fontsize=20.0,
            color="white" if int(paper["year"]) in (2016, 2017, 2019) else "black",
            zorder=5,
        )
    for author, position in author_positions.items():
        ax.scatter(
            [position[0]],
            [position[1]],
            marker="s",
            s=310,
            color="white",
            edgecolor=GOLD,
            linewidth=1.2,
            zorder=5,
        )
        ax.text(position[0], position[1] - 0.13, author, ha="center", va="top", fontsize=20.0, zorder=6)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=11.0,
            label=str(year),
        )
        for year, color in year_colors.items()
    ]
    handles.append(
        Line2D([0], [0], marker="s", color=GOLD, markerfacecolor="white", markersize=11.0, label="Author")
    )
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=5,
        frameon=False,
        handletextpad=0.2,
        columnspacing=0.55,
        borderaxespad=0.0,
    )
    ax.set_xlim(-1.58, 1.58)
    ax.set_ylim(-1.40, 1.40)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "quality_case_network", (COLUMN_WIDTH, 5.80))


def main() -> None:
    configure_style()
    threshold_runtime()
    scale_regime()
    scale_runtime()
    parallel_scaling()
    mechanism_summary()
    result_distribution()
    definition_recovery()
    external_recovery()
    case_network()
    print(f"Generated final-size figures in {HERE}")


if __name__ == "__main__":
    main()
