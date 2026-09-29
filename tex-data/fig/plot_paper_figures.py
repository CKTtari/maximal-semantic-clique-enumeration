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
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ROOT = HERE.parents[1]
QUALITY = ROOT / "experiments" / "quality"
QA_DIR = ROOT / "no-use" / "figure-qa"

# SciPilot helpers provide deterministic PDF/SVG/PNG output and layout audits.
SCIPILOT_CANDIDATES = (
    ROOT / "scripts" / "vendor" / "scipilot-figure-skill" / "scripts",
    Path.home() / ".codex" / "skills" / "scipilot-figure-skill" / "scripts",
)
SCIPILOT_SCRIPTS = next((path for path in SCIPILOT_CANDIDATES if path.is_dir()), None)
if SCIPILOT_SCRIPTS is not None:
    sys.path.insert(0, str(SCIPILOT_SCRIPTS))

try:
    from export_figure import export_figure
    from layout_tools import finalize_figure
    from setup_style import setup_style
    from visual_qa import audit_layout, render_preview
except ImportError as exc:  # pragma: no cover - only used without the local skill
    raise RuntimeError(
        "SciPilot figure helpers are required. Clone scipilot-figure-skill "
        "under scripts/vendor/scipilot-figure-skill before regenerating figures."
    ) from exc


# Final-size PVLDB single-column geometry.  All experimental plots are exported
# at the dimensions used by LaTeX; no post-export downscaling is required.
COLUMN_WIDTH = 3.33
HALF_COLUMN_WIDTH = 1.62
PANEL_HEIGHT = 1.10
SINGLE_PANEL_HEIGHT = 1.42
LEGEND_HEIGHT = 0.24
TWO_ROW_LEGEND_HEIGHT = 0.36
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
DISPLAY_NAMES = {
    "SemBK": "MSC-BKBasic",
    "StrSub": "MSC-BKPlus",
    "MonoSemMCE": "MSC-HighSim",
}


def configure_style() -> None:
    setup_style(journal="general", lang="en", use_sciplots=False, constrained_layout=True)
    plt.rcParams.update(
        {
            "figure.figsize": (COLUMN_WIDTH, SINGLE_PANEL_HEIGHT),
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "stix",
            # 7 pt matches \scriptsize in the 9 pt PVLDB body.  Axis labels,
            # tick labels, and legends intentionally share exactly one size.
            "font.size": 7.0,
            "axes.labelsize": 7.0,
            "axes.titlesize": 7.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.05,
            "lines.markersize": 3.6,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "legend.handlelength": 1.4,
            "legend.columnspacing": 0.65,
            "legend.handletextpad": 0.35,
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
    ax.grid(True, axis=grid_axis, linestyle="--", linewidth=0.35, color=LIGHT_GRAY, alpha=0.48)
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
    render_preview(fig, str(QA_DIR / f"{stem}.png"), dpi=240)
    export_figure(
        fig,
        str(HERE / stem),
        formats=("svg", "pdf", "png"),
        dpi=600,
        size_inches=size,
        grayscale_preview=True,
        tight=False,
    )
    plt.close(fig)


def threshold_panel(dataset: str, stem: str, show_ylabel: bool) -> None:
    rows = [row for row in read_csv("threshold_sensitivity.csv") if row["dataset"] == dataset]
    quantiles = sorted({int(row["quantile"]) for row in rows})
    positions = {quantile: index for index, quantile in enumerate(quantiles)}
    panel_size = (HALF_COLUMN_WIDTH, PANEL_HEIGHT)
    fig, ax = plt.subplots(figsize=panel_size)

    methods = ("SemBK", "StrSub", "MonoSemMCE") if dataset in {"GitHub-Social", "ogbn-arxiv"} else ("StrSub", "MonoSemMCE")
    for method in methods:
        by_quantile = {int(row["quantile"]): row for row in rows if row["algorithm"] == method}
        y = [
            float(by_quantile[q]["seconds"])
            if q in by_quantile and by_quantile[q]["status"] == "completed"
            else np.nan
            for q in quantiles
        ]
        plot_x = np.arange(len(quantiles))
        plot_y = y
        if method == "SemBK":
            completed = [(positions[q], float(row["seconds"])) for q, row in by_quantile.items() if row["status"] == "completed"]
            plot_x = [point[0] for point in sorted(completed)]
            plot_y = [point[1] for point in sorted(completed)]
        ax.plot(
            plot_x,
            plot_y,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            label=DISPLAY_NAMES[method],
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
                s=13,
                linewidths=0.85,
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
    ax.set_ylim(0.05, 1900.0)
    clean_axis(ax)
    save(fig, stem, panel_size)


def threshold_runtime() -> None:
    handles = [
        Line2D([0], [0], color=COLORS[method], marker=MARKERS[method],
               linestyle=LINESTYLES[method], markerfacecolor="white",
               label=DISPLAY_NAMES[method])
        for method in ("SemBK", "StrSub", "MonoSemMCE")
    ]
    legend_fig, legend_ax = plt.subplots(figsize=(COLUMN_WIDTH, LEGEND_HEIGHT))
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", ncol=3, frameon=False,
                     handlelength=1.35, columnspacing=0.65, handletextpad=0.35)
    save(legend_fig, "threshold_runtime_legend", (COLUMN_WIDTH, LEGEND_HEIGHT))
    threshold_panel("GitHub-Social", "threshold_runtime_github", show_ylabel=True)
    threshold_panel("ogbn-arxiv", "threshold_runtime_arxiv", show_ylabel=False)
    threshold_panel("sc-nasasrb", "threshold_runtime_nasasrb", show_ylabel=True)
    threshold_panel("sc-ldoor", "threshold_runtime_ldoor", show_ylabel=False)


def runtime_baseline_comparison() -> None:
    """Compare fixed external runs with the intended operating point of each MSC method."""
    rows = read_csv("main_runtime.csv")
    order = [
        "GitHub-Social",
        "ogbn-arxiv",
        "Flickr",
        "email-Enron",
        "sc-nasasrb",
        "sc-pkustk11",
        "sc-pwtk",
        "sc-ldoor",
    ]
    short_names = ["GitHub", "arxiv", "Flickr", "Enron", "nasasrb", "pkustk11", "pwtk", "ldoor"]
    keyed = {(row["dataset"], int(row["quantile"])): row for row in rows}
    methods = [
        ("StructBK", "alg1", 20, "#222222", ""),
        ("FastQC", "fastqc", 20, GOLD, "//"),
        ("FaPlex", "faplex", 20, "#56B4E9", "\\\\"),
        ("MSC-BKBasic ($q_{20}$)", "alg2", 20, GRAY, ".."),
        ("MSC-BKPlus ($q_{20}$)", "alg3", 20, GREEN, "--"),
        ("MSC-HighSim ($q_{99}$)", "alg4", 99, BLUE, "||"),
    ]

    size = (COLUMN_WIDTH, 2.18)
    fig, ax = plt.subplots(figsize=size)
    x = np.arange(len(order), dtype=float)
    width = 0.125
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    for offset, (label, prefix, quantile, color, hatch) in zip(offsets, methods):
        values = []
        statuses = []
        for dataset in order:
            row = keyed[(dataset, quantile)]
            status = row[f"{prefix}_status"]
            statuses.append(status)
            values.append(
                float(row[f"{prefix}_seconds"])
                if status == "completed"
                else TIMEOUT_SECONDS
            )
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="#202020",
            linewidth=0.45,
            hatch=hatch,
            label=label,
            zorder=3,
        )
        for bar, status in zip(bars, statuses):
            if status == "TLE":
                center = bar.get_x() + bar.get_width() / 2.0
                ax.scatter(center, TIMEOUT_SECONDS, marker="x", s=10, color="#202020", linewidths=0.7, zorder=5)

    ax.set_yscale("log")
    ax.set_ylim(0.02, 2200.0)
    ax.set_ylabel("Time (s)", labelpad=2)
    ax.set_xticks(x, short_names, rotation=28, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", pad=1)
    ax.margins(x=0.015)
    clean_axis(ax)
    handles = [Patch(facecolor=color, edgecolor="#202020", linewidth=0.45, hatch=hatch, label=label)
               for label, _, _, color, hatch in methods]
    # Matplotlib fills multirow legends by columns; interleave the two semantic
    # rows so structural/external methods appear together above the MSC methods.
    handles = [handles[index] for index in (0, 3, 1, 4, 2, 5)]
    ax.legend(
        handles=handles,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        handlelength=1.45,
        handleheight=0.9,
        columnspacing=0.72,
        handletextpad=0.35,
        borderaxespad=0.0,
        fontsize=6.3,
    )
    save(fig, "runtime_baseline_comparison", size)


def scale_regime() -> None:
    rows = read_csv("scale_regime.csv")
    sizes = [25000, 50000, 100000, 169343]
    quantiles = [20, 50, 80, 90, 95, 99]
    by_key = {(int(row["vertices"]), int(row["quantile"])): row for row in rows}
    matrix = np.asarray(
        [[float(by_key[(size, q)]["log2_t4_over_t3"]) for q in quantiles] for size in sizes]
    )

    size = (COLUMN_WIDTH, SINGLE_PANEL_HEIGHT)
    fig, ax = plt.subplots(figsize=size)
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
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7.0, color=color)
    colorbar = fig.colorbar(
        image, ax=ax, orientation="horizontal", location="top", pad=0.04,
        fraction=0.11, aspect=28,
    )
    colorbar.set_label(r"$\log_2(T_{\mathrm{HighSim}}/T_{\mathrm{BKPlus}})$", labelpad=2)
    colorbar.ax.tick_params(labelsize=7.0, length=2)
    save(fig, "scale_regime", size)


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
    legend_fig, legend_ax = plt.subplots(figsize=(COLUMN_WIDTH, LEGEND_HEIGHT))
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=6,
        frameon=False,
        fontsize=7.0,
        handlelength=1.05,
        columnspacing=0.42,
        handletextpad=0.20,
    )
    save(legend_fig, "scale_runtime_legend", (COLUMN_WIDTH, LEGEND_HEIGHT))

    for method, stem, show_ylabel in (
        ("StrSub", "scale_runtime_strsub", True),
        ("MonoSemMCE", "scale_runtime_monosem", False),
    ):
        fig, axis = plt.subplots(figsize=(HALF_COLUMN_WIDTH, PANEL_HEIGHT))
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
                linewidth=0.85,
                markersize=3.2,
                markerfacecolor="white",
                markeredgewidth=0.75,
                label=f"$q_{{{quantile}}}$",
            )
        axis.set_xlim(float(positions[0]) - 0.12, float(positions[-1]) + 0.12)
        axis.set_xticks(positions, ["25", "50", "100", "169"])
        axis.set_yscale("log")
        axis.set_ylim(0.04, 4.1)
        axis.set_xlabel("Vertices (K)")
        if show_ylabel:
            axis.set_ylabel("Time (s)")
        else:
            axis.tick_params(axis="y", labelleft=False)
        clean_axis(axis)
        save(fig, stem, (HALF_COLUMN_WIDTH, PANEL_HEIGHT))


def parallel_scaling() -> None:
    rows = read_csv("parallel_scaling.csv")
    size = (COLUMN_WIDTH, SINGLE_PANEL_HEIGHT)
    fig, ax = plt.subplots(figsize=size)
    threads = np.asarray([1, 2, 4, 8, 16, 32], dtype=float)
    for algorithm in ("SemBK", "StrSub", "MonoSemMCE"):
        selected = [row for row in rows if row["algorithm"] == algorithm]
        mean_speedups = []
        for thread_count in threads.astype(int):
            values = [
                float(row["speedup"])
                for row in selected
                if int(row["threads"]) == thread_count
            ]
            mean_speedups.append(float(np.exp(np.mean(np.log(values)))))
        ax.plot(
            threads,
            mean_speedups,
            color=COLORS[algorithm],
            marker=MARKERS[algorithm],
            linestyle=LINESTYLES[algorithm],
            markerfacecolor="white",
            markeredgecolor=COLORS[algorithm],
            markeredgewidth=0.85,
            label=DISPLAY_NAMES[algorithm],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(threads, ["1", "2", "4", "8", "16\nphysical cores", "32"])
    ax.set_xlabel("Threads")
    ax.set_ylabel("Geometric-mean speedup")
    ax.set_ylim(0.8, 6.1)
    clean_axis(ax)
    legend_above(ax, ncol=3)
    save(fig, "parallel_scaling", size)


def mechanism_summary() -> None:
    alg3_rows = [row for row in read_csv("alg3_mechanism_counters.csv") if row["variant"] != "No Local-Max"]
    labels = ["Full", "No edge", "No vertex"]
    tests = np.asarray([float(row["subset_tests"]) / 1e6 for row in alg3_rows])
    panel_size = (HALF_COLUMN_WIDTH, PANEL_HEIGHT)
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
    ax.set_xticks(np.arange(len(labels)), labels, rotation=22, ha="right")
    ax.set_ylabel("Subset tests (M)")
    ax.set_ylim(0, max(tests) * 1.17)
    ax.bar_label(bars, fmt="%.1f", fontsize=7.0, padding=1.0)
    clean_axis(ax)
    fig._layout_engine = None  # lock manual geometry for paired panels
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.30, top=0.90)
    save(fig, "mechanism_strsub", panel_size, adaptive_layout=False)

    rows = read_csv("alg4_state_contraction.csv")
    quantiles = np.asarray([int(row["quantile"]) for row in rows])
    positions = np.arange(len(quantiles))
    mono_panel_size = (HALF_COLUMN_WIDTH, PANEL_HEIGHT)
    mono_style = {
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.6,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
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
        legend_size = (COLUMN_WIDTH, LEGEND_HEIGHT)
        legend_fig, legend_ax = plt.subplots(figsize=legend_size)
        legend_ax.axis("off")
        legend_ax.legend(
            handles=legend_handles, loc="center", ncol=4, frameon=False,
            bbox_to_anchor=(0.08, 0.0, 0.84, 1.0), mode="expand",
            borderaxespad=0.0, fontsize=7.0, markerscale=0.8,
            handlelength=1.20, columnspacing=0.42, handletextpad=0.25,
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
        linewidth=1.0,
        markersize=3.6,
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
        linewidth=1.0,
        markersize=3.6,
        label="Emitted MSCs",
    )
    ax.set_yscale("log")
    shown = [i for i, q in enumerate(quantiles) if q in {5, 50, 80, 95, 99}]
    ax.set_xticks(shown, [str(quantiles[i]) for i in shown])
    ax.set_xlabel("Quantile (%)", labelpad=1)
    ax.set_ylabel("Count", labelpad=1)
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(-0.40, len(positions) - 0.60)
    ax.set_ylim(7.0e3, max(int(row["states"]) for row in rows) * 1.9)
    clean_axis(ax)
    fig._layout_engine = None
    fig.subplots_adjust(left=0.29, right=0.97, bottom=0.28, top=0.96)
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
        linewidth=1.0, markersize=3.6,
    )
    ax.plot(
        positions, canonical_share, color=GREEN, marker="D", linestyle="--",
        markerfacecolor="white", markeredgecolor=GREEN,
        linewidth=1.0, markersize=3.6,
    )
    ax.set_xticks(shown, [str(quantiles[i]) for i in shown])
    ax.set_xlabel("Quantile (%)", labelpad=1)
    ax.set_ylabel("Candidate share (%)", labelpad=1)
    ax.set_xlim(-0.40, len(positions) - 0.60)
    ax.set_ylim(-3.0, 103.0)
    ax.set_yticks([0, 25, 50, 75, 100])
    clean_axis(ax)
    fig._layout_engine = None
    fig.subplots_adjust(left=0.31, right=0.97, bottom=0.28, top=0.96)
    save(fig, "mechanism_monosem_rejections", mono_panel_size, adaptive_layout=False)


def result_distribution() -> None:
    rows = read_csv("result_distribution.csv")
    quantiles = [20, 50, 80, 95, 99]
    datasets = ["GitHub-Social", "ogbn-arxiv", "sc-nasasrb", "sc-ldoor"]
    colors = {
        "GitHub-Social": RED,
        "ogbn-arxiv": BLUE,
        "sc-nasasrb": GOLD,
        "sc-ldoor": GREEN,
        "Geometric mean": GRAY,
    }
    markers = {
        "GitHub-Social": "o",
        "ogbn-arxiv": "s",
        "sc-nasasrb": "^",
        "sc-ldoor": "v",
        "Geometric mean": "D",
    }
    linestyles = {
        "GitHub-Social": "-",
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
            linewidth=1.25 if dataset == "Geometric mean" else 0.75,
            markersize=4.0 if dataset == "Geometric mean" else 3.2,
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
    legend_fig, legend_ax = plt.subplots(figsize=(COLUMN_WIDTH, TWO_ROW_LEGEND_HEIGHT))
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        labels=[handle.get_label() for handle in handles],
        loc="center",
        ncol=3,
        frameon=False,
        fontsize=7.0,
        handlelength=1.35,
        columnspacing=0.70,
        handletextpad=0.30,
    )
    save(legend_fig, "result_distribution_legend", (COLUMN_WIDTH, TWO_ROW_LEGEND_HEIGHT))

    for field, ylabel, stem in (
        ("count_ratio", "Output-count ratio", "result_distribution_count"),
        ("mean_size_ratio", "Mean-size ratio", "result_distribution_size"),
    ):
        panel_size = (HALF_COLUMN_WIDTH, PANEL_HEIGHT)
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
                linewidth=1.25 if aggregate else 0.75,
                markersize=4.0 if aggregate else 3.2,
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
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "axes.titlesize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.65,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.6,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
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
    size = (HALF_COLUMN_WIDTH, PANEL_HEIGHT)

    definition_style = {
        **rq5_style(),
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "lines.markersize": 3.6,
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
        legend_size = (COLUMN_WIDTH, TWO_ROW_LEGEND_HEIGHT)
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
                s=19,
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
    size = (HALF_COLUMN_WIDTH, PANEL_HEIGHT)

    external_style = {
        **rq5_style(),
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "lines.markersize": 3.6,
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
        legend_size = (COLUMN_WIDTH, LEGEND_HEIGHT)
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
                    s=19,
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


def rq7_combined_summary() -> None:
    """Compact RQ7 panel combining both directions of external-family agreement."""
    recovery_rows = [
        row for row in read_csv("group-recovery.csv")
        if row["comparison"] == "external"
    ]
    methods = ["StructBK", "FastQC", "FaPlex"]
    datasets = ["GitHub-Social", "ogbn-arxiv-25K"]
    colors = {"GitHub-Social": BLUE, "ogbn-arxiv-25K": CRIMSON}
    markers = {"GitHub-Social": "o", "ogbn-arxiv-25K": "D"}
    offsets = {"GitHub-Social": -0.10, "ogbn-arxiv-25K": 0.10}

    style = {
        **rq5_style(),
        "font.size": 7.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
    }
    with plt.rc_context(style):
        handles = [
            Line2D(
                [0], [0], color=colors[dataset], marker=markers[dataset],
                linestyle="none", markeredgecolor="black", markeredgewidth=0.55,
                label=dataset,
            )
            for dataset in datasets
        ]
        handles.extend(
            [
                Line2D(
                    [0], [0], color=GRAY, marker="o", linestyle="none",
                    markerfacecolor="white", markeredgecolor=GRAY, label="Recovery",
                ),
                Line2D(
                    [0], [0], color=GRAY, marker="o", linestyle="none",
                    markerfacecolor=GRAY, markeredgecolor=GRAY, label="Fidelity",
                ),
                Line2D(
                    [0], [0], color=GRAY, linestyle=(0, (2.2, 2.0)),
                    linewidth=1.0, label="MSC self-match",
                ),
            ]
        )
        legend_size = (COLUMN_WIDTH, TWO_ROW_LEGEND_HEIGHT)
        legend_fig, legend_ax = plt.subplots(figsize=legend_size)
        legend_ax.axis("off")
        legend_ax.legend(
            handles=handles, loc="center", ncol=3, frameon=False,
            handlelength=1.45, columnspacing=0.70, handletextpad=0.28,
        )
        save(legend_fig, "rq7_comparison_legend", legend_size)

        size = (HALF_COLUMN_WIDTH, PANEL_HEIGHT)
        fig, ax = plt.subplots(figsize=size, layout="none")
        ax.set_position([0.27, 0.31, 0.69, 0.62])
        positions = np.arange(len(methods), dtype=float)
        for dataset in datasets:
            by_method = {
                row["method"]: row for row in recovery_rows
                if row["dataset"] == dataset
            }
            for index, method in enumerate(methods):
                x_position = positions[index] + offsets[dataset]
                recovery = float(by_method[method]["best_match_f1"])
                fidelity = float(by_method[method]["candidate_best_f1"])
                ax.plot(
                    [x_position, x_position], [fidelity, recovery],
                    color=colors[dataset], linewidth=0.65, alpha=0.70, zorder=2,
                )
                ax.scatter(
                    [x_position], [recovery], s=20, marker=markers[dataset],
                    facecolor="white", edgecolor=colors[dataset], linewidth=0.75,
                    zorder=4,
                )
                ax.scatter(
                    [x_position], [fidelity], s=20, marker=markers[dataset],
                    facecolor=colors[dataset], edgecolor="black", linewidth=0.45,
                    zorder=4,
                )
        ax.axhline(1.0, color=GRAY, linestyle=(0, (2.2, 2.0)), linewidth=1.0)
        ax.set_xticks(positions, methods, rotation=18, ha="right")
        ax.set_ylabel("Best-match F1")
        ax.set_ylim(0.0, 1.10)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(-0.48, 2.48)
        clean_axis(ax)
        save(fig, "rq7_external_comparison", size, adaptive_layout=False)


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


def rq7_bcubed_comparison() -> None:
    """Compare all four baselines with the same overlap-aware B-cubed metrics."""
    rows = read_csv("group-recovery.csv")
    datasets = ["GitHub-Social", "ogbn-arxiv", "Flickr"]
    methods = ["PairSim-MCE", "StructBK", "FastQC", "FaPlex"]
    labels = ["PairSim", "StructBK", "FastQC", "FaPlex"]
    colors = [CRIMSON, BLUE, GOLD, GREEN]
    hatches = ["//", "", "..", "\\\\"]
    keyed = {(row["dataset"], row["method"]): row for row in rows}
    style = {**rq5_style(), "legend.fontsize": 6.5, "xtick.labelsize": 6.4}
    with plt.rc_context(style):
        handles = [
            Patch(facecolor=color, edgecolor="black", hatch=hatch, label=label)
            for label, color, hatch in zip(labels, colors, hatches)
        ]
        handles.append(Line2D([0], [0], color=GRAY, linestyle="--", label="MSC self = 1"))
        legend_size = (COLUMN_WIDTH, TWO_ROW_LEGEND_HEIGHT)
        legend_fig, legend_ax = plt.subplots(figsize=legend_size)
        legend_ax.axis("off")
        legend_ax.legend(handles=handles, loc="center", ncol=3, frameon=False,
                         columnspacing=0.75, handletextpad=0.30, handlelength=1.45)
        save(legend_fig, "rq7_comparison_legend", legend_size)

        x = np.arange(len(datasets), dtype=float)
        width = 0.19
        short = ["GitHub", "ogbn-arxiv", "Flickr"]
        for metric, ylabel, stem in (
            ("bcubed_recall", r"B$^3$ recall", "definition_recovery_summary"),
            ("bcubed_precision", r"B$^3$ precision", "rq7_external_comparison"),
        ):
            size = (HALF_COLUMN_WIDTH, PANEL_HEIGHT)
            fig, ax = plt.subplots(figsize=size, layout="none")
            ax.set_position([0.24, 0.28, 0.73, 0.67])
            for index, (method, color, hatch) in enumerate(zip(methods, colors, hatches)):
                values = [float(keyed[(dataset, method)][metric]) for dataset in datasets]
                ax.bar(x + (index - 1.5) * width, values, width=width,
                       color=color, edgecolor="black", linewidth=0.45, hatch=hatch)
            ax.axhline(1.0, color=GRAY, linestyle="--", linewidth=0.9)
            ax.set_xticks(x, short, rotation=18, ha="right")
            ax.set_ylabel(ylabel, labelpad=1)
            ax.set_ylim(0.0, 1.08)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlim(-0.55, len(datasets) - 0.45)
            clean_axis(ax)
            save(fig, stem, size, adaptive_layout=False)


def case_study_pir() -> None:
    """Illustrate the real q80 PIR MSC and its PairSim fragmentation."""
    metadata = json.loads(
        (QUALITY / "ogbn-arxiv-case-metadata.json").read_text(encoding="utf-8")
    )
    papers = metadata["papers"]
    case_summary = read_csv("ogbn-arxiv-case-study.csv")[0]
    by_vertex = {int(paper["vertex"]): paper for paper in papers}
    similarities: dict[tuple[int, int], float] = {}
    for row in read_csv("ogbn-arxiv-case-similarity.csv"):
        left, right = sorted((int(row["left"]), int(row["right"])))
        similarities[(left, right)] = float(row["similarity"])

    tau = 0.888756
    vertices = sorted(by_vertex)

    def classify(title: str) -> str:
        lowered = title.lower()
        if "set intersection" in lowered:
            return "PSI extension"
        if "cache" in lowered or "side information" in lowered:
            return "Caching / side info"
        if any(token in lowered for token in ("wiretap", "asymmetr", "collud", "robust")):
            return "Security / constraints"
        return "Capacity / storage"

    theme_order = [
        "Capacity / storage",
        "Caching / side info",
        "Security / constraints",
        "PSI extension",
    ]
    theme_colors = {
        "Capacity / storage": BLUE,
        "Caching / side info": GREEN,
        "Security / constraints": GOLD,
        "PSI extension": CRIMSON,
    }
    theme_y = {theme: 3 - index for index, theme in enumerate(theme_order)}
    positions: dict[int, tuple[float, float]] = {}
    grouped: dict[tuple[int, str], list[int]] = {}
    for vertex in vertices:
        paper = by_vertex[vertex]
        theme = classify(paper["title"])
        grouped.setdefault((int(paper["year"]), theme), []).append(vertex)
    for (year, theme), members in grouped.items():
        members.sort(key=lambda vertex: int(by_vertex[vertex]["case_index"]))
        offsets = np.linspace(-0.38, 0.38, len(members)) if len(members) > 1 else [0.0]
        for vertex, offset in zip(members, offsets):
            positions[vertex] = (float(year), theme_y[theme] + float(offset))

    passing = {vertex: set() for vertex in vertices}
    below_pairs: list[tuple[int, int]] = []
    for (left, right), similarity in similarities.items():
        if similarity >= tau:
            passing[left].add(right)
            passing[right].add(left)
        else:
            below_pairs.append((left, right))

    maximal: list[tuple[int, ...]] = []

    def bron_kerbosch(r: set[int], p: set[int], x: set[int]) -> None:
        if not p and not x:
            maximal.append(tuple(sorted(r)))
            return
        pivot = max(p | x, key=lambda vertex: len(p & passing[vertex]), default=None)
        candidates = p - (passing[pivot] if pivot is not None else set())
        for vertex in list(candidates):
            bron_kerbosch(r | {vertex}, p & passing[vertex], x & passing[vertex])
            p.remove(vertex)
            x.add(vertex)

    bron_kerbosch(set(), set(vertices), set())
    largest_pairwise = int(case_summary["largest_retained"])
    intersecting_outputs = int(case_summary["pairsim_outputs_intersecting"])
    average_similarity = float(np.mean(list(similarities.values())))
    minimum_similarity = min(similarities.values())

    style = {
        **rq5_style(),
        "font.size": 6.7,
        "axes.labelsize": 6.8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.0,
    }
    size = (COLUMN_WIDTH, 2.62)
    with plt.rc_context(style):
        fig = plt.figure(figsize=size, layout="none")
        network_ax = fig.add_axes([0.07, 0.14, 0.61, 0.65])
        summary_ax = fig.add_axes([0.75, 0.39, 0.23, 0.40])

        for (left, right), _ in similarities.items():
            x1, y1 = positions[left]
            x2, y2 = positions[right]
            network_ax.plot(
                [x1, x2], [y1, y2], color=LIGHT_GRAY,
                linewidth=0.28, alpha=0.16, zorder=1,
            )
        for left, right in below_pairs:
            x1, y1 = positions[left]
            x2, y2 = positions[right]
            network_ax.plot(
                [x1, x2], [y1, y2], color=CRIMSON,
                linewidth=0.55, linestyle=(0, (2.0, 1.5)), alpha=0.82, zorder=2,
            )
        for vertex in vertices:
            paper = by_vertex[vertex]
            theme = classify(paper["title"])
            x, y = positions[vertex]
            network_ax.scatter(
                [x], [y], s=34, color=theme_colors[theme],
                edgecolor="black", linewidth=0.45, zorder=4,
            )
            network_ax.text(
                x, y, f"P{paper['case_index']}", ha="center", va="center",
                fontsize=4.7, color="white", fontweight="bold", zorder=5,
            )
        network_ax.set_title("(a) One coherent PIR research lineage", loc="left", pad=3, fontweight="bold")
        network_ax.set_xticks([2016, 2017, 2018, 2019])
        network_ax.set_yticks(
            [theme_y[theme] for theme in theme_order],
            ["Capacity", "Caching", "Security", "PSI"],
        )
        network_ax.set_xlim(2015.72, 2019.28)
        network_ax.set_ylim(-0.48, 3.48)
        clean_axis(network_ax, grid_axis="x")

        bars = summary_ax.bar(
            [0, 1], [len(vertices), largest_pairwise], width=0.42,
            color=[GREEN, CRIMSON], edgecolor="black", linewidth=0.5,
        )
        bars[0].set_hatch("///")
        bars[1].set_hatch("xxx")
        summary_ax.set_xticks([0, 1], ["MSC", "Largest\nPairSim"])
        summary_ax.set_ylim(0, 20)
        summary_ax.set_yticks([0, 6, 12, 18])
        summary_ax.set_ylabel("Papers retained")
        summary_ax.set_title("(b) PairSim\nfragmentation", loc="left", pad=3, fontweight="bold")
        fig.text(
            0.855, 0.105,
            f"{intersecting_outputs} PairSim outputs\nintersect this MSC\n"
            f"avgSim {average_similarity:.4f} > $\\tau$ {tau:.4f}\n"
            f"{len(below_pairs)}/153 below $\\tau$\n"
            f"min similarity {minimum_similarity:.4f}",
            ha="center", va="bottom", fontsize=6.0,
        )
        clean_axis(summary_ax, grid_axis="y")

        handles = [
            Patch(facecolor=theme_colors[theme], edgecolor="black", label=theme)
            for theme in theme_order
        ]
        handles.extend(
            [
                Line2D([0], [0], color=LIGHT_GRAY, linewidth=1.0, label="Citation link"),
                Line2D([0], [0], color=CRIMSON, linewidth=1.0, linestyle="--", label=r"Similarity < $\tau$"),
            ]
        )
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.50, 0.995),
            ncol=3, frameon=False, handlelength=1.25,
            columnspacing=0.65, handletextpad=0.28,
        )
        save(fig, "case_study_pir", size, adaptive_layout=False)


def main() -> None:
    configure_style()
    runtime_baseline_comparison()
    threshold_runtime()
    scale_regime()
    scale_runtime()
    parallel_scaling()
    mechanism_summary()
    result_distribution()
    rq7_bcubed_comparison()
    case_study_pir()
    print(f"Generated final-size figures in {HERE}")


if __name__ == "__main__":
    main()
