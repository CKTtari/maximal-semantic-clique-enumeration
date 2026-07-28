# Maximal Semantic Clique Enumeration

This artifact accompanies the paper "Maximal Semantic Clique Enumeration in
Vector-Attributed Graphs." It contains the four exact C++ implementations,
paper experiment plans, result provenance, dataset preparation utilities, and
the data-driven plotting pipeline.

## Algorithms

| Executable | Paper name | Role |
| --- | --- | --- |
| alg1-raw | StructBK | Structural maximal-clique enumeration reference |
| alg2-raw | SemBK | Direct semantic-search baseline |
| alg3-raw | StrSub | Structural decomposition and exact semantic subset enumeration |
| alg4-raw | MonoSemMCE | Canonical-parent reverse search for selective thresholds |

The matching stats executables collect search counters and must not be used for
runtime comparisons.

## Requirements

The C++ programs require a C++17 compiler, OpenMP, and an AVX2-capable
x86-64 processor. CMake 3.18 or newer is optional and provides the unified
build. The paper configuration uses at most 32 OpenMP
threads, a 16 GiB per-process memory limit, and a 1,200-second external timeout.
Python 3.9 or newer is used for orchestration, preprocessing, provenance, and
figures.

## Build

From the release root:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel 2

A direct GCC build, which is the path validated for this package, is:

    g++ -std=c++17 -fopenmp -mavx2 -O3 src/alg4-raw.cpp src/semantic_graph.cpp -I src -o alg4-raw

Replace alg4 with alg1, alg2, or alg3 for the other raw executable. The stats
sources build identically.

Executables are written to build/bin. The experiment runner also supports
on-demand GCC builds so that compile-time ablation definitions are fingerprinted
with every task.

## Data

Dataset binaries are intentionally not stored in this repository. DATASETS.md
lists the eight paper datasets, public source pages, exact transformations,
expected paths, and optional SHA-256 checks. After preparation, verify every
artifact:

    python scripts/verify_datasets.py

The expected directory layout is defined by experiments/datasets.json.

## Run one workload

The programs use an interactive command stream. For example:

    build/bin/alg4-raw dataset 11 threads 32 timeout 0 log results/ogbn-q99.log

Then enter:

    mine 0.949708
    quit

Dataset 11 is ogbn-arxiv. The zero internal timeout leaves TLE enforcement to
the experiment runner.

## Run formal plans

A formal plan records algorithm, build mode, dataset, threshold, thread count,
timeout, repetitions, and optional compile-time switches. Run a bounded batch:

    python scripts/run_experiments.py --plan experiments/plans/batch11-main-core-current.json --batch-size 4 --continue-on-failure

Outputs are written under results/runs. REPRODUCIBILITY.md maps each paper claim
to its plan, archived summary, and canonical CSV.

## Rebuild paper data and figures

The archived manifests and summaries are included. Regenerate canonical CSVs
and then figures with:

    python scripts/paper_results_export.py
    python figures/plot_paper_figures.py

The plotting pipeline consumes only results/paper-data and uses the vendored
MIT-licensed scipilot-figure-skill. Generated PDF and PNG figures remain under
figures.

## Repository layout

| Path | Contents |
| --- | --- |
| src | Four raw and four stats C++ implementations plus shared graph code |
| scripts | Experiment runner, provenance tools, preprocessing, and analysis |
| experiments/plans | Formal and exact-output validation plans |
| experiments/archive | Hash-bearing run manifests |
| experiments/summaries | Aggregated formal measurements |
| results/paper-data | Canonical CSV values used by paper tables and figures |
| results/quality | Output-quality analysis records |
| dataset/provenance | Dataset transformation provenance and expected hashes |
| figures | Data-driven figure script and generated outputs |
| third_party | Vendored figure helper with its original license |

## Public release checklist

The artifact code still needs an author-selected license before the repository
is made public. Third-party licensing is already preserved in
THIRD_PARTY_NOTICES.md and the vendored dependency directory.