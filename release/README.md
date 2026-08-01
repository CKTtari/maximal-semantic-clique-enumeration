# Maximal Semantic Clique Enumeration

This repository contains the four exact C++ algorithms from "Maximal Semantic
Clique Enumeration in Vector-Attributed Graphs."

| Executable | Paper name | Role |
| --- | --- | --- |
| `alg1-raw` | StructBK | Structural maximal-clique enumeration reference |
| `alg2-raw` | SemBK | Direct semantic-search baseline |
| `alg3-raw` | StrSub | Structural decomposition with exact semantic subset enumeration |
| `alg4-raw` | MonoSemMCE | Canonical-parent reverse search |

## Requirements

A C++17 compiler, OpenMP, CMake 3.18 or newer, and an AVX2-capable x86-64
processor are required.

## Build

From the repository root:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel 2

The executables are written to `build/bin`. A direct GCC build follows the same
pattern:

    g++ -std=c++17 -fopenmp -mavx2 -O3 src/alg4-raw.cpp src/semantic_graph.cpp -I src -o alg4-raw

Replace `alg4` with `alg1`, `alg2`, or `alg3` to build another method.

## Input format

Each dataset consists of an undirected graph and one vector per vertex. The
graph file begins with `V E`, followed by zero-based edge pairs `u v`. A vector
binary stores int32 `V`, int32 dimension, and then `V` times dimension float64
values in row-major order. The shared loader applies L2 normalization.

Set the graph and vector paths for a dataset identifier in
`src/experiment_runtime.h`. The programs then accept an interactive command
stream, for example:

    build/bin/alg4-raw dataset 11 threads 32 timeout 0 log run.log
    mine 0.90
    quit

The repository contains code only. Obtain any evaluation graph and its vertex
vectors from their respective public providers before running an experiment.

## Paper reproduction

[DATASETS.md](DATASETS.md) records the public sources, graph transformations,
vector construction, and fixed seeds used in the paper. [REPRODUCIBILITY.md]
(REPRODUCIBILITY.md) gives the paper workload protocol. The final values used
in the manuscript are included in `results/paper-data/`.

## License and citation

The implementation is released under Apache License 2.0. Citation metadata is
provided in `CITATION.cff`.
