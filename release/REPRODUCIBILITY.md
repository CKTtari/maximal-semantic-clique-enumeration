# Reproducing Paper Workloads

The repository contains the final values reported in the paper under
`results/paper-data/`. This document specifies how to rerun the corresponding
algorithm workloads after independently obtaining and processing the public
datasets described in [DATASETS.md](DATASETS.md).

## Build

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --parallel 2

## Run one workload

The paper uses 32 threads and an external 1,200-second limit per workload.
Select a dataset ID from DATASETS.md and a threshold from
`results/paper-data/main_runtime.csv`. For example, run MonoSemMCE on
ogbn-arxiv at its q99 threshold:

    build/bin/alg4-raw dataset 11 threads 32 timeout 0 log ogbn-q99.log
    mine 0.949708
    quit

The interactive command stream is identical for StructBK (`alg1-raw`), SemBK
(`alg2-raw`), StrSub (`alg3-raw`), and MonoSemMCE (`alg4-raw`). `timeout 0`
disables the program's internal timeout; enforce the paper limit externally
when reproducing the timed protocol. Use `sink OUTPUT.txt` after `mine` to
write an enumerated result set.

## Final paper data

`results/paper-data/README.md` maps every retained CSV to its table, result
paragraph, or figure. These files are the final values used in the manuscript.
They contain no raw graph, raw vectors, execution logs, trial archive, or
superseded result set.
