# ACMSC Research Artifact

This repository contains the C++17 implementations and LaTeX source for
"Maximal Semantic Cliques Mining in Vector-Attributed Graphs".  The standard
algorithm entry points are:

| Entry point | Method | Output task |
|---|---|---|
| `src/alg1-raw.cpp` | StructBK | Structural maximal cliques |
| `src/alg2-raw.cpp` | SemBK | ACMSCs, pivot-free baseline |
| `src/alg3-raw.cpp` | StrSub | ACMSCs, structural decomposition |
| `src/alg4-raw.cpp` | MonoSemMCE | ACMSCs, canonical reverse DFS |

`src/algN-stats.cpp` is a thin instrumentation build of the corresponding raw
source.  Raw binaries are the only binaries used for runtime comparisons.

## Repository layout

`src/` contains algorithm and graph-runtime C++ sources; `scripts/` contains
experiment runners, data conversion, analysis, and export utilities.  The
`dataset/` tree is the single dataset workspace: processed graph/vector files
remain in its existing subdirectories, raw source archives are under
`dataset/rawDatasets/`, and conversion/generation programs are under
`dataset/data_processing/`.  Versioned plans, provenance, summaries, manifests,
and ignored raw run output are grouped under `experiments/`; the run output is
specifically `experiments/output/`.  The root `experiment-output` name is only
a compatibility junction for historical run records.  `tex-data/` is the
paper-facing source of truth for figures and numeric data.  Superseded files
and temporary products are retained, without deletion, under the dated
`no-use/archive-*` folders.

## Build and run

The compiler must support C++17, OpenMP, and AVX2.  A direct build is:

```powershell
g++ -std=c++17 -fopenmp -mavx2 -O3 src/alg4-raw.cpp src/semantic_graph.cpp -o alg4-raw.exe
```

All executables accept the same process options.  The standard process limit
is 16 GiB, and the default worker count is at most 32.

```powershell
./alg4-raw.exe dataset 8 threads 32 timeout 0 log run.log
```

Commands are read from standard input:

```text
mine 0.8
sink cliques.txt
quit
```

For a derived dataset, pass `graph <path>` and `vectors <path>`.  Alg1 needs
only `graph`.  Formal experiments use an external hard timeout; `timeout 0`
keeps the algorithm-internal deadline disabled.

## Reproducible experiments

Experiment plans are versioned under `experiments/plans/`.  The runner builds
source-fingerprinted executables under `no-use/compiled/`, executes at most
four tasks by default, and writes independent logs and JSON records under the
ignored `experiments/output/` directory.

```powershell
python scripts/run_experiments.py --plan experiments/plans/batch0-correctness.json --batch-size 3
```

Use `--resume` for the next uncompleted batch.  Add `--skip-attempted` when a
recorded TLE/OOM/crash should remain censored while later tasks continue.
Reattempting a failed task instead requires `--rerun-reason`.  Correctness
plans capture full clique sets and stop when normalized outputs differ.

Raw artifacts remain under the ignored `experiments/output/` tree.  After a
batch is complete, verify and index it with:

```powershell
python scripts/archive_results.py batch0-correctness
```

This writes a Git-trackable `experiments/archive/*.manifest.json` containing
task status, parsed measurements, source/binary fingerprints, and SHA-256 for
every raw artifact.  A changed plan, missing file, count mismatch, or clique-set
mismatch makes archival fail.

Nested scale datasets are derived from real graphs and aligned real vectors:

```powershell
python scripts/generate_subgraphs.py --graph dataset/dataset/FB15K-237_edges.txt `
  --vectors dataset/vectors/FB15K-237_vectors.index `
  --output-dir dataset/scale_datasets/fb15k-prefix `
  --sizes 2000 5000 10000
```

The generator writes complete induced edge sets, aligned binary vectors, and a
checksum-bearing `provenance.json`.  Dataset files and generated outputs are
not stored in Git.

## Paper

The paper source is `main.tex`.  Build it with:

```powershell
latexmk -pdf main.tex
```

LaTeX intermediates, the generated PDF, executables, datasets, and experiment
records are ignored.  Superseded sources and compilation products are retained
under the ignored `no-use/` archive rather than deleted.
