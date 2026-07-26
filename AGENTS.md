# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.
禁止修改环境变量、禁止编辑、删除目标文件夹外的任何东西，如果必须执行操作请先得到用户确认.
## Project Overview

This is a research codebase and LaTeX paper for **"Maximal Semantic Cliques Mining in Vector-Attributed Graphs"** (VLDB-style paper). It implements algorithms for enumerating Average-Constrained Maximal Semantic Cliques (ACMSC) in vector-attributed graphs.

- **Paper**: `main.tex` — VLDB format using `acmart` document class
- **Algorithms**: `src/` contains C++17 implementations of 4 algorithms
- **Figures**: `fig/` contains TikZ snippets and PDFs for paper figures

## Algorithms (src/)

| File | Algorithm | Description |
|------|-----------|-------------|
| `alg1-raw.cpp` | **StructBK** | Structural MCE reference: Tomita pivot BK with degeneracy ordering. No semantic constraint. |
| `alg2-raw.cpp` | **SemBK** | Pivot-free semantic BK baseline with a Check pass for inclusion maximality. |
| `alg3-raw.cpp` | **StrSub** | Pivoted structural decomposition followed by exact intra-clique semantic enumeration and global containment filtering. |
| `alg4-raw.cpp` | **MonoSemMCE** | Canonical-parent reverse DFS. It stores neither size layers nor visited tables and targets the high-$\tau$ regime. |
| `semantic_graph.cpp` | Graph I/O | Graph loader, vector loaders (binary, chunks, JSON), cosine similarity, normalization. |
| `experiment_runtime.h` | Runtime policy | Shared 16 GiB process limit, default 32-thread policy, peak-memory query, and dataset mapping. |

All algorithms share:
- `BlockedSparseBitmap` for O(blocks) neighbor intersection
- `Float32VectorStore` with AVX2 SIMD dot products
- `GraphReorder` — degree-descending BFS reorder for cache locality
- Per-thread `AccStore` (generation-tagged accumulator arrays) for incremental similarity maintenance
- OpenMP parallelization
- Logger that writes to both stdout and a log file

### Build Commands

All algorithms compile with the same pattern. There is no Makefile or CMake.

```bash
# Build any algorithm (example: Alg4 raw)
g++ -std=c++17 -fopenmp -mavx2 -O3 src/alg4-raw.cpp src/semantic_graph.cpp -o alg4-raw

# Build all four algorithms
for f in alg1 alg2 alg3 alg4; do
  g++ -std=c++17 -fopenmp -mavx2 -O3 src/${f}-raw.cpp src/semantic_graph.cpp -o ${f}-raw
done
```

Requirements: GCC/Clang with C++17, OpenMP, AVX2. Tested on GCC 11.4+.

### Running

All algorithms use an interactive command loop:

```bash
./alg4-raw dataset 2 threads 32 timeout 0 log output.log
# Then type commands:
mine 0.2       # run with τ = 0.2
sink out.txt   # write cliques to file
log run.log    # redirect logger to file
quit           # exit
```

Dataset IDs are hardcoded in each `main()`. Default datasets expect a `dataset/` directory with graph edge lists and vector files (binary `.bin` or chunk `.index`).

### Data Format

- **Graph**: text file, first line `V E`, followed by `u v` edges (0-indexed)
- **Vectors binary**: `[int V][int dim][V×dim doubles]`
- **Vectors chunks**: `.index` file listing chunk binary files; each chunk has `[int n][int dim][n×dim doubles]`

## Key Architecture Decisions

### Why no pivot in semantic algorithms
The average-similarity constraint is non-monotone over the subset lattice. Standard BK pivoting relies on structural monotonicity (every maximal clique containing a vertex `u` also contains pivot `p` if `u ∈ N(p)`). Under the average constraint, a clique containing `u` but not `p` may be valid while `{p,u}` is not, so suppressing `u` loses output. This is why SemBK (alg4) omits pivoting entirely, and why StrSub (alg5) decouples structural and semantic search.

### Canonical Parent Property (MonoSemMCE / Alg4)
Every feasible clique has a feasible parent obtained by deleting a vertex of minimum incident-similarity sum. Vertex ID breaks ties, so the parent is unique. Alg4 accepts a child only when the inserted vertex is exactly the child's canonical deletion. This removes duplicate insertion paths without a visited table. The feasibility test `W(C) + acc[v] - tau*|C| >= 0` is a correctness condition, not an optional pruning ablation.

### Exact ablation switches in Alg3
At compile time, Alg3 accepts:
```cpp
#define ACMSC_STRSUB_TOP_EDGE 1
#define ACMSC_STRSUB_VERTEX_BOUND 1
#define ACMSC_STRSUB_LOCAL_MAX 1
```
Setting one to 0 only removes pruning; the global containment filter preserves exact output.

### Top-Edge Layer Pruning (StrSub / Alg3)
For a structural maximal clique `M` of size `m`, sort all `C(m,2)` pairwise similarities descending. The best-case average for any size-k subset is the mean of the top `C(k,2)` edges. If this mean `< τ`, no size-k subset can satisfy the constraint, so the entire size layer is skipped. This is implemented in `process_structural_clique()`.

## Paper Notation Macros (main.tex lines 58–84)

These are defined once in `main.tex` and used throughout:
- `\avgsim(C)` — average pairwise similarity
- `\simsum(C)` — total pairwise similarity sum
- `\W(C)` — semantic surplus = `simsum - τ·C(|C|,2)`
- `\acc[v]` — accumulated similarity of candidate v to current clique
- `\dlim` — monotone-break threshold on marginal contribution

## Common Tasks

- **Add a new dataset**: edit the single mapping in `src/experiment_runtime.h`
- **Change vector dimension**: handled automatically by `Graph::loadVectorsFromBinary()` / `loadVectorsFromChunks()`
- **Run reproducible experiments**: use `python scripts/run_experiments.py --plan experiments/plans/<plan>.json`
- **Regenerate paper PDF**: `latexmk -pdf main.tex` (requires `acmart.cls`, `ACM-Reference-Format.bst`)

## Paper Writing (main.tex)
Write as a VLDB researcher constructing a technical argument, not as a system expanding text.

Global objective:

* Every section must advance one identifiable technical claim.
* Each paragraph must serve one role in the argument chain.
* Remove text that does not change the reader’s understanding of the system, method, or evidence.

Argument structure:

* Introduction: practical problem → concrete limitation in existing systems → consequence → required capability → proposed approach.
* Related work: organize by technical distinction, not by paper-by-paper summary.
* Method/System: design objective → constraint → mechanism → tradeoff.
* Experiment: hypothesis → setup → observation → interpretation.
* Conclusion must summarize verified findings, not restate marketing claims.

Paragraph rules:

* Keep one abstraction level per paragraph.
* Consecutive ideas with dependency relations must be written as connected prose, not bullets.
* Introduce a concept, module, or notation only if it is reused later.
* A paragraph must contain at least one explicit connection to the previous or next argument step.

Technical writing rules:

* Define all nonstandard notation before use.
* Use equations only when they express behavior or constraints more precisely than prose.
* Use pseudocode only for execution logic that cannot be conveyed clearly in text.
* Keep terminology, abbreviations, and module names globally consistent.
* Use short and stable module names.
* Baseline discussion inside Method should explain design motivation or tradeoffs, not advertise superiority.
* Figures and tables must be referenced and interpreted in surrounding text.
* Captions should describe what is shown and why it matters.

Academic register:

* Prefer formal technical exposition over conversational narration.
* Use first-person narration only for explicit author actions, implementation choices, or experimental procedures.
* Prefer precise measurable claims over evaluative wording.
* Avoid rhetorical transitions, conversational fillers, and marketing-style emphasis.
* Do not use formatting, boldface, or lists as substitutes for logical structure.

Formatting and organization:

* Use bullets only for independent unordered items.
* Avoid fragmented subsection structures with many short blocks.
* Maintain consistent tense and terminology throughout the paper.
* Expand abbreviations at first occurrence.

Final verification order:

1. Verify introduction–method–experiment alignment.
2. Verify that every paragraph contributes to the argument chain.
3. Verify notation, terminology, and abbreviation consistency.
4. Remove redundant claims, repeated motivation, and cosmetic wording.
5. Replace list-like exposition with connected technical reasoning where dependency exists.
6. Verify that experiments directly test the stated system goals and tradeoffs.

