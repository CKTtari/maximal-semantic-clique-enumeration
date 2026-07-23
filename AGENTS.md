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
| `alg1-standard.cpp` | **StructBK** | Structural MCE baseline — Tomita pivot BK with degeneracy ordering. No semantic constraint. |
| `alg4-standard.cpp` | **SemBK** | Semantic BK — pivot-free BK with Check pass for semantic maximality. Direct adaptation, no pruning. |
| `alg5-standard-improved.cpp` | **StrSub** | Structural decomposition + subset enumeration. Phase 1: BK with pivot. Phase 2: intra-clique semantic enumeration with Top-Edge Layer Pruning. Phase 3: global deduplication. |
| `alg6-standard.cpp` | **MonoSemMCE** | Level-synchronous clique expansion. BK-free. Uses W-pruning, monotone break, and per-layer deduplication. Best for high-τ regime. |
| `semantic_graph.cpp` | Graph I/O | Graph loader, vector loaders (binary, chunks, JSON), cosine similarity, normalization. |

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
# Build any algorithm (example: alg6)
g++ -std=c++17 -fopenmp -mavx2 -O3 src/alg6-standard.cpp src/semantic_graph.cpp -o alg6

# Build all four algorithms
for f in alg1 alg4 alg5 alg6; do
  g++ -std=c++17 -fopenmp -mavx2 -O3 src/${f}-standard.cpp src/semantic_graph.cpp -o $f
done
```

Requirements: GCC/Clang with C++17, OpenMP, AVX2. Tested on GCC 11.4+.

### Running

All algorithms use an interactive command loop:

```bash
./alg6 dataset 2 log output.log   # dataset ID + optional log file
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

### Monotone Path Property (MonoSemMCE / alg6)
Theorem: every ACMSC admits a vertex insertion ordering where `avgSim` is non-increasing at every step. This enables the **sorted-suffix cut** (monotone break): once a candidate's marginal contribution `δ_v` exceeds the threshold `δ_lim`, the entire remaining sorted suffix is discarded in O(1). Combined with **W-pruning** (`W + δ_v < τ·k`), this defines a feasibility window that contracts rapidly as τ increases.

### Ablation switches in alg6
At the top of `alg6-standard.cpp`:
```cpp
#define ENABLE_W_PRUNE     1
#define ENABLE_MONO_PRUNE  1
#define ENABLE_LAYER_DEDUP 1
```
Set to 0 to disable individual pruning rules for experiments.

### Top-Edge Layer Pruning (StrSub / alg5)
For a structural maximal clique `M` of size `m`, sort all `C(m,2)` pairwise similarities descending. The best-case average for any size-k subset is the mean of the top `C(k,2)` edges. If this mean `< τ`, no size-k subset can satisfy the constraint, so the entire size layer is skipped. This is implemented in `process_structural_clique()`.

## Paper Notation Macros (main.tex lines 58–84)

These are defined once in `main.tex` and used throughout:
- `\avgsim(C)` — average pairwise similarity
- `\simsum(C)` — total pairwise similarity sum
- `\W(C)` — semantic surplus = `simsum - τ·C(|C|,2)`
- `\acc[v]` — accumulated similarity of candidate v to current clique
- `\dlim` — monotone-break threshold on marginal contribution

## Common Tasks

- **Add a new dataset**: edit the `switch(dataset_id)` block in each algorithm's `main()`
- **Change vector dimension**: handled automatically by `Graph::loadVectorsFromBinary()` / `loadVectorsFromChunks()`
- **Run a single experiment non-interactively**: pipe commands, e.g. `printf 'mine 0.2\nquit\n' | ./alg6 dataset 2`
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


