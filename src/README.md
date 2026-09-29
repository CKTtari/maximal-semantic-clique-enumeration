# Source Layout

The standard experiment entry points are:

- `alg1-raw.cpp`: structural MCE reference.
- `alg2-raw.cpp`: direct semantic BK baseline.
- `alg3-raw.cpp`: standard StrSub implementation. It combines structural
  decomposition with exact semantic subset enumeration, Top-Edge pruning, a
  vertex-aware size-layer upper bound, and width-adaptive subset masks. Native
  64-bit masks cover host cliques up to 63 vertices; a dynamic mask preserves
  correctness without imposing a clique-size cap.
- `alg4-raw.cpp`: canonical-parent DFS (the standard Alg4 implementation).
- `alg5-unified.cpp`: current unified-algorithm research candidate; it is not
  part of the paper's archived Alg1--Alg4 runtime tables. Earlier Alg5
  variants are retained in `no-use/archive-20260929/source-legacy/alg5-explorations/`.
- `alg6-framework.cpp`: exact MSC framework baseline. It keeps structural host
  decomposition and canonical ownership while disabling the optional Unified
  pruning certificates; it is intended for mechanism ablations, not as a new
  result-family definition.
- `semantic_graph.cpp` / `semantic_graph.h`: shared graph and vector loading.

Files named `*-stats.cpp` are thin instrumentation entry points that compile
the corresponding `*-raw.cpp` with `ACMSC_ENABLE_STATS=1`; they do not contain
independent algorithm copies and must not be used for runtime tables.

The original visited-DFS and layered Alg4 source-of-truth files are archived in
`no-use/legacy-alg4/`. Nothing in that directory is a standard build input.
The superseded fixed-128-bit Alg3 implementation is archived in
`no-use/legacy-alg3/` for controlled performance comparisons only.

The formal paper-data exporters remain in `src/experiments/standard/` because
their recorded paths are part of existing provenance manifests. They are
analysis scripts, not algorithm implementations; new orchestration utilities
belong in `scripts/` and new plans in `experiments/plans/`.

All standard executables obtain the 16 GiB process limit, default 32-thread
policy, peak-memory query, and dataset mapping from `experiment_runtime.h`.
Derived datasets can be supplied with the `graph` and `vectors` command-line
arguments. Formal TLE is enforced by `scripts/run_experiments.py`; the internal
timeout defaults to unlimited and an incomplete run never reports a final
count.
