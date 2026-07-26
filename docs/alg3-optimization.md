# Alg3 StrSub Optimization

## Standard entry point

`src/alg3-raw.cpp` is the standard Alg3 implementation. The previous fixed
128-bit version is archived as
`no-use/legacy-alg3/alg3-original-mask128.cpp`.

The implementation does not cap the number of graph vertices, neighbors,
structural maximal cliques, results, or the size of a host clique. The symbol
`m` in the implementation and analysis means the size of one structural
maximal clique being processed by Phase 2, not the dataset size. The constant
`ALG3_VERTEX_BOUND_MIN_M=12` only avoids computing an upper bound on tiny host
cliques; it does not skip or truncate any input or output.

## Exact optimizations

For `m <= 63`, Phase 2 uses a native `uint64_t` mask. For `64 <= m <= 127`, it
retains the original two-word mask and Gosper iteration. For `m > 127`, it uses
a dynamic array of 64-bit words and exact lexicographic combination generation.
All three paths enumerate the same subsets in decreasing cardinality and apply
the same feasibility and containment tests. The dynamic path removes the
original `m <= 127` assertion rather than replacing it with another limit.

The second optimization adds a vertex-aware upper bound for each subset size.
For a host clique `M`, let `q_v(k)` be the sum of the `k-1` largest similarities
incident to `v` within `M`, and let `q_(1)(k) >= ... >= q_(m)(k)` be these values
in descending order. For every size-`k` subset `S`:

```text
2 * simSum(S)
  = sum over v in S of sum over u in S - {v} of sim(u,v)
  <= sum over v in S of q_v(k)
  <= sum from i=1 to k of q_(i)(k).
```

Therefore

```text
avgSim(S) <= (sum from i=1 to k of q_(i)(k)) / (k * (k-1)).
```

If this upper bound is below `tau`, the whole size layer is impossible and can
be skipped. The implementation rounds bound accumulation and division toward
positive infinity, so floating-point evaluation cannot turn the bound into an
underestimate at the pruning boundary.

For large fixed-cardinality layers in the native-mask path, combinations are
partitioned by their smallest selected vertex and evaluated as OpenMP tasks.
These partitions are disjoint and exhaustive. During one size-`k` layer, every
task reads the same frozen collection of feasible supersets from larger layers;
distinct size-`k` subsets cannot strictly contain one another. The task group is
joined and its feasible masks are merged before size `k-1` begins. Consequently,
parallel evaluation makes exactly the same local-containment decisions as the
descending sequential order. The scheduling threshold controls overhead only;
it does not skip combinations or limit host-clique size.

The final strict-containment filter sorts candidates once by decreasing size
and then lexicographically, so exact duplicates are adjacent without a second
full sort. It processes one candidate-size group at a time. Queries for a
size-$k$ group read an inverted index containing only retained candidates of
size greater than $k$; the group is inserted only after all of its queries
finish. Distinct equal-size sets cannot strictly contain one another, so this
batched order is equivalent to the original sequential filter. If a discarded
larger candidate contains a smaller one, some retained candidate also contains
both by transitivity, so indexing retained supersets only remains complete.

## Validation

- The upper-bound oracle checked 225,000 randomly generated size layers against
  exhaustive subset enumeration and found no underestimated optimum.
- A 256-vertex complete host clique exercised the dynamic-mask path. Phase 2
  considered all 32,640 size-two subsets selected by the safe layer bounds and
  returned the unique expected ACMSC in 33 ms.
- A 600-vertex complete host clique (degree 599) returned one size-600 ACMSC,
  and the uncapped histogram reported both its layer count and average size
  correctly.
- On WN18RR at `tau=0.6`, standard Alg3 and canonical Alg4 both returned 9,724
  ACMSCs; normalized set comparison reported zero missing and zero extra sets.
- On FB15K-237, Alg3 returned exactly the canonical Alg4 set at both tested
  thresholds. At `tau=0.6`, the optimized Phase 2 reduced end-to-end time from
  85.237 s to 67.864 s. At `tau=0.8`, it reduced time from 31.431 s to 16.747 s.
- On FB15K-237 at the pre-registered `q80` threshold, the current standard
  build returned exactly the same 76,106 cliques and per-size histogram at
  every tested thread count. Median algorithm time decreased from 19.662 s at
  one thread to 8.698 s at 32 threads (2.26x); 16 threads was slightly faster
  at 8.232 s. The final global filter used only 59--103 ms in these runs, so
  the scaling limit lies in the combined structural-search and local-subset
  phase rather than in final containment filtering.
- The grouped global filter returned exactly the same normalized output as the
  preceding implementation on com-Amazon `q20` (423,409 ACMSCs, 7,203 strict
  subsets removed) and FB15K-237 `q80` (76,106 ACMSCs, 27,119 strict subsets
  removed). In paired validation runs, filter time decreased from 373 to 266 ms
  on com-Amazon and from 87 to 65 ms on FB15K-237. These are diagnostic runs,
  not the repeated timings used in paper tables.

These measurements use 32 OpenMP threads and a 16 GiB process memory limit.
The 180-second rule is only a parameter-selection rule for before/after tests:
we compare settings on which the original implementation itself finishes
within 180 seconds. It is not a required completion time for the algorithm.
The standard program has no default time or output-count limit; an explicit
`timeout 180` argument is used only as a guard during bounded validation runs.
Exact subset enumeration remains exponential in the worst case, as expected.
