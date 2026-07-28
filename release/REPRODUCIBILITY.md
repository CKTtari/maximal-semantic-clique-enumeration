# Reproducibility protocol

## Resource and timing policy

All mining executables enforce a 16 GiB process memory limit. Formal plans use
at most 32 OpenMP threads. A task has a 1,200-second external timeout. Completed
runtime cells use three repetitions and report the median; a task that reaches
TLE is not repeated. Statistics builds are isolated from runtime builds because
counter instrumentation changes execution cost.

The runner fingerprints source files, compile definitions, dataset catalog,
graph files, vector indexes, and vector chunks. It limits each launched batch
to eight tasks and records machine, compiler, command, stdout, stderr, parsed
counts, per-size histograms, peak memory, and hashes.

## Exactness checks

The validation plans compare full normalized output sets or complete size
histograms, not only total counts. The principal plans are:

| Plan | Purpose |
| --- | --- |
| batch0-correctness.json | Cross-method checks on compact real workloads |
| batch0-fb15k-alg3-ablation-equivalence.json | Exact StrSub switch equivalence |
| batch10-current-fb15k-q99-correctness.json | StrSub and MonoSemMCE full-set equality |
| batch10-validation-alg4-filter.json | Canonical search and final containment filter |

Run a plan with the same bounded runner used for timing:

    python scripts/run_experiments.py --plan experiments/plans/batch10-current-fb15k-q99-correctness.json --batch-size 2

## Paper experiment map

| Paper evidence | Formal plan families | Canonical data |
| --- | --- | --- |
| Eight-dataset overview | batch11-main, batch13-main, batch2-main, batch7-main | main_runtime.csv |
| Threshold sensitivity | batch12-threshold, batch13-threshold, batch14-threshold | threshold_sensitivity.csv |
| Scale and threshold regime | batch12-scale-ogbn-regime-current | scale_regime.csv |
| Parallel scaling | batch12-thread-fb15k-alg3-current, batch12-thread-ogbn-alg4-current | parallel_scaling.csv |
| StrSub phase attribution | batch12-alg3-phase-attribution-current | alg3_phase_attribution.csv |
| StrSub exact ablation | batch12-fb15k-alg3-ablation-current, batch17-nasasrb-alg3-ablation-formal | alg3_ablation_runtime.csv |
| Search counters | batch12-mechanism-stats-current | alg3_mechanism_counters.csv, alg4_state_contraction.csv |
| MonoSemMCE generation ablation | batch16-alg4-generation-ablation-formal | alg4_generation_ablation.csv |
| Output distributions | archived exact-output manifests | result_distribution.csv |
| Subject purity and case study | experiments/quality records | result_quality files and case metadata |

Every formal summary named by results/paper-data/provenance.json is included
under experiments/summaries, with its source manifest under experiments/archive.
The two raw stdout files required for hash-verified phase extraction are
included under experiment-output/batch12-alg3-phase-attribution-current.

## Running and archiving a plan

Run:

    python scripts/run_experiments.py --plan experiments/plans/PLAN.json --batch-size 4 --continue-on-failure

Resume without repeating any attempted TLE, OOM, or crash:

    python scripts/run_experiments.py --plan experiments/plans/PLAN.json --batch-size 4 --resume --skip-attempted --continue-on-failure

Archive and summarize a completed run:

    python scripts/archive_results.py RUN_NAME
    python scripts/summarize_results.py experiments/archive/RUN_NAME.manifest.json

New outputs go to results/runs. New manifests and summaries go to
experiments/archive and experiments/summaries.

## Rebuilding canonical CSV files

Run:

    python scripts/paper_results_export.py

The exporter reads the formal summaries, resolves their source manifests, and
verifies the raw phase-attribution artifact hashes before rewriting
results/paper-data. It does not use pilot timing.

## Rebuilding figures

Run:

    python figures/plot_paper_figures.py

Each plotted value is read from a canonical CSV or quality-analysis JSON. The
script exports vector PDF and 600 dpi PNG at final single-column dimensions,
uses Times-compatible typography, and runs the vendored SciPilot layout audit.

## Interpreting status

completed means a valid total runtime, output count, size histogram, and peak
memory were parsed. tle means the external 1,200-second limit was reached. oom
means the process reported allocation or memory failure. parse_error and crash
are invalid measurements and must be diagnosed before use.