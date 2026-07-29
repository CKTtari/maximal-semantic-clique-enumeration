# Experimental Figure Contract

All panels use Python/Matplotlib exclusively, read canonical CSV files, and export editable SVG plus PDF and PNG. No observations are sampled or omitted inside the plotting code. Censored runs remain at the registered 1,200-second boundary.

| Figure | Core conclusion | Evidence and source | Reviewer risk controlled |
|---|---|---|---|
| Threshold regimes | StrSub covers broad thresholds, while MonoSemMCE becomes dominant after the feasible-state family contracts. | Nine registered quantiles on ogbn-arxiv and sc-ldoor; `threshold_sensitivity.csv`. | The crossover is shown as a trajectory rather than a selected endpoint. |
| Scale regimes | The high-threshold crossover persists across nested graph sizes. | Four deterministic ogbn-arxiv degree-prefix graphs and six quantiles; `scale_regime.csv`. | Every cell uses the same source ordering, vectors, and threshold values. |
| Parallel scaling | Edge-root scheduling scales further than host-clique subset scheduling. | Threads 1, 2, 4, 8, 16, and 32; `parallel_scaling.csv`. | The two curves are labeled by their distinct registered workloads. |
| Search mechanisms | Each exact bound removes measurable work, and canonical states contract sharply with tau. | StrSub counters and MonoSemMCE state counts; `alg3_mechanism_counters.csv`, `alg4_state_contraction.csv`. | Counter builds are separated from runtime builds. |
| Output distribution | MSC count and size change non-monotonically over tau relative to structural MCE. | Five quantiles on four datasets; `result_distribution.csv`. | Both individual datasets and the geometric mean are retained. |
| Quality sweep | MSC provides a better purity-coverage trajectory than pairwise thresholding across tau. | Official ogbn-arxiv labels at five registered quantiles; `result_quality_sweep.csv`. | q80 is not used as the sole quality claim. |
| Multi-baseline quality | At q80, MSC retains near-top label purity with broader coverage than PairSim-MCE and much higher purity than FastQC on two natural-feature graphs. | GitHub-Social and ogbn-arxiv-25K outputs with minimum size 10; `github-social-quality.csv`, `ogbn-arxiv-25k-quality.csv`. | FastQC gamma is treated as its own structural parameter, not as an equivalent semantic tau. |
| Case network | Average similarity preserves one coherent paper group even when some pairs fall below tau. | Verified ogbn-arxiv metadata and cosine values; `ogbn-arxiv-case-similarity.csv`. | The layout changes positions only; labels, similarities, and membership are source-derived. |
