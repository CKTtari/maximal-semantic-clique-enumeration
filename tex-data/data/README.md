# Canonical paper data

All numerical values used by active paper tables and experimental figures are
stored here. Regenerate them with
`src/experiments/standard/paper_results_export.py`; do not edit CSV values by
hand.

| File | Paper use |
| --- | --- |
| `dataset_inventory.csv` | Dataset table and evaluation roles |
| `main_runtime.csv` | Common-grid runtime and output-count table |
| `threshold_sensitivity.csv` | Threshold-regime figure |
| `scale_regime.csv` | Graph-size/threshold regime map |
| `parallel_scaling.csv` | Thread-scaling figure |
| `alg3_phase_attribution.csv` | Alg3 parallel bottleneck attribution in the text |
| `alg3_ablation_runtime.csv` | Exact Alg3 ablation table |
| `alg3_mechanism_counters.csv` | Alg3 pruning-work counters |
| `alg4_state_contraction.csv` | Alg4 explored-state counters |
| `provenance.json` | Exact archived source summaries |

`provenance.json` is the audit boundary between archived formal experiment
records and paper-facing data. The exporter reads summaries by default and,
for phase attribution, verifies the archived stdout hash recorded in the
manifest before extracting the two reported phase measurements. The current
export uses the eight common-grid
datasets only; the retired pressure-group CSV is archived under
`no-use/paper-data/pre-batch12-current-export`.
