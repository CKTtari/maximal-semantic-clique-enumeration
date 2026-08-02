# Canonical paper data

All numerical values used by active paper tables and experimental figures are
stored here. Regenerate them with
`src/experiments/standard/paper_results_export.py` and
`src/experiments/standard/export_engineering_ablation.py`; do not edit CSV
values by hand.

| File | Paper use |
| --- | --- |
| `dataset_inventory.csv` | Dataset table and evaluation roles |
| `main_runtime.csv` | Primary-grid MSC runtimes plus fixed FastQC and FaPlex runtime references |
| `threshold_sensitivity.csv` | Threshold-regime figure |
| `scale_regime.csv` | Graph-size/threshold regime map |
| `parallel_scaling.csv` | Thread-scaling figure |
| `alg3_phase_attribution.csv` | Alg3 parallel bottleneck attribution in the text |
| `alg3_ablation_runtime.csv` | Exact Alg3 ablation table |
| `alg3_mechanism_counters.csv` | Alg3 pruning-work counters |
| `alg4_state_contraction.csv` | Alg4 explored-state counters |
| `engineering_ablation.csv` | Selected method-specific engineering comparisons in Table 5 |
| `engineering_profile_selection.csv` | Complete 31-configuration engineering sweep behind Table 5 |
| `provenance.json` | Exact archived source summaries |

`provenance.json` is the audit boundary between archived formal experiment
records and paper-facing data. The exporter reads summaries by default and,
for phase attribution, verifies the archived stdout hash recorded in the
manifest before extracting the two reported phase measurements. The current
export uses the eight primary-grid
datasets only; the retired pressure-group CSV is archived under
`no-use/paper-data/pre-batch12-current-export`.
