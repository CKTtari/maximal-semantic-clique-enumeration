# Paper figures

`plot_paper_figures.py` reads only canonical CSV files from `tex-data/data` and
generates the experimental PDF/SVG/PNG panels composed by `main.tex`:

| Output | Canonical input |
| --- | --- |
| `runtime_baseline_comparison.pdf` | `main_runtime.csv`; fixed structural/external runs and method-specific MSC operating points |
| `threshold_runtime.pdf` | `threshold_sensitivity.csv`; ogbn-arxiv transition grid and sc-ldoor common-grid points |
| `scale_runtime_*.pdf` | `scale_regime.csv`; absolute runtime on four nested ogbn-arxiv induced graphs |
| `parallel_scaling.pdf` | `parallel_scaling.csv` |
| `mechanism_monosem_*.pdf` | `alg4_state_contraction.csv` |
| `result_distribution_*.pdf` | `result_distribution.csv` |
| `definition_recovery_summary.pdf` | `group-recovery.csv` |
| `rq7_external_comparison.pdf` | `group-recovery.csv`; paired recovery/fidelity endpoints |

Regenerate from the repository root with:

```powershell
python tex-data/fig/plot_paper_figures.py
```

The local SciPilot helper code used by this generator is kept under
`scripts/vendor/scipilot-figure-skill/`. The running-example graph's editable
source is `msc-draw.drawio`; its PDF is kept here with the figures included by
`main.tex`.

`strsub_workflow.tex` and `monosem_workflow.tex` are conceptual algorithm
diagrams synchronized with the current Alg3 and Alg4 implementations. Their
compiled PDFs are included directly by `main.tex`.
