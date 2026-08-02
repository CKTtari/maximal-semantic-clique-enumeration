# Paper figures

`plot_paper_figures.py` reads only canonical CSV files from `tex-data/data` and
generates the four experimental PDFs and PNG previews in this directory:

| Output | Canonical input |
| --- | --- |
| `threshold_runtime.pdf` | `threshold_sensitivity.csv`; ogbn-arxiv transition grid and sc-ldoor common-grid points |
| `scale_regime.pdf` | `scale_regime.csv`; four nested ogbn-arxiv induced graphs |
| `parallel_scaling.pdf` | `parallel_scaling.csv` |
| `mechanism_summary.pdf` | `alg3_mechanism_counters.csv`, `alg4_state_contraction.csv` |

Regenerate from the repository root with:

```powershell
python tex-data/fig/plot_paper_figures.py
```

`strsub_workflow.tex` and `monosem_workflow.tex` are conceptual algorithm
diagrams synchronized with the current Alg3 and Alg4 implementations. Their
compiled PDFs are included directly by `main.tex`.
