# Paper figures

plot_paper_figures.py reads canonical CSV files from results/paper-data and
quality records from results/quality. It writes each experimental panel as PDF
and 600 dpi PNG at its final single-column size.

The two multi-panel paper figures are composed with LaTeX subfigure blocks.
result_distribution_legend.pdf is the shared legend for the two distribution
panels. mechanism_strsub.pdf and mechanism_monosem.pdf form the two side-by-side
mechanism panels.

The plotting helpers are vendored under third_party/scipilot-figure-skill with
their upstream MIT license.