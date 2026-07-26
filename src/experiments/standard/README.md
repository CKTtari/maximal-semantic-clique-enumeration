# Standard paper-result export

This directory contains the scripts that define the reproducible path from
archived experiment summaries to the paper's canonical data files.

Run from the repository root:

```powershell
python src/experiments/standard/paper_results_export.py
```

The exporter reads the archived JSON summaries listed in
`tex-data/data/provenance.json` and rewrites the canonical CSV files in
`tex-data/data`. Pilot summaries contribute execution status only; numeric
paper results come from the formal summaries. Temporary output from future
standard runs belongs in `src/experiments/standard/tmp-output`.

The runtime executables remain separate from statistics executables. Runtime
tables and curves must use raw builds; search-state and pruning counters must
use stats builds and are reported only as mechanism measurements.
