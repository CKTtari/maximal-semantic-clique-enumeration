# Datasets and Preparation

No raw or processed dataset is redistributed in this repository. The table
below identifies the eight datasets used by the paper and the public source of
each input.

| Dataset | ID | Vertices | Edges | Dim. | Attribute source |
| --- | ---: | ---: | ---: | ---: | --- |
| FB15K-237 | 7 | 14,541 | 248,611 | 384 | [Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=52312) |
| GitHub-Social | 17 | 37,700 | 289,003 | 4,005 | [SNAP](https://snap.stanford.edu/data/github-social.html) |
| ogbn-arxiv | 11 | 169,343 | 1,157,799 | 128 | [OGB](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) |
| email-Enron | 16 | 36,692 | 183,831 | 256 | [SNAP](https://snap.stanford.edu/data/email-Enron.html) |
| sc-nasasrb | 2 | 54,870 | 1,311,227 | 256 | [Network Repository](https://networkrepository.com/sc_nasasrb.php) |
| sc-pkustk11 | 3 | 87,804 | 2,565,054 | 256 | [Network Repository](https://networkrepository.com/sc_pkustk11.php) |
| sc-pwtk | 14 | 217,891 | 5,653,221 | 256 | [Network Repository](https://networkrepository.com/sc_pwtk.php) |
| sc-ldoor | 1 | 909,537 | 20,770,807 | 256 | [Network Repository](https://networkrepository.com/sc_ldoor.php) |

## Graph processing

All graph inputs are converted to a simple undirected graph: self-loops and
duplicate or reciprocal edges are removed, active vertices are mapped to
contiguous zero-based identifiers, and the edge list is sorted. FB15K-237 is
the undirected union of its train, validation, and test triples. GitHub-Social
retains the official 4,005-dimensional profile features after graph cleanup.
ogbn-arxiv ignores citation direction, removes self-loops and duplicate rows,
and retains the official 128-dimensional node features.

For FB15K-237, entity descriptions are converted to 384-dimensional vectors
with `sentence-transformers/all-MiniLM-L6-v2`: token embeddings are
attention-mask mean pooled over at most 256 tokens and then L2-normalized.
Missing descriptions use the text `unknown entity`.

email-Enron and the four scientific-computing graphs use generated 256-
dimensional vectors. Coordinates are independent draws from Uniform[-1,1)
using NumPy PCG64, followed by the shared loader's L2 normalization. The fixed
seeds are:

| Dataset | Seed |
| --- | ---: |
| sc-ldoor | 2026072301 |
| sc-nasasrb | 2026072302 |
| sc-pkustk11 | 2026072303 |
| sc-pwtk | 2026072307 |
| email-Enron | 2026072315 |

## Input layout

Place processed files under the paths selected by `src/experiment_runtime.h`.
Each graph file begins with `V E`, followed by zero-based undirected edge pairs
`u v`. Each vector binary stores int32 `V`, int32 dimension, and then `V` times
dimension float64 values in row-major order. The common loader applies L2
normalization before enumeration.
