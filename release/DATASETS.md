# Datasets and preprocessing

No raw or processed dataset is redistributed in this artifact. The table fixes
the exact common-grid inventory used by the paper.

| Dataset | Vertices | Edges | Dim. | Attributes | Public source |
| --- | ---: | ---: | ---: | --- | --- |
| FB15K-237 | 14,541 | 248,611 | 384 | MiniLM text embeddings | Microsoft FB15K-237 download |
| WN18RR | 40,943 | 75,762 | 384 | MiniLM text embeddings | KG-BERT WN18RR files |
| ogbn-arxiv | 169,343 | 1,157,799 | 128 | Official node features | Open Graph Benchmark |
| email-Enron | 36,692 | 183,831 | 256 | Fixed generated vectors | SNAP |
| sc-nasasrb | 54,870 | 1,311,227 | 256 | Fixed generated vectors | Network Repository |
| sc-pkustk11 | 87,804 | 2,565,054 | 256 | Fixed generated vectors | Network Repository |
| sc-pwtk | 217,891 | 5,653,221 | 256 | Fixed generated vectors | Network Repository |
| sc-ldoor | 909,537 | 20,770,807 | 256 | Fixed generated vectors | Network Repository |

The scientific-computing graphs are Network Repository graph exports of
SuiteSparse-origin matrices. The release records the Network Repository pages
because those graph exports, rather than matrix numeric values, are the actual
inputs used in the experiments.

## Public source pages

FB15K-237:
https://www.microsoft.com/en-us/download/details.aspx?id=52312

WN18RR:
https://github.com/yao8839836/kg-bert/tree/master/data/WN18RR

ogbn-arxiv documentation and archive:
https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv
https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip

email-Enron:
https://snap.stanford.edu/data/email-Enron.html

Scientific-computing graph exports:
https://networkrepository.com/sc_nasasrb.php
https://networkrepository.com/sc_pkustk11.php
https://networkrepository.com/sc_pwtk.php
https://networkrepository.com/sc_ldoor.php

## Natural semantic attributes

For FB15K-237, the graph is the undirected simple union of train, validation,
and test triples. Relation labels and edge direction are not used. Entities are
assigned contiguous identifiers by lexicographic entity ID. Text is extracted
from lexical tokens in text_emnlp.txt dependency paths; at most five extracted
descriptions are concatenated per entity.

For WN18RR, the graph is built by the same union, direction removal,
deduplication, and sorted-ID rule over train.tsv, dev.tsv, and test.tsv. Entity
text and WordNet definitions are read from entity2text.txt and
wordnet-mlj12-definitions.txt, with at most five descriptions per entity.

Both datasets use sentence-transformers/all-MiniLM-L6-v2 through the
Transformers encoder. Token vectors are attention-mask mean pooled, truncated
at 256 tokens, and L2-normalized. Missing descriptions use the literal text
"unknown entity." The portable command is:

    python scripts/prepare_kg_dataset.py --dataset FB15K-237 --input-dir dataset/downloads/FB15K-237
    python scripts/prepare_kg_dataset.py --dataset WN18RR --input-dir dataset/downloads/WN18RR

requirements-kg.txt contains the additional packages. The preparation record
stores the resolved model commit exposed by Transformers, package versions, and
all output hashes. dataset/checksums.json is the compatibility boundary for the
paper's archived vectors.

## ogbn-arxiv

Place the official archive at dataset/downloads/ogbn-arxiv/arxiv.zip and run:

    python scripts/import_ogbn_arxiv.py

The importer verifies the archived source SHA-256, drops self-loops, ignores
citation direction, collapses duplicate or reciprocal rows, sorts undirected
edges, and preserves the official 128-dimensional feature rows. The shared C++
loader applies L2 normalization.

## email-Enron

Download email-Enron.txt.gz from SNAP and run:

    python scripts/import_snap_graph.py --input dataset/downloads/email-Enron.txt.gz --output dataset/Processed/email-Enron/graph.txt --provenance dataset/Processed/email-Enron/provenance.json --name email-Enron --source-url https://snap.stanford.edu/data/email-Enron.html
    python scripts/generate_structural_vectors.py --dataset email-Enron

The graph conversion removes self-loops and duplicate edges, treats the graph
as undirected, and maps ascending source identifiers to contiguous zero-based
identifiers.

## Scientific-computing graphs

Download and unpack each Network Repository edge list. Normalize it with:

    python scripts/normalize_public_graph.py --input INPUT.edges --output dataset/dataset/sc-nasasrb.txt --name sc-nasasrb --source-url https://networkrepository.com/sc_nasasrb.php

Repeat with the corresponding name, output path, and source URL. The normalizer
removes self-loops, merges reciprocal and duplicate rows, removes isolated
vertices, and maps ascending active source identifiers to contiguous IDs. For a
Matrix Market coordinate file, add --format matrix-market. If an edge list
starts with a V E row, add --skip-first-data-row.

Generate the five controlled attribute files:

    python scripts/generate_structural_vectors.py --dataset sc-nasasrb --dataset sc-pkustk11 --dataset sc-pwtk --dataset sc-ldoor --dataset email-Enron

The generator uses NumPy 1.26.4 PCG64 with independent coordinates from
Uniform[-1,1), dimension 256, and these fixed seeds:

| Dataset | Seed |
| --- | ---: |
| sc-ldoor | 2026072301 |
| sc-nasasrb | 2026072302 |
| sc-pkustk11 | 2026072303 |
| sc-pwtk | 2026072307 |
| email-Enron | 2026072315 |

The binary stores little-endian int32 vertex count, int32 dimension, and
row-major float64 coordinates. The shared C++ loader performs L2 normalization.

## Input formats

A graph file starts with V E and then contains one zero-based undirected edge
u v per line. A single vector binary is int32 V, int32 dimension, followed by
V times dimension float64 values. A chunk index starts with V dimension and
then lists chunk binaries with the same per-chunk header.

Exact expected byte sizes and SHA-256 values are in dataset/checksums.json.