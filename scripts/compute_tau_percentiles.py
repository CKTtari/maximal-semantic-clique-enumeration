#!/usr/bin/env python3
"""
Compute edge similarity percentiles for all datasets.
Outputs tau_values.json with 20th, 50th, 70th, 80th, 90th, 95th percentiles.
"""

import os
import json
import struct
import math
from pathlib import Path

DATASETS = {
    1: {
        'name': 'sc-ldoor',
        'edge_file': 'dataset/dataset/sc-ldoor.txt',
        'vector_file': 'dataset/vectors-256/sc-ldoor_vectors.bin',
        'type': 'structural',
        'dim': 256,
    },
    2: {
        'name': 'sc-nasasrb',
        'edge_file': 'dataset/dataset/sc-nasasrb.txt',
        'vector_file': 'dataset/vectors-256/sc-nasasrb_vectors.bin',
        'type': 'structural',
        'dim': 256,
    },
    3: {
        'name': 'sc-pkustk11',
        'edge_file': 'dataset/dataset/sc-pkustk11.txt',
        'vector_file': 'dataset/vectors-256/sc-pkustk11_vectors.bin',
        'type': 'structural',
        'dim': 256,
    },
    4: {
        'name': 'soc-buzznet',
        'edge_file': 'dataset/dataset/soc-buzznet.txt',
        'vector_file': 'dataset/vectors-256/soc-buzznet_vectors.bin',
        'type': 'structural',
        'dim': 256,
    },
    5: {
        'name': 'soc-digg',
        'edge_file': 'dataset/dataset/soc-digg.txt',
        'vector_file': 'dataset/vectors-256/soc-digg_vectors.bin',
        'type': 'structural',
        'dim': 256,
    },
    6: {
        'name': 'tech-as-skitter',
        'edge_file': 'dataset/dataset/tech-as-skitter.txt',
        'vector_file': 'dataset/vectors-256/tech-as-skitter_vectors.bin',
        'type': 'structural',
        'dim': 256,
    },
    7: {
        'name': 'FB15K-237',
        'edge_file': 'dataset/dataset/FB15K-237_edges.txt',
        'vector_file': 'dataset/vectors/FB15K-237_vectors.index',
        'type': 'semantic',
        'dim': 384,
    },
    8: {
        'name': 'WN18RR',
        'edge_file': 'dataset/dataset/WN18RR_edges.txt',
        'vector_file': 'dataset/vectors/WN18RR_vectors.index',
        'type': 'semantic',
        'dim': 384,
    },
    9: {
        'name': 'DBLP',
        'edge_file': 'dataset/dataset/dblp_coauthor.txt',
        'vector_file': 'dataset/vectors/dblp_coauthor_vectors.index',
        'type': 'semantic',
        'dim': 384,
    },
    10: {
        'name': 'Amazon',
        'edge_file': 'dataset/dataset/products.txt',
        'vector_file': 'dataset/vectors/products_vectors.index',
        'type': 'semantic',
        'dim': 384,
    },
}

PERCENTILES = [20, 50, 70, 80, 90, 95]


def load_vectors_binary(filepath, dim):
    """Load vectors from binary file: [int V][int dim][V*dim doubles]"""
    vectors = {}
    with open(filepath, 'rb') as f:
        header = f.read(8)
        V, d = struct.unpack('ii', header)
        assert d == dim, f"Dimension mismatch: {d} vs {dim}"
        for v in range(V):
            data = f.read(dim * 8)
            vec = struct.unpack(f'{dim}d', data)
            vectors[v] = vec
    return vectors


def load_vectors_chunks(index_file, dim):
    """Load vectors from chunk index file."""
    vectors = {}
    base_dir = os.path.dirname(index_file)
    with open(index_file, 'r') as f:
        total_vertices, vector_dim = map(int, f.readline().split())
        assert vector_dim == dim
        for line in f:
            chunk_file = line.strip()
            if not chunk_file:
                continue
            if not os.path.isabs(chunk_file) and not os.path.exists(chunk_file):
                chunk_file = os.path.join(base_dir, chunk_file)
            with open(chunk_file, 'rb') as cf:
                chunk_v, chunk_d = struct.unpack('ii', cf.read(8))
                assert chunk_d == dim
                for v in range(chunk_v):
                    data = cf.read(dim * 8)
                    vec = struct.unpack(f'{dim}d', data)
                    # Assign global vertex ID sequentially
                    vid = len(vectors)
                    vectors[vid] = vec
    return vectors


def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    return dot  # vectors are normalized


def compute_percentiles(dataset_id, config):
    print(f"\n[{dataset_id}] {config['name']} ...")

    # Load graph edges
    edges = []
    with open(config['edge_file'], 'r') as f:
        first = f.readline().strip()
        while first.startswith('#'):
            first = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u != v:
                    edges.append((min(u, v), max(u, v)))

    edges = list(set(edges))  # dedup
    print(f"  Edges: {len(edges)}")

    # Load vectors
    vec_file = config['vector_file']
    if vec_file.endswith('.bin'):
        vectors = load_vectors_binary(vec_file, config['dim'])
    else:
        vectors = load_vectors_chunks(vec_file, config['dim'])
    print(f"  Vectors: {len(vectors)}")

    # Compute similarities for all edges
    sims = []
    missing = 0
    for u, v in edges:
        if u in vectors and v in vectors:
            sim = cosine_similarity(vectors[u], vectors[v])
            sims.append(sim)
        else:
            missing += 1

    if missing > 0:
        print(f"  Warning: {missing} edges missing vectors")

    sims.sort()
    n = len(sims)
    print(f"  Valid similarities: {n}")

    result = {}
    for p in PERCENTILES:
        idx = int((p / 100.0) * (n - 1))
        idx = max(0, min(idx, n - 1))
        result[f'tau_{p}'] = round(sims[idx], 6)

    # Also compute some statistics
    result['min'] = round(sims[0], 6) if n > 0 else 0
    result['max'] = round(sims[-1], 6) if n > 0 else 0
    result['mean'] = round(sum(sims) / n, 6) if n > 0 else 0
    result['median'] = round(sims[n // 2], 6) if n > 0 else 0

    for p in PERCENTILES:
        print(f"    tau_{p}: {result[f'tau_{p}']}")

    return result


def main():
    os.chdir('E:/projects/semanticCliqueMining-writing')
    results = {}

    for ds_id, config in DATASETS.items():
        if not os.path.exists(config['edge_file']):
            print(f"  SKIP: edge file not found: {config['edge_file']}")
            continue
        if not os.path.exists(config['vector_file']):
            print(f"  SKIP: vector file not found: {config['vector_file']}")
            continue
        results[ds_id] = compute_percentiles(ds_id, config)
        results[ds_id]['name'] = config['name']

    output_file = 'scripts/tau_values.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_file}")

    # Print markdown table
    print("\n| Dataset | tau_20 | tau_50 | tau_70 | tau_80 | tau_90 | tau_95 |")
    print("|---------|--------|--------|--------|--------|--------|--------|")
    for ds_id in sorted(results.keys()):
        r = results[ds_id]
        print(f"| {r['name']} | {r['tau_20']} | {r['tau_50']} | {r['tau_70']} | {r['tau_80']} | {r['tau_90']} | {r['tau_95']} |")


if __name__ == '__main__':
    main()
