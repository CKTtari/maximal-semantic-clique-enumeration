#!/usr/bin/env python3
"""
DBLP 团质量分析（Exp-10）。
读取算法输出的团文件，计算每个团的 avgSim，按大小和 avgSim 排序。
"""

import argparse
import json
import os
from pathlib import Path


def load_vectors(index_file):
    """加载 DBLP 向量"""
    import struct
    base_dir = os.path.dirname(index_file)
    vectors = []
    with open(index_file, 'r') as f:
        lines = f.readlines()
    total_vertices, vector_dim = map(int, lines[0].strip().split())
    for line in lines[1:]:
        cf = line.strip()
        if not cf:
            continue
        if not os.path.isabs(cf):
            cf = os.path.join(base_dir, cf)
        with open(cf, 'rb') as cfile:
            chunk_v, chunk_d = struct.unpack('ii', cfile.read(8))
            assert chunk_d == vector_dim
            for _ in range(chunk_v):
                vec_data = cfile.read(vector_dim * 8)
                vec = struct.unpack(f'{vector_dim}d', vec_data)
                vectors.append(vec)
    return vectors


def cosine_similarity(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))


def compute_clique_avg_sim(clique, vectors):
    """计算团的平均 pairwise similarity"""
    if len(clique) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            u, v = clique[i], clique[j]
            if u < len(vectors) and v < len(vectors):
                total += cosine_similarity(vectors[u], vectors[v])
                count += 1
    return total / count if count > 0 else 0.0


def analyze_cliques(clique_file, vectors, top_k=20):
    """分析团文件"""
    cliques = []
    with open(clique_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                clique = [int(x) for x in parts]
                avg_sim = compute_clique_avg_sim(clique, vectors)
                cliques.append({
                    'size': len(clique),
                    'avg_sim': avg_sim,
                    'vertices': clique,
                })

    # 按 avg_sim 降序
    cliques.sort(key=lambda x: x['avg_sim'], reverse=True)

    print(f"Total cliques: {len(cliques)}")
    print(f"\nTop {top_k} cliques by avgSim:")
    print("-" * 60)
    for i, c in enumerate(cliques[:top_k]):
        print(f"{i+1}. Size={c['size']}, AvgSim={c['avg_sim']:.3f}, Vertices={c['vertices'][:10]}{'...' if len(c['vertices']) > 10 else ''}")

    # 按大小分组统计
    from collections import defaultdict
    size_groups = defaultdict(list)
    for c in cliques:
        size_groups[c['size']].append(c['avg_sim'])

    print(f"\nSize distribution:")
    for sz in sorted(size_groups.keys()):
        sims = size_groups[sz]
        print(f"  Size {sz}: {len(sims)} cliques, avgSim range [{min(sims):.3f}, {max(sims):.3f}], mean={sum(sims)/len(sims):.3f}")

    return cliques


def main():
    parser = argparse.ArgumentParser(description='Analyze DBLP cliques')
    parser.add_argument('clique_file', help='Clique output file')
    parser.add_argument('--vectors', default='dataset/vectors/dblp_coauthor_vectors.index', help='Vector index file')
    parser.add_argument('--top-k', type=int, default=20, help='Number of top cliques to show')
    args = parser.parse_args()

    os.chdir('E:/projects/semanticCliqueMining-writing')

    print(f"Loading vectors from {args.vectors}...")
    vectors = load_vectors(args.vectors)
    print(f"Loaded {len(vectors)} vectors")

    print(f"\nAnalyzing cliques from {args.clique_file}...")
    analyze_cliques(args.clique_file, vectors, args.top_k)


if __name__ == '__main__':
    main()
