#!/usr/bin/env python3
"""
生成不同密度的图变体（Exp-3）。
随机删边至目标比例，保持最小度 >= 1。
"""

import os
import random

RATIOS = [0.25, 0.50, 1.0]
DATASETS = [1, 3, 7, 9]


def read_edges(edge_file):
    edges = []
    with open(edge_file, 'r') as f:
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
    return list(set(edges))


def generate_variant(edges, ratio, output_file):
    target = int(len(edges) * ratio)
    # 构建度数字典
    deg = {}
    for u, v in edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1

    # 随机打乱后尝试删除
    shuffled = edges[:]
    random.shuffle(shuffled)
    kept = set(edges)

    for u, v in shuffled:
        if len(kept) <= target:
            break
        # 检查删除后是否违反最小度
        if deg[u] > 1 and deg[v] > 1:
            kept.remove((min(u, v), max(u, v)))
            deg[u] -= 1
            deg[v] -= 1

    # 收集所有顶点
    vertices = set()
    for u, v in kept:
        vertices.add(u)
        vertices.add(v)

    vertex_list = sorted(vertices)
    vertex_map = {v: i for i, v in enumerate(vertex_list)}

    with open(output_file, 'w') as f:
        f.write(f"{len(vertices)} {len(kept)}\n")
        for u, v in sorted(kept):
            f.write(f"{vertex_map[u]} {vertex_map[v]}\n")

    print(f"  Saved: {output_file} ({len(kept)} edges, {len(vertices)} vertices)")
    return vertex_map, vertex_list


def main():
    random.seed(42)
    os.chdir('E:/projects/semanticCliqueMining-writing')

    for ds_id in DATASETS:
        # 获取 edge file 路径
        # 这里简化处理，使用 dataset ID 映射
        edge_files = {
            1: 'dataset/dataset/sc-ldoor.txt',
            3: 'dataset/dataset/sc-pkustk11.txt',
            7: 'dataset/dataset/FB15K-237_edges.txt',
            9: 'dataset/dataset/dblp_coauthor.txt',
        }
        edge_file = edge_files.get(ds_id)
        if not edge_file or not os.path.exists(edge_file):
            print(f"SKIP dataset {ds_id}")
            continue

        print(f"\nDataset {ds_id}: {edge_file}")
        edges = read_edges(edge_file)
        print(f"  Original: {len(edges)} edges")

        output_dir = f'dataset/density_datasets/ds{ds_id}'
        os.makedirs(output_dir, exist_ok=True)

        for ratio in RATIOS:
            ratio_name = f"{int(ratio*100)}pct"
            output_file = os.path.join(output_dir, f"ds{ds_id}_{ratio_name}.txt")
            generate_variant(edges, ratio, output_file)


if __name__ == '__main__':
    main()
