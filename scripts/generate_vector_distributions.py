#!/usr/bin/env python3
"""
生成不同分布的向量文件（Exp-4）。
为 structural 数据集生成 uniform / high-sim / normal 三种分布的向量。
"""

import os
import struct
import random
import math
import numpy as np

DATASETS = [1, 3]  # sc-ldoor, sc-pkustk11
DIM = 256


def normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-9:
        return vec
    return [x / norm for x in vec]


def generate_uniform(n, dim):
    """Uniform [-1,1]^dim, L2-normalized"""
    vecs = []
    for _ in range(n):
        vec = [random.uniform(-1, 1) for _ in range(dim)]
        vecs.append(normalize(vec))
    return vecs


def generate_high_sim(n, dim, sigma=0.1):
    """x = x_bar + epsilon, where x_bar is a shared unit centroid"""
    # 随机生成一个单位质心
    centroid = normalize([random.gauss(0, 1) for _ in range(dim)])
    vecs = []
    for _ in range(n):
        vec = [centroid[i] + random.gauss(0, sigma) for i in range(dim)]
        vecs.append(normalize(vec))
    return vecs


def generate_normal(n, dim):
    """x ~ N(0, I), L2-normalized"""
    vecs = []
    for _ in range(n):
        vec = [random.gauss(0, 1) for _ in range(dim)]
        vecs.append(normalize(vec))
    return vecs


def save_vectors(vecs, output_file):
    n = len(vecs)
    dim = len(vecs[0])
    with open(output_file, 'wb') as f:
        f.write(struct.pack('ii', n, dim))
        for vec in vecs:
            f.write(struct.pack(f'{dim}d', *vec))
    print(f"  Saved: {output_file} ({n} vectors, dim={dim})")


def get_vertex_count(edge_file):
    """从 edge file 第一行读取顶点数"""
    with open(edge_file, 'r') as f:
        first = f.readline().strip()
        while first.startswith('#'):
            first = f.readline().strip()
        parts = first.split()
        return int(parts[0])


def main():
    random.seed(42)
    os.chdir('E:/projects/semanticCliqueMining-writing')

    edge_files = {
        1: 'dataset/dataset/sc-ldoor.txt',
        3: 'dataset/dataset/sc-pkustk11.txt',
    }

    for ds_id in DATASETS:
        edge_file = edge_files.get(ds_id)
        if not edge_file or not os.path.exists(edge_file):
            print(f"SKIP dataset {ds_id}")
            continue

        n = get_vertex_count(edge_file)
        print(f"\nDataset {ds_id}: {n} vertices")

        output_dir = f'dataset/vector_distributions/ds{ds_id}'
        os.makedirs(output_dir, exist_ok=True)

        # Uniform (与原图一致)
        vecs = generate_uniform(n, DIM)
        save_vectors(vecs, os.path.join(output_dir, 'uniform_vectors.bin'))

        # High-sim
        vecs = generate_high_sim(n, DIM, sigma=0.1)
        save_vectors(vecs, os.path.join(output_dir, 'highsim_vectors.bin'))

        # Normal
        vecs = generate_normal(n, DIM)
        save_vectors(vecs, os.path.join(output_dir, 'normal_vectors.bin'))


if __name__ == '__main__':
    main()
