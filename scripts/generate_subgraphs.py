#!/usr/bin/env python3
"""
BFS扩展生成不同规模的子图数据集（更新版）
对应论文 Exp-2：对 DBLP (dataset 9) 和 Amazon (dataset 10)
生成 6 个子图，边数从 10% 到 100%。
"""

import os
import random
import struct
from collections import deque

# 6 个比例点：10%, 20%, 30%, 50%, 80%, 100%
RATIOS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

DATASETS = {
    9: {
        'name': 'dblp_coauthor',
        'edge_file': 'dataset/dataset/dblp_coauthor.txt',
        'vector_file': 'dataset/vectors/dblp_coauthor_vectors.index',
        'output_dir': 'dataset/scale_datasets/dblp',
    },
    10: {
        'name': 'products',
        'edge_file': 'dataset/dataset/products.txt',
        'vector_file': 'dataset/vectors/products_vectors.index',
        'output_dir': 'dataset/scale_datasets/products',
    },
}


def read_graph(edge_file):
    """读取图，返回邻接表和总边数"""
    print(f"Reading graph from {edge_file}...")
    adj = {}
    total_edges = 0
    with open(edge_file, 'r') as f:
        first_line = f.readline().strip()
        while first_line.startswith('#'):
            first_line = f.readline().strip()
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                u, v = int(parts[0]), int(parts[1])
                if u not in adj:
                    adj[u] = set()
                if v not in adj:
                    adj[v] = set()
                if v not in adj[u]:
                    adj[u].add(v)
                    adj[v].add(u)
                    total_edges += 1
    print(f"  Vertices: {len(adj)}, Edges: {total_edges}")
    return adj, total_edges


def bfs_expand_to_ratio(adj, start_node, target_edges):
    """BFS扩展直到达到目标边数"""
    visited = set([start_node])
    queue = deque([start_node])
    subgraph_vertices = set([start_node])
    subgraph_edges = set()

    # 添加起点邻居的边
    for v in adj.get(start_node, set()):
        if v in subgraph_vertices:
            edge = (min(start_node, v), max(start_node, v))
            subgraph_edges.add(edge)

    while queue and len(subgraph_edges) < target_edges:
        u = queue.popleft()
        for v in adj.get(u, set()):
            if v not in subgraph_vertices:
                subgraph_vertices.add(v)
                visited.add(v)
                queue.append(v)
            # 添加 u 与所有已在子图中的邻居的边
            for w in adj.get(u, set()):
                if w in subgraph_vertices:
                    edge = (min(u, w), max(u, w))
                    if edge not in subgraph_edges:
                        subgraph_edges.add(edge)
                        if len(subgraph_edges) >= target_edges:
                            break
            if len(subgraph_edges) >= target_edges:
                break

    return subgraph_vertices, subgraph_edges


def save_subgraph(vertices, edges, output_edge_file):
    """保存子图，顶点重新编号为0开始的连续整数"""
    vertex_list = sorted(vertices)
    vertex_map = {v: i for i, v in enumerate(vertex_list)}
    with open(output_edge_file, 'w') as f:
        f.write(f"{len(vertices)} {len(edges)}\n")
        for u, v in sorted(edges):
            f.write(f"{vertex_map[u]} {vertex_map[v]}\n")
    return vertex_map, vertex_list


def save_vectors_index(original_index_file, vertex_list, output_index_file, output_dir):
    """为子图生成向量 index 文件（引用原始 chunk 文件）"""
    # 对于 chunk 格式的向量，子图顶点需要重新映射
    # 简化处理：直接复制原始 index 文件，因为 chunk 文件包含所有顶点
    # 实际运行时，算法会读取 index 中指定的 chunk 文件
    # 由于顶点已重新编号为 0..n-1，需要提取对应 chunk 中的向量
    # 这里我们生成一个新的 index 文件，指向提取后的 chunk

    # 读取原始 index
    base_dir = os.path.dirname(original_index_file)
    with open(original_index_file, 'r') as f:
        lines = f.readlines()

    total_vertices, vector_dim = map(int, lines[0].strip().split())

    # 收集所有 chunk 文件路径
    chunk_files = []
    for line in lines[1:]:
        cf = line.strip()
        if cf:
            if not os.path.isabs(cf):
                cf = os.path.join(base_dir, cf)
            chunk_files.append(cf)

    # 简化：假设所有顶点都在一个 chunk 中（或按顺序分布）
    # 对于子图，我们只保留需要的顶点，生成新的 chunk
    # 由于 chunk 文件格式为 [int n][int dim][n*dim doubles]
    # 我们需要读取所有 chunk，提取需要的顶点

    # 读取所有向量
    all_vectors = []
    for cf in chunk_files:
        with open(cf, 'rb') as f:
            chunk_v, chunk_d = struct.unpack('ii', f.read(8))
            assert chunk_d == vector_dim
            for _ in range(chunk_v):
                vec_data = f.read(vector_dim * 8)
                vec = struct.unpack(f'{vector_dim}d', vec_data)
                all_vectors.append(vec)

    # 提取子图需要的向量（按原始顶点顺序）
    sub_vectors = []
    for v in vertex_list:
        if v < len(all_vectors):
            sub_vectors.append(all_vectors[v])

    # 保存为新的 chunk 文件
    os.makedirs(output_dir, exist_ok=True)
    chunk_name = os.path.join(output_dir, 'vectors.chunk.0.bin')
    with open(chunk_name, 'wb') as f:
        f.write(struct.pack('ii', len(sub_vectors), vector_dim))
        for vec in sub_vectors:
            f.write(struct.pack(f'{vector_dim}d', *vec))

    # 保存新的 index 文件
    with open(output_index_file, 'w') as f:
        f.write(f"{len(sub_vectors)} {vector_dim}\n")
        f.write(f"{chunk_name}\n")


def process_dataset(dataset_id, config):
    """处理单个数据集，生成多个比例的子图"""
    print(f"\n{'='*60}")
    print(f"Processing dataset {dataset_id}: {config['name']}")
    print(f"{'='*60}")

    edge_file = config['edge_file']
    vector_file = config['vector_file']
    output_dir = config['output_dir']

    if not os.path.exists(edge_file):
        print(f"  SKIP: edge file not found: {edge_file}")
        return

    adj, total_edges = read_graph(edge_file)

    # 选择度数最高的顶点作为 BFS 起点，确保子图连通且有意义
    start_node = max(adj.keys(), key=lambda x: len(adj[x]))
    print(f"  Start node: {start_node} (degree {len(adj[start_node])})")

    os.makedirs(output_dir, exist_ok=True)

    for ratio in RATIOS:
        target_edges = int(total_edges * ratio)
        level_name = f"L{int(ratio*100)}"

        print(f"\n  Generating {level_name}: target {target_edges:,} edges ({ratio*100:.0f}%)")

        vertices, edges = bfs_expand_to_ratio(adj, start_node, target_edges)
        print(f"    Result: {len(vertices):,} vertices, {len(edges):,} edges")

        # 保存子图 edge list
        edge_output = os.path.join(output_dir, f"{config['name']}_{level_name}.txt")
        vertex_map, vertex_list = save_subgraph(vertices, edges, edge_output)
        print(f"    Saved: {edge_output}")

        # 保存向量
        if os.path.exists(vector_file):
            index_output = os.path.join(output_dir, f"{config['name']}_{level_name}_vectors.index")
            save_vectors_index(vector_file, vertex_list, index_output, output_dir)
            print(f"    Saved: {index_output}")


def main():
    random.seed(42)
    os.chdir('E:/projects/semanticCliqueMining-writing')

    for ds_id, config in DATASETS.items():
        process_dataset(ds_id, config)

    print(f"\n{'='*60}")
    print("Done! Subgraphs generated in dataset/scale_datasets/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
