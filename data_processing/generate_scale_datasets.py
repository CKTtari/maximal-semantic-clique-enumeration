#!/usr/bin/env python3
"""
BFS扩展生成不同规模的子图数据集
从固定起点进行单次BFS扩展，在扩展过程中记录不同边规模下的子图
保证各规模子图之间具有嵌套关系与一致的结构分布
"""

import os
import random
import struct
from collections import deque


TARGET_EDGES = [
    (1, int(1e4)),      
    (2, int(4e4)),      
    (3, int(1.6e5)),    
    (4, int(6.3e5)),    
    (5, int(2.5e6)),    
    (6, int(1e7)),      
]

DATASETS = {
    'sc-ldoor': {
        'edge_file': 'dataset/dataset/sc-ldoor.txt',
        'vector_file': 'dataset/vectors-256/sc-ldoor_vectors.bin',
        'vector_dim': 256,
    },
    'tech-as-skitter': {
        'edge_file': 'dataset/dataset/tech-as-skitter.txt',
        'vector_file': 'dataset/vectors-256/tech-as-skitter_vectors.bin',
        'vector_dim': 256,
    },
    'dblp_coauthor': {
        'edge_file': 'dataset/dataset/dblp_coauthor.txt',
        'vector_file': 'dataset/vectors-256/dblp_coauthor_vectors.bin',
        'vector_dim': 256,
    },
    'products': {
        'edge_file': 'dataset/dataset/products.txt',
        'vector_file': 'dataset/vectors-256/products_vectors.bin',
        'vector_dim': 256,
    },
}

OUTPUT_DIR = 'dataset/scale_datasets'


def read_graph(edge_file):
    """读取图，返回邻接表"""
    print(f"Reading graph from {edge_file}...")
    
    adj = {}
    total_edges = 0
    
    with open(edge_file, 'r') as f:
        first_line = f.readline().strip()
        while first_line.startswith('#'):
            first_line = f.readline().strip()
        
        parts = first_line.split()
        num_vertices = int(parts[0])
        num_edges = int(parts[1]) if len(parts) > 1 else 0
        
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


def bfs_expand_with_checkpoints(adj, start_node, target_edges_list):
    """
    单次BFS扩展，在达到目标边数时记录checkpoint
    边只在两个端点都已在子图中时加入，确保得到严格的induced subgraph
    返回: checkpoints = [(level, vertices, edges), ...]
    """
    visited = set([start_node])
    queue = deque([start_node])
    
    subgraph_vertices = set([start_node])
    subgraph_edges = set()
    
    for v in adj.get(start_node, []):
        if v in subgraph_vertices:
            edge = (min(start_node, v), max(start_node, v))
            subgraph_edges.add(edge)
    
    checkpoints = []
    target_iter = iter(sorted(target_edges_list, key=lambda x: x[1]))
    current_target = next(target_iter, None)
    
    while queue and current_target:
        u = queue.popleft()
        
        if u not in adj:
            continue
        
        for v in adj[u]:
            if v not in subgraph_vertices:
                subgraph_vertices.add(v)
                visited.add(v)
                queue.append(v)
        
        for v in adj[u]:
            if v in subgraph_vertices:
                edge = (min(u, v), max(u, v))
                subgraph_edges.add(edge)
        
        while current_target and len(subgraph_edges) >= current_target[1]:
            checkpoints.append((current_target[0], set(subgraph_vertices), set(subgraph_edges)))
            print(f"    Checkpoint L{current_target[0]}: {len(subgraph_vertices):,} vertices, {len(subgraph_edges):,} edges")
            current_target = next(target_iter, None)
    
    return checkpoints


def save_subgraph(vertices, edges, output_edge_file):
    """保存子图，顶点重新编号为0开始的连续整数"""
    vertex_list = sorted(vertices)
    vertex_map = {v: i for i, v in enumerate(vertex_list)}
    
    with open(output_edge_file, 'w') as f:
        f.write(f"{len(vertices)} {len(edges)}\n")
        for u, v in sorted(edges):
            f.write(f"{vertex_map[u]} {vertex_map[v]}\n")
    
    return vertex_map, vertex_list


def load_all_vectors(vector_file, vector_dim):
    """一次性读取所有向量"""
    print(f"  Loading vectors from {vector_file}...")
    vectors = {}
    
    with open(vector_file, 'rb') as f:
        header = f.read(8)
        total_vertices, dim = struct.unpack('ii', header)
        
        assert dim == vector_dim, f"Dimension mismatch: {dim} vs {vector_dim}"
        
        for v in range(total_vertices):
            vec_data = f.read(dim * 8)
            vectors[v] = vec_data
    
    print(f"    Loaded {len(vectors)} vectors, dim={dim}")
    return vectors, dim


def save_vectors(vectors, vertex_list, output_file, dim):
    """保存指定顶点的向量"""
    with open(output_file, 'wb') as f:
        f.write(struct.pack('ii', len(vertex_list), dim))
        for v in vertex_list:
            f.write(vectors[v])


def process_dataset(name, config, output_base):
    """处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"{'='*60}")
    
    edge_file = config['edge_file']
    vector_file = config['vector_file']
    vector_dim = config['vector_dim']
    
    if not os.path.exists(edge_file):
        print(f"  Edge file not found: {edge_file}")
        return
    
    adj, total_edges = read_graph(edge_file)
    
    targets_for_dataset = [(l, t) for l, t in TARGET_EDGES if t <= total_edges]
    
    if not targets_for_dataset:
        print(f"  Graph too small for any target level")
        return
    
    start_node = random.choice(list(adj.keys()))
    print(f"  Start node: {start_node}")
    print(f"  Target levels: {[f'L{l}' for l, _ in targets_for_dataset]}")
    
    print(f"  Running single BFS expansion...")
    checkpoints = bfs_expand_with_checkpoints(adj, start_node, targets_for_dataset)
    
    print(f"  Got {len(checkpoints)} checkpoints")
    
    all_vectors = None
    if os.path.exists(vector_file):
        all_vectors, actual_dim = load_all_vectors(vector_file, vector_dim)
    else:
        print(f"  Vector file not found: {vector_file}")
    
    dataset_dir = os.path.join(output_base, name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    for level, vertices, edges in checkpoints:
        print(f"  L{level}: {len(vertices):,} vertices, {len(edges):,} edges")
        
        output_edge_file = os.path.join(dataset_dir, f"{name}_L{level}.txt")
        vertex_map, vertex_list = save_subgraph(vertices, edges, output_edge_file)
        print(f"    Saved: {output_edge_file}")
        
        if all_vectors:
            output_vector_file = os.path.join(dataset_dir, f"{name}_L{level}_vectors.bin")
            save_vectors(all_vectors, vertex_list, output_vector_file, actual_dim)
            print(f"    Saved: {output_vector_file}")


def main():
    random.seed(42)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("BFS Scale Dataset Generator (Single-Pass)")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Target edge counts:")
    for level, target in TARGET_EDGES:
        print(f"  L{level}: {target:,}")
    
    for name, config in DATASETS.items():
        process_dataset(name, config, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
