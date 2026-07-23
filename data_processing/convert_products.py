#!/usr/bin/env python3
import os
import gzip
import pandas as pd

# 数据集路径
DATASET_DIR = 'dataset/dataset/products'
RAW_DIR = os.path.join(DATASET_DIR, 'raw')
OUTPUT_DIR = 'dataset/dataset'

# 输出文件名
OUTPUT_GRAPH_FILE = os.path.join(OUTPUT_DIR, 'products_co-purchase.txt')
OUTPUT_VECTORS_FILE = os.path.join('dataset/vectors', 'products_vectors.index')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('dataset/vectors', exist_ok=True)

def convert_graph():
    """将 products 数据集转换为 Graph 类期望的格式"""
    print("=== 转换 products 数据集 ===")
    
    # 1. 读取节点数量
    node_list_file = os.path.join(RAW_DIR, 'num-node-list.csv')
    print(f"读取节点数量: {node_list_file}")
    with open(node_list_file, 'r') as f:
        num_nodes = int(f.read().strip())
    print(f"节点数: {num_nodes}")
    
    # 2. 读取边列表（分块处理）
    edge_list_file = os.path.join(RAW_DIR, 'edge.csv')
    print(f"读取边列表: {edge_list_file}")
    
    # 计算边数
    num_edges = 0
    with open(edge_list_file, 'r') as f:
        for line in f:
            if line.strip():
                num_edges += 1
    print(f"边数: {num_edges}")
    
    # 3. 写入图文件（分块处理）
    print(f"写入图文件: {OUTPUT_GRAPH_FILE}")
    batch_size = 1000000  # 每批处理100万条边
    
    with open(OUTPUT_GRAPH_FILE, 'w') as f:
        # 写入第一行：节点数和边数
        f.write(f"{num_nodes} {num_edges}\n")
        
        # 分块写入边
        edge_count = 0
        batch = []
        with open(edge_list_file, 'r') as edge_file:
            for line in edge_file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    u = int(parts[0])
                    v = int(parts[1])
                    batch.append(f"{u} {v}\n")
                    edge_count += 1
                    
                    if len(batch) >= batch_size:
                        f.writelines(batch)
                        batch.clear()
                        print(f"  已写入 {edge_count}/{num_edges} 条边...")
        
        # 写入剩余的边
        if batch:
            f.writelines(batch)
            print(f"  已写入 {edge_count}/{num_edges} 条边...")
    
    print(f"图文件转换完成: {OUTPUT_GRAPH_FILE}")
    
    # 4. 处理节点特征向量（分块处理）
    print("处理节点特征向量...")
    feat_file = os.path.join(RAW_DIR, 'node-feat.csv')
    print(f"读取特征文件: {feat_file}")
    
    # 先读取第一行确定向量维度
    vector_dim = 0
    with open(feat_file, 'r') as f:
        first_line = f.readline()
        if first_line:
            parts = first_line.strip().split(',')
            vector_dim = len(parts)
    
    if vector_dim > 0:
        print(f"特征向量维度: {vector_dim}")
        
        # 5. 分块生成向量文件
        import struct
        import gc
        
        chunk_size = 100000  # 每块处理10万个向量
        chunk_files = []
        
        # 读取并处理向量文件
        with open(feat_file, 'r') as f:
            chunk_idx = 0
            current_chunk = []
            line_count = 0
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                vec = [float(x) for x in parts]
                current_chunk.append(vec)
                line_count += 1
                
                # 当达到块大小或文件结束时，写入块文件
                if len(current_chunk) >= chunk_size:
                    chunk_output = os.path.join('dataset/vectors', f'products_vectors.chunk.{chunk_idx}.bin')
                    chunk_files.append(chunk_output)
                    
                    print(f"  处理块 {chunk_idx}: {len(current_chunk)} 个向量...")
                    
                    # 写入二进制块文件
                    with open(chunk_output, 'wb') as chunk_f:
                        # 写入块头部：向量数量和维度
                        chunk_f.write(struct.pack('ii', len(current_chunk), vector_dim))
                        # 写入向量
                        for vec in current_chunk:
                            for val in vec:
                                chunk_f.write(struct.pack('d', val))
                    
                    # 清空当前块并释放内存
                    current_chunk.clear()
                    gc.collect()
                    chunk_idx += 1
            
            # 处理最后一个块
            if current_chunk:
                chunk_output = os.path.join('dataset/vectors', f'products_vectors.chunk.{chunk_idx}.bin')
                chunk_files.append(chunk_output)
                
                print(f"  处理块 {chunk_idx}: {len(current_chunk)} 个向量...")
                
                with open(chunk_output, 'wb') as chunk_f:
                    chunk_f.write(struct.pack('ii', len(current_chunk), vector_dim))
                    for vec in current_chunk:
                        for val in vec:
                            chunk_f.write(struct.pack('d', val))
                
                current_chunk.clear()
                gc.collect()
        
        # 6. 生成向量索引文件
        print(f"生成向量索引文件: {OUTPUT_VECTORS_FILE}")
        with open(OUTPUT_VECTORS_FILE, 'w') as f:
            f.write(f"{num_nodes} {vector_dim}\n")
            for chunk_file in chunk_files:
                f.write(f"{chunk_file}\n")
        
        print(f"向量索引文件生成完成: {OUTPUT_VECTORS_FILE}")
        print(f"生成了 {len(chunk_files)} 个向量块文件")
    else:
        print("未找到特征向量，使用随机向量")
        vector_dim = 384
        with open(OUTPUT_VECTORS_FILE, 'w') as f:
            f.write(f"{num_nodes} {vector_dim}\n")
    
    return num_nodes, num_edges

def main():
    try:
        num_nodes, num_edges = convert_graph()
        print(f"\n=== 转换完成 ===")
        print(f"节点数: {num_nodes}")
        print(f"边数: {num_edges}")
        print(f"图文件: {OUTPUT_GRAPH_FILE}")
        print(f"向量索引: {OUTPUT_VECTORS_FILE}")
        print("\n使用命令运行:")
        print(f"./s-mce-system-v4.3.exe mine 0.7")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
