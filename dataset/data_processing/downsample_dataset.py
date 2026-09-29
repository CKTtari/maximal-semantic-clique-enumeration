import random
import os

def downsample_dataset(input_file, output_file, ratio=0.1):
    """
    将数据集精简到指定比例
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        ratio: 保留比例，默认0.1（即1/10）
    """
    # 读取原始数据集
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # 解析第一行（顶点数和边数）
    first_line = lines[0].strip()
    parts = first_line.split()
    num_vertices = int(parts[0])
    num_edges = int(parts[1])
    
    print(f"原始数据集: {num_vertices} 个顶点, {num_edges} 条边")
    
    # 计算要保留的顶点数
    target_vertices = int(num_vertices * ratio)
    print(f"目标顶点数: {target_vertices} (保留比例: {ratio})")
    
    # 随机选择要保留的顶点
    all_vertices = list(range(num_vertices))
    selected_vertices = set(random.sample(all_vertices, target_vertices))
    
    print(f"选择的顶点: {sorted(selected_vertices)[:10]}... (显示前10个)")
    
    # 过滤边：只保留两个端点都在selected_vertices中的边
    edges = []
    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            u = int(parts[0])
            v = int(parts[1])
            
            # 只保留两个端点都在selected_vertices中的边
            if u in selected_vertices and v in selected_vertices:
                edges.append((u, v))
    
    # 创建顶点映射：将原始顶点编号映射到连续的编号
    vertex_mapping = {}
    new_vertex_id = 0
    for v in sorted(selected_vertices):
        vertex_mapping[v] = new_vertex_id
        new_vertex_id += 1
    
    # 应用顶点映射到边
    mapped_edges = []
    for u, v in edges:
        mapped_u = vertex_mapping[u]
        mapped_v = vertex_mapping[v]
        mapped_edges.append((mapped_u, mapped_v))
    
    # 写入精简后的数据集
    with open(output_file, 'w') as f:
        # 写入新的顶点数和边数
        f.write(f"{target_vertices} {len(mapped_edges)}\n")
        
        # 写入边
        for u, v in mapped_edges:
            f.write(f"{u}\t{v}\n")
    
    print(f"精简数据集: {target_vertices} 个顶点, {len(mapped_edges)} 条边")
    print(f"输出文件: {output_file}")
    print(f"边数比例: {len(mapped_edges) / num_edges:.4f} (原始边数)")

if __name__ == "__main__":
    # 设置随机种子，保证可重复性
    random.seed(42)
    
    # 输入输出文件路径
    input_file = "dataset/dataset/sc-nasasrb.txt"
    output_file = "dataset/dataset/sc-nasasrb_small.txt"
    
    # 转换数据集，保留1/10的数据量
    downsample_dataset(input_file, output_file, ratio=0.25)
    
    print("\n转换完成！")
    print(f"原始文件: {input_file}")
    print(f"精简文件: {output_file}")