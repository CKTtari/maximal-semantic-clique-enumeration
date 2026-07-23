import os
import json
from collections import defaultdict

# 数据集路径
fb15k_dir = 'e:\\projects\\semanticCliqueMining\\dataset\\dataset\\FB15K-237'
train_file = os.path.join(fb15k_dir, 'train.txt')
valid_file = os.path.join(fb15k_dir, 'valid.txt')
test_file = os.path.join(fb15k_dir, 'test.txt')
text_file = os.path.join(fb15k_dir, 'text_emnlp.txt')

# 输出路径
output_dir = 'e:\\projects\\semanticCliqueMining\\dataset\\dataset'
edge_file = os.path.join(output_dir, 'FB15K-237_edges.txt')
semantic_file = os.path.join(output_dir, 'FB15K-237_semantics.json')

# 存储实体和边
entities = set()
edges = set()
entity_semantics = defaultdict(list)

# 读取知识图谱三元组
def read_triples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                mid1, relation, mid2 = parts
                entities.add(mid1)
                entities.add(mid2)
                # 只添加无向边，避免重复
                if mid1 < mid2:
                    edges.add((mid1, mid2))
                else:
                    edges.add((mid2, mid1))

# 读取文本信息
def read_textual_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                mid1, path, mid2, count = parts
                # 提取路径中的文本信息作为语义
                # 简单处理：提取路径中的单词
                words = []
                for token in path.split(':'):
                    if token and not token.startswith('<') and not token.startswith('['):
                        words.append(token)
                semantic_text = ' '.join(words)
                if semantic_text:
                    entity_semantics[mid1].append(semantic_text)
                    entity_semantics[mid2].append(semantic_text)

# 读取所有文件
print("Reading knowledge graph triples...")
read_triples(train_file)
read_triples(valid_file)
read_triples(test_file)

print("Reading textual data...")
read_textual_data(text_file)

# 为实体分配ID（按字母顺序排序）
sorted_entities = sorted(entities)
entity_id_map = {}
for i, entity in enumerate(sorted_entities):
    entity_id_map[entity] = i

# 生成边文件
print("Generating edge file...")
with open(edge_file, 'w', encoding='utf-8') as f:
    # 写入节点数和边数（用空格分隔）
    f.write(f"{len(entities)} {len(edges)}\n")
    # 写入边（按排序后的顺序）
    sorted_edges = sorted([(entity_id_map[mid1], entity_id_map[mid2]) for mid1, mid2 in edges])
    for id1, id2 in sorted_edges:
        f.write(f"{id1} {id2}\n")

# 生成语义文件
print("Generating semantic file...")
semantics_dict = {}
for entity, id in entity_id_map.items():
    # 为每个实体取前5个语义描述，避免文件过大
    semantics = entity_semantics.get(entity, [])[:5]
    semantics_dict[str(id)] = semantics

with open(semantic_file, 'w', encoding='utf-8') as f:
    json.dump(semantics_dict, f, indent=2, ensure_ascii=False)

print(f"Conversion completed!")
print(f"Edge file: {edge_file}")
print(f"Semantic file: {semantic_file}")
print(f"Total entities: {len(entities)}")
print(f"Total edges: {len(edges)}")