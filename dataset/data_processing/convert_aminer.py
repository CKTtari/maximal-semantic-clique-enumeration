import os
import json
from collections import defaultdict

# 数据集路径
aminer_dir = 'e:\\projects\\semanticCliqueMining\\dataset\\dataset\\AMiner-534K'
train_file = os.path.join(aminer_dir, 'training.txt')
valid_file = os.path.join(aminer_dir, 'validation.txt')
test_file = os.path.join(aminer_dir, 'testing.txt')
text_file = os.path.join(aminer_dir, 'textual_literals.txt')

# 输出路径
output_dir = 'e:\\projects\\semanticCliqueMining\\dataset\\dataset'
edge_file = os.path.join(output_dir, 'AMiner-534K_edges.txt')
semantic_file = os.path.join(output_dir, 'AMiner-534K_semantics.json')

# 存储作者和论文的关系
paper_authors = defaultdict(list)
author_papers = defaultdict(set)
entities = set()

# 读取三元组数据
def read_triples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                subj, rel, obj = parts
                # 只处理dcterms:creator关系
                if rel == 'dcterms:creator':
                    paper = subj
                    author = obj
                    paper_authors[paper].append(author)
                    author_papers[author].add(paper)
                    entities.add(author)

# 读取文本信息
def read_textual_data(file_path):
    paper_titles = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 2)
            if len(parts) == 3:
                entity, prop, value = parts
                if prop == 'dcterms:title':
                    paper_titles[entity] = value
    return paper_titles

# 构建合作关系图
def build_coauthor_graph():
    edges = set()
    for paper, authors in paper_authors.items():
        if len(authors) >= 2:
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    u, v = authors[i], authors[j]
                    if u > v:
                        u, v = v, u
                    edges.add((u, v))
    return edges

# 读取所有文件
print("Reading triples...")
read_triples(train_file)
read_triples(valid_file)
read_triples(test_file)

print("Reading textual data...")
paper_titles = read_textual_data(text_file)

# 构建合作关系图
print("Building coauthor graph...")
edges = build_coauthor_graph()

# 为作者分配ID（按字母顺序排序）
sorted_authors = sorted(entities)
author_id_map = {}
for i, author in enumerate(sorted_authors):
    author_id_map[author] = i

# 生成边文件
print("Generating edge file...")
with open(edge_file, 'w', encoding='utf-8') as f:
    # 写入节点数和边数（用空格分隔）
    f.write(f"{len(entities)} {len(edges)}\n")
    # 写入边（按排序后的顺序）
    sorted_edges = sorted([(author_id_map[u], author_id_map[v]) for u, v in edges])
    for id1, id2 in sorted_edges:
        f.write(f"{id1} {id2}\n")

# 生成语义文件
print("Generating semantic file...")
author_semantics = {}
for author, id in author_id_map.items():
    # 提取作者的语义信息（使用其发表的论文标题）
    papers = author_papers[author]
    semantics = []
    for paper in papers:
        if paper in paper_titles:
            semantics.append(paper_titles[paper])
        if len(semantics) >= 5:  # 每个作者最多取5个语义描述
            break
    author_semantics[str(id)] = semantics

with open(semantic_file, 'w', encoding='utf-8') as f:
    json.dump(author_semantics, f, indent=2, ensure_ascii=False)

print(f"Conversion completed!")
print(f"Edge file: {edge_file}")
print(f"Semantic file: {semantic_file}")
print(f"Total authors: {len(entities)}")
print(f"Total edges: {len(edges)}")