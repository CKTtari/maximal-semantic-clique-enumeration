#!/usr/bin/env python3
import os
import json
from collections import defaultdict

raw_dir = 'e:\\projects\\semanticCliqueMining\\rawDatasets\\WN18RR'
output_dir = 'e:\\projects\\semanticCliqueMining\\dataset\\dataset'

train_file = os.path.join(raw_dir, 'train.tsv')
dev_file = os.path.join(raw_dir, 'dev.tsv')
test_file = os.path.join(raw_dir, 'test.tsv')
entity2text_file = os.path.join(raw_dir, 'entity2text.txt')
definitions_file = os.path.join(raw_dir, 'wordnet-mlj12-definitions.txt')

edge_file = os.path.join(output_dir, 'WN18RR_edges.txt')
semantic_file = os.path.join(output_dir, 'WN18RR_semantics.json')

entities = set()
edges = set()
entity_semantics = defaultdict(list)

def read_triples(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                e1, relation, e2 = parts
                entities.add(e1)
                entities.add(e2)
                if e1 < e2:
                    edges.add((e1, e2))
                else:
                    edges.add((e2, e1))

def read_entity2text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                entity_id, text = parts
                entity_semantics[entity_id].append(text)

def read_definitions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 2)
            if len(parts) >= 3:
                entity_id, _, definition = parts
                entity_semantics[entity_id].append(definition)

print("Reading knowledge graph triples...")
read_triples(train_file)
read_triples(dev_file)
read_triples(test_file)

print("Reading entity text descriptions...")
read_entity2text(entity2text_file)

print("Reading WordNet definitions...")
read_definitions(definitions_file)

sorted_entities = sorted(entities)
entity_id_map = {}
for i, entity in enumerate(sorted_entities):
    entity_id_map[entity] = i

print("Generating edge file...")
with open(edge_file, 'w', encoding='utf-8') as f:
    f.write(f"{len(entities)} {len(edges)}\n")
    sorted_edges = sorted([(entity_id_map[e1], entity_id_map[e2]) for e1, e2 in edges])
    for id1, id2 in sorted_edges:
        f.write(f"{id1} {id2}\n")

print("Generating semantic file...")
semantics_dict = {}
for entity, eid in entity_id_map.items():
    semantics = entity_semantics.get(entity, [])[:5]
    semantics_dict[str(eid)] = semantics

with open(semantic_file, 'w', encoding='utf-8') as f:
    json.dump(semantics_dict, f, indent=2, ensure_ascii=False)

print(f"Conversion completed!")
print(f"Edge file: {edge_file}")
print(f"Semantic file: {semantic_file}")
print(f"Total entities: {len(entities)}")
print(f"Total edges: {len(edges)}")