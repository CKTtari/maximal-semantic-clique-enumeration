#!/usr/bin/env python3
import os
import json
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_data(input_dir):
    print("Loading data...")
    
    pub_authors = defaultdict(list)
    with open(os.path.join(input_dir, "pub_author.csv"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pub_id = int(parts[0])
                author_id = int(parts[1])
                pub_authors[pub_id].append(author_id)
    
    pub_info = {}
    with open(os.path.join(input_dir, "pub.csv"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t', 2)
            if len(parts) >= 3:
                pub_id = int(parts[0])
                year = parts[1]
                title = parts[2]
                pub_info[pub_id] = {"year": year, "title": title}
    
    print(f"  Pubs: {len(pub_info)}, Pub-Authors: {len(pub_authors)}")
    return pub_authors, pub_info

def build_coauthor_graph(pub_authors):
    print("Building coauthor graph...")
    
    author_papers = defaultdict(set)
    edges = set()
    
    for pub_id, authors in pub_authors.items():
        for a in authors:
            author_papers[a].add(pub_id)
        
        if len(authors) >= 2:
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    u, v = authors[i], authors[j]
                    if u > v:
                        u, v = v, u
                    edges.add((u, v))
    
    print(f"  Authors: {len(author_papers)}, Edges: {len(edges)}")
    return author_papers, edges

def generate_embeddings(author_papers, pub_info, output_file):
    print("Loading all-MiniLM-L6-v2 model...")
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on: {device}")
    model.eval()
    
    print("Generating embeddings...")
    
    author_ids = sorted(author_papers.keys())
    batch_size = 128
    save_interval = 64000
    
    total = len(author_ids)
    current_embeddings = {}
    temp_files = []
    
    for i in range(0, total, batch_size):
        batch_ids = author_ids[i:i+batch_size]
        
        if i % 1000 == 0:
            print(f"  Progress: {i}/{total}")
        
        # Generate keywords on the fly
        batch_texts = []
        for author_id in batch_ids:
            papers = author_papers[author_id]
            keywords = []
            for pub_id in papers:
                if pub_id in pub_info:
                    info = pub_info[pub_id]
                    title = info.get("title", "")
                    if title:
                        keywords.append(title)
            text = " ".join(keywords[:20])
            batch_texts.append(text)
        
        with torch.no_grad():
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            # Move inputs to the same device as model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            batch_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
        for j, author_id in enumerate(batch_ids):
            current_embeddings[author_id] = batch_embeddings[j].cpu().numpy().tolist()
        
        # Save and clear memory
        if (i + batch_size) % save_interval == 0 or (i + batch_size) >= total:
            temp_file = f"{output_file}.temp.{len(temp_files)}.json"
            print(f"  Saving batch to {temp_file}...")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(current_embeddings, f)
            temp_files.append(temp_file)
            current_embeddings.clear()
            torch.cuda.empty_cache()
            
            # Use garbage collection instead of clearing pub_info
            # This ensures all paper information is retained for all authors
            import gc
            gc.collect()
    
    # Merge temp files
    print(f"Merging {len(temp_files)} temporary files...")
    final_embeddings = {}
    for temp_file in temp_files:
        with open(temp_file, 'r', encoding='utf-8') as f:
            batch_data = json.load(f)
            final_embeddings.update(batch_data)
        os.remove(temp_file)
    
    print(f"  Done: {len(final_embeddings)} embeddings")
    
    print(f"Saving final embeddings to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_embeddings, f)
    
    return final_embeddings



def save_graph(author_papers, edges, output_file):
    print(f"Saving graph to {output_file}...")
    
    # Create directory if not exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    total_edges = len(edges)
    print(f"  Preparing to write {len(author_papers)} vertices and {total_edges} edges...")
    
    # Write in batches to avoid memory issues
    batch_size = 1000000  # 1 million edges per batch
    edge_count = 0
    
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"{len(author_papers)} {total_edges}\n")
        
        # Write edges in batches
        batch = []
        for u, v in edges:
            batch.append(f"{u} {v}\n")
            edge_count += 1
            
            if len(batch) >= batch_size:
                f.writelines(batch)
                batch.clear()
                print(f"  Written {edge_count}/{total_edges} edges...")
        
        # Write remaining edges
        if batch:
            f.writelines(batch)
            print(f"  Written {edge_count}/{total_edges} edges...")
    
    print(f"  Graph saved: {len(author_papers)} vertices, {total_edges} edges")

def main():
    input_dir = "final/final"
    output_graph = "dataset/dataset/dblp_coauthor.txt"
    output_vectors = "dataset/vectors/dblp_coauthor_vectors.json"
    
    pub_authors, pub_info = load_data(input_dir)
    author_papers, edges = build_coauthor_graph(pub_authors)
    generate_embeddings(author_papers, pub_info, output_vectors)
    save_graph(author_papers, edges, output_graph)
    
    print("\n=== Done! ===")
    print(f"Graph: {output_graph}")
    print(f"Vectors: {output_vectors}")

if __name__ == "__main__":
    main()
