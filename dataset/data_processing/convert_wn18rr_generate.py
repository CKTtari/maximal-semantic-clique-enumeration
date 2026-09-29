#!/usr/bin/env python3
import os
import json
import gc

def generate_embeddings(semantic_file, output_file):
    print("Loading all-MiniLM-L6-v2 model...")

    from transformers import AutoTokenizer, AutoModel
    import torch

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on: {device}")
    model.eval()

    print(f"Loading semantic data from {semantic_file}...")
    with open(semantic_file, 'r', encoding='utf-8') as f:
        entity_semantics = json.load(f)

    entity_ids = sorted(entity_semantics.keys(), key=lambda x: int(x))
    print(f"Total entities: {len(entity_ids)}")

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    batch_size = 64
    save_interval = 32000

    total = len(entity_ids)
    current_embeddings = {}
    temp_files = []

    for i in range(0, total, batch_size):
        batch_ids = entity_ids[i:i+batch_size]

        if i % 1000 == 0:
            print(f"  Progress: {i}/{total}")

        batch_texts = []
        for entity_id in batch_ids:
            semantics = entity_semantics[entity_id]
            text = " ".join(semantics[:10]) if semantics else "unknown entity"
            batch_texts.append(text)

        with torch.no_grad():
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            batch_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)

        for j, entity_id in enumerate(batch_ids):
            current_embeddings[entity_id] = batch_embeddings[j].cpu().numpy().tolist()

        if (i + batch_size) % save_interval == 0 or (i + batch_size) >= total:
            temp_file = f"{output_file}.temp.{len(temp_files)}.json"
            print(f"  Saving batch to {temp_file}...")
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(current_embeddings, f)
            temp_files.append(temp_file)
            current_embeddings.clear()
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\n=== Generate Complete ===")
    print(f"Generated {len(temp_files)} temporary files")
    print(f"Next step: Run convert_wn18rr_merge.py to merge these files")

def main():
    semantic_file = "dataset/dataset/WN18RR_semantics.json"
    output_vectors = "dataset/vectors/WN18RR_vectors.json"

    generate_embeddings(semantic_file, output_vectors)

    print("\n=== Done! ===")
    print(f"Temporary files created for embeddings")

if __name__ == "__main__":
    main()