#!/usr/bin/env python3
import os
import json
import gc
import struct
import re

def merge_to_binary_chunks(output_file):
    print("Merging temporary files into binary chunks...")

    temp_files = []
    dir_name = os.path.dirname(output_file)
    base_name = os.path.basename(output_file)

    for file in os.listdir(dir_name):
        if file.startswith(base_name + ".temp."):
            temp_file = os.path.join(dir_name, file)
            temp_files.append(temp_file)

    def get_file_number(file_path):
        filename = os.path.basename(file_path)
        match = re.search(r'temp\.(\d+)\.json$', filename)
        if match:
            return int(match.group(1))
        return 0

    temp_files.sort(key=get_file_number)
    print("Sorted temporary files:")
    for i, temp_file in enumerate(temp_files[:5]):
        print(f"  {i+1}. {os.path.basename(temp_file)}")
    if len(temp_files) > 5:
        print(f"  ... and {len(temp_files) - 5} more files")
    print(f"Found {len(temp_files)} temporary files")

    if not temp_files:
        print("No temporary files found!")
        return

    all_entity_data = {}
    for temp_file in temp_files:
        print(f"  Scanning {temp_file}...")
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_entity_data.update(data)
        gc.collect()

    vector_entity_ids = sorted(map(int, all_entity_data.keys()))

    max_entity_id = max(vector_entity_ids) if vector_entity_ids else 0
    total_vertices = max_entity_id + 1

    vector_dim = 0
    if vector_entity_ids:
        first_vector = all_entity_data[str(vector_entity_ids[0])]
        vector_dim = len(first_vector)

    print(f"\nTotal entities with vectors: {len(vector_entity_ids)}")
    print(f"Maximum entity ID: {max_entity_id}")
    print(f"Total vertices needed: {total_vertices}")
    print(f"Vector dimension: {vector_dim}")

    entity_map = {}
    for entity_id in vector_entity_ids:
        entity_map[entity_id] = all_entity_data[str(entity_id)]

    chunk_size = 100000
    chunk_files = []

    for i in range(0, total_vertices, chunk_size):
        chunk_end = min(i + chunk_size, total_vertices)
        chunk_output = f"{output_file}.chunk.{i//chunk_size}.bin"
        chunk_files.append(chunk_output)

        print(f"\nProcessing chunk {i//chunk_size}: entities {i} to {chunk_end-1}...")

        with open(chunk_output, 'wb') as f:
            f.write(struct.pack('ii', chunk_end - i, vector_dim))

            for entity_id in range(i, chunk_end):
                if entity_id in entity_map:
                    vector = entity_map[entity_id]
                else:
                    vector = [0.0] * vector_dim

                for value in vector:
                    f.write(struct.pack('d', value))

        print(f"  Chunk saved: {chunk_output}")
        gc.collect()

    index_file = output_file.replace(".json", ".index")
    print(f"\nGenerating index file: {index_file}...")

    with open(index_file, 'w') as f:
        f.write(f"{total_vertices} {vector_dim}\n")
        for chunk_file in chunk_files:
            f.write(f"{chunk_file}\n")

    print("Cleaning up temporary files...")
    for temp_file in temp_files:
        os.remove(temp_file)

    print(f"\n=== Merge Complete ===")
    print(f"Generated {len(chunk_files)} binary chunks")
    print(f"Total vertices: {total_vertices}")
    print(f"Entities with vectors: {len(all_entity_data)}")
    print(f"Index file: {index_file}")
    print(f"Temporary files cleaned up")

def main():
    output_vectors = "dataset/vectors/FB15K-237_vectors.json"
    merge_to_binary_chunks(output_vectors)

    print("\n=== All Done! ===")
    print(f"Generated binary chunks for embeddings")

if __name__ == "__main__":
    main()