#!/usr/bin/env python3
import os
import json
import gc
import struct

def merge_to_binary_chunks(output_file):
    print("Merging temporary files into binary chunks...")
    
    # Find all temporary files
    temp_files = []
    for file in os.listdir(os.path.dirname(output_file)):
        if file.startswith(os.path.basename(output_file) + ".temp."):
            temp_file = os.path.join(os.path.dirname(output_file), file)
            temp_files.append(temp_file)
    
    # Sort by numeric part of filename
    def get_file_number(file_path):
        filename = os.path.basename(file_path)
        # Extract number after "temp."
        import re
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
    
    # First pass: collect all author IDs and find max ID
    all_author_data = {}
    for temp_file in temp_files:
        print(f"  Scanning {temp_file}...")
        with open(temp_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_author_data.update(data)
        gc.collect()
    
    # Get author IDs from vectors
    vector_author_ids = sorted(map(int, all_author_data.keys()))
    
    # Get maximum author ID from published authors only
    max_author_id = max(vector_author_ids) if vector_author_ids else 0
    total_vertices = max_author_id + 1
    
    # Get vector dimension
    vector_dim = 0
    if vector_author_ids:
        first_vector = all_author_data[str(vector_author_ids[0])]
        vector_dim = len(first_vector)
    
    print(f"\nTotal authors with vectors: {len(vector_author_ids)}")
    print(f"Maximum author ID: {max_author_id}")
    print(f"Total vertices needed: {total_vertices}")
    print(f"Vector dimension: {vector_dim}")
    
    # Create author ID to index mapping
    author_map = {}
    for author_id in vector_author_ids:
        author_map[author_id] = all_author_data[str(author_id)]
    
    # Process in chunks
    chunk_size = 100000  # Process 100k authors per chunk
    chunk_files = []
    
    for i in range(0, total_vertices, chunk_size):
        chunk_end = min(i + chunk_size, total_vertices)
        chunk_output = f"{output_file}.chunk.{i//chunk_size}.bin"
        chunk_files.append(chunk_output)
        
        print(f"\nProcessing chunk {i//chunk_size}: authors {i} to {chunk_end-1}...")
        
        # Write binary chunk
        with open(chunk_output, 'wb') as f:
            # Write chunk header: number of vertices and dimension
            f.write(struct.pack('ii', chunk_end - i, vector_dim))
            
            # Write vectors for each author in this range
            for author_id in range(i, chunk_end):
                if author_id in author_map:
                    # Use existing vector
                    vector = author_map[author_id]
                else:
                    # Generate zero vector for authors without embeddings
                    vector = [0.0] * vector_dim
                
                # Write vector
                for value in vector:
                    f.write(struct.pack('d', value))
        
        print(f"  Chunk saved: {chunk_output}")
        gc.collect()
    
    # Generate index file
    index_file = output_file.replace(".json", ".index")
    print(f"\nGenerating index file: {index_file}...")
    
    with open(index_file, 'w') as f:
        f.write(f"{total_vertices} {vector_dim}\n")  # Total vertices and dimension
        for chunk_file in chunk_files:
            f.write(f"{chunk_file}\n")
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    for temp_file in temp_files:
        os.remove(temp_file)
    
    print(f"\n=== Merge Complete ===")
    print(f"Generated {len(chunk_files)} binary chunks")
    print(f"Total vertices: {total_vertices}")
    print(f"Authors with vectors: {len(all_author_ids)}")
    print(f"Index file: {index_file}")
    print(f"Temporary files cleaned up")

def save_graph():
    print("Saving coauthor graph...")
    
    input_dir = "final/final"
    output_graph = "dataset/dataset/dblp_coauthor.txt"
    
    # Read existing binary chunk files to get the author ID range
    vector_index_file = "dataset/vectors/dblp_coauthor_vectors.index"
    
    # Get total vertices from index file
    with open(vector_index_file, 'r') as f:
        first_line = f.readline().strip()
        total_vertices, vector_dim = map(int, first_line.split())
    
    print(f"  Total vertices from index: {total_vertices}")
    
    # The vectors are stored in author ID order (0 to total_vertices-1)
    # Author IDs in pub_author.csv start from 1, so we need to map them
    # Valid author IDs are from 1 to total_vertices (since vector index = author_id - 1)
    valid_author_set = set(range(1, total_vertices))
    
    # Load pub_authors and filter to only include authors with vectors
    pub_authors = {}
    with open(os.path.join(input_dir, "pub_author.csv"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pub_id = int(parts[0])
                author_id = int(parts[1])
                # Only include authors that have vectors
                if author_id in valid_author_set:
                    if pub_id not in pub_authors:
                        pub_authors[pub_id] = []
                    pub_authors[pub_id].append(author_id)
    
    # Build edges
    author_papers = {}
    edges = set()
    
    for pub_id, authors in pub_authors.items():
        for a in authors:
            if a not in author_papers:
                author_papers[a] = set()
            author_papers[a].add(pub_id)
        
        if len(authors) >= 2:
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    u, v = authors[i], authors[j]
                    if u > v:
                        u, v = v, u
                    edges.add((u, v))
    
    # Save graph
    os.makedirs(os.path.dirname(output_graph), exist_ok=True)
    
    # Use total_vertices from vector index file directly
    total_edges = len(edges)
    
    print(f"  Preparing to write {total_vertices} vertices and {total_edges} edges...")
    
    batch_size = 1000000
    edge_count = 0
    
    with open(output_graph, 'w') as f:
        f.write(f"{total_vertices} {total_edges}\n")
        
        batch = []
        for u, v in edges:
            batch.append(f"{u} {v}\n")
            edge_count += 1
            
            if len(batch) >= batch_size:
                f.writelines(batch)
                batch.clear()
                print(f"  Written {edge_count}/{total_edges} edges...")
        
        if batch:
            f.writelines(batch)
            print(f"  Written {edge_count}/{total_edges} edges...")
    
    print(f"  Graph saved: {total_vertices} vertices, {total_edges} edges")

def main():
    output_vectors = "dataset/vectors/dblp_coauthor_vectors.json"
    
    # Process vectors first
    merge_to_binary_chunks(output_vectors)
    
    # Then save graph
    save_graph()
    
    print("\n=== All Done! ===")
    print(f"Generated binary chunks for embeddings")
    print(f"Graph saved: dataset/dataset/dblp_coauthor.txt")

if __name__ == "__main__":
    main()
