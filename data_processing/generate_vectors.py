#!/usr/bin/env python3
import os
import json
import random
import struct
import glob
import argparse
from pathlib import Path

def generate_random_vector(dim):
    return [random.uniform(-1.0, 1.0) for _ in range(dim)]

def generate_vectors_for_graph(graph_file, output_dir, vector_dim=64, binary=True):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(graph_file, 'r') as f:
        first_line = f.readline().strip()
        while first_line.startswith('#'):
            first_line = f.readline().strip()
        
        parts = first_line.split()
        if len(parts) < 1:
            print(f"Error: Invalid graph file format for {graph_file}")
            return False
        
        num_vertices = int(parts[0])
    
    graph_name = Path(graph_file).stem
    
    if binary:
        bin_file = os.path.join(output_dir, f"{graph_name}_vectors.bin")
        
        if os.path.exists(bin_file):
            print(f"  Binary exists, skip: {bin_file}")
            return True
            
        print(f"Generating binary vectors for {graph_name}")
        print(f"  Vector dimension: {vector_dim}")
        print(f"  Number of vertices: {num_vertices}")
        
        with open(bin_file, 'wb') as f:
            f.write(struct.pack('ii', num_vertices, vector_dim))
            
            batch_size = 50000
            for i in range(0, num_vertices, batch_size):
                end = min(i + batch_size, num_vertices)
                
                for j in range(i, end):
                    vector = generate_random_vector(vector_dim)
                    for val in vector:
                        f.write(struct.pack('d', val))
        
        print(f"  Done: {bin_file}")
    else:
        json_file = os.path.join(output_dir, f"{graph_name}_vectors.json")
        
        if os.path.exists(json_file):
            print(f"  JSON exists, skip: {json_file}")
            return True
            
        print(f"Generating JSON vectors for {graph_name}")
        print(f"  Vector dimension: {vector_dim}")
        
        with open(json_file, 'w') as f:
            f.write('{\n')
            
            for j in range(num_vertices):
                vector = generate_random_vector(vector_dim)
                f.write(f'  "{j}": [{", ".join(map(str, vector))}]')
                if j < num_vertices - 1:
                    f.write(",")
                f.write("\n")
            
            f.write('}\n')
        
        print(f"  Done: {json_file}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate vectors for graph files')
    parser.add_argument('--dir', default='dataset/dataset', help='Directory containing graph files')
    parser.add_argument('--dim', type=int, default=64, help='Vector dimension (default: 64)')
    parser.add_argument('--json', action='store_true', help='Generate JSON instead of binary')
    parser.add_argument('--force', action='store_true', help='Force regenerate existing files')

    args = parser.parse_args()

    output_dir = "dataset/vectors-256"
    os.makedirs(output_dir, exist_ok=True)

    target_datasets = [
        "sc-ldoor.txt",
        "sc-nasasrb.txt",
        "sc-pkustk11.txt",
        "soc-buzznet.txt",
        "soc-digg.txt",
        "tech-as-skitter.txt"
    ]

    all_files = glob.glob(os.path.join(args.dir, "*.txt"))
    graph_files = [f for f in all_files if Path(f).name in target_datasets]

    if not graph_files:
        print(f"No target graph files found in {args.dir}")
        exit(1)
    
    print(f"Found {len(graph_files)} graph files")
    print(f"Vector dimension: {args.dim}")
    print(f"Output directory: {output_dir}")
    print()
    
    for graph_file in graph_files:
        graph_name = Path(graph_file).stem
        print(f"\n[{graph_name}]")
        
        if not args.force:
            bin_file = os.path.join(output_dir, f"{graph_name}_vectors.bin")
            json_file = os.path.join(output_dir, f"{graph_name}_vectors.json")
            
            if os.path.exists(bin_file) or os.path.exists(json_file):
                print(f"  Exists, skip")
                continue
        
        generate_vectors_for_graph(graph_file, output_dir, args.dim, not args.json)
    
    print(f"\n=== All done! ===")
