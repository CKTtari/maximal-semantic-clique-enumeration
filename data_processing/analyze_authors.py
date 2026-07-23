#!/usr/bin/env python3
import os
import csv
from collections import defaultdict

def analyze_author_data():
    input_dir = "final/final"
    
    # 分析 author.csv
    author_ids = set()
    author_count = 0
    
    with open(os.path.join(input_dir, "author.csv"), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row:
                author_id = int(row[0])
                author_ids.add(author_id)
                author_count += 1
    
    print(f"=== Author.csv Analysis ===")
    print(f"Total authors: {author_count}")
    print(f"Unique author IDs: {len(author_ids)}")
    if author_ids:
        print(f"Min author ID: {min(author_ids)}")
        print(f"Max author ID: {max(author_ids)}")
    
    # 分析 pub_author.csv
    pub_author_ids = set()
    pub_count = 0
    
    with open(os.path.join(input_dir, "pub_author.csv"), 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                author_id = int(parts[1])
                pub_author_ids.add(author_id)
                pub_count += 1
    
    print(f"\n=== Pub_author.csv Analysis ===")
    print(f"Total author occurrences: {pub_count}")
    print(f"Unique author IDs in pub_author: {len(pub_author_ids)}")
    if pub_author_ids:
        print(f"Min author ID: {min(pub_author_ids)}")
        print(f"Max author ID: {max(pub_author_ids)}")
    
    # 找出差异
    only_in_author = author_ids - pub_author_ids
    only_in_pub = pub_author_ids - author_ids
    
    print(f"\n=== ID Discrepancies ===")
    print(f"Authors only in author.csv: {len(only_in_author)}")
    print(f"Authors only in pub_author.csv: {len(only_in_pub)}")
    
    # 检查ID连续性
    if author_ids:
        expected_count = max(author_ids) - min(author_ids) + 1
        actual_count = len(author_ids)
        print(f"\n=== ID Continuity Check ===")
        print(f"Expected authors (continuous): {expected_count}")
        print(f"Actual authors: {actual_count}")
        print(f"Missing IDs: {expected_count - actual_count}")

if __name__ == "__main__":
    analyze_author_data()
