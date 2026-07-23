#!/usr/bin/env python3
"""
解析算法日志，提取关键指标。

用法:
  python scripts/parse_log.py logs/exp1_alg2_ds2_r0.log
  python scripts/parse_log.py --batch logs/exp1_*.log --output results/parsed.csv
"""

import argparse
import csv
import os
import re
from pathlib import Path


def parse_single_log(log_path):
    """解析单个日志文件，返回字典。"""
    result = {
        'log_file': os.path.basename(log_path),
        'total_ms': None,
        'clique_count': None,
        'avg_size': None,
        'phase1_ms': None,
        'phase2_ms': None,
        'phase3_ms': None,
        'w_prune': None,
        'mono_prune': None,
        'layer_dedup': None,
    }

    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 总时间和团数量
    m = re.search(r'TOTAL:\s*(\d+)\s*ms,\s*(\d+)\s*maximal cliques', content)
    if m:
        result['total_ms'] = int(m.group(1))
        result['clique_count'] = int(m.group(2))

    # 平均团大小
    m = re.search(r'Average clique size:\s*([\d.]+)', content)
    if m:
        result['avg_size'] = float(m.group(1))

    # alg3 分阶段时间
    m = re.search(r'Phase3 dedup:\s*(\d+)\s*ms', content)
    if m:
        result['phase3_ms'] = int(m.group(1))

    # alg4 ablation 开关状态
    m = re.search(r'Ablation:\s*W_PRUNE=(\d+)\s+MONO_PRUNE=(\d+)\s+LAYER_DEDUP=(\d+)', content)
    if m:
        result['w_prune'] = int(m.group(1))
        result['mono_prune'] = int(m.group(2))
        result['layer_dedup'] = int(m.group(3))

    return result


def parse_batch(pattern, output_path):
    """批量解析日志文件并输出 CSV。"""
    import glob

    log_files = glob.glob(pattern)
    if not log_files:
        print(f'No files matched: {pattern}')
        return

    log_files.sort()
    rows = []
    for lf in log_files:
        rows.append(parse_single_log(lf))

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Parsed {len(rows)} logs -> {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Parse algorithm logs')
    parser.add_argument('log_file', nargs='?', help='Single log file to parse')
    parser.add_argument('--batch', type=str, help='Glob pattern for batch parsing')
    parser.add_argument('--output', type=str, default='results/parsed.csv', help='Output CSV path')
    args = parser.parse_args()

    os.chdir('E:/projects/semanticCliqueMining-writing')

    if args.batch:
        parse_batch(args.batch, args.output)
    elif args.log_file:
        result = parse_single_log(args.log_file)
        for k, v in result.items():
            print(f'{k}: {v}')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
