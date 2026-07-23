#!/usr/bin/env python3
"""
Exp-6/Exp-7 消融实验结果解析与表格生成

功能：
1. 解析 run_ablation_exps.py 生成的 CSV 结果
2. 生成论文格式的 LaTeX 表格（Table 5, 6, 7）
3. 提供统计分析（加速比、贡献度等）

使用方法：
    python scripts/parse_ablation_results.py --exp exp6a    # 生成 Table 5
    python scripts/parse_ablation_results.py --exp exp6b    # 生成 Table 6
    python scripts/parse_ablation_results.py --exp exp7     # 生成 Table 7
    python scripts/parse_ablation_results.py --exp all      # 生成所有表格
"""

import argparse
import csv
import os
from pathlib import Path


RESULTS_DIR = 'results'
TABLES_DIR = 'tables'


def load_results(filename):
    """加载 CSV 结果文件"""
    filepath = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(filepath):
        print(f"⚠️  结果文件不存在: {filepath}")
        return []

    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 转换数值字段
            try:
                row['time_ms'] = int(row['time_ms'])
                row['cliques'] = int(row['cliques'])
                row['dataset'] = int(row['dataset'])
                row['tau'] = float(row['tau'])
            except (ValueError, KeyError):
                pass
            results.append(row)

    return results


def format_time(ms):
    """格式化时间显示"""
    if ms < 0:
        return "N/A"
    elif ms < 1000:
        return f"{ms}ms"
    elif ms < 60000:
        return f"{ms/1000:.2f}s"
    else:
        return f"{ms/60000:.1f}m"


def generate_table5_exp6a(results):
    """
    生成 Table 5: MonoSemMCE Ablation Study

    表格结构：
    | Variant | DS1 | DS5 | DS7 | DS8 |
    |---------|-----|-----|-----|-----|
    | No pruning | T/O | T/O | ... |
    | W-pruning only | ... |
    | Monotone break only | ... |
    | Both, no dedup | ... |
    | Full system | ... |
    """
    print("\n" + "="*80)
    print("Table 5: MonoSemMCE Ablation Study (Exp-6a)")
    print("="*80)

    # 按变体和数据集组织数据
    variants_order = [
        'No pruning',
        'W-pruning only',
        'Monotone break only',
        'W+Mono, no dedup',
        'Full system'
    ]

    datasets = [1, 5, 7, 8]

    # 构建表格数据
    table_data = {}
    for r in results:
        variant = r.get('variant', '')
        ds = r.get('dataset')
        if variant not in table_data:
            table_data[variant] = {}
        table_data[variant][ds] = r

    # 生成 LaTeX 表格
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Ablation study on MonoSemMCE (\\texttt{ALG4}).}")
    latex.append("\\label{tab:exp6a}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l|cccc}")
    latex.append("\\toprule")
    latex.append("Variant & \\textsc{sc-ldoor} & \\textsc{soc-digg} & \\textsc{FB15K-237} & \\textsc{WN18RR} \\\\")
    latex.append("\\midrule")

    for variant in variants_order:
        if variant not in table_data:
            continue

        row = f"{variant}"
        for ds in datasets:
            if ds in table_data[variant]:
                r = table_data[variant][ds]
                time_s = r['time_ms'] / 1000.0 if r['time_ms'] > 0 else -1
                cliques = r['cliques']
                status = r['status']

                if status == 'TLE':
                    cell = "$>$1h"
                elif status == 'OOM':
                    cell = "OOM"
                elif status == 'FAILED':
                    cell = "FAIL"
                else:
                    if time_s < 1:
                        cell = f"{time_s*1000:.0f}ms"
                    elif time_s < 60:
                        cell = f"{time_s:.2f}s"
                    else:
                        cell = f"{time_s/60:.1f}m"

                    # 添加团数量（如果合理）
                    if 0 < cliques < 10000:
                        cell += f" ({cliques})"
            else:
                cell = "--"

            row += f" & {cell}"

        row += " \\\\"
        latex.append(row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")
    latex.append("\\end{table}")

    # 保存 LaTeX 文件
    os.makedirs(TABLES_DIR, exist_ok=True)
    table_file = os.path.join(TABLES_DIR, 'table5_exp6a_ablation.tex')
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex))

    print(f"\n✅ LaTeX 表格已保存到: {table_file}")

    # 打印 ASCII 表格预览
    print("\n📊 表格预览:")
    print("-" * 90)
    header = f"{'Variant':<25} | {'DS1':>12} | {'DS5':>12} | {'DS7':>12} | {'DS8':>12}"
    print(header)
    print("-" * 90)

    for variant in variants_order:
        if variant not in table_data:
            continue

        row = f"{variant:<25} |"
        for ds in datasets:
            if ds in table_data[variant]:
                r = table_data[variant][ds]
                time_s = r['time_ms'] / 1000.0 if r['time_ms'] > 0 else -1
                if r['status'] == 'OK':
                    if time_s < 60:
                        cell = f"{time_s:>10.2f}s"
                    else:
                        cell = f"{time_s/60:>9.1f}m"
                elif r['status'] == 'TLE':
                    cell = f"{'>1h':>12}"
                else:
                    cell = f"{r['status']:>12}"
            else:
                cell = f"{'--':>12}"

            row += f" {cell} |"

        print(row)

    print("-" * 90)

    # 计算加速比分析
    print("\n📈 加速比分析（相对于 No pruning）:")
    if 'No pruning' in table_data and 'Full system' in table_data:
        for ds in datasets:
            if ds in table_data['No pruning'] and ds in table_data['Full system']:
                base = table_data['No pruning'][ds]['time_ms']
                full = table_data['Full system'][ds]['time_ms']
                if base > 0 and full > 0:
                    speedup = base / full
                    print(f"  DS{ds}: {speedup:.2f}x")


def generate_table6_exp6b(results):
    """
    生成 Table 6: StrSub Ablation Study

    表格结构（按阈值分组）：
    对每个阈值 τ_10, τ_30, τ_60, τ_90：
    | Variant | DS4 | DS8 |
    """
    print("\n" + "="*80)
    print("Table 6: StrSub Ablation Study (Exp-6b)")
    print("="*80)

    variants_order = [
        ('No pruning', 'ablation-no'),
        ('Top-Edge only', 'ablation-topedge'),
        ('LocalMax only', 'ablation-localmax'),
        ('Full StrSub', 'ablation-full')
    ]

    datasets = [4, 8]
    percentiles = [10, 30, 60, 90]

    # 按百分位分组
    by_percentile = {}
    for r in results:
        pct = r.get('tau_percentile', 80)
        if pct not in by_percentile:
            by_percentile[pct] = []
        by_percentile[pct].append(r)

    # 为每个百分位生成子表
    all_latex = []
    for pct in sorted(percentiles.keys() if isinstance(percentiles, dict) else percentiles):
        if pct not in by_percentile:
            continue

        pct_results = by_percentile[pct]

        print(f"\n--- τ_{pct} ---")

        # 构建当前百分位的表格数据
        table_data = {}
        for r in pct_results:
            variant = r.get('variant', '')
            ds = r.get('dataset')
            if variant not in table_data:
                table_data[variant] = {}
            table_data[variant][ds] = r

        # 生成 LaTeX
        latex = []
        latex.append("\\begin{table}[t]")
        latex.append("\\centering")
        if pct == 90:
            latex.append("\\caption{Ablation study on \\textsc{StrSub} (\\texttt{ALG3}) at $\\tau_{90}$.}")
            latex.append("\\label{tab:exp6b}")
        else:
            latex.append(f"\\caption{{Ablation on \\textsc{{StrSub}} at $\\tau_{{{pct}}}$.}}")

        latex.append("\\begin{tabular}{l|cc}")
        latex.append("\\toprule")
        latex.append("Variant & \\textsc{soc-buzznet} & \\textsc{WN18RR} \\\\")
        latex.append("\\midrule")

        for variant_name, _ in variants_order:
            if variant_name not in table_data:
                continue

            row = f"{variant_name}"
            for ds in datasets:
                if ds in table_data[variant_name]:
                    r = table_data[variant_name][ds]
                    time_s = r['time_ms'] / 1000.0 if r['time_ms'] > 0 else -1
                    status = r['status']

                    if status == 'TLE':
                        cell = "$>$1h"
                    elif status == 'OK':
                        if time_s < 1:
                            cell = f"{time_s*1000:.0f}ms"
                        elif time_s < 60:
                            cell = f"{time_s:.2f}s"
                        else:
                            cell = f"{time_s/60:.1f}m"
                    else:
                        cell = status
                else:
                    cell = "--"

                row += f" & {cell}"

            row += " \\\\"
            latex.append(row)

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        all_latex.append('\n'.join(latex))

        # 打印预览
        print(f"{'Variant':<20} | {'DS4':>12} | {'DS8':>12}")
        print("-" * 50)
        for variant_name, _ in variants_order:
            if variant_name not in table_data:
                continue
            row = f"{variant_name:<20} |"
            for ds in datasets:
                if ds in table_data[variant_name]:
                    r = table_data[variant_name][ds]
                    if r['status'] == 'OK':
                        ts = r['time_ms'] / 1000.0
                        cell = f"{ts:>10.2f}s" if ts < 60 else f"{ts/60:>9.1f}m"
                    else:
                        cell = f"{r['status']:>12}"
                else:
                    cell = f"{'--':>12}"
                row += f" {cell} |"
            print(row)
        print("-" * 50)

    # 保存所有表格
    os.makedirs(TABLES_DIR, exist_ok=True)
    table_file = os.path.join(TABLES_DIR, 'table6_exp6b_ablation.tex')
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(all_latex))
    print(f"\n✅ LaTeX 表格已保存到: {table_file}")


def generate_table7_exp7(results):
    """
    生成 Table 7: Engineering Optimizations

    表格结构：
    | Configuration | DS1 | DS2 | DS8 | DS9 |
    |---------------|-----|-----|-----|-----|
    | Dense bitmap | ... |
    | Naïve intersection | ... |
    | No incremental acc | ... |
    | No reorder caching | ... |
    | Full system | ... |
    """
    print("\n" + "="*80)
    print("Table 7: Engineering Optimizations (Exp-7)")
    print("="*80)

    configs_order = [
        'Full system (baseline)',
        'Dense bitmap',
        'Naïve intersection',
        'No incremental acc',
        'No reorder caching'
    ]

    datasets = [1, 2, 8, 9]

    # 构建表格数据
    table_data = {}
    for r in results:
        config = r.get('variant', '')
        ds = r.get('dataset')
        if config not in table_data:
            table_data[config] = {}
        table_data[config][ds] = r

    # 生成 LaTeX
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Impact of engineering optimizations on \\texttt{ALG4}.}")
    latex.append("\\label{tab:exp7}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l|cccc}")
    latex.append("\\toprule")
    latex.append("Configuration & \\textsc{sc-ldoor} & \\textsc{sc-nasasrb} & \\textsc{WN18RR} & \\textsc{DBLP} \\\\")
    latex.append("\\midrule")

    for config in configs_order:
        if config not in table_data:
            # 尝试部分匹配
            found = False
            for key in table_data:
                if config.split()[0].lower() in key.lower():
                    config = key
                    found = True
                    break
            if not found:
                continue

        row = f"{config}"
        for ds in datasets:
            if ds in table_data[config]:
                r = table_data[config][ds]
                time_s = r['time_ms'] / 1000.0 if r['time_ms'] > 0 else -1
                status = r['status']

                if status == 'TLE':
                    cell = "$>$1h"
                elif status == 'OK':
                    if time_s < 1:
                        cell = f"{time_s*1000:.0f}ms"
                    elif time_s < 60:
                        cell = f"{time_s:.2f}s"
                    else:
                        cell = f"{time_s/60:.1f}m"
                else:
                    cell = status
            else:
                cell = "--"

            row += f" & {cell}"

        row += " \\\\"
        latex.append(row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")
    latex.append("\\end{table}")

    # 保存
    os.makedirs(TABLES_DIR, exist_ok=True)
    table_file = os.path.join(TABLES_DIR, 'table7_exp7_engineering.tex')
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex))

    print(f"\n✅ LaTeX 表格已保存到: {table_file}")

    # 打印预览
    print("\n📊 表格预览:")
    print("-" * 100)
    header = f"{'Configuration':<25} | {'DS1':>12} | {'DS2':>12} | {'DS8':>12} | {'DS9':>12}"
    print(header)
    print("-" * 100)

    for config in configs_order:
        if config not in table_data:
            continue

        row = f"{config:<25} |"
        for ds in datasets:
            if ds in table_data[config]:
                r = table_data[config][ds]
                if r['status'] == 'OK':
                    ts = r['time_ms'] / 1000.0
                    cell = f"{ts:>10.2f}s" if ts < 60 else f"{ts/60:>9.1f}m"
                else:
                    cell = f"{r['status']:>12}"
            else:
                cell = f"{'--':>12}"
            row += f" {cell} |"

        print(row)

    print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description='消融实验结果解析与表格生成')
    parser.add_argument('--exp', type=str, default='all',
                       choices=['exp6a', 'exp6b', 'exp7', 'all'],
                       help='要生成的表格 (默认: all)')
    args = parser.parse_args()

    print("="*80)
    print("消融实验结果解析系统")
    print("="*80)

    if args.exp in ['exp6a', 'all']:
        results = load_results('exp6a_monosemmce_ablation.csv')
        if results:
            generate_table5_exp6a(results)
        else:
            print("\n⚠️  未找到 Exp-6a 结果，请先运行: python scripts/run_ablation_exps.py --exp exp6a")

    if args.exp in ['exp6b', 'all']:
        results = load_results('exp6b_strsub_ablation.csv')
        if results:
            generate_table6_exp6b(results)
        else:
            print("\n⚠️  未找到 Exp-6b 结果，请先运行: python scripts/run_ablation_exps.py --exp exp6b")

    if args.exp in ['exp7', 'all']:
        results = load_results('exp7_engineering.csv')
        if results:
            generate_table7_exp7(results)
        else:
            print("\n⚠️  未找到 Exp-7 结果，请先运行: python scripts/run_ablation_exps.py --exp exp7")

    print("\n" + "="*80)
    print("完成！所有表格已保存到 tables/ 目录")
    print("="*80)


if __name__ == '__main__':
    main()
