#!/usr/bin/env python3
"""
批量实验执行脚本。

用法:
  python scripts/run_exp.py --exp exp1 --datasets 1-8 --algorithms 2-4 --tau 70th
  python scripts/run_exp.py --exp exp5 --datasets 1,5,7,8 --tau 50th-95th
  python scripts/run_exp.py --exp exp7 --dataset 3 --threads 1,2,4,8,16,32
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

# 加载 tau 值
TAU_FILE = 'scripts/tau_values.json'
TAU_VALUES = {}
if os.path.exists(TAU_FILE):
    with open(TAU_FILE, 'r') as f:
        TAU_VALUES = json.load(f)

BUILD_DIR = '.'
LOG_DIR = 'logs'
RESULTS_DIR = 'results'
TIMEOUT_SEC = 120  # 本地超时 120s，论文写 3600s


def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_tau(dataset_id, percentile):
    """获取指定数据集的指定 percentile tau 值。"""
    ds_id = str(dataset_id)
    pkey = f'tau_{percentile}'
    if ds_id in TAU_VALUES and pkey in TAU_VALUES[ds_id]:
        return TAU_VALUES[ds_id][pkey]
    raise ValueError(f'Tau not found for dataset {dataset_id}, percentile {percentile}')


def run_algorithm(alg_id, dataset_id, tau=None, threads=None, log_suffix=''):
    """
    运行单个算法，返回日志路径。
    alg_id: 1-4
    dataset_id: 1-10
    tau: float or None (alg1 不需要)
    threads: int or None
    """
    alg_name = f'alg{alg_id}-raw'
    exe = os.path.abspath(f'{alg_name}.exe')
    if not os.path.exists(exe):
        raise FileNotFoundError(f'Executable not found: {exe}')

    log_name = f'exp{log_suffix}_alg{alg_id}_ds{dataset_id}'
    if threads:
        log_name += f'_t{threads}'
    log_name += '.log'
    log_path = os.path.join(LOG_DIR, log_name)

    # 构建命令
    cmd = [exe, 'dataset', str(dataset_id), 'log', log_path]

    # 准备输入
    if alg_id == 1:
        stdin_input = 'mine\nquit\n'
    else:
        if tau is None:
            raise ValueError(f'tau required for alg{alg_id}')
        stdin_input = f'mine {tau}\nquit\n'

    env = os.environ.copy()
    # Ensure libgomp-1.dll is found on Windows
    libgomp_path = r'D:\Scoop\apps\mingw\current\bin'
    if os.path.exists(libgomp_path):
        env['PATH'] = libgomp_path + os.pathsep + env.get('PATH', '')
    if threads:
        env['OMP_NUM_THREADS'] = str(threads)

    print(f'  Running: alg{alg_id} ds{dataset_id} tau={tau} threads={threads} ...')

    start = time.time()
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        stdout, stderr = proc.communicate(input=stdin_input, timeout=TIMEOUT_SEC)
        elapsed = time.time() - start
        print(f'    Done in {elapsed:.1f}s')
        return {'log': log_path, 'elapsed': elapsed, 'timeout': False, 'error': None}
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f'    TIMEOUT after {TIMEOUT_SEC}s')
        return {'log': log_path, 'elapsed': TIMEOUT_SEC, 'timeout': True, 'error': 'TLE'}
    except Exception as e:
        print(f'    ERROR: {e}')
        return {'log': log_path, 'elapsed': None, 'timeout': False, 'error': str(e)}


def parse_log(log_path):
    """从日志中提取总时间和团数量。"""
    total_ms = None
    count = None
    if not os.path.exists(log_path):
        return {'total_ms': None, 'count': None}
    with open(log_path, 'r') as f:
        for line in f:
            if 'TOTAL:' in line and 'maximal cliques' in line:
                # === TOTAL: 1234 ms,  567 maximal cliques ===
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == 'TOTAL:':
                        total_ms = int(parts[i + 1])
                    if p == 'maximal' and i > 0:
                        count = int(parts[i - 1])
    return {'total_ms': total_ms, 'count': count}


def run_exp1(datasets, algorithms, tau_percentile, runs=3):
    """Exp-1: Overall Efficiency"""
    print('\n=== Exp-1: Overall Efficiency ===')
    results = {}
    for ds in datasets:
        tau = get_tau(ds, tau_percentile)
        for alg in algorithms:
            key = f'alg{alg}_ds{ds}'
            times = []
            counts = []
            for r in range(runs):
                res = run_algorithm(alg, ds, tau=tau, log_suffix=f'1_r{r}')
                if not res['timeout'] and not res['error']:
                    parsed = parse_log(res['log'])
                    if parsed['total_ms']:
                        times.append(parsed['total_ms'] / 1000.0)
                        counts.append(parsed['count'])
            if times:
                results[key] = {
                    'time_avg': sum(times) / len(times),
                    'time_min': min(times),
                    'time_max': max(times),
                    'count': counts[0] if counts else None,
                }
            else:
                results[key] = {'time_avg': 'TLE', 'count': None}
    return results


def run_exp1_multi_tau(datasets, algorithms, tau_percentiles, runs=3):
    """Exp-1: Overall Efficiency with multiple tau values"""
    print('\n=== Exp-1: Overall Efficiency ===')
    results = {}
    for p in tau_percentiles:
        for ds in datasets:
            tau = get_tau(ds, p)
            for alg in algorithms:
                key = f'{p}th_alg{alg}_ds{ds}'
                times = []
                counts = []
                for r in range(runs):
                    res = run_algorithm(alg, ds, tau=tau, log_suffix=f'1_{p}th_r{r}')
                    if not res['timeout'] and not res['error']:
                        parsed = parse_log(res['log'])
                        if parsed['total_ms']:
                            times.append(parsed['total_ms'] / 1000.0)
                            counts.append(parsed['count'])
                if times:
                    results[key] = {
                        'time_avg': sum(times) / len(times),
                        'time_min': min(times),
                        'time_max': max(times),
                        'count': counts[0] if counts else None,
                    }
                else:
                    results[key] = {'time_avg': 'TLE', 'count': None}
    return results


def run_exp3(datasets, algorithms, percentiles, runs=1):
    """Exp-3: Sensitivity to tau"""
    print('\n=== Exp-3: Sensitivity to tau ===')
    results = {}
    for ds in datasets:
        for p in percentiles:
            tau = get_tau(ds, p)
            for alg in algorithms:
                key = f'p{p}_alg{alg}_ds{ds}'
                res = run_algorithm(alg, ds, tau=tau, log_suffix=f'3_p{p}')
                if not res['timeout'] and not res['error']:
                    parsed = parse_log(res['log'])
                    results[key] = {
                        'time': parsed['total_ms'] / 1000.0 if parsed['total_ms'] else None,
                        'count': parsed['count'],
                    }
                else:
                    results[key] = {'time': 'TLE', 'count': None}
    return results


def run_exp5(dataset, algorithms, thread_counts, tau_percentile=80, runs=1):
    """Exp-5: Parallel Scalability"""
    print('\n=== Exp-5: Parallel Scalability ===')
    results = {}
    tau = get_tau(dataset, tau_percentile)
    for alg in algorithms:
        for t in thread_counts:
            key = f'alg{alg}_t{t}'
            res = run_algorithm(alg, dataset, tau=tau, threads=t, log_suffix='5')
            if not res['timeout'] and not res['error']:
                parsed = parse_log(res['log'])
                results[key] = {
                    'time': parsed['total_ms'] / 1000.0 if parsed['total_ms'] else None,
                    'count': parsed['count'],
                }
            else:
                results[key] = {'time': 'TLE', 'count': None}
    return results


def save_results(exp_name, results):
    path = os.path.join(RESULTS_DIR, f'{exp_name}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved results to {path}')


def parse_range(s):
    """解析范围字符串如 '1-8' 或 '1,3,5'"""
    items = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            items.extend(range(int(a), int(b) + 1))
        else:
            items.append(int(part))
    return items


def main():
    parser = argparse.ArgumentParser(description='Run experiments')
    parser.add_argument('--exp', type=str, required=True,
                        help='Experiment name: exp1, exp1b, exp5, exp7, ...')
    parser.add_argument('--datasets', type=str, default='1-8',
                        help='Dataset IDs, e.g. 1-8 or 1,3,5')
    parser.add_argument('--algorithms', type=str, default='2-4',
                        help='Algorithm IDs, e.g. 2-4 or 2,3,4')
    parser.add_argument('--tau', type=str, default='70th',
                        help='Tau percentile, e.g. 70th or 50th-95th')
    parser.add_argument('--threads', type=str, default='1,2,4,8,16,32',
                        help='Thread counts for parallel exp')
    parser.add_argument('--runs', type=int, default=3,
                        help='Number of runs per data point')
    args = parser.parse_args()

    os.chdir('E:/projects/semanticCliqueMining-writing')
    ensure_dirs()

    datasets = parse_range(args.datasets)
    algorithms = parse_range(args.algorithms)

    if args.exp == 'exp1':
        percentiles = parse_range(args.tau.replace('th', ''))
        results = run_exp1_multi_tau(datasets, algorithms, percentiles, runs=args.runs)
        save_results('exp1', results)
    elif args.exp == 'exp3':
        percentiles = parse_range(args.tau.replace('th', ''))
        results = run_exp3(datasets, algorithms, percentiles, runs=1)
        save_results('exp3', results)
    elif args.exp == 'exp5':
        if len(datasets) != 1:
            raise ValueError('exp5 requires exactly one dataset')
        thread_counts = parse_range(args.threads)
        p = int(args.tau.replace('th', ''))
        results = run_exp5(datasets[0], algorithms, thread_counts, p)
        save_results('exp5', results)
    else:
        print(f'Unknown experiment: {args.exp}')


if __name__ == '__main__':
    main()
