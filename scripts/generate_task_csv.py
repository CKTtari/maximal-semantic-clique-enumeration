#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验任务自动生成与管理工具

功能：
1. 根据实验设计文档生成完整的实验任务 CSV 表
2. 读取已有结果，标记已完成/未完成状态
3. 支持断点续跑：只执行未完成的任务
4. TLE 智能重跑：当超时标准提高时自动重新运行

使用方法：
    python scripts/generate_task_csv.py          # 生成/更新任务表
    python scripts/run_exp_onfuwuqi_auto.py       # 自动执行未完成任务
"""

import csv
import json
import os
from pathlib import Path
from datetime import datetime


# ==================== 配置 ====================
TASK_CSV = 'scripts/experiment_tasks.csv'
RESULTS_DIR = 'results'
TAU_FILE = 'scripts/tau_values.json'

# 当前超时标准（秒）
CURRENT_TLE_STANDARD = 600

# 数据集定义
DATASETS = {
    1: 'sc-ldoor',
    2: 'sc-nasasrb',
    3: 'sc-pkustk11',
    4: 'soc-buzznet',
    5: 'soc-digg',
    6: 'tech-as-skitter',  # 通常跳过
    7: 'FB15K-237',
    8: 'WN18RR',
    9: 'DBLP',
    10: 'Amazon/products'
}

# 算法定义
ALGORITHMS = {
    1: {'name': 'StructBK', 'exe': 'alg1-raw.exe', 'need_tau': False, 'priority_factor': 2.0},
    2: {'name': 'SemBK', 'exe': 'alg2-raw.exe', 'need_tau': True, 'priority_factor': 0.5},
    3: {'name': 'StrSub', 'exe': 'alg3-raw.exe', 'need_tau': True, 'priority_factor': 1.5},
    4: {'name': 'MonoSemMCE', 'exe': 'alg4-raw.exe', 'need_tau': True, 'priority_factor': 1.0}
}


def calc_priority(alg_id, tau_percentile, status='pending'):
    """
    计算任务优先级
    
    规则：
    - pending/needs_rerun: 正常计算 (基础分 + 状态加成)
    - completed: 不参与排序 (已跳过)
    - tle/oom/failed: 极低优先级 (-1000 基础分)
    
    公式: x * (s + 0.5) + status_bonus
    数值越大越先执行
    """
    # 状态加成：pending 最高，失败最低
    STATUS_BONUS = {
        'pending': 100,      # 待执行 → 最先跑
        'needs_rerun': 100,   # 需重跑 → 同样优先
        'completed': -9999,   # 已完成 → 跳过
        'tle': -1000,         # 超时 → 最后处理
        'oom': -1000,         # 内存溢出 → 最后处理  
        'failed': -1000       # 其他失败 → 最后处理
    }
    
    bonus = STATUS_BONUS.get(status, 0)
    
    # 如果是已完成或失败，直接返回低优先级
    if status == 'completed':
        return -9999
    elif status in ['tle', 'oom', 'failed']:
        # 失败任务内部仍按公式排序，但整体很低
        pass
    else:
        # pending/needs_rerun: 正常计算
        pass
    
    # 算法因子 x
    ALG_FACTOR = {1: 2.0, 2: 0.5, 3: 1.5, 4: 1.0}
    x = ALG_FACTOR.get(alg_id, 1.0)
    
    if alg_id == 1:
        s = 1.0
    elif alg_id == 3:
        s_raw = tau_percentile / 100.0
        s = abs(s_raw - 0.5) * 2
    else:
        s = tau_percentile / 100.0
    
    return round(x * (s + 0.5) + bonus, 3)


def get_task_priority(alg_id=None, exp_priority=1):
    """获取基础优先级（用于无tau的实验）"""
    ALG_FACTOR = {1: 2.0, 2: 0.5, 3: 1.5, 4: 1.0}
    x = ALG_FACTOR.get(alg_id, 1.0)
    return round(x * (1 + 0.5), 3) if alg_id == 1 else round(x * 0.75, 3)

# 通用字段模板（所有实验都必须包含这些字段）
TASK_TEMPLATE = {
    'task_id': '',
    'exp_id': '',
    'exp_name': '',
    'dataset_id': '',
    'dataset_name': '',
    'scale_level': '',       # Exp-2 使用
    'algorithm_id': '',
    'algorithm_name': '',
    'code_version': '',
    'tau_percentile': '',
    'tau_value': '',
    'threads': '',
    'variant': '',
    'extra_params': '',
    'status': '',
    'elapsed_s': '',
    'tle_standard': '',
    'runs': 1,
    'priority': 1,
    'created_at': '',
    'updated_at': ''
}


def create_task(**kwargs):
    """创建任务字典，自动填充缺失字段"""
    task = TASK_TEMPLATE.copy()
    task.update(kwargs)
    return task


# Exp-5 线程数配置
THREAD_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]

# Exp-6a Ablation 变体 (MonoSemMCE)
# 命名必须与 build_ablation.sh 生成的 exe 文件名一致：alg4-ablation-{name}.exe
ABLATION_VARIANTS = [
    ('no',   'No pruning (W=0,M=0,L=0)'),
    ('w',    'W-pruning only'),
    ('mono', 'Monotone break only'),
    ('both', 'Both, no layer dedup'),
    ('full', 'Full MonoSemMCE')
]

# Exp-6b Ablation 变体 (StrSub)
# 命名必须与 build_ablation.sh 生成的 exe 文件名一致：alg3-ablation-{name}.exe
ABLBATION_STRSUB = [
    ('no',        'No pruning, no filter'),
    ('topedge',   'Top-Edge pruning only'),
    ('localmax',  'Local max filter only'),
    ('full',      'Full StrSub')
]

# Exp-7 Engineering 优化变体
# 命名必须与 build_ablation.sh 生成的 exe 文件名一致：alg4-eng-{name}.exe
ENGINEERING_VARIANTS = [
    ('dense',     'Dense bitmap'),
    ('naive',     'Naïve intersection'),
    ('noacc',     'No incremental acc'),
    ('noreorder', 'No reorder caching'),
    ('full',      'Full system')
]


def load_tau_values():
    """加载 tau 值"""
    if os.path.exists(TAU_FILE):
        with open(TAU_FILE, 'r') as f:
            return json.load(f)
    return {}


def load_existing_results():
    """加载已有的实验结果"""
    results = {}
    results_dir = Path(RESULTS_DIR)
    
    if not results_dir.exists():
        return results
    
    for json_file in results_dir.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    results.update(data)
        except Exception as e:
            print(f"[WARN] Cannot read {json_file}: {e}")
    
    return results


def get_task_status(task_id, existing_results, current_tle):
    """根据已有结果判断任务状态
    
    返回: (status, elapsed, tle_standard)
    其中 elapsed 优先使用 time_avg（纯计算时间），失败时才用 elapsed
    """
    if task_id not in existing_results:
        return 'pending', None, None
    
    result = existing_results[task_id]
    status = result.get('status', '')
    error_type = result.get('error_type', '')
    
    # 获取耗时：优先 time_avg（纯计算时间）
    elapsed = None
    if status == 'success':
        # 成功时：优先用 C++ 报告的 time_avg
        time_avg = result.get('time_avg')
        if time_avg is not None:
            try:
                elapsed = float(time_avg)
            except (ValueError, TypeError):
                pass
        
        # 如果 time_avg 无效，fallback 到 elapsed
        if elapsed is None:
            raw_elapsed = result.get('elapsed')
            if raw_elapsed is not None:
                try:
                    elapsed = float(raw_elapsed)
                except (ValueError, TypeError):
                    pass
    else:
        # 失败时：只能用 elapsed（time_avg 是错误字符串）
        raw_elapsed = result.get('elapsed')
        if raw_elapsed is not None:
            try:
                elapsed = float(raw_elapsed)
            except (ValueError, TypeError):
                pass
    
    # 成功完成
    if status == 'success':
        return 'completed', elapsed, current_tle
    
    # TLE - 检查是否需要重跑
    if error_type == 'TLE' or 'TLE' in str(result.get('time_avg', '')):
        prev_tle = result.get('tle_standard', current_tle)
        if current_tle > prev_tle:
            return 'needs_rerun', elapsed, current_tle
        else:
            return 'tle', elapsed, prev_tle
    
    # OOM 或其他错误
    if error_type == 'OOM':
        return 'oom', elapsed, current_tle
    
    # 其他失败
    if status == 'failed' or error_type:
        return 'failed', elapsed, current_tle
    
    # 默认待处理
    return 'pending', elapsed, current_tle


def merge_alg1_tasks(tasks):
    """
    合并 alg1 的重复任务
    alg1 (StructBK) 不依赖 tau 参数，同一数据集只需运行一次
    保留 50th 作为代表，合并其他变体的状态和时间
    """
    alg1_groups = {}
    merged_tasks = []
    removed_count = 0
    
    for task in tasks:
        alg_id = str(task.get('algorithm_id', ''))
        if alg_id != '1':
            merged_tasks.append(task)
            continue
        
        # 用 (exp_id, dataset_id) 作为分组键
        key = (task['exp_id'], task['dataset_id'])
        
        if key not in alg1_groups:
            alg1_groups[key] = task
        else:
            # 合并状态：取最优状态
            existing = alg1_groups[key]
            new_status = task['status']
            old_status = existing['status']
            
            STATUS_PRIORITY = {
                'completed': 0,
                'needs_rerun': 1,
                'pending': 2,
                'failed': 3,
                'tle': 4,
                'oom': 5
            }
            
            # 如果新任务状态更优或有时间数据而旧任务没有
            new_prio = STATUS_PRIORITY.get(new_status, 99)
            old_prio = STATUS_PRIORITY.get(old_status, 99)
            
            has_new_time = bool(task.get('elapsed_s'))
            has_old_time = bool(existing.get('elapsed_s'))
            
            should_replace = False
            
            if new_prio < old_prio:
                should_replace = True
            elif new_prio == old_prio and has_new_time and not has_old_time:
                should_replace = True
            
            if should_replace:
                # 保留新任务的更好状态/时间
                existing['status'] = new_status
                existing['tau_percentile'] = task['tau_percentile']  # 更新为当前选中的
                existing['task_id'] = task['task_id']
                if task.get('elapsed_s'):
                    existing['elapsed_s'] = task['elapsed_s']
            
            removed_count += 1
    
    print(f"     [DEDUP] Removed {removed_count} duplicate alg1 tasks")
    return merged_tasks


def generate_exp1_tasks(tau_values, existing_results):
    """生成 Exp-1: Overall Efficiency 任务"""
    tasks = []
    
    # 数据集: 1-5, 7-8 (跳过6)
    datasets = [1, 2, 3, 4, 5, 7, 8]
    algorithms = [1, 2, 3, 4]
    percentiles = [10, 50, 90]  # τ_10, τ_50, τ_90
    
    for ds in datasets:
        for alg in algorithms:
            for p in percentiles:
                task_id = f'{p}th_alg{alg}_ds{ds}'
                
                # 获取 tau 值
                tau = None
                if ALGORITHMS[alg]['need_tau']:
                    ds_str = str(ds)
                    pkey = f'tau_{p}'
                    if ds_str in tau_values and pkey in tau_values[ds_str]:
                        tau = tau_values[ds_str][pkey]
                
                status, elapsed, tle_std = get_task_status(
                    task_id, existing_results, CURRENT_TLE_STANDARD
                )
                
                tasks.append(create_task(
                    task_id=task_id,
                    exp_id='exp1',
                    exp_name='Overall Efficiency',
                    dataset_id=ds,
                    dataset_name=DATASETS[ds],
                    algorithm_id=alg,
                    algorithm_name=ALGORITHMS[alg]['name'],
                    code_version=f'alg{alg}-raw.exe',
                    tau_percentile=p,
                    tau_value=tau,
                    status=status,
                    elapsed_s=round(elapsed, 2) if elapsed else '',
                    tle_standard=tle_std,
                    priority=calc_priority(alg, p, status),
                    created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
    
    return tasks


def generate_exp2_tasks(tau_values, existing_results):
    """生成 Exp-2: Scalability 任务"""
    tasks = []
    
    # DBLP 和 Amazon 的子图
    base_datasets = [(9, 'DBLP'), (10, 'Amazon')]
    scale_levels = [1, 6]  # L1(10%), L6(100%)
    algorithms = [2, 3, 4]
    percentiles = [20, 50, 80, 90]  # τ_90 仅用于 MonoSemMCE
    
    for ds_id, ds_name in base_datasets:
        for scale_id in scale_levels:
            for alg in algorithms:
                for p in percentiles:
                    # τ_90 只用于 alg4 (MonoSemMCE)
                    if p == 90 and alg != 4:
                        continue
                    
                    task_id = f'exp2_L{scale_id}_alg{alg}_ds{ds_id}_{p}th'
                    
                    tau = None
                    ds_str = str(ds_id)
                    pkey = f'tau_{p}'
                    if ds_str in tau_values and pkey in tau_values[ds_str]:
                        tau = tau_values[ds_str][pkey]
                    
                    status, elapsed, tle_std = get_task_status(
                        task_id, existing_results, CURRENT_TLE_STANDARD
                    )
                    
                    tasks.append(create_task(
                        task_id=task_id,
                        exp_id='exp2',
                        exp_name='Scalability',
                        dataset_id=ds_id,
                        dataset_name=ds_name,
                        scale_level=f'L{scale_id}',
                        algorithm_id=alg,
                        algorithm_name=ALGORITHMS[alg]['name'],
                        code_version=f'alg{alg}-raw.exe',
                        tau_percentile=p,
                        tau_value=tau,
                        extra_params=f'scale=L{scale_id}',
                        status=status,
                        elapsed_s=round(elapsed, 2) if elapsed else '',
                        tle_standard=tle_std,
                        priority=calc_priority(alg, p, status),
                        created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ))
    
    return tasks


def generate_exp3_tasks(tau_values, existing_results):
    """生成 Exp-3: Sensitivity to τ 任务"""
    tasks = []
    
    datasets = [1, 5, 7, 8]
    algorithms = [2, 3, 4]
    percentiles = [50, 60, 70, 80, 90, 95]
    
    for ds in datasets:
        for alg in algorithms:
            for p in percentiles:
                task_id = f'exp3_{p}th_alg{alg}_ds{ds}'
                
                tau = None
                ds_str = str(ds)
                pkey = f'tau_{p}'
                if ds_str in tau_values and pkey in tau_values[ds_str]:
                    tau = tau_values[ds_str][pkey]
                
                status, elapsed, tle_std = get_task_status(
                    task_id, existing_results, CURRENT_TLE_STANDARD
                )
                
                tasks.append(create_task(
                    task_id=task_id,
                    exp_id='exp3',
                    exp_name='Sensitivity to Tau',
                    dataset_id=ds,
                    dataset_name=DATASETS[ds],
                    algorithm_id=alg,
                    algorithm_name=ALGORITHMS[alg]['name'],
                    code_version=f'alg{alg}-raw.exe',
                    tau_percentile=p,
                    tau_value=tau,
                    status=status,
                    elapsed_s=round(elapsed, 2) if elapsed else '',
                    tle_standard=tle_std,
                    priority=calc_priority(alg, p, status),
                    created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
    
    return tasks


def generate_exp4_tasks(tau_values, existing_results):
    """生成 Exp-4: Diagnostics 任务 (使用 stats 版本)"""
    tasks = []
    
    datasets = [1, 2, 3, 4, 5, 7, 8, 9]
    algorithms_stats = [2, 3, 4]  # 使用 stats 版本
    percentiles = [10, 50, 90]
    
    for ds in datasets:
        for alg in algorithms_stats:
            for p in percentiles:
                task_id = f'exp4_{p}th_alg{alg}-stats_ds{ds}'
                
                tau = None
                ds_str = str(ds)
                pkey = f'tau_{p}'
                if ds_str in tau_values and pkey in tau_values[ds_str]:
                    tau = tau_values[ds_str][pkey]
                
                status, elapsed, tle_std = get_task_status(
                    task_id, existing_results, CURRENT_TLE_STANDARD
                )
                
                tasks.append(create_task(
                    task_id=task_id,
                    exp_id='exp4',
                    exp_name='Diagnostics',
                    dataset_id=ds,
                    dataset_name=DATASETS.get(ds, f'Dataset-{ds}'),
                    algorithm_id=alg,
                    algorithm_name=ALGORITHMS[alg]['name'],
                    code_version=f'alg{alg}-stats.exe',
                    tau_percentile=p,
                    tau_value=tau,
                    extra_params='stats_mode=True',
                    status=status,
                    elapsed_s=round(elapsed, 2) if elapsed else '',
                    tle_standard=tle_std,
                    priority=calc_priority(alg, p, status),
                    created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
    
    return tasks


def generate_exp5_tasks(tau_values, existing_results):
    """生成 Exp-5: Parallel Scalability 任务"""
    tasks = []
    
    dataset = 2  # 固定数据集 2（sc-ldoor，比 ds3 更快）
    algorithms = [2, 3, 4]
    percentile = 80  # τ_80
    
    for alg in algorithms:
        for threads in THREAD_COUNTS:
            task_id = f'exp5_t{threads}_alg{alg}_ds{dataset}_{percentile}th'
            
            tau = None
            ds_str = str(dataset)
            pkey = f'tau_{percentile}'
            if ds_str in tau_values and pkey in tau_values[ds_str]:
                tau = tau_values[ds_str][pkey]
            
            status, elapsed, tle_std = get_task_status(
                task_id, existing_results, CURRENT_TLE_STANDARD
            )
            
            tasks.append(create_task(
                task_id=task_id,
                exp_id='exp5',
                exp_name='Parallel Scalability',
                dataset_id=dataset,
                dataset_name=DATASETS[dataset],
                algorithm_id=alg,
                algorithm_name=ALGORITHMS[alg]['name'],
                code_version=f'alg{alg}-raw.exe',
                tau_percentile=percentile,
                tau_value=tau,
                threads=threads,
                extra_params=f'OMP_NUM_THREADS={threads}',
                status=status,
                elapsed_s=round(elapsed, 2) if elapsed else '',
                tle_standard=tle_std,
                priority=calc_priority(alg, percentile, status),
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
    
    return tasks


def generate_exp6a_tasks(tau_values, existing_results):
    """生成 Exp-6a: MonoSemMCE Ablation 任务"""
    tasks = []
    
    datasets = [1, 5, 7, 8]
    percentile = 90  # τ_90
    alg = 4  # MonoSemMCE
    
    for ds in datasets:
        for variant_id, variant_name in ABLATION_VARIANTS:
            task_id = f'exp6a_{variant_id}_ds{ds}'
            
            tau = None
            ds_str = str(ds)
            pkey = f'tau_{percentile}'
            if ds_str in tau_values and pkey in tau_values[ds_str]:
                tau = tau_values[ds_str][pkey]
            
            status, elapsed, tle_std = get_task_status(
                task_id, existing_results, CURRENT_TLE_STANDARD
            )
            
            tasks.append(create_task(
                task_id=task_id,
                exp_id='exp6a',
                exp_name='Ablation (MonoSemMCE)',
                dataset_id=ds,
                dataset_name=DATASETS[ds],
                algorithm_id=alg,
                algorithm_name=ALGORITHMS[alg]['name'],
                code_version=f'alg4-ablation-{variant_id}.exe',
                tau_percentile=percentile,
                tau_value=tau,
                variant=variant_name,
                extra_params=f'ablation={variant_id}',
                status=status,
                elapsed_s=round(elapsed, 2) if elapsed else '',
                tle_standard=tle_std,
                priority=calc_priority(alg, percentile, status),
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
    
    return tasks


def generate_exp6b_tasks(tau_values, existing_results):
    """生成 Exp-6b: StrSub Ablation 任务"""
    tasks = []
    
    datasets = [4, 8]
    alg = 3  # StrSub
    percentiles = [10, 30, 60, 90]  # 四个阈值对比
    
    for ds in datasets:
        for variant_id, variant_name in ABLBATION_STRSUB:
            for p in percentiles:
                task_id = f'exp6b_{variant_id}_ds{ds}_{p}th'
                
                tau = None
                ds_str = str(ds)
                pkey = f'tau_{p}'
                if ds_str in tau_values and pkey in tau_values[ds_str]:
                    tau = tau_values[ds_str][pkey]
                
                status, elapsed, tle_std = get_task_status(
                    task_id, existing_results, CURRENT_TLE_STANDARD
                )
                
                tasks.append(create_task(
                    task_id=task_id,
                    exp_id='exp6b',
                    exp_name='Ablation (StrSub)',
                    dataset_id=ds,
                    dataset_name=DATASETS[ds],
                    algorithm_id=alg,
                    algorithm_name=ALGORITHMS[alg]['name'],
                    code_version=f'alg3-ablation-{variant_id}.exe',
                    tau_percentile=p,
                    tau_value=tau,
                    variant=variant_name,
                    extra_params=f'ablation={variant_id}',
                    status=status,
                    elapsed_s=round(elapsed, 2) if elapsed else '',
                    tle_standard=tle_std,
                    priority=calc_priority(alg, p, status),
                    created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
    
    return tasks


def generate_exp7_tasks(tau_values, existing_results):
    """生成 Exp-7: Engineering Optimizations 任务"""
    tasks = []
    
    datasets = [1, 2, 8, 9]
    percentile = 90  # τ_90
    alg = 4  # MonoSemMCE
    
    for ds in datasets:
        for variant_id, variant_name in ENGINEERING_VARIANTS:
            task_id = f'exp7_{variant_id}_ds{ds}'
            
            tau = None
            ds_str = str(ds)
            pkey = f'tau_{percentile}'
            if ds_str in tau_values and pkey in tau_values[ds_str]:
                tau = tau_values[ds_str][pkey]
            
            status, elapsed, tle_std = get_task_status(
                task_id, existing_results, CURRENT_TLE_STANDARD
            )
            
            tasks.append(create_task(
                task_id=task_id,
                exp_id='exp7',
                exp_name='Engineering Optimizations',
                dataset_id=ds,
                dataset_name=DATASETS.get(ds, f'Dataset-{ds}'),
                algorithm_id=alg,
                algorithm_name=ALGORITHMS[alg]['name'],
                code_version=f'alg4-eng-{variant_id}.exe',
                tau_percentile=percentile,
                tau_value=tau,
                variant=variant_name,
                extra_params=f'engineering={variant_id}',
                status=status,
                elapsed_s=round(elapsed, 2) if elapsed else '',
                tle_standard=tle_std,
                priority=calc_priority(alg, percentile, status),
                created_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
    
    return tasks


def write_task_csv(tasks, csv_path=TASK_CSV):
    """写入 CSV 文件"""
    if not tasks:
        print("[ERROR] No tasks to write")
        return
    
    fieldnames = list(tasks[0].keys())
    
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tasks)
    
    print(f"[OK] Task CSV generated: {csv_path}")
    print(f"     Total tasks: {len(tasks)}")
    
    # 统计各状态数量
    status_count = {}
    for task in tasks:
        s = task['status']
        status_count[s] = status_count.get(s, 0) + 1
    
    print("\n[STATUS SUMMARY]")
    for status, count in sorted(status_count.items()):
        print(f"   {status}: {count}")


def main():
    """主函数"""
    print("=" * 70)
    print(" Experiment Task Generator")
    print("=" * 70)
    
    # 加载 tau 值
    print("\n[Step 1] Loading tau values...")
    tau_values = load_tau_values()
    if tau_values:
        print(f"[OK] Loaded tau values for {len(tau_values)} datasets")
    else:
        print("[WARN] No tau values found, some tasks will have empty tau_value")
    
    # 加载已有结果
    print("\n[Step 2] Scanning existing results...")
    existing_results = load_existing_results()
    if existing_results:
        print(f"[OK] Loaded {len(existing_results)} existing results")
    else:
        print("[WARN] No existing results found")
    
    # 生成所有实验任务
    print("\n[Step 3] Generating experiment tasks...")
    all_tasks = []
    
    exp1_tasks = generate_exp1_tasks(tau_values, existing_results)
    all_tasks.extend(exp1_tasks)
    print(f"  [+] Exp-1 (Overall Efficiency): {len(exp1_tasks)} tasks")
    
    exp2_tasks = generate_exp2_tasks(tau_values, existing_results)
    all_tasks.extend(exp2_tasks)
    print(f"  [+] Exp-2 (Scalability): {len(exp2_tasks)} tasks")
    
    exp3_tasks = generate_exp3_tasks(tau_values, existing_results)
    all_tasks.extend(exp3_tasks)
    print(f"  [+] Exp-3 (Sensitivity): {len(exp3_tasks)} tasks")
    
    exp4_tasks = generate_exp4_tasks(tau_values, existing_results)
    all_tasks.extend(exp4_tasks)
    print(f"  [+] Exp-4 (Diagnostics): {len(exp4_tasks)} tasks")
    
    exp5_tasks = generate_exp5_tasks(tau_values, existing_results)
    all_tasks.extend(exp5_tasks)
    print(f"  [+] Exp-5 (Parallel): {len(exp5_tasks)} tasks")
    
    exp6a_tasks = generate_exp6a_tasks(tau_values, existing_results)
    all_tasks.extend(exp6a_tasks)
    print(f"  [+] Exp-6a (Ablation-Mono): {len(exp6a_tasks)} tasks")
    
    exp6b_tasks = generate_exp6b_tasks(tau_values, existing_results)
    all_tasks.extend(exp6b_tasks)
    print(f"  [+] Exp-6b (Ablation-StrSub): {len(exp6b_tasks)} tasks")
    
    exp7_tasks = generate_exp7_tasks(tau_values, existing_results)
    all_tasks.extend(exp7_tasks)
    print(f"  [+] Exp-7 (Engineering): {len(exp7_tasks)} tasks")
    
    # alg1 去重 (StructBK 不依赖 tau)
    print(f"\n[Step 3.5] Deduplicating alg1 tasks...")
    original_count = len(all_tasks)
    all_tasks = merge_alg1_tasks(all_tasks)
    deduped_count = len(all_tasks)
    if deduped_count < original_count:
        print(f"     [OK] {original_count - deduped_count} duplicate tasks removed ({original_count} -> {deduped_count})")
    
    # 写入 CSV
    print(f"\n[Step 4] Writing task CSV...")
    write_task_csv(all_tasks)
    
    # 输出摘要
    pending = sum(1 for t in all_tasks if t['status'] in ['pending', 'needs_rerun'])
    completed = sum(1 for t in all_tasks if t['status'] == 'completed')
    failed = sum(1 for t in all_tasks if t['status'] in ['tle', 'oom', 'failed'])
    
    print(f"\n{'='*70}")
    print(f" [SUMMARY]")
    print(f"   [DONE] Completed: {completed}")
    print(f"   [TODO] Pending: {pending}")
    print(f"   [FAIL] Failed/TLE/OOM: {failed}")
    print(f"   [TOTAL] Total: {len(all_tasks)}")
    print(f"{'='*70}")
    
    if pending > 0:
        print(f"\n[RUN] Execute this command to start:")
        print(f"      python scripts/run_exp_onfuwuqi_auto.py")


if __name__ == '__main__':
    main()
