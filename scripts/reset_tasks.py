#!/usr/bin/env python3
"""
重置任务状态脚本
将因文件不存在（ERROR_127）而标记为 failed/needs_rerun 的任务重置为 pending
用法: python scripts/reset_tasks.py [--all | --error-127]
"""

import csv
import os
import argparse
from datetime import datetime

CSV_FILE = 'scripts/experiment_tasks.csv'

def load_tasks():
    if not os.path.exists(CSV_FILE):
        print(f"❌ 任务表不存在: {CSV_FILE}")
        return []
    with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_tasks(tasks):
    if not tasks:
        return
    
    # 动态收集所有可能的字段名（兼容新旧 CSV 格式）
    all_fieldnames = set()
    for task in tasks:
        all_fieldnames.update(task.keys())
    
    # 保持原有字段顺序（从第一个任务获取），追加新字段
    base_fieldnames = list(tasks[0].keys())
    for fn in sorted(all_fieldnames):
        if fn not in base_fieldnames:
            base_fieldnames.append(fn)
    
    with open(CSV_FILE, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=base_fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(tasks)

def reset_error_127_tasks():
    """重置所有 ERROR_127 相关的失败任务为 pending"""
    tasks = load_tasks()
    if not tasks:
        return
    
    reset_count = 0
    for task in tasks:
        status = task.get('status', '')
        error_type = task.get('error_type', '')
        error_msg = task.get('error_msg', '')
        
        # 检查是否是文件不存在的错误
        is_file_not_found = (
            error_type == 'ERROR_127' or 
            'No such file' in str(error_msg) or
            '127' in str(error_type)
        )
        
        # 启发式检测：如果是 failed 状态且符合以下特征，很可能是文件不存在
        # 特征1: code_version 包含 "stats"（stats 版本未编译）
        # 特征2: elapsed_s 非常短 (< 1s) 且没有 error_type 记录
        if not is_file_not_found and status == 'failed':
            code_version = task.get('code_version', '')
            elapsed = task.get('elapsed_s', '')
            
            try:
                elapsed_val = float(elapsed) if elapsed else 0
            except (ValueError, TypeError):
                elapsed_val = 0
            
            is_heuristic_match = (
                ('stats' in str(code_version) and elapsed_val < 1.0) or
                (not error_type and elapsed_val < 1.0 and status == 'failed')
            )
            
            if is_heuristic_match:
                is_file_not_found = True
        
        # 如果状态是 failed/needs_rerun 且原因是文件不存在，则重置为 pending
        if status in ['failed', 'needs_rerun'] and is_file_not_found:
            old_status = status
            task['status'] = 'pending'
            task['elapsed_s'] = ''
            task['error_type'] = ''
            task['error_msg'] = ''
            task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            reset_count += 1
            print(f"  ✓ 重置: {task['task_id']} ({old_status} → pending)")
    
    save_tasks(tasks)
    print(f"\n✅ 已重置 {reset_count} 个任务为 pending")

def reset_all_failed():
    """重置所有失败任务为 pending（谨慎使用！）"""
    tasks = load_tasks()
    if not tasks:
        return
    
    reset_count = 0
    for task in tasks:
        status = task.get('status', '')
        
        # 只重置 failed 和 needs_rerun 状态，保留 completed/tle/oom
        if status in ['failed', 'needs_rerun']:
            old_status = status
            task['status'] = 'pending'
            task['elapsed_s'] = ''
            task['error_type'] = ''
            task['error_msg'] = ''
            task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            reset_count += 1
            print(f"  ✓ 重置: {task['task_id']} ({old_status} → pending)")
    
    save_tasks(tasks)
    print(f"\n✅ 已重置 {reset_count} 个任务为 pending")

def show_status_summary():
    """显示当前任务状态统计"""
    tasks = load_tasks()
    if not tasks:
        return
    
    from collections import Counter
    status_counts = Counter(task.get('status', 'unknown') for task in tasks)
    
    print("\n📊 当前任务状态:")
    print("-" * 40)
    for status, count in sorted(status_counts.items()):
        icon = {'pending': '⏳', 'completed': '✅', 'failed': '❌', 
                'tle': '⏰', 'oom': '💾', 'needs_rerun': '🔄'}.get(status, '❓')
        print(f"  {icon} {status:15s}: {count:4d}")
    print("-" * 40)
    print(f"  总计: {len(tasks)}")
    
    # 显示 ERROR_127 的任务
    error_127_tasks = [t for t in tasks if '127' in t.get('error_type', '') or 'No such file' in t.get('error_msg', '')]
    if error_127_tasks:
        print(f"\n⚠️  发现 {len(error_127_tasks)} 个文件不存在错误的任务:")
        for t in error_127_tasks[:10]:  # 只显示前10个
            print(f"   - {t['task_id']}: {t.get('code_version', '?')} ({t.get('status', '?')})")
        if len(error_127_tasks) > 10:
            print(f"   ... 还有 {len(error_127_tasks) - 10} 个")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='重置任务状态')
    parser.add_argument('--all', action='store_true', help='重置所有失败任务')
    parser.add_argument('--error-127', action='store_true', help='只重置文件不存在的错误 (默认)')
    parser.add_argument('--show', action='store_true', help='仅显示当前状态统计')
    
    args = parser.parse_args()
    
    show_status_summary()
    
    if args.show:
        pass
    elif args.all:
        print("\n🔄 正在重置所有失败任务...")
        reset_all_failed()
    else:
        print("\n🔄 正在重置文件不存在错误的任务...")
        reset_error_127_tasks()