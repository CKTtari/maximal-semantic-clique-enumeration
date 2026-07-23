#!/usr/bin/env python3
"""
Exp-6/Exp-7 消融实验执行脚本

功能：
1. Exp-6a: MonoSemMCE 消融实验（5个变体 × 4个数据集）
2. Exp-6b: StrSub 消融实验（4个变体 × 2个数据集 × 4个阈值）
3. Exp-7: 工程优化实验（5个配置 × 4个数据集）

使用方法：
    python scripts/run_ablation_exps.py --exp exp6a    # 只运行 Exp-6a
    python scripts/run_ablation_exps.py --exp exp6b    # 只运行 Exp-6b
    python scripts/run_ablation_exps.py --exp exp7     # 只运行 Exp-7
    python scripts/run_ablation_exps.py --exp all      # 运行所有消融实验
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("需要安装 paramiko: pip install paramiko")
    exit(1)


# ==================== 配置 ====================
RESULTS_DIR = 'results'
TAU_FILE = 'scripts/tau_values.json'
LOG_DIR = 'logs'

# 远程服务器配置
SERVER_HOST = '10.108.21.72'
SERVER_PORT = 55566
SERVER_USER = 'tygao'
SERVER_PASSWORD = 'tygaobit2023'

# Docker 容器配置
DOCKER_CONTAINER = 'naughty_mahavira'
WORKSPACE_DIR = '/workspace'

TIMEOUT_SEC = 600  # 服务器超时时间（秒）

# 加载 tau 值
TAU_VALUES = {}
if os.path.exists(TAU_FILE):
    with open(TAU_FILE, 'r') as f:
        TAU_VALUES = json.load(f)


class RemoteSSH:
    """SSH 远程连接管理器"""

    def __init__(self):
        self.client = None

    def connect(self):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        print(f'正在连接 {SERVER_HOST}:{SERVER_PORT} ...')
        self.client.connect(
            hostname=SERVER_HOST,
            port=SERVER_PORT,
            username=SERVER_USER,
            password=SERVER_PASSWORD,
            timeout=30
        )
        print('连接成功！')

    def disconnect(self):
        if self.client:
            try:
                self.client.close()
            except:
                pass

    def exec_command(self, cmd, timeout=TIMEOUT_SEC):
        full_cmd = f'docker exec {DOCKER_CONTAINER} bash -c "cd {WORKSPACE_DIR} && {cmd}"'
        try:
            stdin, stdout, stderr = self.client.exec_command(full_cmd, timeout=timeout)
            out = stdout.read().decode('utf-8', errors='ignore')
            err = stderr.read().decode('utf-8', errors='ignore')
            code = stdout.channel.recv_exit_status()
            return out, err, code
        except Exception as e:
            return "", str(e), -1

    def exec_command_with_input(self, cmd, input_text, timeout=TIMEOUT_SEC):
        full_cmd = f'docker exec -i {DOCKER_CONTAINER} bash -c "cd {WORKSPACE_DIR} && {cmd}"'
        try:
            stdin, stdout, stderr = self.client.exec_command(full_cmd, timeout=timeout)
            stdin.write(input_text)
            stdin.flush()
            stdin.channel.shutdown_write()

            out = stdout.read().decode('utf-8', errors='ignore')
            err = stderr.read().decode('utf-8', errors='ignore')
            code = stdout.channel.recv_exit_status()
            return out, err, code
        except Exception as e:
            return "", str(e), -1

    def upload_file(self, local_path, remote_path):
        sftp = self.client.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()

    def download_file(self, remote_path, local_path):
        sftp = self.client.open_sftp()
        sftp.get(remote_path, local_path)
        sftp.close()


def get_tau(dataset_id, percentile=80):
    """获取指定数据集和百分位的 tau 值"""
    key = f"ds{dataset_id}_tau{percentile}"
    if key in TAU_VALUES:
        return TAU_VALUES[key]
    # 回退到默认值
    ds_key = f"ds{dataset_id}"
    if ds_key in TAU_VALUES:
        taus = TAU_VALUES[ds_key]
        if isinstance(taus, dict) and f"p{percentile}" in taus:
            return taus[f"p{percentile}"]
    print(f"⚠️  未找到 dataset={dataset_id}, percentile={percentile} 的 tau 值")
    return 0.5


def run_single_experiment(ssh, exe_name, dataset_id, tau, log_prefix, variant_label=""):
    """
    执行单次实验并返回结果
    Returns: (time_ms, clique_count, status)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_prefix}_{timestamp}.log"

    # 构建输入命令
    if 'alg1' in exe_name:
        input_text = f'mine\nquit\n'
    else:
        input_text = f'mine {tau}\nquit\n'

    cmd = f'./build/{exe_name}.exe dataset {dataset_id} log {log_file}'

    print(f"\n  📊 执行: {exe_name} | DS={dataset_id} | τ={tau:.4f} | Variant: {variant_label}")
    print(f"  日志: {log_file}")

    start_time = time.time()
    out, err, code = ssh.exec_command_with_input(cmd, input_text, timeout=TIMEOUT_SEC)
    elapsed = time.time() - start_time

    # 判断状态
    if elapsed >= TIMEOUT_SEC - 10:
        status = 'TLE'
        time_ms = int(elapsed * 1000)
        clique_count = -1
        print(f"  ⏰ 超时 (TLE): {elapsed:.1f}s")
    elif 'out of memory' in err.lower() or 'oom' in err.lower():
        status = 'OOM'
        time_ms = int(elapsed * 1000)
        clique_count = -1
        print(f"  💾 内存溢出 (OOM)")
    elif code != 0:
        status = 'FAILED'
        time_ms = int(elapsed * 1000)
        clique_count = -1
        print(f"  ❌ 执行失败 (code={code}): {err[:200]}")
    else:
        status = 'OK'
        # 尝试从输出中解析时间
        time_ms = -1
        clique_count = -1

        # 解析日志文件获取精确时间
        try:
            remote_log = f"{WORKSPACE_DIR}/{log_file}"
            local_log = f"{LOG_DIR}/{log_file}"
            os.makedirs(LOG_DIR, exist_ok=True)
            ssh.download_file(remote_log, local_log)

            with open(local_log, 'r') as f:
                log_content = f.read()

            # 查找 TOTAL 时间行
            for line in log_content.split('\n'):
                if 'TOTAL:' in line and 'ms' in line:
                    import re
                    match = re.search(r'TOTAL:\s*(\d+)\s*ms.*?(\d+)\s*maximal\s*cliques', line)
                    if match:
                        time_ms = int(match.group(1))
                        clique_count = int(match.group(2))
                        break
        except Exception as e:
            print(f"  ⚠️  解析日志失败: {e}")

        if time_ms == -1:
            time_ms = int(elapsed * 1000)

        print(f"  ✅ 完成: {time_ms}ms, {clique_count} cliques")

    return {
        'exe': exe_name,
        'dataset': dataset_id,
        'tau': tau,
        'variant': variant_label,
        'time_ms': time_ms,
        'cliques': clique_count,
        'status': status,
        'log': log_file
    }


def run_exp6a(ssh):
    """
    Exp-6a: MonoSemMCE Ablation Study
    变体：no, w, mono, both, full
    数据集：1, 5, 7, 8
    阈值：τ_90
    """
    print("\n" + "="*80)
    print("Exp-6a: MonoSemMCE Ablation Study")
    print("="*80)

    datasets = [1, 5, 7, 8]
    variants = [
        ('ablation-no',     'No pruning'),
        ('ablation-w',      'W-pruning only'),
        ('ablation-mono',   'Monotone break only'),
        ('ablation-both',   'W+Mono, no dedup'),
        ('ablation-full',   'Full system'),
    ]

    results = []
    tau = get_tau(8, 90)  # 使用 τ_90

    for ds_id in datasets:
        ds_tau = get_tau(ds_id, 90)
        print(f"\n📁 数据集 {ds_id} (τ={ds_tau:.4f})")

        for exe_name, label in variants:
            result = run_single_experiment(
                ssh,
                f"alg4-{exe_name}",
                ds_id,
                ds_tau,
                f"logs/exp6a_alg4_{exe_name}_ds{ds_id}",
                label
            )
            results.append(result)

    # 保存结果
    save_results(results, 'exp6a_monosemmce_ablation.csv')
    return results


def run_exp6b(ssh):
    """
    Exp-6b: StrSub Ablation Study
    变体：no, topedge, localmax, full
    数据集：4, 8
    阈值：τ_10, τ_30, τ_60, τ_90
    """
    print("\n" + "="*80)
    print("Exp-6b: StrSub Ablation Study")
    print("="*80)

    datasets = [4, 8]
    percentiles = [10, 30, 60, 90]
    variants = [
        ('ablation-no',      'No pruning'),
        ('ablation-topedge', 'Top-Edge only'),
        ('ablation-localmax','LocalMax only'),
        ('ablation-full',    'Full StrSub'),
    ]

    results = []

    for ds_id in datasets:
        print(f"\n📁 数据集 {ds_id}")

        for pct in percentiles:
            ds_tau = get_tau(ds_id, pct)
            print(f"\n  📊 阈值 τ_{pct} = {ds_tau:.4f}")

            for exe_name, label in variants:
                result = run_single_experiment(
                    ssh,
                    f"alg3-{exe_name}",
                    ds_id,
                    ds_tau,
                    f"logs/exp6b_alg3_{exe_name}_ds{ds_id}_tau{pct}",
                    label
                )
                result['tau_percentile'] = pct
                results.append(result)

    # 保存结果
    save_results(results, 'exp6b_strsub_ablation.csv')
    return results


def run_exp7(ssh):
    """
    Exp-7: Engineering Optimizations
    配置：dense, naive, noacc, noreorder, full
    数据集：1, 2, 8, 9
    阈值：τ_90
    """
    print("\n" + "="*80)
    print("Exp-7: Engineering Optimizations")
    print("="*80)

    datasets = [1, 2, 8, 9]
    configs = [
        ('raw',             'Full system (baseline)'),
        # ('eng-dense',       'Dense bitmap'),           # 需要手动实现
        # ('eng-naive',       'Naïve intersection'),     # 需要手动实现
        # ('eng-noacc',       'No incremental acc'),     # 需要手动实现
        # ('eng-noreorder',   'No reorder caching'),     # 需要手动实现
    ]

    results = []

    for ds_id in datasets:
        ds_tau = get_tau(ds_id, 90)
        print(f"\n📁 数据集 {ds_id} (τ={ds_tau:.4f})")

        for exe_name, label in configs:
            result = run_single_experiment(
                ssh,
                f"alg4-{exe_name}",
                ds_id,
                ds_tau,
                f"logs/exp7_alg4_{exe_name}_ds{ds_id}",
                label
            )
            results.append(result)

    # 保存结果
    save_results(results, 'exp7_engineering.csv')
    return results


def save_results(results, filename):
    """保存结果到 CSV 文件"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)

    fieldnames = ['exe', 'dataset', 'tau', 'tau_percentile', 'variant',
                  'time_ms', 'cliques', 'status', 'log']

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\n💾 结果已保存到: {filepath}")
    print(f"   共 {len(results)} 条记录")


def main():
    parser = argparse.ArgumentParser(description='Exp-6/Exp-7 消融实验执行脚本')
    parser.add_argument('--exp', type=str, default='all',
                       choices=['exp6a', 'exp6b', 'exp7', 'all'],
                       help='要执行的实验 (默认: all)')
    parser.add_argument('--datasets', type=str, default=None,
                       help='指定数据集 ID，逗号分隔 (如: 1,5,7)')
    parser.add_argument('--dry-run', action='store_true',
                       help='只显示将要执行的任务，不实际运行')
    args = parser.parse_args()

    # 解析数据集过滤
    dataset_filter = None
    if args.datasets:
        dataset_filter = [int(x.strip()) for x in args.datasets.split(',')]

    print("="*80)
    print("消融实验执行系统")
    print("="*80)
    print(f"实验: {args.exp}")
    print(f"服务器: {SERVER_HOST}:{SERVER_PORT}")
    print(f"超时: {TIMEOUT_SEC}s")
    if dataset_filter:
        print(f"数据集过滤: {dataset_filter}")

    # 连接服务器
    ssh = RemoteSSH()
    try:
        ssh.connect()

        # 确保远程目录存在
        ssh.exec_command(f'mkdir -p {WORKSPACE_DIR}/build {WORKSPACE_DIR}/logs {WORKSPACE_DIR}/results')

        # 上传构建脚本并编译变体
        print("\n🔨 编译消融变体...")
        if args.exp in ['exp6a', 'all']:
            print("  编译 Exp-6a 变体...")
            ssh.upload_file('scripts/build_ablation.sh', f'{WORKSPACE_DIR}/scripts/build_ablation.sh')
            out, err, code = ssh.exec_command('bash scripts/build_ablation.sh exp6a')
            if code != 0:
                print(f"  ⚠️  编译警告/错误: {err[:300]}")

        if args.exp in ['exp6b', 'all']:
            print("  编译 Exp-6b 变体...")
            out, err, code = ssh.exec_command('bash scripts/build_ablation.sh exp6b')
            if code != 0:
                print(f"  ⚠️  编译警告/错误: {err[:300]}")

        # 执行实验
        all_results = []

        if args.exp in ['exp6a', 'all']:
            results = run_exp6a(ssh)
            all_results.extend(results)

        if args.exp in ['exp6b', 'all']:
            results = run_exp6b(ssh)
            all_results.extend(results)

        if args.exp in ['exp7', 'all']:
            results = run_exp7(ssh)
            all_results.extend(results)

        # 打印汇总
        print("\n" + "="*80)
        print("实验完成！汇总:")
        print("="*80)

        ok_count = sum(1 for r in all_results if r['status'] == 'OK')
        tle_count = sum(1 for r in all_results if r['status'] == 'TLE')
        oom_count = sum(1 for r in all_results if r['status'] == 'OOM')
        fail_count = sum(1 for r in all_results if r['status'] == 'FAILED')

        print(f"  总计: {len(all_results)} 个任务")
        print(f"  ✅ 成功: {ok_count}")
        print(f"  ⏰ 超时: {tle_count}")
        print(f"  💾 OOM: {oom_count}")
        print(f"  ❌ 失败: {fail_count}")

    finally:
        ssh.disconnect()
        print("\n👋 已断开连接")


if __name__ == '__main__':
    main()
