#!/usr/bin/env python3
"""
实验自动执行脚本 - 基于任务表的断点续跑版本

功能：
1. 从 experiment_tasks.csv 读取待执行任务
2. 只执行 pending/needs_rerun 状态的任务
3. 每完成一个任务立即更新 CSV 状态
4. 支持中断后继续（断点续跑）
5. TLE 智能判断：当超时标准提高时自动重跑

使用方法：
    python scripts/generate_task_csv.py          # 先生成任务表（首次或更新时）
    python scripts/run_exp_onfuwuqi_auto.py       # 执行未完成任务
    python scripts/run_exp_onfuwuqi_auto.py --exp exp1 --status tle  # 只跑特定条件
"""

import argparse
import csv
import json
import os
import signal
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
TASK_CSV = 'scripts/experiment_tasks.csv'
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
DEFAULT_THREADS = 32  # 默认并行度（exp5 并行度测试除外）

# 全局标志：是否收到 Ctrl+C
interrupted = False

# 加载 tau 值
TAU_VALUES = {}
if os.path.exists(TAU_FILE):
    with open(TAU_FILE, 'r') as f:
        TAU_VALUES = json.load(f)


# ==================== 信号处理 ====================
def handle_sigint(signum, frame):
    global interrupted
    if interrupted:
        print("\n强制退出...")
        sys.exit(1)
    print("\n\n⚠️  收到 Ctrl+C，正在安全停止当前任务并保存状态...")
    interrupted = True

signal.signal(signal.SIGINT, handle_sigint)
# =================================================


class RemoteSSH:
    """SSH 远程连接管理器（带自动重连）"""
    
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
        print('已断开 SSH 连接')
    
    def reconnect(self, max_retries=3):
        for attempt in range(max_retries):
            try:
                print(f'  🔌 尝试重连 ({attempt+1}/{max_retries})...')
                self.disconnect()
                time.sleep(2)
                self.connect()
                self._init_remote_env()
                return True
            except Exception as e:
                print(f'  ⚠ 重连失败: {str(e)[:100]}')
                time.sleep(3)
        return False
    
    def is_alive(self):
        if not self.client:
            return False
        try:
            transport = self.client.get_transport()
            if not transport or not transport.is_active():
                return False
            stdin, stdout, stderr = self.client.exec_command('echo OK', timeout=5)
            stdout.read()
            return True
        except:
            return False
    
    def ensure_connection(self):
        if self.is_alive():
            return True
        print('⚠️ SSH 连接已断开，正在自动重连...')
        return self.reconnect()
    
    def exec_command(self, cmd, timeout=TIMEOUT_SEC):
        if interrupted:
            return "", "Interrupted", -1
        if not self.is_alive():
            if not self.ensure_connection():
                return "", "SSH reconnect failed", -1
        
        full_cmd = f'docker exec {DOCKER_CONTAINER} bash -c "cd {WORKSPACE_DIR} && {cmd}"'
        try:
            stdin, stdout, stderr = self.client.exec_command(full_cmd, timeout=timeout)
            out = stdout.read().decode('utf-8', errors='ignore')
            err = stderr.read().decode('utf-8', errors='ignore')
            code = stdout.channel.recv_exit_status()
            return out, err, code
        except Exception as e:
            if self.ensure_connection():
                try:
                    stdin, stdout, stderr = self.client.exec_command(full_cmd, timeout=timeout)
                    out = stdout.read().decode('utf-8', errors='ignore')
                    err = stderr.read().decode('utf-8', errors='ignore')
                    code = stdout.channel.recv_exit_status()
                    return out, err, code
                except:
                    pass
            return "", str(e), -1
    
    def exec_command_with_input(self, cmd, input_text, timeout=TIMEOUT_SEC):
        if interrupted:
            return "", "Interrupted", -1
        
        if not self.is_alive():
            if not self.ensure_connection():
                return "", "SSH reconnect failed", -1
        
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
            if not interrupted and self.ensure_connection():
                try:
                    stdin, stdout, stderr = self.client.exec_command(full_cmd, timeout=timeout)
                    stdin.write(input_text)
                    stdin.flush()
                    stdin.channel.shutdown_write()
                    out = stdout.read().decode('utf-8', errors='ignore')
                    err = stderr.read().decode('utf-8', errors='ignore')
                    code = stdout.channel.recv_exit_status()
                    return out, err, code
                except Exception as e2:
                    return "", str(e2), -1
            if interrupted:
                return "", "Interrupted by Ctrl+C", -1
            return "", str(e), -1
    
    def get_file_content(self, remote_path):
        if interrupted:
            return None
        cmd = f'cat {remote_path}'
        out, err, code = self.exec_command(cmd)
        if code == 0:
            return out
        return None
    
    def _init_remote_env(self):
        """初始化远程环境"""
        print('正在初始化远程环境...')
        
        try:
            kill_cmd = 'pkill -f "alg" 2>/dev/null || true; sleep 0.3; echo "CLEANED"'
            out, err, code = self.exec_command(kill_cmd)
            print('  ✓ 已清理残留进程')
        except Exception as e:
            print(f'  ⚠ 清理失败（继续）: {str(e)[:100]}')
        
        init_cmd = f'mkdir -p {LOG_DIR} {RESULTS_DIR}'
        out, err, code = self.exec_command(init_cmd)
        if code == 0:
            print(f'✓ 远程目录已创建: {LOG_DIR}, {RESULTS_DIR}')
        else:
            print(f'⚠ 创建远程目录失败: {err}')
    
    def __enter__(self):
        self.connect()
        self._init_remote_env()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


class TaskManager:
    """任务管理器：读写 CSV 任务表"""
    
    def __init__(self, csv_path=TASK_CSV):
        self.csv_path = csv_path
        self.tasks = []
        self._load()
    
    def _load(self):
        """加载任务表"""
        if not os.path.exists(self.csv_path):
            print(f"❌ 任务表不存在: {self.csv_path}")
            print("   请先运行: python scripts/generate_task_csv.py")
            sys.exit(1)
        
        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            self.tasks = list(reader)
        
        print(f"✓ 已加载 {len(self.tasks)} 个任务")
    
    def get_pending_tasks(self, exp_filter=None, status_filter=None):
        """获取待执行的任务（按优先级降序排列）"""
        pending = []
        
        for task in self.tasks:
            status = task['status']
            
            # 状态过滤
            if status_filter and status != status_filter:
                continue
            
            # 实验过滤
            if exp_filter and task['exp_id'] != exp_filter:
                continue
            
            # 只选择需要执行的状态
            if status in ['pending', 'needs_rerun']:
                pending.append(task)
            
            # 可选：重新运行失败的任务
            elif status_filter == 'failed' and status in ['failed', 'tle', 'oom']:
                pending.append(task)
        
        # 按优先级降序排列（越大越先执行）
        pending.sort(key=lambda t: float(t.get('priority', 0) or 0), reverse=True)
        
        return pending
    
    def update_task_status(self, task_id, status, elapsed=None, error_type=None, 
                           error_msg=None, tle_standard=None):
        """更新单个任务状态"""
        for task in self.tasks:
            if task['task_id'] == task_id:
                task['status'] = status
                task['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                if elapsed is not None:
                    task['elapsed_s'] = round(elapsed, 2) if elapsed else ''
                
                if error_type is not None:
                    task['error_type'] = error_type
                
                if error_msg is not None:
                    task['error_msg'] = error_msg
                
                if tle_standard:
                    task['tle_standard'] = tle_standard
                
                break
        
        self._save()
    
    def reset_all_tasks(self):
        """重置所有任务状态为 pending，清除执行记录"""
        count = 0
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for task in self.tasks:
            if task['status'] != 'pending':
                task['status'] = 'pending'
                task['elapsed_s'] = ''
                task['error_type'] = ''
                task['error_msg'] = ''
                task['updated_at'] = now
                count += 1
        self._save()
        return count
    
    def _save(self):
        """保存任务表到文件"""
        if not self.tasks:
            return
        
        # 动态收集所有字段名（兼容新增字段如 error_type, error_msg）
        all_fieldnames = set()
        for task in self.tasks:
            all_fieldnames.update(task.keys())
        
        base_fieldnames = list(self.tasks[0].keys())
        for fn in sorted(all_fieldnames):
            if fn not in base_fieldnames:
                base_fieldnames.append(fn)
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=base_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.tasks)


def parse_log(content):
    """解析日志提取时间和团数量"""
    total_ms = None
    count = None
    
    if not content:
        return {'total_ms': None, 'count': None}
    
    for line in content.split('\n'):
        if 'TOTAL:' in line and 'maximal cliques' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'TOTAL:':
                    try:
                        total_ms = int(parts[i + 1])
                    except (ValueError, IndexError):
                        pass
                if p == 'maximal' and i > 0:
                    try:
                        count = int(parts[i - 1])
                    except (ValueError, IndexError):
                        pass
            break
    
    return {'total_ms': total_ms, 'count': count}


def run_single_task(ssh, task, task_manager):
    """执行单个任务并返回结果"""
    global interrupted
    
    task_id = task['task_id']
    alg_id = int(task['algorithm_id'])
    ds_id = int(task['dataset_id'])
    tau_val = task.get('tau_value')
    threads = task.get('threads')
    log_suffix = task['exp_id']
    
    # 线程数逻辑：
    # - exp5 (并行度测试): 使用任务指定的线程数 (1,2,4,8,16,32,64,128)
    # - 其他实验: 默认使用 DEFAULT_THREADS=32
    is_parallel_test = task.get('exp_id') == 'exp5'
    if not threads or not is_parallel_test:
        threads = DEFAULT_THREADS
        if not is_parallel_test:
            log_suffix += f'_t{threads}'
    
    # 使用 task 中指定的 code_version
    # 注意：exe 文件在服务器的 /workspace/build/ 目录下
    code_version = task.get('code_version', f'alg{alg_id}-raw.exe')
    alg_name = f'./build/{code_version}'  # 添加 build/ 路径前缀
    
    print(f"\n{'='*70}")
    print(f"📋 任务: {task_id}")
    print(f"   实验: {task['exp_name']}")
    print(f"   数据集: {task['dataset_name']} (ID={ds_id})")
    print(f"   算法: {task['algorithm_name']} (alg{alg_id})")
    print(f"   版本: {code_version}")
    print(f"   阈值: τ_{task['tau_percentile']}th = {tau_val}")
    print(f"   线程数: {threads}{' (并行度测试)' if is_parallel_test else ' (默认)'}")
    print(f"{'='*70}")
    
    if interrupted:
        return {'success': False, 'error': 'Interrupted'}
    
    # 清理残留进程
    try:
        kill_cmd = 'pkill -f "alg" 2>/dev/null; sleep 0.3'
        ssh.exec_command(kill_cmd)
    except:
        pass
    
    # 构建命令 (alg_name 已在上面从 task['code_version'] 获取)
    remote_log_name = f'{log_suffix}_{task_id}.log'
    remote_log_path = f'{LOG_DIR}/{remote_log_name}'
    
    cmd = f'{alg_name} dataset {ds_id} log {remote_log_path}'
    
    if alg_id == 1:
        stdin_input = 'mine\nquit\n'
    else:
        if not tau_val or tau_val == '':
            print(f"   ❌ 缺少 tau 值，跳过")
            task_manager.update_task_status(task_id, 'failed', error_type='MISSING_TAU')
            return {'success': False, 'error': 'Missing tau'}
        stdin_input = f'mine {tau_val}\nquit\n'
    
    env_cmd = f'export OMP_NUM_THREADS={threads} && '
    
    safe_cmd = (
        f'{env_cmd}'
        f'ulimit -v {24 * 1024 * 1024} && '
        f'timeout {TIMEOUT_SEC} '
        f'{cmd}'
    )
    
    print(f"   ⏱️  开始执行...")
    start = time.time()
    
    try:
        out, err, code = ssh.exec_command_with_input(safe_cmd, stdin_input)
        elapsed = time.time() - start
        
        # DEBUG: 显示实际命令和退出码（遇到问题时便于排查）
        if code != 0 and code != 124:
            print(f"   [DEBUG] cmd: {safe_cmd[:200]}...")
            print(f"   [DEBUG] exit_code={code}, stderr={err[:200] if err else '(empty)'}")
        
        if interrupted:
            print(f"   ⚠️  已中断")
            task_manager.update_task_status(task_id, 'interrupted', elapsed=elapsed)
            return {'success': False, 'error': 'Interrupted'}
        
        # 保存输出
        output_file = os.path.join(RESULTS_DIR, f'output_{remote_log_name.replace(".log", ".txt")}')
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f'=== STDOUT ===\n{(out or "")}\n=== STDERR ===\n{(err or "")}\n=== Exit Code: {code} ===\n')
        
        # 错误检测
        error_type = None
        error_msg = None
        
        out_lower = (out or '').lower()
        err_lower = (err or '').lower()
        
        # OOM 检测
        oom_indicators = ['killed', 'oom', 'out of memory', 'cannot allocate', 'malloc failed', 'std::bad_alloc']
        if any(ind in out_lower or ind in err_lower for ind in oom_indicators):
            error_type = 'OOM'
            error_msg = '内存不足(OOM)'
            print(f"   ⚠️  {error_type}: {error_msg}")
        
        # 超时检测
        elif code == 124:
            error_type = 'TLE'
            error_msg = f'运行超时(>{TIMEOUT_SEC}s)'
            print(f"   ⏰  {error_type}: {error_msg}")
        
        # 其他错误
        elif code != 0:
            signal_map = {
                127: '可执行文件不存在(未编译)',
                137: '被系统杀死(可能OOM)',
                139: '段错误(SegFault)',
                134: '程序异常中止(Abort)',
                -1: 'SSH连接异常',
            }
            error_type = f'SIGNAL_{code}' if code in signal_map else f'ERROR_{code}'
            error_msg = signal_map.get(code, (err or '')[:300] or f'错误码:{code}')
            print(f"   ❌ {error_type}: {error_msg}")
        
        # 更新任务状态
        if error_type:
            print(f"   ⏱️  耗时: {elapsed:.1f}s")
            
            # 特殊处理：文件不存在 (ERROR_127) → 标记为 needs_rerun（视为未执行）
            if error_type == 'ERROR_127' or 'No such file' in str(error_msg):
                status = 'needs_rerun'
                print(f"   ⚠️  文件不存在，标记为待重试（编译后可自动执行）")
            elif error_type in ['TLE', 'OOM']:
                status = error_type.lower()
            else:
                status = 'failed'
            
            task_manager.update_task_status(
                task_id, 
                status,
                elapsed=elapsed,
                error_type=error_type,
                error_msg=error_msg,
                tle_standard=TIMEOUT_SEC
            )
            return {'success': False, 'error': error_type, 'elapsed': elapsed}
        
        # 成功
        print(f"   ✅ 成功！耗时: {elapsed:.1f}s")
        
        # 解析日志
        parsed = parse_log(ssh.get_file_content(remote_log_path))
        if not parsed or not parsed.get('total_ms'):
            parsed = parse_log(out)
        
        if parsed and parsed.get('total_ms'):
            print(f"   📊 结果: {parsed['total_ms']/1000:.3f}s, {parsed['count']} cliques")
        
        # 对于 stats 版本（Exp-4 Diagnostics），下载完整日志保存统计信息
        log_content = None
        if 'stats' in code_version:
            try:
                log_content = ssh.get_file_content(remote_log_path)
                if log_content:
                    # 保存完整日志到本地
                    local_log_file = os.path.join(RESULTS_DIR, f'log_{task_id}.txt')
                    os.makedirs(RESULTS_DIR, exist_ok=True)
                    with open(local_log_file, 'w', encoding='utf-8') as f:
                        f.write(log_content)
                    print(f"   📝 已保存详细日志: {local_log_file}")
            except Exception as e:
                print(f"   ⚠️  日志下载失败: {str(e)[:100]}")
        
        # 更新为完成
        task_manager.update_task_status(
            task_id, 
            'completed',
            elapsed=elapsed,
            tle_standard=TIMEOUT_SEC
        )
        
        # 保存结果到 JSON
        result_data = {
            'time_avg': parsed['total_ms'] / 1000.0 if parsed and parsed.get('total_ms') else elapsed,
            'time_min': parsed['total_ms'] / 1000.0 if parsed and parsed.get('total_ms') else elapsed,
            'time_max': parsed['total_ms'] / 1000.0 if parsed and parsed.get('total_ms') else elapsed,
            'count': parsed.get('count'),
            'status': 'success',
            'elapsed': elapsed
        }
        
        # stats 版本额外保存日志路径和原始内容
        if log_content:
            result_data['log_file'] = f'log_{task_id}.txt'
            result_data['raw_log'] = log_content
        
        save_result_to_json(task_id, result_data)
        
        return {'success': True, 'elapsed': elapsed, 'parsed': parsed}
        
    except Exception as e:
        elapsed = time.time() - start
        error_msg = str(e)
        
        ssh_exceptions = ['Socket is closed', 'Connection reset', 'Broken pipe', 'timed out']
        if any(exc in error_msg for exc in ssh_exceptions):
            error_type = 'SSH_ERROR'
            print(f"   🔌 SSH异常: {error_msg[:200]}")
        else:
            error_type = 'EXCEPTION'
            print(f"   💥 异常: {error_msg[:200]}")
        
        task_manager.update_task_status(
            task_id,
            'failed',
            elapsed=elapsed,
            error_type=error_type,
            error_msg=error_msg
        )
        
        return {'success': False, 'error': error_type, 'elapsed': elapsed}


def save_result_to_json(task_id, result_data):
    """保存结果到 JSON 文件（按数据集分组）"""
    # 提取数据集 ID
    parts = task_id.split('_ds')
    if len(parts) < 2:
        return
    
    ds_part = parts[1].split('_')[0]
    json_file = os.path.join(RESULTS_DIR, f'exp1_ds{ds_part}.json')
    
    # 加载已有结果
    existing = {}
    if os.path.exists(json_file):
        with open(json_file, 'r', encoding='utf-8') as f:
            existing = json.load(f)
    
    # 更新
    existing[task_id] = result_data
    
    # 保存
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def main():
    global interrupted
    
    parser = argparse.ArgumentParser(description='实验自动执行脚本')
    parser.add_argument('--exp', type=str, help='只运行指定实验 (exp1-exp7)')
    parser.add_argument('--status', type=str, help='只运行指定状态的任务 (pending/completed/tle/oom/failed)')
    parser.add_argument('--limit', type=int, default=0, help='限制运行任务数量 (0=不限制)')
    parser.add_argument('--dry-run', action='store_true', help='只显示要运行的任务，不实际执行')
    parser.add_argument('--reset', action='store_true', help='重置所有任务状态为 pending（用于清空 results 后重新开始）')
    args = parser.parse_args()
    
    print("=" * 70)
    print(" 实验自动执行系统")
    print("=" * 70)
    print(f" 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 超时标准: {TIMEOUT_SEC}s")
    print("=" * 70)
    
    # 加载任务表
    print("\n[Step 1] 加载任务表...")
    task_manager = TaskManager()
    
    # 重置模式：将所有任务状态设为 pending
    if args.reset:
        print("\n[RESET] 重置所有任务状态为 pending...")
        reset_count = task_manager.reset_all_tasks()
        print(f"   ✅ 已重置 {reset_count} 个任务")
        return
    
    # 获取待执行任务
    print("\n[Step 2] 筛选待执行任务...")
    pending_tasks = task_manager.get_pending_tasks(
        exp_filter=args.exp,
        status_filter=args.status
    )
    
    if not pending_tasks:
        print("\n✅ 所有指定条件的任务都已完成！")
        print("   （或者没有匹配的任务）")
        return
    
    # 限制数量
    if args.limit > 0:
        pending_tasks = pending_tasks[:args.limit]
    
    print(f"\n📋 待执行任务: {len(pending_tasks)} 个")
    if args.exp:
        print(f"   过滤: 实验={args.exp}")
    if args.status:
        print(f"   过滤: 状态={args.status}")
    
    # Dry run 模式
    if args.dry_run:
        print("\n[DRY RUN] 以下任务将被执行:\n")
        for i, task in enumerate(pending_tasks, 1):
            print(f"  {i}. [{task['status']}] {task['task_id']}")
            print(f"     {task['algorithm_name']} @ {task['dataset_name']}")
        return
    
    # 统计
    stats = {'success': 0, 'failed': 0, 'skipped': 0}
    
    # 执行任务
    print(f"\n[Step 3] 开始执行任务...\n")
    
    with RemoteSSH() as ssh:
        for i, task in enumerate(pending_tasks, 1):
            if interrupted:
                print("\n⚠️  收到中断信号，停止执行")
                break
            
            print(f"\n[{i}/{len(pending_tasks)}]", end=" ")
            
            result = run_single_task(ssh, task, task_manager)
            
            if result['success']:
                stats['success'] += 1
            else:
                stats['failed'] += 1
            
            # 短暂休息，避免服务器压力过大
            time.sleep(1)
    
    # 输出统计
    print(f"\n{'='*70}")
    print(f"📊 执行统计:")
    print(f"   ✅ 成功: {stats['success']}")
    print(f"   ❌ 失败: {stats['failed']}")
    print(f"   总计: {len(pending_tasks)}")
    print(f"{'='*70}")
    
    if interrupted:
        print("\n💡 提示: 可以重新运行此脚本继续执行剩余任务")
    
    remaining = len(task_manager.get_pending_tasks(exp_filter=args.exp, status_filter=args.status))
    if remaining > 0:
        print(f"\n⏳ 剩余待执行任务: {remaining} 个")


if __name__ == '__main__':
    main()
