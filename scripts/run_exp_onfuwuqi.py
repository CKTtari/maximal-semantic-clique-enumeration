#!/usr/bin/env python3
"""
批量实验执行脚本 - 远程服务器版本 (已修复 Ctrl+C 无法停止问题)
"""

import argparse
import json
import os
import subprocess
import signal
import sys
import time
from pathlib import Path

try:
    import paramiko
except ImportError:
    print("需要安装 paramiko: pip install paramiko")
    exit(1)

# 远程服务器配置
SERVER_HOST = '10.108.21.72'
SERVER_PORT = 55566
SERVER_USER = 'tygao'
SERVER_PASSWORD = 'tygaobit2023'

# Docker 容器配置
DOCKER_CONTAINER = 'naughty_mahavira'
WORKSPACE_DIR = '/workspace'

# 本地配置
TAU_FILE = 'scripts/tau_values.json'
LOG_DIR = 'logs'
RESULTS_DIR = 'results'
TIMEOUT_SEC = 600  # 服务器超时时间（秒）

# 全局标志：是否收到 Ctrl+C
interrupted = False

# 加载 tau 值
TAU_VALUES = {}
if os.path.exists(TAU_FILE):
    with open(TAU_FILE, 'r') as f:
        TAU_VALUES = json.load(f)


# ====================== 修复：Ctrl+C 信号处理 ======================
def handle_sigint(signum, frame):
    global interrupted
    if interrupted:
        print("\n强制退出...")
        sys.exit(1)
    print("\n\n⚠️  收到 Ctrl+C，正在安全停止实验并断开 SSH...")
    interrupted = True

# 注册信号
signal.signal(signal.SIGINT, handle_sigint)
# ==================================================================


class RemoteSSH:
    """SSH 远程连接管理器"""

    def __init__(self):
        self.client = None

    def connect(self):
        """建立 SSH 连接"""
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
        """断开连接"""
        if self.client:
            try:
                self.client.close()
            except:
                pass
        print('已断开 SSH 连接')

    def reconnect(self, max_retries=3):
        """自动重连"""
        for attempt in range(max_retries):
            try:
                print(f'  🔌 尝试重连 ({attempt+1}/{max_retries})...')
                self.disconnect()
                import time as _t
                _t.sleep(2)
                self.connect()
                self._init_remote_env()
                return True
            except Exception as e:
                print(f'  ⚠ 重连失败: {str(e)[:100]}')
                _t.sleep(3)
        return False

    def is_alive(self):
        """检查连接是否存活"""
        if not self.client:
            return False
        try:
            transport = self.client.get_transport()
            if not transport or not transport.is_active():
                return False
            # 发送一个简单命令测试
            stdin, stdout, stderr = self.client.exec_command('echo OK', timeout=5)
            stdout.read()
            return True
        except:
            return False

    def ensure_connection(self):
        """确保连接可用，断开则自动重连"""
        if self.is_alive():
            return True
        print('⚠️ SSH 连接已断开，正在自动重连...')
        return self.reconnect()

    def exec_command(self, cmd, timeout=TIMEOUT_SEC):
        if interrupted:
            return "", "Interrupted", -1
        # 确保连接存活
        if not self.is_alive():
            print('  ⚠️ 连接断开，尝试重连...')
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
            # 尝试重连后重试一次
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

        # 确保连接存活
        if not self.is_alive():
            print('  ⚠️ 连接断开，尝试重连...')
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
            # 尝试重连后重试一次
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

    def __enter__(self):
        self.connect()
        self._init_remote_env()
        return self

    def _init_remote_env(self):
        """初始化远程环境：创建必要目录并清理残余进程"""
        print('正在初始化远程环境...')
        
        # 清理所有残留的 alg 进程
        print('  清理残留进程...')
        try:
            kill_cmd = 'pkill -f "alg" 2>/dev/null; sleep 0.5; echo "CLEANED"'
            out, err, code = self.exec_command(kill_cmd)
            if 'CLEANED' in (out or ''):
                print('  ✓ 已清理残留的 alg 进程')
            else:
                print(f'  ⚠ 清理命令执行异常: {(err or "")[:100]}')
        except Exception as e:
            print(f'  ⚠ 清理失败（继续）: {str(e)[:100]}')
        
        # 创建 logs 和 results 目录
        init_cmd = f'mkdir -p {LOG_DIR} {RESULTS_DIR}'
        out, err, code = self.exec_command(init_cmd)
        if code == 0:
            print(f'✓ 远程目录已创建: {LOG_DIR}, {RESULTS_DIR}')
        else:
            print(f'⚠ 创建远程目录失败: {err}')
        
        # 验证工作目录和可执行文件
        check_cmd = 'ls -la *.exe 2>/dev/null | head -5'
        out, err, code = self.exec_command(check_cmd)
        if code == 0 and out.strip():
            print(f'✓ 可执行文件检查通过:\n{out[:200]}')
        else:
            print(f'⚠ 未找到可执行文件或目录错误')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def ensure_dirs():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def get_tau(dataset_id, percentile):
    ds_id = str(dataset_id)
    pkey = f'tau_{percentile}'
    if ds_id in TAU_VALUES and pkey in TAU_VALUES[ds_id]:
        return TAU_VALUES[ds_id][pkey]
    raise ValueError(f'Tau not found for dataset {dataset_id}, percentile {percentile}')


def run_algorithm(ssh, alg_id, dataset_id, tau=None, threads=None, log_suffix=''):
    global interrupted
    if interrupted:
        return {'log': '', 'elapsed': 0, 'timeout': False, 'error': 'Interrupted', 'output': '', 'stderr': ''}

    # 每次运行前清理残留的 alg 进程
    try:
        kill_cmd = 'pkill -f "alg" 2>/dev/null; sleep 0.3'
        ssh.exec_command(kill_cmd)
    except Exception as e:
        print(f'    ⚠ 清理进程失败（继续执行）: {str(e)[:100]}')

    alg_name = f'./alg{alg_id}-raw.exe'
    remote_log_name = f'exp{log_suffix}_alg{alg_id}_ds{dataset_id}'
    if threads:
        remote_log_name += f'_t{threads}'
    remote_log_name += '.log'
    remote_log_path = f'{LOG_DIR}/{remote_log_name}'

    cmd = f'{alg_name} dataset {dataset_id} log {remote_log_path}'

    if alg_id == 1:
        stdin_input = 'mine\nquit\n'
    else:
        if tau is None:
            raise ValueError(f'tau required for alg{alg_id}')
        stdin_input = f'mine {tau}\nquit\n'

    env_cmd = ''
    if threads:
        env_cmd = f'export OMP_NUM_THREADS={threads} && '

    # 添加内存限制和超时保护：限制最大24GB内存，超时300s自动终止
    safe_cmd = (
        f'ulimit -v {24 * 1024 * 1024} && '  # 24GB virtual memory limit (in KB)
        f'timeout {TIMEOUT_SEC} '
        f'{env_cmd}{cmd}'
    )
    
    full_cmd = safe_cmd
    print(f'  Running: alg{alg_id} ds{dataset_id} tau={tau} threads={threads} ...')

    start = time.time()
    try:
        out, err, code = ssh.exec_command_with_input(full_cmd, stdin_input)
        elapsed = time.time() - start

        if interrupted:
            print(f'    已中断')
            return {'log': remote_log_path, 'elapsed': elapsed, 'timeout': False, 'error': 'Interrupted', 'output': out or '', 'stderr': err or ''}
        
        # 保存输出到本地日志文件
        local_output_file = os.path.join(RESULTS_DIR, f'output_{remote_log_name.replace(".log", ".txt")}')
        with open(local_output_file, 'w', encoding='utf-8') as f:
            f.write(f'=== STDOUT ===\n{(out or "")}\n=== STDERR ===\n{(err or "")}\n=== Exit Code: {code} ===\n')
        
        # 错误检测与分类
        error_type = None
        error_msg = None
        
        # 检测 OOM (Out of Memory)
        oom_indicators = ['killed', 'oom', 'out of memory', 'cannot allocate', 'malloc failed', 'std::bad_alloc']
        out_lower = (out or '').lower()
        err_lower = (err or '').lower()
        
        if any(indicator in out_lower or indicator in err_lower for indicator in oom_indicators):
            error_type = 'OOM'
            error_msg = f'内存不足(OOM) - 程序被系统终止'
            print(f'    ⚠️  {error_type}: {error_msg}')
        
        # 检测超时 (timeout command returns 124)
        elif code == 124:
            error_type = 'TLE'
            error_msg = f'运行超时(>{TIMEOUT_SEC}s)'
            print(f'    ⏰  {error_type}: {error_msg}')
        
        # 检测其他错误退出码
        elif code != 0 and code != 124:
            # 常见信号: 137=SIGKILL(可能OOM), 139=SIGSEGV(段错误), 134=SIGABRT
            signal_map = {
                137: '被系统强制杀死(可能是OOM)',
                139: '段错误(Segmentation Fault)',
                134: '程序异常中止(Abort)',
                -1: '连接断开或SSH异常',
                -9: '进程被强制终止',
                1: '一般错误',
                2: '误用shell命令',
                126: '命令不可执行',
                127: '命令未找到',
            }
            
            if code in signal_map:
                error_type = f'SIGNAL_{code}'
                error_msg = signal_map[code]
            else:
                error_type = f'ERROR_{code}'
                error_msg = (err or '')[:500] or f'未知错误(退出码:{code})'
            
            print(f'    ❌ {error_type}: {error_msg}')
        
        # 如果有错误，返回错误信息
        if error_type:
            print(f'    Done in {elapsed:.1f}s (FAILED: {error_type})')
            return {
                'log': remote_log_path,
                'elapsed': elapsed,
                'timeout': (error_type == 'TLE'),
                'error': error_msg,
                'error_type': error_type,
                'output': out or '',
                'stderr': err or '',
                'parsed': None
            }
        
        # 成功执行
        print(f'    Done in {elapsed:.1f}s (exit code: {code})')
        
        parsed = parse_remote_log(ssh, remote_log_path)
        
        # 如果日志解析失败，尝试从 stdout 解析
        if not parsed or not parsed.get('total_ms'):
            print(f'    ⚠ 远程日志解析失败，尝试从 stdout 解析...')
            parsed = parse_stdout(out)
            if parsed and parsed.get('total_ms'):
                print(f'    ✓ 从 stdout 成功解析: total_ms={parsed["total_ms"]}, count={parsed["count"]}')
            else:
                print(f'    ✗ 解析完全失败！')
                print(f'    程序输出（前500字符）:\n{(out or "")[:500]}')
        
        return {
            'log': remote_log_path,
            'elapsed': elapsed,
            'timeout': False,
            'error': None,
            'error_type': None,
            'parsed': parsed,
            'output': out or '',
            'stderr': err or ''
        }
    except Exception as e:
        elapsed = time.time() - start
        error_msg = str(e)
        
        # 检测是否是 SSH 连接异常
        ssh_exceptions = ['Socket is closed', 'Connection reset', 'Broken pipe', 'timed out']
        if any(exc in error_msg for exc in ssh_exceptions):
            error_type = 'SSH_ERROR'
            print(f'    🔌 SSH连接异常: {error_msg}')
        else:
            error_type = 'EXCEPTION'
            print(f'    💥 异常: {error_msg}')
        
        return {
            'log': remote_log_path, 
            'elapsed': elapsed, 
            'timeout': False, 
            'error': error_msg,
            'error_type': error_type,
            'output': '',
            'stderr': error_msg,
            'parsed': None
        }


def parse_remote_log(ssh, remote_log_path):
    """从远程服务器上的日志文件提取总时间和团数量。"""
    content = ssh.get_file_content(remote_log_path)
    if not content:
        return {'total_ms': None, 'count': None}
    
    total_ms = None
    count = None
    for line in content.split('\n'):
        if 'TOTAL:' in line and 'maximal cliques' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == 'TOTAL:':
                    total_ms = int(parts[i + 1])
                if p == 'maximal' and i > 0:
                    count = int(parts[i - 1])
    return {'total_ms': total_ms, 'count': count}


def parse_stdout(stdout_content):
    """从 stdout 输出中提取总时间和团数量（备选方案）"""
    if not stdout_content:
        return {'total_ms': None, 'count': None}
    
    total_ms = None
    count = None
    for line in stdout_content.split('\n'):
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


# ========== 所有实验函数都增加中断检查 ==========
def run_exp1(ssh, datasets, algorithms, tau_percentile, runs=3):
    global interrupted
    print('\n=== Exp-1: Overall Efficiency ===')
    results = {}
    for ds in datasets:
        if interrupted: break
        tau = get_tau(ds, tau_percentile)
        for alg in algorithms:
            if interrupted: break
            key = f'alg{alg}_ds{ds}'
            times = []
            counts = []
            for r in range(runs):
                if interrupted: break
                res = run_algorithm(ssh, alg, ds, tau=tau, log_suffix=f'1_r{r}')
                if not res['timeout'] and not res['error'] and res.get('parsed'):
                    parsed = res['parsed']
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
                results[key] = {'time_avg': 'TLE/Interrupted', 'count': None}
    return results


def run_exp1_multi_tau(ssh, datasets, algorithms, tau_percentiles, runs=1):
    global interrupted
    print('\n=== Exp-1: Overall Efficiency ===')
    results = {}
    
    for ds in datasets:
        if interrupted: break
        
        print(f'\n--- Processing Dataset {ds} ---')
        ds_results = {}
        
        for p in tau_percentiles:
            if interrupted: break
            tau = get_tau(ds, p)
            for alg in algorithms:
                if interrupted: break
                key = f'{p}th_alg{alg}_ds{ds}'
                times = []
                counts = []
                last_res = None
                for r in range(runs):
                    if interrupted: break
                    try:
                        res = run_algorithm(ssh, alg, ds, tau=tau, log_suffix=f'1_{p}th_r{r}')
                    except Exception as e:
                        print(f'    💥 运行异常: {str(e)[:200]}')
                        res = {'log': '', 'elapsed': 0, 'timeout': False, 'error': str(e), 'error_type': 'EXCEPTION', 'output': '', 'stderr': str(e), 'parsed': None}
                    last_res = res
                    if not res['timeout'] and not res['error'] and res.get('parsed'):
                        parsed = res['parsed']
                        if parsed['total_ms']:
                            times.append(parsed['total_ms'] / 1000.0)
                            counts.append(parsed['count'])
                
                if times:
                    results[key] = {
                        'time_avg': sum(times) / len(times),
                        'time_min': min(times),
                        'time_max': max(times),
                        'count': counts[0] if counts else None,
                        'status': 'success'
                    }
                    ds_results[key] = results[key]
                else:
                    # 保存失败时的详细信息（包含错误类型）
                    error_info = {
                        'time_avg': last_res.get('error', 'TLE/Interrupted') if last_res else 'TLE/Interrupted',
                        'count': None,
                        'status': 'failed',
                        'error_type': last_res.get('error_type', 'UNKNOWN') if last_res else 'UNKNOWN',
                        'elapsed': last_res.get('elapsed') if last_res else None,
                        'error': last_res.get('error') if last_res else None,
                        'output_preview': (last_res.get('output', '') or '')[:500] if last_res else '',
                        'stderr_preview': (last_res.get('stderr', '') or '')[:300] if last_res else ''
                    }
                    results[key] = error_info
                    ds_results[key] = error_info
                
                # 每次运行后立即保存（包含详细信息）
                save_results(f'exp1_ds{ds}', ds_results)
        
        print(f'✓ Dataset {ds} completed and saved!')
    
    return results


def run_exp3(ssh, datasets, algorithms, percentiles, runs=1):
    global interrupted
    print('\n=== Exp-3: Sensitivity to tau ===')
    results = {}
    for ds in datasets:
        if interrupted: break
        for p in percentiles:
            if interrupted: break
            tau = get_tau(ds, p)
            for alg in algorithms:
                if interrupted: break
                key = f'p{p}_alg{alg}_ds{ds}'
                res = run_algorithm(ssh, alg, ds, tau=tau, log_suffix=f'3_p{p}')
                if not res['timeout'] and not res['error'] and res.get('parsed'):
                    parsed = res['parsed']
                    results[key] = {
                        'time': parsed['total_ms'] / 1000.0 if parsed['total_ms'] else None,
                        'count': parsed['count'],
                    }
                else:
                    results[key] = {'time': 'TLE/Interrupted', 'count': None}
    return results


def run_exp5(ssh, dataset, algorithms, thread_counts, tau_percentile=80, runs=1):
    global interrupted
    print('\n=== Exp-5: Parallel Scalability ===')
    results = {}
    tau = get_tau(dataset, tau_percentile)
    for alg in algorithms:
        if interrupted: break
        for t in thread_counts:
            if interrupted: break
            key = f'alg{alg}_t{t}'
            res = run_algorithm(ssh, alg, dataset, tau=tau, threads=t, log_suffix='5')
            if not res['timeout'] and not res['error'] and res.get('parsed'):
                parsed = res['parsed']
                results[key] = {
                    'time': parsed['total_ms'] / 1000.0 if parsed['total_ms'] else None,
                    'count': parsed['count'],
                }
            else:
                results[key] = {'time': 'TLE/Interrupted', 'count': None}
    return results
# ==================================================


def save_results(exp_name, results):
    path = os.path.join(RESULTS_DIR, f'{exp_name}.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved results to {path}')


def parse_range(s):
    items = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            items.extend(range(int(a), int(b) + 1))
        else:
            items.append(int(part))
    return items


def main():
    global interrupted
    parser = argparse.ArgumentParser(description='Run experiments on remote server')
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--datasets', type=str, default='1-8',
                        help='Dataset IDs, e.g. 1-8 or 1,3,5 or 1-5,7-9 (跳过6)')
    parser.add_argument('--algorithms', type=str, default='2-4')
    parser.add_argument('--tau', type=str, default='70th')
    parser.add_argument('--threads', type=str, default='1,2,4,8,16,32,64,128',
                        help='Thread counts for parallel exp')
    parser.add_argument('--runs', type=int, default=1)
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ensure_dirs()

    datasets = parse_range(args.datasets)
    algorithms = parse_range(args.algorithms)

    with RemoteSSH() as ssh:
        if interrupted:
            return
        if args.exp == 'exp1':
            percentiles = parse_range(args.tau.replace('th', ''))
            results = run_exp1_multi_tau(ssh, datasets, algorithms, percentiles, runs=args.runs)
            save_results('exp1', results)
        elif args.exp == 'exp3':
            percentiles = parse_range(args.tau.replace('th', ''))
            results = run_exp3(ssh, datasets, algorithms, percentiles, runs=1)
            save_results('exp3', results)
        elif args.exp == 'exp5':
            if len(datasets) != 1:
                raise ValueError('exp5 requires exactly one dataset')
            thread_counts = parse_range(args.threads)
            p = int(args.tau.replace('th', ''))
            results = run_exp5(ssh, datasets[0], algorithms, thread_counts, p)
            save_results('exp5', results)
        else:
            print(f'Unknown experiment: {args.exp}')

    if interrupted:
        print("\n✅ 实验已安全停止")
    else:
        print('\n✅ 实验完成！')


if __name__ == '__main__':
    main()