#!/usr/bin/env python3
"""
在远程服务器上编译运行 C++ 版本的 Tau 百分位数计算程序。

用法:
  python scripts/compute_tau_remote.py
  
输出:
  - 远程: /workspace/scripts/tau_values.json
  - 本地: scripts/tau_values.json (自动下载)
"""

import os
import sys
import time

try:
    import paramiko
except ImportError:
    print("需要安装 paramiko: pip install paramiko")
    exit(1)

# 远程服务器配置（与 run_exp_onfuwuqi.py 保持一致）
SERVER_HOST = '10.108.21.72'
SERVER_PORT = 55566
SERVER_USER = 'tygao'
SERVER_PASSWORD = 'tygaobit2023'

# Docker 容器配置
DOCKER_CONTAINER = 'naughty_mahavira'
WORKSPACE_DIR = '/workspace'

TIMEOUT_SEC = 1800  # 30分钟超时（大图计算可能需要较长时间）


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
        print('✓ 连接成功！')

    def disconnect(self):
        """断开连接"""
        if self.client:
            self.client.close()
            print('已断开连接')

    def exec_command(self, cmd, timeout=TIMEOUT_SEC):
        """
        在 Docker 容器中执行命令
        返回 (stdout, stderr, exit_code)
        """
        full_cmd = f'docker exec {DOCKER_CONTAINER} bash -c "cd {WORKSPACE_DIR} && {cmd}"'
        stdin, stdout, stderr = self.client.exec_command(full_cmd, timeout=timeout)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        code = stdout.channel.recv_exit_status()
        return out, err, code

    def get_file_content(self, remote_path):
        """读取远程文件内容"""
        cmd = f'cat {remote_path}'
        out, err, code = self.exec_command(cmd)
        if code == 0:
            return out
        return None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()


def main():
    print('=' * 60)
    print('Tau 百分位数计算工具 (远程 C++ 版本)')
    print('=' * 60)

    # 获取当前脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    with RemoteSSH() as ssh:
        # Step 1: 检查源文件和依赖文件是否存在
        print('\n[Step 1] 检查远程环境...')
        
        check_cmd = 'ls -lh compute_tau.cpp semantic_graph.cpp semantic_graph.h 2>&1'
        out, err, code = ssh.exec_command(check_cmd)
        
        if code != 0:
            print(f'❌ 错误：找不到必要的源文件')
            print(out)
            print(err)
            return False
        
        print('✓ 源文件检查通过:')
        for line in out.strip().split('\n'):
            if line.strip():
                print(f'   {line}')

        # Step 2: 编译 C++ 程序
        print('\n[Step 2] 编译 C++ 程序...')
        
        compile_cmd = (
            f'g++ -O3 -march=native -fopenmp -std=c++17 '
            f'compute_tau.cpp semantic_graph.cpp '
            f'-o compute_tau.exe '
            f'2>&1'
        )
        
        start = time.time()
        out, err, code = ssh.exec_command(compile_cmd, timeout=120)
        compile_time = time.time() - start
        
        if code != 0:
            print(f'❌ 编译失败 (耗时 {compile_time:.1f}s):')
            print(f'   错误信息:\n{err}')
            print(f'   输出:\n{out}')
            return False
        
        print(f'✓ 编译成功！耗时 {compile_time:.1f}s')
        
        # 验证可执行文件
        verify_cmd = 'ls -lh compute_tau.exe'
        out, err, code = ssh.exec_command(verify_cmd)
        if code == 0:
            print(f'   可执行文件: {out.strip()}')

        # Step 3: 运行计算程序
        print('\n[Step 3] 运行 Tau 计算...')
        print('   （这可能需要几分钟时间，请耐心等待...）\n')
        
        run_cmd = './compute_tau.exe 2>/dev/null'
        
        start = time.time()
        out, err, code = ssh.exec_command(run_cmd, timeout=TIMEOUT_SEC)
        elapsed = time.time() - start
        
        if code != 0:
            print(f'❌ 运行失败 (耗时 {elapsed:.1f}s):')
            print(f'   退出码: {code}')
            print(f'   错误信息:\n{err[:500]}')
            print(f'   输出:\n{out[:500]}')
            return False
        
        print(f'✓ 计算完成！耗时 {elapsed:.1f}s')
        
        # 显示输出的前几行预览
        lines = out.strip().split('\n')
        print(f'\n   输出预览 ({len(lines)} 行):')
        preview_lines = min(15, len(lines))
        for i in range(preview_lines):
            print(f'   {lines[i]}')
        if len(lines) > preview_lines:
            print(f'   ... (共 {len(lines)} 行)')

        # Step 4: 保存到远程文件
        print('\n[Step 4] 保存到远程服务器...')
        
        save_cmd = './compute_tau.exe > tau_values.json 2>/dev/null && echo "SAVE_OK"'
        out_save, err_save, code_save = ssh.exec_command(save_cmd, timeout=TIMEOUT_SEC)
        
        if code_save != 0 or 'SAVE_OK' not in out_save:
            print(f'⚠ 远程保存可能失败，尝试使用管道输出...')
            remote_json = out
        else:
            print('✓ 已保存到远程: /workspace/tau_values.json')
            # 重新读取以确认
            remote_json = ssh.get_file_content('tau_values.json') or out
        
        # Step 5: 下载到本地
        print('\n[Step 5] 下载到本地...')
        
        local_output_file = os.path.join(script_dir, 'tau_values.json')
        
        try:
            with open(local_output_file, 'w', encoding='utf-8') as f:
                f.write(remote_json)
            
            file_size = os.path.getsize(local_output_file)
            print(f'✓ 已保存到本地: {local_output_file}')
            print(f'   文件大小: {file_size:,} bytes')
            
        except Exception as e:
            print(f'❌ 本地保存失败: {e}')
            return False

        # Step 6: 清理临时文件（可选）
        print('\n[Step 6] 清理临时文件...')
        
        cleanup_cmd = 'rm -f compute_tau.exe'
        out_clean, err_clean, code_clean = ssh.exec_command(cleanup_cmd)
        
        if code_clean == 0:
            print('✓ 已清理临时编译文件')
        else:
            print('⚠ 清理失败（不影响结果）')

    # 完成
    print('\n' + '=' * 60)
    print('🎉 Tau 计算完成！')
    print('=' * 60)
    print(f'\n结果文件:')
    print(f'  远程: {DOCKER_CONTAINER}:{WORKSPACE_DIR}/scripts/tau_values.json')
    print(f'  本地: {os.path.abspath(local_output_file)}')
    
    # 显示简要统计
    try:
        import json
        with open(local_output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f'\n数据集数量: {len(data)} 个')
        print(f'可用百分位: tau_10, tau_20, tau_50, tau_70, tau_80, tau_90, tau_95')
        print(f'\n快速预览:')
        for ds_id, info in data.items():
            taus = [f"τ_{k.split('_')[1]}={v:.6f}" for k, v in info.items() if k.startswith('tau_')]
            print(f"  数据集 {ds_id} ({info['name']:12s}): {', '.join(taus[:4])}")
            
    except Exception as e:
        print(f'⚠ 无法解析 JSON 预览: {e}')

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
