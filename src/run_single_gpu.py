#!/usr/bin/env python
"""
单卡训练启动脚本
自动选择空闲GPU或指定GPU进行训练
"""

import os
import sys
import subprocess
import argparse

def get_gpu_info():
    """获取所有GPU的显存使用情况"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        gpus.append({
                            'id': int(parts[0]),
                            'name': parts[1],
                            'total': int(parts[2]),
                            'used': int(parts[3]),
                            'free': int(parts[4])
                        })
            return gpus
        return []
    except Exception as e:
        print(f"错误: 无法获取GPU信息: {e}")
        return []

def get_gpu_processes(gpu_id):
    """获取指定GPU上的进程"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
             '--format=csv,noheader', '-i', str(gpu_id)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        processes.append({
                            'pid': parts[0],
                            'name': parts[1],
                            'memory': parts[2]
                        })
            return processes
        return []
    except Exception as e:
        return []

def find_best_gpu(gpus):
    """找到显存最空闲的GPU"""
    if not gpus:
        return None
    # 按空闲显存排序
    sorted_gpus = sorted(gpus, key=lambda x: x['free'], reverse=True)
    return sorted_gpus[0]['id']

def main():
    parser = argparse.ArgumentParser(description='单卡训练启动脚本')
    parser.add_argument('--gpu', type=int, default=None, 
                       help='指定GPU ID（默认自动选择最空闲的GPU）')
    parser.add_argument('--auto-kill', action='store_true',
                       help='自动终止目标GPU上的其他进程（需要权限）')
    args = parser.parse_args()
    
    print("=" * 70)
    print("单卡训练启动脚本")
    print("=" * 70)
    
    # 获取GPU信息
    gpus = get_gpu_info()
    if not gpus:
        print("错误: 未检测到GPU或nvidia-smi不可用")
        sys.exit(1)
    
    # 显示所有GPU信息
    print(f"\n检测到 {len(gpus)} 个GPU:")
    print(f"{'ID':<4} {'名称':<25} {'总显存':<10} {'已用':<10} {'空闲':<10}")
    print("-" * 70)
    for gpu in gpus:
        print(f"{gpu['id']:<4} {gpu['name']:<25} {gpu['total']:>8}MB {gpu['used']:>8}MB {gpu['free']:>8}MB")
    
    # 选择GPU
    if args.gpu is not None:
        target_gpu = args.gpu
        if target_gpu >= len(gpus):
            print(f"\n错误: GPU {target_gpu} 不存在")
            sys.exit(1)
        print(f"\n✔ 用户指定GPU: GPU:{target_gpu}")
    else:
        target_gpu = find_best_gpu(gpus)
        print(f"\n✔ 自动选择最空闲GPU: GPU:{target_gpu} (空闲显存: {gpus[target_gpu]['free']}MB)")
    
    # 检查目标GPU上的进程
    processes = get_gpu_processes(target_gpu)
    if processes:
        print(f"\n⚠ GPU:{target_gpu} 上检测到 {len(processes)} 个进程:")
        print(f"{'PID':<10} {'进程名':<30} {'显存占用':<15}")
        print("-" * 70)
        for proc in processes:
            print(f"{proc['pid']:<10} {proc['name']:<30} {proc['memory']:<15}")
        
        if args.auto_kill:
            print(f"\n正在终止GPU:{target_gpu}上的进程...")
            for proc in processes:
                try:
                    subprocess.run(['kill', '-9', proc['pid']], timeout=5)
                    print(f"✔ 已终止进程 PID {proc['pid']}")
                except Exception as e:
                    print(f"✗ 无法终止进程 PID {proc['pid']}: {e}")
            import time
            time.sleep(2)
        else:
            print("\n提示: 使用 --auto-kill 参数可自动终止这些进程")
    else:
        print(f"\n✔ GPU:{target_gpu} 上无其他进程")
    
    # 设置环境变量
    os.environ['TARGET_GPU_ID'] = str(target_gpu)
    
    print("\n" + "=" * 70)
    print(f"启动训练程序 (GPU:{target_gpu})...")
    print("=" * 70 + "\n")
    
    # 启动main.py
    import main

if __name__ == "__main__":
    main()

