"""
The function is taken from deepxde
which can be downloaded from https://github.com/lululxvi/deepxde
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import subprocess
import tensorflow as tf

_BACKEND = "tensorflow"
_VERSION = tf.__version__
_IS_TF_1 = _VERSION.startswith("1.")
_GPU_NUM = 1
_CPU_NUM = 1

if _IS_TF_1:
    print("Using TensorFlow V1 backend.")
    # tf = tf
else:
    print("Using TensorFlow V2 backend.")
    # tf = tf.compat.v1
    # tf.disable_v2_behavior()

# ==================== 单卡GPU配置（关键优化） ====================
# 选择要使用的GPU ID（默认使用GPU 1，因为GPU 0通常被其他进程占用）
TARGET_GPU_ID = int(os.environ.get('TARGET_GPU_ID', '1'))  # 可通过环境变量修改

print(f"\n{'='*70}")
print(f"单卡GPU配置")
print(f"{'='*70}")
print(f"目标GPU: GPU:{TARGET_GPU_ID}")

# 设置CUDA_VISIBLE_DEVICES，让TensorFlow只看到目标GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(TARGET_GPU_ID)
print(f"✔ 已设置 CUDA_VISIBLE_DEVICES={TARGET_GPU_ID}")

# 启用cuda_malloc_async显存分配器，减少显存碎片
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
print(f"✔ 已启用 cuda_malloc_async 显存分配器")

# 检查目标GPU上的进程
def check_gpu_processes(gpu_id):
    """检查指定GPU上的进程"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader', '-i', str(gpu_id)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        pid = parts[0].strip()
                        mem = parts[1].strip()
                        processes.append((pid, mem))
            return processes
        return []
    except Exception as e:
        print(f"⚠ 无法检查GPU进程: {e}")
        return []

# 检查目标GPU上是否有其他进程
gpu_processes = check_gpu_processes(TARGET_GPU_ID)
if gpu_processes:
    total_mem = sum(int(mem.split()[0]) for _, mem in gpu_processes)
    print(f"\n⚠ 警告: GPU:{TARGET_GPU_ID} 上检测到 {len(gpu_processes)} 个进程 (总占用: ~{total_mem}MB):")
    for pid, mem in gpu_processes:
        print(f"  - PID {pid}: {mem}")

    # 检查是否设置了AUTO_KILL_GPU_PROCESSES环境变量
    auto_kill = os.environ.get('AUTO_KILL_GPU_PROCESSES', '0') == '1'

    if auto_kill:
        print(f"\n✔ 检测到 AUTO_KILL_GPU_PROCESSES=1，自动终止进程...")
        killed_count = 0
        for pid, mem in gpu_processes:
            try:
                subprocess.run(['kill', '-9', pid], timeout=5, check=False)
                print(f"✔ 已终止进程 PID {pid}")
                killed_count += 1
            except Exception as e:
                print(f"✗ 无法终止进程 PID {pid}: {e}")

        if killed_count > 0:
            print(f"✔ 成功终止 {killed_count}/{len(gpu_processes)} 个进程")
            import time
            time.sleep(2)  # 等待GPU显存释放
        else:
            print(f"⚠ 未能终止任何进程，可能需要权限或进程不属于当前用户")
    else:
        print(f"\n提示: 这些进程可能占用显存，如需自动终止，请设置环境变量:")
        print(f"  export AUTO_KILL_GPU_PROCESSES=1")
        print(f"⚠ 继续运行，但可能因显存不足导致OOM")
else:
    print(f"✔ GPU:{TARGET_GPU_ID} 上无其他进程")

print(f"{'='*70}\n")

# 重新获取GPU列表（此时只能看到TARGET_GPU_ID对应的GPU）
gpus = tf.config.list_physical_devices("GPU")
_GPU_NUM = len(gpus)
cpus = tf.config.list_physical_devices("CPU")
_CPU_NUM = len(cpus)

if gpus:
    """单GPU配置"""
    for gpu in gpus:
        # 启用显存按需增长
        tf.config.experimental.set_memory_growth(gpu, True)
        print('TensorFlow可见GPU:')
        print(gpu)
        print(f"✔ 显存按需增长已启用")
else:
    """CPU fallback"""
    print("⚠ TensorFlow未检测到GPU，将使用CPU训练")
    for cpu in cpus:
        print('CPU info:')
        print(cpu)


def backend():
    """Returns the name and version of the current backend, e.g., ("tensorflow", 1.14.0).

    Returns:
        tuple: A ``tuple`` of the name and version of the backend DeepXDE is currently using.

    Examples:

    .. code-block:: python

        >>> dde.backend.backend()
        ("tensorflow", 1.14.0)
    """
    return _BACKEND, _VERSION


def is_tf_1():
    return _IS_TF_1
