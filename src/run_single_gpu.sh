#!/bin/bash
# 单卡训练启动脚本
# 用法: ./run_single_gpu.sh [GPU_ID]
# 示例: ./run_single_gpu.sh 1  # 使用GPU 1

# 默认使用GPU 1（因为GPU 0通常被其他进程占用）
GPU_ID=${1:-1}

echo "=========================================="
echo "单卡训练启动脚本"
echo "=========================================="
echo "目标GPU: GPU:$GPU_ID"
echo ""

# 检查nvidia-smi是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: nvidia-smi 未找到，请确保NVIDIA驱动已安装"
    exit 1
fi

# 显示目标GPU信息
echo "GPU信息:"
nvidia-smi -i $GPU_ID --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv

echo ""
echo "GPU上的进程:"
nvidia-smi -i $GPU_ID --query-compute-apps=pid,process_name,used_memory --format=csv

echo ""
echo "=========================================="
echo "启动训练程序..."
echo "=========================================="

# 设置环境变量并启动训练
export TARGET_GPU_ID=$GPU_ID
python main.py

