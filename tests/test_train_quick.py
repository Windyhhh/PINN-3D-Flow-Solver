#!/usr/bin/env python3
"""
快速训练测试脚本 - 验证GPU训练是否可用
运行时间: 约 2-3 分钟
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加Code目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Code'))

import train
from pinns import NavierStokes3DPINNs

print("=" * 70)
print("🚀 PINN 快速训练测试")
print("=" * 70)

# 配置参数
TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), "Data", "Isotropic Flow")
TRAIN_DATA_FILE = "Downsampled_PIV_R4_T1-20_Int1.mat-PY.mat"
SAVE_FILE = os.path.join(os.path.dirname(__file__), "Code", "weights", "Isotropicflow_test")
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "Code", "weights")

# 超参数 - 缩小规模用于快速测试
HP_TRAIN = {
    'layers': [4] + 3 * [64] + [4],  # 3层而不是15层
    'ExistModel': 0,  # 从零开始
    'train': True,
    'map_name': 'rnn',
    'savename': 'Isotropicflow_test',
    'norm_mode': 'input_only',
    'Re': 433,
    'alpha': 0,
    'z_weight': 0,
    'nt_lr': 0.4,
    'nt_max_iternum': 0,  # 关闭L-BFGS
    'nt_steps_per_loop': 20,
    'nt_batch_size': 32,
    'tf_epochs': 1,  # 仅1个epoch用于快速测试
    'tf_initial_epoch': 0,
    'tf_steps_per_epoch': 2,  # 仅2步
    'tf_batch_size': 128,
    'tf_init_lr': 3e-4,
    'base_path': os.path.join(os.path.dirname(__file__), 'Code'),
    'save_file': SAVE_FILE,
    'weights_dir': WEIGHTS_DIR,
    'norm_paras': np.zeros((2, 8), dtype='float32')  # 虚拟归一化参数
}

print("\n1️⃣  检查GPU")
print("-" * 70)
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f"✅ 检测到 {len(gpus)} 个GPU")
for i, gpu in enumerate(gpus[:4]):  # 显示前4个
    print(f"   GPU {i}: {gpu}")

print("\n2️⃣  检查数据文件")
print("-" * 70)
data_file = os.path.join(TRAIN_DATA_PATH, TRAIN_DATA_FILE)
if os.path.exists(data_file):
    print(f"✅ 数据文件存在: {data_file}")
    print(f"   文件大小: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB")
else:
    print(f"❌ 数据文件不存在: {data_file}")
    sys.exit(1)

print("\n3️⃣  运行快速训练测试（1个epoch, 2步）")
print("-" * 70)

try:
    # 调用训练函数
    train.train_flow_3d(
        data_pathname=TRAIN_DATA_PATH,
        data_filename=TRAIN_DATA_FILE,
        nlevels=[0],
        alphas=[0],
        N_cell=64,  # 缩小网络
        hp=HP_TRAIN,
        data_generator_name="wohuan3D_DataGenerator",
        save_file=SAVE_FILE,
    )
    
    print("\n✅ 训练成功完成！")
    print("\n" + "=" * 70)
    print("🎉 恭喜！您的GPU已成功配置，可以开始训练项目了！")
    print("=" * 70)
    print("\n📚 下一步:")
    print("   1. 编辑 Code/main.py 调整训练参数")
    print("   2. 运行: python3 Code/main.py")
    print("   3. 选择训练模式和参数")
    print("\n💾 监控训练进度:")
    print("   • 权重保存位置: Code/weights/")
    print("   • 使用 nvidia-smi -l 1 监控显存")
    
except Exception as e:
    print(f"\n❌ 训练出错: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
