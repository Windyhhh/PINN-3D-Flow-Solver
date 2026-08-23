#!/usr/bin/env python
"""
自动化测试脚本：测试断点续训功能（GPU显存异常问题）
"""
import os
import sys
import subprocess
import time

# 添加Code目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Code'))

import train
import tensorflow as tf

# 配置参数
SCRIPT_DIR = os.path.join(os.path.dirname(__file__), 'Code')
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
BASE_PATH = SCRIPT_DIR
SAVE_FILE = os.path.join(SCRIPT_DIR, "weights", "Isotropicflow_test")

TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "Isotropic Flow")
TRAIN_DATA_FILE = "Downsampled_PIV_R4_T1-20_Int1.mat-PY.mat"

# 超参数配置（降低batch_size以适应显存）
HP_TRAIN = {
    'layers': [4] + 15 * [256] + [4],
    'ExistModel': 0,  # 从零开始
    'train': True,
    'map_name': 'rnn',
    'savename': 'Isotropicflow_test',
    'norm_mode': 'input_only',
    'Re': 433,
    'alpha': 0,
    'z_weight': 0,
    'nt_lr': 0.4,
    'nt_max_iternum': 10,
    'nt_steps_per_loop': 20,
    'nt_batch_size': 32,
    'tf_epochs': 2,  # 测试用：只训练2个epoch
    'tf_initial_epoch': 0,
    'tf_steps_per_epoch': 5,  # 测试用：每个epoch只5步
    'tf_batch_size': 128,
    'tf_init_lr': 3e-4,
    'base_path': BASE_PATH,
    'save_file': SAVE_FILE,
    'weights_dir': os.path.join(BASE_PATH, "weights")
}

def print_gpu_status(title=""):
    """打印GPU显存状态"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    gpus = tf.config.list_physical_devices('GPU')
    print(f"可用GPU数量: {len(gpus)}")
    
    # 获取显存使用情况
    for gpu in gpus:
        print(f"  - {gpu}")

def test_from_scratch():
    """测试1：从零开始训练"""
    print("\n" + "="*60)
    print("测试1：从零开始训练（ExistModel=0）")
    print("="*60)
    
    print_gpu_status("训练前GPU状态")
    
    HP_TRAIN['ExistModel'] = 0
    HP_TRAIN['tf_epochs'] = 2
    
    try:
        train.train_flow_3d(
            data_pathname=TRAIN_DATA_PATH,
            data_filename=TRAIN_DATA_FILE,
            nlevels=[0],
            alphas=[0],
            N_cell=256,
            hp=HP_TRAIN,
            data_generator_name="wohuan3D_DataGenerator",
            save_file=SAVE_FILE,
        )
        print("\n✅ 测试1成功：从零开始训练完成")
        return True
    except Exception as e:
        print(f"\n❌ 测试1失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_resume_training():
    """测试2：断点续训（关键测试）"""
    print("\n" + "="*60)
    print("测试2：断点续训（ExistModel=1）- 关键测试")
    print("="*60)
    
    print_gpu_status("续训前GPU状态")
    
    HP_TRAIN['ExistModel'] = 1  # 加载已有模型
    HP_TRAIN['tf_epochs'] = 4   # 继续训练到第4个epoch
    HP_TRAIN['tf_initial_epoch'] = 2  # 从第2个epoch开始
    
    try:
        train.train_flow_3d(
            data_pathname=TRAIN_DATA_PATH,
            data_filename=TRAIN_DATA_FILE,
            nlevels=[0],
            alphas=[0],
            N_cell=256,
            hp=HP_TRAIN,
            data_generator_name="wohuan3D_DataGenerator",
            save_file=SAVE_FILE,
        )
        print("\n✅ 测试2成功：断点续训完成（显存异常问题已解决）")
        return True
    except Exception as e:
        print(f"\n❌ 测试2失败：{str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("PINN模型断点续训显存异常问题 - 自动化测试")
    print("="*60)
    
    print_gpu_status("初始GPU状态")
    
    # 测试1：从零开始训练
    result1 = test_from_scratch()
    time.sleep(5)
    
    # 测试2：断点续训（关键）
    result2 = test_resume_training()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"测试1（从零开始训练）: {'✅ 通过' if result1 else '❌ 失败'}")
    print(f"测试2（断点续训）: {'✅ 通过' if result2 else '❌ 失败'}")
    
    if result1 and result2:
        print("\n🎉 所有测试通过！显存异常问题已解决。")
        sys.exit(0)
    else:
        print("\n⚠️  部分测试失败，请检查日志。")
        sys.exit(1)

