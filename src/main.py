"""
主程序
"""
import os
import train
import predict
import time
import numpy as np
from pinns import NavierStokes3DPINNs

# ================= 全局配置参数 =================
# 自动获取项目根目录（Code文件夹的父目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

BASE_PATH = SCRIPT_DIR
FLOW_DIM = "3D"
DATA_GENERATOR = "wohuan3D_DataGenerator"
REYNOLDS_NUM = 433

TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "Isotropic Flow")
TRAIN_DATA_FILE = "Downsampled_PIV_R4_T1-20_Int1.mat-PY.mat"
N_LEVELS = [0]
ALPHAS = [0]
Z_WEIGHT = [0]
N_CELL = 256
N_LAYERS = 15

EPOCH_NUM = 3

PREDICT_DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "Isotropic Flow")
PREDICT_DATA_FILE = "Downsampled_PIV_R4_T1-20_Int1.mat-PY.mat"
NUMBER = 20
DT = 1
SAVE_FILE = os.path.join(SCRIPT_DIR, "weights", "Isotropicflow")
PREDICT_SAVE_PATH = os.path.join(SCRIPT_DIR, "predict_results")
TIME_INDEX = 1
SLICE_INDEX = 0

# ================= 模型超参数 =================
# 显存优化说明：
# - tf_batch_size: 从1000降至128，避免3D+二阶导数场景下的显存溢出
# - nt_batch_size: 从100降至32，L-BFGS优化阶段也需要降低batch_size
# - 这些值适配6GB GPU显存，如果GPU显存更大可适当增加
WEIGHTS_DIR = os.path.join(BASE_PATH, "weights")

HP_TRAIN = {
    'layers': [4] + N_LAYERS * [N_CELL] + [4],
    'ExistModel': 1,
    'train': True,
    'map_name': 'rnn',
    'savename': 'Isotropicflow',
    'norm_mode': 'input_only',                          # 'minmax'/ 'standard' / 'none' / input_only
    'Re': REYNOLDS_NUM,
    'alpha': ALPHAS[0],
    'z_weight': Z_WEIGHT[0],
    'nt_lr': 0.4,
    'nt_max_iternum': 10,
    'nt_steps_per_loop': 20,
    'nt_batch_size': 32,                                # 优化：从100降至32
    'tf_epochs': EPOCH_NUM,
    'tf_initial_epoch': 0,
    'tf_steps_per_epoch': 20,
    'tf_batch_size': 128,                               # 优化：从1000降至128（关键！）
    'tf_init_lr': 3e-4,
    'base_path': BASE_PATH,
    'save_file': SAVE_FILE,
    'weights_dir': WEIGHTS_DIR
}

HP_PREDICT = {
    'layers': [4] + N_LAYERS * [N_CELL] + [4],
    'ExistModel': 1,
    'train': False,
    'map_name': 'rnn',
    'savename': 'Isotropicflow',
    'norm_mode': 'minmax',
    'Re': REYNOLDS_NUM,
    'alpha': ALPHAS[0],
    'z_weight': Z_WEIGHT[0],
    'dim_flag': '3d3c',
    'nt_lr': 0.4,
    'nt_max_iternum': 200,
    'nt_steps_per_loop': 1,
    'nt_batch_size': 50000,
    'tf_epochs': 100,
    'tf_initial_epoch': 0,
    'tf_steps_per_epoch': 5,
    'tf_batch_size': 1000,
    'tf_init_lr': 3e-4,
    'base_path': BASE_PATH,
    'weights_dir': WEIGHTS_DIR,
    'save_file': SAVE_FILE
}


# ================= 交互界面（双击运行） =================
def choose_mode():
    print("\n========= 请选择运行模式 =========")
    print("1. 训练（train）")
    print("2. 预测（predict）")
    print("==================================")
    choice = input("请输入数字 1 或 2：")
    
    if choice == "1":
        return "train"
    elif choice == "2":
        return "predict"
    else:
        print("输入无效，默认选择训练模式 train")
        return "train"


def choose_dim():
    print("\n========= 请选择流场维度 =========")
    print("1. 2D")
    print("2. 3D")
    print("=================================")
    choice = input("请输入数字 1 或 2：")
    
    if choice == "1":
        return "2D"
    elif choice == "2":
        return "3D"
    else:
        print("输入无效，默认选择 3D")
        return "3D"

def choose_resume_training():
    print("\n========= 是否继续训练（断点续练） =========")
    print("1. 从已有权重继续训练")
    print("2. 从零开始训练")
    print("==========================================")
    choice = input("请输入数字 1 或 2：")

    if choice == "1":
        return 1   # ExistModel = 1
    elif choice == "2":
        return 0
    else:
        print("输入无效，默认从零开始训练")
        return 0


# ================= 执行主程序 =================
if __name__ == "__main__":

    mode = choose_mode()
    dim = choose_dim()

    print(f"\n===== 开始执行主程序: {mode} 模式 =====")
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"流场维度: {dim}")

    # ========== 训练 ==========
    if mode == 'train':
        print(f"===== 开始{dim}流场训练 =====")

        resume_flag = choose_resume_training()
        HP_TRAIN['ExistModel'] = resume_flag

        if dim == '3D':
            train.train_flow_3d(
                data_pathname=TRAIN_DATA_PATH,
                data_filename=TRAIN_DATA_FILE,
                nlevels=N_LEVELS,
                alphas=ALPHAS,
                N_cell=N_CELL,
                hp=HP_TRAIN,
                data_generator_name=DATA_GENERATOR,
                save_file=SAVE_FILE,
            )
        else:
            # 2D训练功能暂未实现
            print("❌ 错误：2D训练功能暂未实现，请选择3D模式")
            raise NotImplementedError("train_flow_2d 功能暂未实现")

        print(f"===== {dim}流场训练完成 =====")

    # ========== 预测 ==========
    elif mode == 'predict':
        print(f"===== 开始{dim}流场预测 =====")

        if dim == '3D':
            data = np.zeros([10, 8])
            eqns = np.zeros([10, 4])
            model = NavierStokes3DPINNs(HP_PREDICT, data, eqns)

            predict.predict_flow_3d(
                model=model,
                data_pathname=PREDICT_DATA_PATH,
                data_filename=PREDICT_DATA_FILE,
                number=NUMBER,
                dt=DT,
                filepath=PREDICT_SAVE_PATH,
                dd=TIME_INDEX,
                zd=SLICE_INDEX,
                nlevels=N_LEVELS,
                alphas=ALPHAS,
                hp=HP_PREDICT,
                save_file=SAVE_FILE
            )
        else:
            # 2D预测功能暂未实现
            print("❌ 错误：2D预测功能暂未实现，请选择3D模式")
            raise NotImplementedError("predict_flow_2d 功能暂未实现")

        print(f"===== {dim}流场预测完成 =====")

    print("\n===== 程序执行结束 =====")
    input("按回车键关闭窗口...")
