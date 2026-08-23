"""
训练模块 - 3D流场PINN训练
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'FALSE'  # 禁止绕过MKL冲突

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
import h5py
import time
import gc
from tensorflow.keras import backend as K

from datagenerator import wohuan3D_DataGenerator
from pinns import NavierStokes3DPINNs


# 数据生成器映射
DATA_GENERATOR_MAP = {
    "wohuan3D_DataGenerator": wohuan3D_DataGenerator
}



def train_flow_3d(data_pathname, data_filename, nlevels, alphas, N_cell, hp, save_file, data_generator_name):
    """
    3D 流场训练函数：支持断点续训 + 固定间隔记录损失日志 + 保存权重
    """

    # ======================= 基础设置 ==========================
    SAVE_INTERVAL = 10                                                     #  每隔多少 epoch 保存一次损失到 txt

    # 选择数据生成器
    data_generator = DATA_GENERATOR_MAP[data_generator_name]

    # ========== 生成数据 ==========
    norm_mode = hp['norm_mode']
    data, eqns, bc, dim_flag, norm_paras = data_generator(data_pathname, data_filename,norm_mode=norm_mode)

    # 更新 HP 结构
    hp['dim_flag'] = dim_flag
    hp['norm_paras'] = norm_paras

    # 权重保存路径
    weights_dir = hp['weights_dir']
    savename = hp['savename']
    weight_path = os.path.join(weights_dir, savename)

    log_file = weight_path + "_log.txt"                                     # 间隔损失日志
    status_file = weight_path + "_status.txt"                               # 保存累计 epoch，为断点训练提供

    print(f"\n【权重保存路径】: {weight_path}\n")

    # ======================= 创建模型实例 ==========================
    # 关键说明：
    # - 如果 hp['ExistModel'] == 1，NavierStokes3DPINNs 内部会调用 loadNN() 加载权重
    # - 如果 hp['ExistModel'] == 0，则创建全新模型
    # - 此处不再重复加载权重，避免双重加载导致的显存浪费
    pinn_model = NavierStokes3DPINNs(hp, data, eqns)

    # ========== 读取累计训练 epoch（如果有续训）==========
    global_epoch = 0
    if os.path.exists(status_file):
        with open(status_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "last_global_epoch" in line:
                    global_epoch = int(line.split("=")[1])
        print(f"✔ 已恢复训练状态：累计训练 {global_epoch} 个 epoch\n")

    # ========== 断点续训状态提示 ==========
    # 注意：权重加载已由 NavierStokes3DPINNs 的 __init__ 方法处理（当 ExistModel=1 时）
    # 此处仅做状态提示，不再重复加载权重
    if hp.get('ExistModel', 0) == 1:
        print("✔ 已通过 ExistModel=1 加载历史权重，优化器已重新初始化，开始断点续训...\n")
    elif os.path.exists(weight_path + ".index"):
        # 仅当 ExistModel=0 但权重文件存在时（用户选择从零训练但有残留权重），给出提示
        print("⚠ 检测到已有权重文件，但用户选择从零开始训练，将使用新初始化的模型。\n")
    else:
        print("无权重文件，从零开始训练。\n")


    # ======================= 正式训练 ==========================
    for nlevel in nlevels:
        for alpha in alphas:

            hp['alpha'] = alpha

            print(f"\n===== 开始训练：alpha={alpha}, nlevel={nlevel} =====")

            # 计时
            t_start = time.time()

            # 调用 PINN 模型训练一次（内部含多个 epoch）
            H = pinn_model.train()

            # ======================================================
            #   固定间隔记录损失日志（Append 模式，不覆盖文件）
            # ======================================================

            for i in range(len(H.history["loss"])):
                global_epoch += 1                                                #  全局 epoch 自增，

                if global_epoch % SAVE_INTERVAL == 0:
                    total_loss = float(H.history["loss"][i])
                    data_loss  = float(H.history["loss_fn_data"][i])
                    eqns_loss  = float(H.history["loss_fn_eqns"][i])

                    with open(log_file, "a") as f:
                        f.write(f"epoch={global_epoch}  "
                                f"total={total_loss:.6e}  "
                                f"data={data_loss:.6e}  "
                                f"eqns={eqns_loss:.6e}\n")

                    print(f"✔ 已记录损失：epoch={global_epoch}, total={total_loss:.3e}")

            # 单次训练耗时
            t_end = time.time()
            print(f"训练时间: {t_end - t_start:.2f} 秒")

            # ======================================================
            #   保存 Loss 曲线（当前这次训练的损失）
            # ======================================================
            plt.figure()
            plt.yscale("log")
            N = np.arange(0, len(H.history['loss']))
            plt.plot(N, H.history['loss'], label='total_loss')
            plt.plot(N, H.history['loss_fn_data'], label='data_loss')
            plt.plot(N, H.history['loss_fn_eqns'], label='eqns_loss')
            plt.plot(N, H.history['loss_fn_conds'], label='bc_loss')
            plt.grid(linestyle='-.')
            plt.legend()
            plt.savefig(weight_path + '_loss.png', dpi=300)
            plt.close()

            # ======================================================
            #   保存权重（Keras 3 使用 .weights.h5 格式）
            # ======================================================
            pinn_model.model.save_weights(weight_path + ".weights.h5")
            print(f"✔ 权重已保存：{weight_path}.weights.h5")

            # ======================================================
            #   保存训练状态（累计 epoch）
            # ======================================================
            with open(status_file, "w") as f:
                f.write(f"last_global_epoch={global_epoch}\n")

            print(f"✔ 状态文件已更新（累计 epoch={global_epoch}）\n")

    return pinn_model
