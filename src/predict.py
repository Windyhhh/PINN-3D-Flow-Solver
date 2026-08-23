# -*- coding: utf-8 -*-
"""
重构版预测模块 - 统一管理预测流程
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from pinns import NavierStokes3DPINNs
import time


def predict_flow_3d(data_pathname, data_filename, number, dt, filepath,
                    nlevels, alphas,  hp, save_file, dd, zd, model):
    """
    3D流场预测函数

    参数:
        data_pathname: 数据文件路径
        data_filename: 数据文件名
        number: 时间点数
        dt: 时间间隔
        filepath: 结果保存路径
        nlevels: 噪声水平列表
        alphas: 正则化参数列表
        N_cell: 神经元数量
        hp: 超参数字典
        save_file: 模型保存路径
        dd: 时间点索引 (默认5)
        zd: z轴索引 (默认0)
    """
    # 加载数据
    print(f"加载3D数据: {os.path.join(data_pathname, data_filename)}")
    wohuan_data = h5py.File(os.path.join(data_pathname, data_filename), 'r')

    # 数据处理
    piv_xmesh = np.transpose(wohuan_data['x'])
    piv_ymesh = np.transpose(wohuan_data['y'])
    piv_zmesh = np.transpose(wohuan_data['z'])
    piv_u = np.transpose(wohuan_data['fu_col'])
    piv_v = np.transpose(wohuan_data['fv_col'])
    piv_w = np.transpose(wohuan_data['fw_col'])

    # 提取空间维度
    sizD = piv_xmesh.shape
    mask = np.zeros((sizD[0], sizD[1], sizD[2]))

    # 时间向量
    tvec = np.linspace(0, dt * (number - 1), number)
    sizT = tvec.size

    # 保存原始数据
    all_data_u = piv_u[:, :, :, 0:number]
    all_data_v = piv_v[:, :, :, 0:number]
    all_data_w = piv_w[:, :, :, 0:number]
    all_data_p = np.zeros_like(all_data_u)

    # 使用HDF5格式保存原始数据
    filename = f"{hp['savename']}_exact.h5"
    with h5py.File(os.path.join(filepath, filename), 'w') as f:
        f.create_dataset('xmesh', data=piv_xmesh, compression='gzip')
        f.create_dataset('ymesh', data=piv_ymesh, compression='gzip')
        f.create_dataset('zmesh', data=piv_zmesh, compression='gzip')
        f.create_dataset('all_data_u', data=all_data_u, compression='gzip')
        f.create_dataset('all_data_v', data=all_data_v, compression='gzip')
        f.create_dataset('all_data_w', data=all_data_w, compression='gzip')
        f.create_dataset('all_data_p', data=all_data_p, compression='gzip')
    print(f"原始数据已保存至: {os.path.join(filepath, filename)}")

    # 加载归一化参数
    save_file = f"{save_file}"
    print(f"加载归一化参数: {save_file}_paras.mat")
    domain = sio.loadmat(save_file + '_paras.mat', squeeze_me=True)
    norm_paras = domain['norm_paras']
    hp['norm_paras'] = norm_paras

    # 展平坐标数据
    x_data = piv_xmesh.flatten()[:, None]
    y_data = piv_ymesh.flatten()[:, None]
    z_data = piv_zmesh.flatten()[:, None]

    # 初始化预测结果数组
    all_data_u_pred = np.zeros((sizD[0], sizD[1], sizD[2], sizT))
    all_data_v_pred = np.zeros((sizD[0], sizD[1], sizD[2], sizT))
    all_data_w_pred = np.zeros((sizD[0], sizD[1], sizD[2], sizT))
    all_data_p_pred = np.zeros((sizD[0], sizD[1], sizD[2], sizT))

    # 预测过程
    start_time = time.time()
    print(f"开始3D流场预测...")

    for nlevel in nlevels:
        for alpha in alphas:
            hp['alpha'] = alpha

            # 创建模型
            print(f"创建3D PINN模型 (alpha={alpha}, nlevel={nlevel})")
            data = np.zeros([10, 8])  # 占位数据
            eqns = np.zeros([10, 4])  # 占位方程
            pinn_model = NavierStokes3DPINNs(hp, data, eqns)

            count = -1
            for tt in tvec:
                count += 1
                t_pred = tt * np.ones_like(x_data)
                pred = np.concatenate((t_pred, x_data, y_data, z_data), axis=1)

                # 预测
                u_pred, v_pred, w_pred, p_pred = pinn_model.predict(pred)

                if count % max(1, sizT // 10) == 0:
                    progress = (count + 1) / sizT * 100
                    print(f"进度: {progress:.1f}% (时间点 {count + 1}/{sizT})")

                # 保存预测结果
                all_data_u_pred[:, :, :, count] = u_pred.reshape(sizD[0], sizD[1], sizD[2])
                all_data_v_pred[:, :, :, count] = v_pred.reshape(sizD[0], sizD[1], sizD[2])
                all_data_w_pred[:, :, :, count] = w_pred.reshape(sizD[0], sizD[1], sizD[2])
                all_data_p_pred[:, :, :, count] = p_pred.reshape(sizD[0], sizD[1], sizD[2])

            # 使用HDF5格式保存预测数据
            filename = f"{hp['savename']}_predict.h5"
            try:
                with h5py.File(os.path.join(filepath, filename), 'w') as f:
                    f.create_dataset('xmesh', data=piv_xmesh, compression='gzip')
                    f.create_dataset('ymesh', data=piv_ymesh, compression='gzip')
                    f.create_dataset('zmesh', data=piv_zmesh, compression='gzip')
                    f.create_dataset('all_data_u', data=all_data_u_pred, compression='gzip')
                    f.create_dataset('all_data_v', data=all_data_v_pred, compression='gzip')
                    f.create_dataset('all_data_w', data=all_data_w_pred, compression='gzip')
                    f.create_dataset('all_data_p', data=all_data_p_pred, compression='gzip')
                    f.create_dataset('mask', data=mask, compression='gzip')
                print(f"预测数据已保存至: {os.path.join(filepath, filename)}")
            except Exception as e:
                print(f"HDF5保存失败: {e}")
                # 备用方案：尝试分块保存
                print("尝试分块保存...")
                self._save_large_data_chunked(filepath, filename.replace('.h5', '_chunked.h5'), 
                                            piv_xmesh, piv_ymesh, piv_zmesh,
                                            all_data_u_pred, all_data_v_pred, 
                                            all_data_w_pred, all_data_p_pred, mask)

    end_time = time.time()
    print(f"3D预测完成! 用时: {end_time - start_time:.2f}秒")

    # 绘制对比图
    print("绘制结果对比图...")
    umin, umax = np.min(piv_u), np.max(piv_u)
    vmin, vmax = np.min(piv_v), np.max(piv_v)
    wmin, wmax = np.min(piv_w), np.max(piv_w)
    pmin, pmax = np.min(all_data_p_pred), np.max(all_data_p_pred)

    plt.figure(figsize=(18, 12))

    # 原始数据
    ax1 = plt.subplot(2, 4, 1)
    p1 = ax1.pcolor(piv_xmesh[:, :, zd], piv_ymesh[:, :, zd], all_data_u[:, :, zd, dd],
                    cmap='RdYlGn_r', shading='auto', vmin=umin, vmax=umax)
    ax1.set_title(r'$u_{ori}$', fontsize=12, color='r')
    ax1.axis('equal')
    plt.colorbar(p1, ax=ax1)

    ax2 = plt.subplot(2, 4, 2)
    p2 = ax2.pcolor(piv_xmesh[:, :, zd], piv_ymesh[:, :, zd], all_data_v[:, :, zd, dd],
                    cmap='RdYlGn_r', shading='auto', vmin=vmin, vmax=vmax)
    ax2.set_title(r'$v_{ori}$', fontsize=12, color='r')
    ax2.axis('equal')
    plt.colorbar(p2, ax=ax2)

    ax3 = plt.subplot(2, 4, 3)
    p3 = ax3.pcolor(piv_xmesh[:, :, zd], piv_ymesh[:, :, zd], all_data_w[:, :, zd, dd],
                    cmap='RdYlGn_r', shading='auto', vmin=wmin, vmax=wmax)
    ax3.set_title(r'$w_{ori}$', fontsize=12, color='r')
    ax3.axis('equal')
    plt.colorbar(p3, ax=ax3)

    ax4 = plt.subplot(2, 4, 4)
    p4 = ax4.pcolor(piv_xmesh[:, :, zd], piv_ymesh[:, :, zd], all_data_p[:, :, zd, dd],
                    cmap='RdYlGn_r', shading='auto', vmin=pmin, vmax=pmax)
    ax4.set_title(r'$p_{ori}$', fontsize=12, color='r')
    ax4.axis('equal')
    plt.colorbar(p4, ax=ax4)

    # 预测数据
    ax5 = plt.subplot(2, 4, 5)
    p5 = ax5.pcolor(piv_xmesh[:, :, zd], piv_ymesh[:, :, zd], all_data_u_pred[:, :, zd, dd],
                    cmap='RdYlGn_r', shading='auto', vmin=umin, vmax=umax)
    ax5.set_title(r'$u_{pre}$', fontsize=12, color='b')
    ax5.axis('equal')
    plt.colorbar(p5, ax=ax5)

    ax6 = plt.subplot(2, 4, 6)
    p6 = ax6.pcolor(piv_xmesh[:, :, zd], piv_ymesh[:, :, zd], all_data_v_pred[:, :, zd, dd],
                    cmap='RdYlGn_r', shading='auto', vmin=vmin, vmax=vmax)
    ax6.set_title(r'$v_{pre}$', fontsize=12, color='b')
    ax6.axis('equal')
    plt.colorbar(p6, ax=ax6)

    ax7 = plt.subplot(2, 4, 7)
    p7 = ax7.pcolor(piv_xmesh[:, :, zd], piv_ymesh[:, :, zd], all_data_w_pred[:, :, zd, dd],
                    cmap='RdYlGn_r', shading='auto', vmin=wmin, vmax=wmax)
    ax7.set_title(r'$w_{pre}$', fontsize=12, color='b')
    ax7.axis('equal')
    plt.colorbar(p7, ax=ax7)

    ax8 = plt.subplot(2, 4, 8)
    p8 = ax8.pcolor(piv_xmesh[:, :, zd], piv_ymesh[:, :, zd], all_data_p_pred[:, :, zd, dd],
                    cmap='RdYlGn_r', shading='auto', vmin=pmin, vmax=pmax)
    ax8.set_title(r'$p_{pre}$', fontsize=12, color='b')
    ax8.axis('equal')
    plt.colorbar(p8, ax=ax8)

    plt.subplots_adjust(wspace=0.5, hspace=0.3)

    # 保存并显示图像
    img_path = os.path.join(filepath, f"{hp['savename']}_3D_compare.png")
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {img_path}")
    plt.show()

    return all_data_u_pred, all_data_v_pred, all_data_w_pred, all_data_p_pred


