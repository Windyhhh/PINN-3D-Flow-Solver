
# -*- coding: utf-8 -*-
"""
Created Jun14 2021

@author: H.P. Wang
github:  https://github.com/hpwang87
"""

import numpy as np
import random
import h5py
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
import scipy.io as sio




def wohuan3D_DataGenerator(data_pathname, data_filename, norm_mode="minmax"):
    """
    3D data generator with optional normalization modes:
    norm_mode = "minmax" / "standard" / "none"
    """
    # ===== 读取数据 =====
    data_file = h5py.File(os.path.join(data_pathname, data_filename),'r')

    piv_xmesh = np.transpose(data_file['x_remain'])
    piv_ymesh = np.transpose(data_file['y_remain'])
    piv_zmesh = np.transpose(data_file['z_remain'])
    piv_tmesh = np.transpose(data_file['t_remain'])
    piv_u = np.transpose(data_file['u_remain'])
    piv_v = np.transpose(data_file['v_remain'])
    piv_w = np.transpose(data_file['w_remain'])

    dim_flag = '3d3c'
        
    # flatten
    x_data = piv_xmesh.flatten()[:,None]
    y_data = piv_ymesh.flatten()[:,None]
    z_data = piv_zmesh.flatten()[:,None]
    t_data = piv_tmesh.flatten()[:,None]
    u_data = piv_u.flatten()[:,None]
    v_data = piv_v.flatten()[:,None]
    w_data = piv_w.flatten()[:,None]
    p_data = np.zeros_like(u_data)

    # ========== data sampling ==========
    N_data = 10_000_000
    if x_data.shape[0] <= N_data:
        data = np.concatenate((t_data, x_data, y_data, z_data, 
                               u_data, v_data, w_data, p_data), 1)
    else:
        idx = np.random.choice(x_data.shape[0], N_data, replace=True)
        data = np.concatenate((t_data[idx,:], x_data[idx,:], y_data[idx,:], z_data[idx,:],
                               u_data[idx,:], v_data[idx,:], w_data[idx,:], p_data[idx,:]), 1)

    # free memory
    del x_data, y_data, z_data, t_data, u_data, v_data, w_data, p_data
    
    # ========== boundary points ==========
    x_bc = np.transpose(data_file['x_remain'])
    y_bc = np.transpose(data_file['y_remain'])
    z_bc = np.transpose(data_file['z_remain'])
    t_bc = np.transpose(data_file['t_remain'])

    u_bc = np.zeros_like(x_bc)
    v_bc = np.zeros_like(x_bc)
    w_bc = np.zeros_like(x_bc)
    p_bc = np.zeros_like(x_bc)

    bc = np.concatenate((t_bc, x_bc, y_bc, z_bc, 
                         u_bc, v_bc, w_bc, p_bc), 1)

    # free mem
    del x_bc, y_bc, z_bc, t_bc, u_bc, v_bc, w_bc, p_bc
    
    # ========== equation points ==========
    x_eqn = np.transpose(data_file['x_remain'])
    y_eqn = np.transpose(data_file['y_remain'])
    z_eqn = np.transpose(data_file['z_remain'])
    t_eqn = np.transpose(data_file['t_remain'])

    eqns = np.concatenate((t_eqn, x_eqn, y_eqn, z_eqn), 1)

    # free mem
    del x_eqn, y_eqn, z_eqn, t_eqn


    #  关键部分：根据 norm_mode 生成 norm_paras

    norm_paras = np.zeros((2,8))

    if norm_mode == "minmax":
        # 0~3: t,x,y,z → min/max
        norm_paras[0,0:4] = eqns.min(axis=0)
        norm_paras[1,0:4] = eqns.max(axis=0)

        # 4~7: u,v,w,p → min/max
        norm_paras[0,4:8] = np.min(data[:,4:8], axis=0)
        norm_paras[1,4:8] = np.max(data[:,4:8], axis=0)

    elif norm_mode == "standard":
        # 0~3: min/max for coordinates (建议不变）
        norm_paras[0,0:4] = eqns.min(axis=0)
        norm_paras[1,0:4] = eqns.max(axis=0)

        # 4~7: mean / std
        norm_paras[0,4:8] = np.mean(data[:,4:8], axis=0)    # mean
        norm_paras[1,4:8] = np.std(data[:,4:8], axis=0)     # std
        
    elif norm_mode == "input_only":
        # 0~3: identity mapping
        norm_paras[0,0:4] = eqns.min(axis=0)
        norm_paras[1,0:4] = eqns.max(axis=0)

        # 4~7: identity mapping → Y_pred 直接是物理量
        norm_paras[0,4:8] = 0.0   # vmin
        norm_paras[1,4:8] = 1.0   # vmax
        
    elif norm_mode == "none":
        # 0~3: identity mapping
        norm_paras[0,0:4] = 0.0
        norm_paras[1,0:4] = 1.0

        # 4~7: identity mapping → Y_pred 直接是物理量
        norm_paras[0,4:8] = 0.0   # vmin
        norm_paras[1,4:8] = 1.0   # vmax

    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    return data, eqns, bc, dim_flag, norm_paras
