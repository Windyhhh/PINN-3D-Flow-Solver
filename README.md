# 🌊 PINN 3D 流场求解器 | PINN 3D Flow Field Solver

> **物理信息神经网络求解 3D 各向同性湍流——无需网格划分，PDE 约束嵌入损失，直接从稀疏数据重建三维流场。**
>
> *Physics-Informed Neural Network for 3D isotropic turbulence — no mesh needed, PDE constraints embedded in loss, reconstruct 3D flow field from sparse data.*

---

## ⭐ 核心卖点 | Why Star This

| 卖点 | Feature | 一句话 |
|------|---------|--------|
| 🧠 **物理嵌入** | Physics-Informed | Navier-Stokes 方程作为损失约束 |
| 🧊 **3D 流场** | 3D Flow Field | 三维各向同性湍流的速度场 (u,v,w) 求解 |
| 📉 **数据高效** | Data-Efficient | 稀疏 PIV 数据即可训练，无需全流场采样 |
| ⚡ **无网格** | Mesh-Free | 神经网络连续表示，不受网格分辨率限制 |
| 🎯 **可复现** | Reproducible | 完整训练代码 + 预训练权重 + 实验数据 |

---

## 🏆 技术栈 | Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange?logo=tensorflow)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-red?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-green?logo=plotly)

---

## 📊 方法对比 | Method Comparison

| 方法 | 网格需求 | 数据需求 | 物理一致性 | 3D 可扩展性 |
|------|---------|---------|-----------|------------|
| CFD (Fluent) | 🔴 高 | 🟢 低 | ✅ 强 | 🟡 计算量大 |
| 纯数据驱动 NN | 🟢 无 | 🔴 高 | ❌ 无 | ✅ 好 |
| **PINN (本项目)** | 🟢 无 | 🟡 中 | ✅ 强 | ✅ 好 |

---

## 🚀 快速开始 | Quick Start

```bash
git clone https://github.com/Windyhhh/PINN-3D-Flow-Solver.git
cd PINN-3D-Flow-Solver
pip install -r requirements.txt

# 训练
python src/train.py --data Data/Isotropic\ Flow/ --epochs 10000

# 预测/可视化
python src/predict.py --weights weights/Isotropicflow.weights.h5
```

---

## 📂 项目结构 | Project Structure

```
PINN-3D-Flow-Solver/
├── src/
│   ├── main.py                # 主入口
│   ├── train.py               # 训练脚本
│   ├── predict.py             # 预测/可视化
│   ├── pinns.py               # PINN 模型核心
│   ├── datagenerator.py       # 数据生成器
│   ├── custom_lbfgs.py        # 自定义 L-BFGS 优化器
│   ├── maps.py                # 特征映射
│   ├── logger.py              # 日志
│   └── userbackend.py         # 后端工具
├── Data/
│   └── Isotropic Flow/        # 各向同性湍流数据
├── weights/                   # 预训练权重
│   ├── Isotropicflow.weights.h5
│   └── Isotropicflow_loss.png
├── tests/                     # 测试脚本
├── requirements.txt           # 依赖
└── README.md
```

---

## 🔬 核心原理 | Core Idea

### 物理信息神经网络 | Physics-Informed Neural Network

```
输入: (x, y, z, t)  →  神经网络  →  输出: (u, v, w, p)

损失函数:
L = L_data + λ₁·L_NS + λ₂·L_BC + λ₃·L_IC

L_data = MSE(NN(x,y,z,t) - PIV_observed)        # 数据拟合
L_NS   = MSE(∂u/∂t + u∂u/∂x + ... - ν∇²u + ∇p/ρ)  # N-S 方程残差
L_BC   = 边界条件残差
L_IC   = 初始条件残差
```

### Navier-Stokes 约束 | Navier-Stokes Constraints

```
连续性方程:  ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
动量方程:    ∂u/∂t + u∂u/∂x + v∂u/∂y + w∂u/∂z = -1/ρ ∂p/∂x + ν∇²u
            (v, w 方向同理)
```

---

## 🎯 应用场景 | Use Cases

- ✈️ **航空航天**：飞行器周围流场的快速预测
- 🚗 **汽车工业**：车身空气动力学优化
- 🏭 **能源工程**：涡轮机、管道内的流动分析
- 🌊 **海洋工程**：海洋环流、波浪传播模拟
- 🏥 **生物医学**：血管内血液流动分析
- 🌪️ **气象预测**：大气流动的快速模拟

---

## 📚 参考文献 | References

- Raissi, M., et al. "Physics-informed neural networks." JCP 2019.
- Cai, S., et al. "Physics-informed neural networks for heat transfer problems." JHT 2021.
- Karniadakis, G. E., et al. "Physics-informed machine learning." Nature Reviews Physics 2021.

---

## 📄 License

MIT License — 自由使用、修改和分发。

---

> 💡 **物理 + AI 求解 3D 湍流，Star ⭐ 支持开源计算流体力学！**
