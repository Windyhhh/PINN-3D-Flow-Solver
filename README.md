# 🌊 PINN 3D Flow Solver | PINN 3D 流场求解器

> **3D flow field solver using Physics-Informed Neural Networks (PINN). Solve Navier-Stokes equations with neural networks, no mesh required. 3D velocity/pressure field prediction, boundary conditions, and visualization.**
>
> 基于物理信息神经网络（PINN）的 3D 流场求解器。用神经网络求解 Navier-Stokes 方程，无需网格。3D 速度/压力场预测、边界条件和可视化。

---

## 🌟 Features | 核心特性

- **Physics-Informed NN** — Embed PDEs into neural network loss
- **Navier-Stokes** — 3D incompressible flow equations
- **Mesh-Free** — No computational mesh required
- **3D Fields** — Velocity (u,v,w) + pressure (p)
- **Boundary Conditions** — Dirichlet, Neumann, periodic
- **Visualization** — 3D flow field rendering, streamlines
- **PyTorch** — Modern deep learning framework

---

## 🚀 Quick Start | 快速开始

```bash
pip install torch numpy matplotlib mayavi

# Train PINN for 3D flow
python train.py --geometry cylinder --Re 100 --epochs 10000

# Solve and visualize
python solve.py --model best_model.pth --output results/

# Generate streamlines
python visualize.py --field results/flow_field.npy --streamlines
```

---

## 🔬 Physics | 物理方程

### Navier-Stokes Equations | 纳维-斯托克斯方程

```
Continuity:  ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
Momentum x:  ∂u/∂t + u∂u/∂x + v∂u/∂y + w∂u/∂z = -1/ρ ∂p/∂x + ν∇²u
Momentum y:  ∂v/∂t + u∂v/∂x + v∂v/∂y + w∂v/∂z = -1/ρ ∂p/∂y + ν∇²v
Momentum z:  ∂w/∂t + u∂w/∂x + v∂w/∂y + w∂w/∂z = -1/ρ ∂p/∂z + ν∇²w
```

### PINN Loss | PINN 损失函数

```
L = λ_data * MSE(u_pred, u_data)
  + λ_pde  * MSE(NS_residual, 0)
  + λ_bc   * MSE(boundary_residual, 0)
```

---

## 📄 License | 许可证

MIT License.

[GitHub](https://github.com/Windyhhh/PINN-3D-Flow-Solver)
