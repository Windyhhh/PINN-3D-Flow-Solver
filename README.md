<div align="center">

# 🔬 PINN-3D-Flow-Solver

### Physics-informed neural networks for 3D CFD.

A PINN-based solver for 3D computational fluid dynamics with custom L-BFGS and data generation.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

---

**PINN-3D-Flow-Solver** applies **physics-informed neural networks (PINN)** to 3D computational fluid dynamics — with data generation, a custom **L-BFGS** optimizer, and prediction modules.

> [!NOTE]
> 中文项目：PINN 3D 流场求解器——物理信息神经网络用于 3D 计算流体力学。

---

## Quickstart

```bash
git clone https://github.com/Windyhhh/PINN-3D-Flow-Solver.git
cd PINN-3D-Flow-Solver

pip install -r requirements.txt
pip install -e .

# train
python src/main.py

# predict
python src/predict.py
```

---

## Features

- **PINN solver** — physics-informed loss for 3D flow.
- **Custom L-BFGS** — `custom_lbfgs.py` optimizer.
- **Data generation** — `datagenerator.py` for training data.

---

## Project Structure

```
PINN-3D-Flow-Solver/
├── src/
│   ├── main.py           # train entry
│   ├── pinns.py          # PINN model
│   ├── datagenerator.py  # data
│   ├── custom_lbfgs.py   # optimizer
│   ├── predict.py        # inference
│   └── maps.py, logger.py
├── Data/                 # isotropic flow data
└── setup.py
```

---

## License

MIT — free to use, modify and distribute.
