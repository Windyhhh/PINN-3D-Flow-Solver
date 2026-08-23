# 项目结构说明

## 目录结构

```
src/
├── README.md                    # 完整使用指南和优化说明
├── QUICK_START.md               # 快速开始指南
├── OPTIMIZATION_SUMMARY.md      # 优化总结
├── PROJECT_STRUCTURE.md         # 本文件
├── main.py                      # 主程序入口（交互式菜单）
├── train.py                     # 训练流程
├── predict.py                   # 预测模块
├── userbackend.py               # GPU配置和显存优化
├── pinns.py                     # PINN模型核心实现
├── custom_lbfgs.py              # L-BFGS优化器
├── datagenerator.py             # 数据生成器
├── logger.py                    # 日志模块
├── maps.py                      # 映射函数
└── run_single_gpu.py            # 智能GPU选择脚本

Data/
└── Isotropic Flow/
    └── Downsampled_PIV_R4_T1-20_Int1.mat-PY.mat

tests/
├── test_env.py
├── test_resume_training.py
└── test_train_quick.py

weights/
├── Isotropicflow.weights.h5           # 模型权重
├── Isotropicflow_checkpoint.weights.h5 # 检查点权重
├── Isotropicflow_paras.mat            # 归一化参数
├── Isotropicflow_status.txt           # 训练状态
├── Isotropicflow_log.txt              # 训练日志
├── Isotropicflow_loss.png             # 损失曲线
└── Isotropicflow_uvw_range.txt        # 速度范围

docs/                  # 文档目录
config/                # 配置目录
examples/              # 示例目录
```

---

## 核心文件说明

### 主程序

#### `main.py`
- **功能**: 主程序入口，提供交互式菜单
- **菜单选项**:
  1. 训练 (train)
  2. 预测 (predict)
- **流场维度**: 2D / 3D
- **训练模式**: 从零开始 / 断点续训
- **关键参数**: `tf_batch_size=128` (第64行)

#### `train.py`
- **功能**: 训练流程管理
- **主要函数**:
  - `train_3d()` - 3D流场训练
  - `train_2d()` - 2D流场训练
- **优化器重新初始化**: 加载权重后清除历史动量

#### `predict.py`
- **功能**: 预测模块
- **输入**: 训练好的权重
- **输出**: 预测结果

#### `userbackend.py`
- **功能**: GPU配置和显存优化
- **关键优化**:
  - `CUDA_VISIBLE_DEVICES` - 单卡配置
  - `TF_GPU_ALLOCATOR=cuda_malloc_async` - 异步分配
  - `set_memory_growth()` - 按需增长
  - GPU进程检测和提示

### 模型实现

#### `pinns.py`
- **功能**: PINN模型核心实现
- **关键优化**:
  - `saveNN()` - 仅保存权重 (第757-773行)
  - `loadNN()` - 加载权重并重新初始化优化器 (第775-809行)
  - `ns_eqns()` - PDE残差计算，梯度带及时释放
  - `loss_fn_eqns()` - 分块计算PDE残差 (chunk_size=256)
- **模型架构**: 8层残差网络，256个隐藏单元

#### `custom_lbfgs.py`
- **功能**: L-BFGS优化器实现
- **用途**: 可选的高级优化算法

#### `datagenerator.py`
- **功能**: 训练数据生成
- **输出**: 训练点、边界条件、PDE残差点

#### `logger.py`
- **功能**: 日志记录
- **输出**: 训练日志、损失曲线

#### `maps.py`
- **功能**: 映射函数
- **用途**: 数据归一化和反归一化

### 启动脚本

#### `run_single_gpu.py`
- **功能**: 智能GPU选择脚本
- **特性**:
  - 自动检测所有GPU显存占用
  - 自动选择最空闲的GPU
  - 显示GPU进程信息
  - 可选自动终止其他进程
- **用法**:
  ```bash
  python run_single_gpu.py              # 自动选择
  python run_single_gpu.py --gpu 2     # 指定GPU
  ```

#### `run_single_gpu.sh`
- **功能**: Bash启动脚本
- **用法**:
  ```bash
  ./run_single_gpu.sh 1  # 使用GPU 1
  ```

---

## 文档说明

### `README.md`
- **内容**: 完整使用指南和优化说明
- **包含**:
  - 项目概述
  - 快速开始
  - 优化方案详解 (6项)
  - 性能数据
  - 高级用法
  - 故障排除

### `QUICK_START.md`
- **内容**: 快速开始指南
- **包含**:
  - 30秒快速开始
  - 常用命令
  - 配置调整
  - 常见问题

### `OPTIMIZATION_SUMMARY.md`
- **内容**: 优化总结
- **包含**:
  - 问题背景
  - 6项优化方案
  - 优化效果
  - 性能数据
  - 测试结果

### `PROJECT_STRUCTURE.md`
- **内容**: 项目结构说明 (本文件)

---

## 权重和数据

### `weights/` 目录

#### `Isotropicflow.weights.h5`
- **说明**: 主权重文件 (Keras 3 格式)
- **大小**: ~3.8 MB
- **包含**: 模型参数 (923K 参数)

#### `Isotropicflow_checkpoint.weights.h5`
- **说明**: 检查点权重文件
- **用途**: ModelCheckpoint 回调保存

#### `Isotropicflow_paras.mat`
- **说明**: 归一化参数
- **包含**: 输入/输出的均值和标准差

#### `Isotropicflow_status.txt`
- **说明**: 训练状态文件
- **包含**: 累计 epoch 数

#### `Isotropicflow_log.txt`
- **说明**: 训练日志
- **包含**: 每个 epoch 的损失值

#### `Isotropicflow_loss.png`
- **说明**: 损失曲线图
- **用途**: 可视化训练进度

#### `Isotropicflow_uvw_range.txt`
- **说明**: 速度范围
- **包含**: 预测速度的最小/最大值

---

## 关键参数配置

### `main.py`
```python
'tf_batch_size': 128        # 训练 batch size
'nt_batch_size': 32         # PDE 残差 batch size
'init_lr': 0.0003           # 初始学习率
'epochs': 3                 # 每次训练的 epoch 数
```

### `userbackend.py`
```python
TARGET_GPU_ID = 1           # 默认使用 GPU 1
CUDA_VISIBLE_DEVICES = '1'  # 限制可见 GPU
TF_GPU_ALLOCATOR = 'cuda_malloc_async'  # 异步分配器
```

### `pinns.py`
```python
chunk_size = 256            # PDE 残差分块大小
learning_rate = 0.0003      # 优化器学习率
```

---

## 使用流程

### 1. 从零开始训练
```bash
python main.py
# 选择: 1 (训练) → 2 (3D) → 2 (从零开始)
```

### 2. 断点续训
```bash
python main.py
# 选择: 1 (训练) → 2 (3D) → 1 (从已有权重继续)
```

### 3. 预测
```bash
python main.py
# 选择: 2 (预测)
```

### 4. 监控显存
```bash
nvidia-smi -l 1
```

---

## 总结

✅ **项目已完全优化并验证**

- 代码结构清晰，模块化设计
- 文档完整，易于使用
- 显存优化充分，可稳定运行
- Keras 3 完全兼容
- 支持断点续训和多次续训

系统已准备好用于生产环境！

