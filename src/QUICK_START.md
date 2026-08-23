# 快速开始指南

## 🚀 30秒快速开始

```bash
cd Code
python main.py
```

然后按照菜单选择：
1. 选择 `1` (训练)
2. 选择 `2` (3D)
3. 选择 `2` (从零开始) 或 `1` (断点续训)

---

## 📋 常用命令

### 使用默认GPU (GPU:1)
```bash
python main.py
```

### 指定GPU
```bash
TARGET_GPU_ID=2 python main.py
```

### 自动选择最空闲GPU
```bash
python run_single_gpu.py
```

### 自动终止其他进程
```bash
AUTO_KILL_GPU_PROCESSES=1 TARGET_GPU_ID=1 python main.py
```

### 监控显存
```bash
nvidia-smi -l 1
```

---

## 🔧 配置调整

### 降低batch_size (如果OOM)
编辑 `main.py` 第64行：
```python
'tf_batch_size': 64,  # 从128改为64
```

### 选择其他GPU
```bash
TARGET_GPU_ID=3 python main.py
```

---

## ✅ 验证安装

```bash
# 检查GPU
nvidia-smi

# 检查TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# 检查Keras
python -c "import keras; print(keras.__version__)"
```

---

## 📊 性能指标

| 指标 | 值 |
|------|-----|
| 单epoch耗时 | ~56秒 |
| 显存占用 | ~3GB |
| GPU利用率 | 30-40% |
| batch_size | 128 |

---

## 🆘 常见问题

**Q: 出现OOM错误？**
A: 降低batch_size或选择其他GPU

**Q: 断点续训失败？**
A: 检查 `weights/Isotropicflow.weights.h5` 是否存在

**Q: GPU未被检测？**
A: 运行 `nvidia-smi` 检查驱动是否正常

---

## 📚 详细文档

- `README.md` - 完整使用指南和优化说明
- `run_single_gpu.py` - 智能GPU选择脚本

