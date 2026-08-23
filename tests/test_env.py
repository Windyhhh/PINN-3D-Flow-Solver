#!/usr/bin/env python
import sys
import os

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'NOT SET'))
print("CUDA_HOME:", os.environ.get('CUDA_HOME', 'NOT SET'))
print("PATH:", os.environ.get('PATH', 'NOT SET')[:200])

try:
    import tensorflow as tf
    print("\nTensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs: {len(gpus)}")
    for gpu in gpus:
        print(f"  - {gpu}")
except Exception as e:
    print(f"\nError importing TensorFlow: {e}")
    import traceback
    traceback.print_exc()

