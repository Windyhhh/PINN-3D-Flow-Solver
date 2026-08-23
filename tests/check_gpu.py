import tensorflow as tf
import sys

print("TensorFlow version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs: {len(gpus)}")
for gpu in gpus:
    print(f"  - {gpu}")

if len(gpus) == 0:
    print("\n⚠️  WARNING: No GPU detected!")
    print("Checking CUDA availability...")
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    sys.exit(1)
else:
    print("\n✅ GPU detected successfully!")
    sys.exit(0)

