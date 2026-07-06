import sys
import os

# 設定環境變數，減少 TensorFlow 的囉嗦警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

print(f"🐍 Python Path: {sys.executable}")
print("="*40)

# --- 1. 檢查 PyTorch ---
print("🔍 正在檢查 PyTorch...")
try:
    import torch
    print(f"   PyTorch Version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✅ PyTorch CUDA: True")
        print(f"   👻 GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   🔢 CUDA Version (Compile): {torch.version.cuda}")
    else:
        print(f"   ❌ PyTorch CUDA: False (PyTorch 找不到 GPU)")
except ImportError:
    print("   ❌ PyTorch 未安裝")
except Exception as e:
    print(f"   ❌ PyTorch 錯誤: {e}")

print("="*40)

# --- 2. 檢查 TensorFlow ---
print("🔍 正在檢查 TensorFlow...")
try:
    import tensorflow as tf
    print(f"   TensorFlow Version: {tf.__version__}")
    
    # 防止 TF 一口氣吃光 VRAM，導致 PyTorch 沒得用 (這行對你的評估代碼很重要)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if len(gpus) > 0:
        print(f"   ✅ TensorFlow GPU: Detected {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"      Device {i}: {gpu.name}")
    else:
        print(f"   ❌ TensorFlow GPU: 0 (TF 找不到 GPU)")
        
except ImportError:
    print("   ❌ TensorFlow 未安裝")
except Exception as e:
    print(f"   ❌ TensorFlow 錯誤: {e}")

print("="*40)