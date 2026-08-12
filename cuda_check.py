import sys

print(f"Python: {sys.version.split()[0]}")
print(f"Executable: {sys.executable}")
print("=" * 56)

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is not installed in this environment.") from exc

print(f"PyTorch: {torch.__version__}")
print(f"Compiled CUDA: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for device_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(device_id)
        memory_gib = props.total_memory / (1024 ** 3)
        print(f"  [{device_id}] {props.name} ({memory_gib:.1f} GiB)")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
else:
    print("WARNING: PyTorch cannot access CUDA; training will run on CPU.")

print("TensorFlow is not required by VQMotion.")
