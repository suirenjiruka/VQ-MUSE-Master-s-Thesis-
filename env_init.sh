
#!/usr/bin/env bash
set -euo pipefail

conda create -n vqmotion python=3.11 -y
conda activate vqmotion

echo "The reference momask environment uses PyTorch 2.4.1 with CUDA 12.1."
echo "Install PyTorch first (change the wheel only when the target CUDA differs):"
echo "  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121"
echo "Then install the project dependencies:"
echo "  pip install -r requirements.txt"
echo "TensorFlow is not required by VQMotion."
