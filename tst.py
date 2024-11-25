import torch, time
import os

# 1. 检查CUDA环境
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA版本: {torch.version.cuda}")

