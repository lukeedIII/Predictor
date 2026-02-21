"""Quick WSL2 environment verification."""
import platform
print(f"Platform: {platform.system()} {platform.release()}")

import torch
print(f"PyTorch:  {torch.__version__}")

import triton
print(f"Triton:   {triton.__version__}")

print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM:     {props.total_memory / 1e9:.1f} GB")

    # torch.compile test
    import torch.nn as nn
    model = nn.Linear(64, 32).cuda()
    compiled = torch.compile(model, mode='max-autotune')
    x = torch.randn(16, 64, device='cuda')
    y = compiled(x)
    print(f"torch.compile: WORKS (output shape: {y.shape})")
    print("ALL CHECKS PASSED")
else:
    print("CUDA NOT AVAILABLE — check driver")
