# import torch
# print(f"Is CUDA available? {torch.cuda.is_available()}")
# print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version PyTorch wants: {torch.version.cuda}")