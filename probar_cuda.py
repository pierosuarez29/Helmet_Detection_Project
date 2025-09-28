import torch

# Check if CUDA is available
print(torch.cuda.is_available())

# Number of available CUDA devices (GPUs)
print(torch.cuda.device_count())

# Name of the first available CUDA device
print(torch.cuda.get_device_name(0))
