import torch

# Check if CUDA is available
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
else:
    print("CUDA is not available. No GPUs detected.")
