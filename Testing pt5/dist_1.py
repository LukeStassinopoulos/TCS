import torch
import random
import torch.distributed as dist

# Function to generate random tensor
def generate_random_tensor():
    shape = tuple(random.randint(1, 10) for _ in range(random.randint(1, 4)))
    dtype = random.choice([torch.float32, torch.float64, torch.int32])
    return torch.randn(shape, dtype=dtype) if dtype != torch.int32 else torch.randint(0, 100, shape)

# Fuzz testing for distributed training and parallelism
def fuzz_distributed_training():
    # Assume distributed environment is already initialized
    for _ in range(10):  # Adjust number of tests as needed
        model = torch.nn.Sequential(
            torch.nn.Linear(random.randint(1, 10), random.randint(1, 10)),
            torch.nn.ReLU()
        )
        input_tensor = generate_random_tensor()
        model = torch.nn.parallel.DistributedDataParallel(model)
        
        try:
            output = model(input_tensor)
            if torch.isnan(output).any():
                print("NaN detected in distributed model output.")
        except Exception:
            # Silently ignore expected exceptions
            pass

if __name__ == "__main__":
    # Ensure to call `dist.init_process_group` before running this function in a real distributed setup
    # dist.init_process_group(backend='nccl')
    fuzz_distributed_training()

#Dist error