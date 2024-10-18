import torch
import numpy as np
import random
import time
import tracemalloc

# Function to generate random tensor
def generate_random_tensor():
    shape = tuple(random.randint(1, 10) for _ in range(random.randint(1, 4)))
    dtype = random.choice([torch.float32, torch.float64, torch.int32])
    return torch.randn(shape, dtype=dtype) if dtype != torch.int32 else torch.randint(0, 100, shape)

# Function to monitor memory usage
def monitor_memory():
    current, peak = tracemalloc.get_traced_memory()
    if peak > 10 * 1024 * 1024:  # Threshold of 10MB
        print(f"Memory usage: Peak {peak / 1024 / 1024:.2f}MB")

# Main fuzz testing loop for tensor operations
def fuzz_tensor_operations():
    operations = [
        lambda x: x + x,
        lambda x: x - x,
        lambda x: x * x,
        lambda x: x / (x + 1e-10),  # Prevent division by zero
        lambda x: x.sum(),
        lambda x: x.mean(),
        lambda x: x.reshape(-1),
        lambda x: x.view(-1),
        lambda x: x[0],  # Indexing
    ]

    tracemalloc.start()
    for _ in range(100):  # Adjust number of tests as needed
        tensor = generate_random_tensor()
        for operation in operations:
            try:
                start_time = time.time()
                result = operation(tensor)
                if torch.isnan(result).any() or torch.isinf(result).any():
                    print(f"Unexpected output: {result}")
                if time.time() - start_time > 1:  # Threshold for abnormal delay
                    print("Abnormal delay detected in operation.")
                monitor_memory()
            except Exception as e:
                # Silently ignore expected exceptions
                pass
    tracemalloc.stop()

fuzz_tensor_operations()
