import torch
import random

# Function to generate random tensor
def generate_random_tensor():
    shape = tuple(random.randint(1, 10) for _ in range(random.randint(1, 4)))
    dtype = random.choice([torch.float32, torch.float64, torch.int32])
    return torch.randn(shape, dtype=dtype) if dtype != torch.int32 else torch.randint(0, 100, shape)

# Fuzz testing for autograd and gradient calculations
def fuzz_autograd():
    for _ in range(100):  # Adjust number of tests as needed
        tensor = generate_random_tensor()
        tensor.requires_grad = True
        
        # Determine input size for the Linear layer
        input_size = tensor.size(-1) if tensor.dim() > 1 else 1  # Use last dimension or 1 for 1D tensors
        
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, random.randint(1, 10)),
            torch.nn.ReLU()
        )
        
        try:
            output = model(tensor)
            output.sum().backward()
            if tensor.grad is not None and torch.isnan(tensor.grad).any():
                print("NaN detected in gradients.")
        except Exception:
            # Silently ignore expected exceptions
            pass

if __name__ == "__main__":
    fuzz_autograd()

#Another error