import torch
import random
import numpy as np

# Define boundary and invalid values
invalid_inputs = [
    None,                        # Non-type input
    float('inf'),                # Infinity
    float('-inf'),               # Negative infinity
    float('nan'),                # NaN
    "invalid_string",            # String input
    {},                          # Empty dictionary
    [],                          # Empty list
    -1,                          # Negative integer
    torch.tensor([]),            # Empty tensor
    torch.tensor([[[1, 2], [3, 4]]]),  # Incorrect shape (for some operations)
    np.array([1, 2, 3]),         # Numpy array (non-PyTorch tensor)
]

# Function to generate random tensor with different properties
def random_tensor():
    shapes = [(), (1,), (3, 3), (100, 100), (0,), (1, 3, 3), (10, 0), (10000, 10000)]
    dtype = random.choice([torch.float32, torch.int32, torch.float64])
    return torch.rand(random.choice(shapes)).to(dtype)

# List of functions to fuzz (selected from previous list)
pytorch_functions = [
    torch.add,
    torch.sub,
    torch.mul,
    torch.div,
    torch.pow,
    torch.matmul,
    torch.sum,
    torch.mean,
    torch.autograd.grad,
    torch.nn.functional.relu,
    torch.nn.functional.softmax,
    torch.save,
    torch.load
]

# Function to test each PyTorch function with random and invalid inputs
def fuzz_pytorch_function(func, inputs):
    for inp in inputs:
        try:
            # Try applying the function with the random/invalid input
            if isinstance(inp, tuple):
                result = func(*inp)  # Unpack tuple inputs for multi-arg functions
            else:
                result = func(inp)
            print(f"Function {func.__name__} handled input {inp} successfully.")
        except Exception as e:
            print(f"Function {func.__name__} raised an exception for input {inp}: {e}")

# Fuzzing test loop
for func in pytorch_functions:
    print(f"\nTesting function: {func.__name__}")
    
    # Generate random inputs
    random_inputs = [
        random_tensor(),                 # Valid random tensor
        (random_tensor(), random_tensor()) if func in [torch.add, torch.sub, torch.mul, torch.div] else random_tensor(),  # Random tensor pair for binary ops
        torch.tensor([1.0], requires_grad=True)  # Input with gradient enabled for autograd functions
    ]
    
    # Combine random and invalid inputs for the fuzzing test
    all_inputs = random_inputs + invalid_inputs
    
    # Test the function with all generated inputs
    fuzz_pytorch_function(func, all_inputs)
