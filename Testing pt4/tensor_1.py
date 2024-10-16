# Prompt: Generate Python code to perform fuzz testing for PyTorch tensor operations. Include a wide range of tensor operations such as arithmetic (add, subtract, multiply, etc.), reductions (mean, sum, max), reshaping (view, reshape, permute), and indexing. Generate random tensors of various shapes, sizes, and data types, and ensure maximum coverage of PyTorch’s tensor manipulation functionalities. Only report crashes or unexpected outputs, and ignore expected exceptions like size mismatches. Additionally, monitor memory usage and performance, and report only when memory usage exceeds a threshold or computation takes an unusually long time.

import torch
import random
import time
import tracemalloc

# Initialize memory tracking
tracemalloc.start()

# List of tensor operations to test
tensor_operations = [
    'add', 'sub', 'mul', 'div', 'matmul', 'pow', 't',
    'mean', 'sum', 'max', 'min', 'argmax', 'argmin',
    'view', 'reshape', 'permute', 'squeeze', 'unsqueeze',
    'cat', 'split', 'slice', 'index_select'
]

# Function to generate random inputs
def generate_random_tensor(shape, dtype=None):
    if dtype is None:
        dtype = random.choice([torch.float32, torch.float64, torch.int32, torch.int64])
    return torch.randn(shape, dtype=dtype)

# Function to perform a fuzz test on an operation
def fuzz_operation(op_name, *inputs):
    try:
        result = getattr(torch, op_name)(*inputs)

        # Check for unexpected outputs
        if isinstance(result, torch.Tensor) and (torch.isnan(result).any() or torch.isinf(result).any()):
            print(f"WARNING: {op_name} returned NaN or Inf.")
        
    except Exception as e:
        # Silently ignore expected exceptions (e.g., size mismatches)
        if not ("size mismatch" in str(e) or "unsupported" in str(e)):
            print(f"ERROR in {op_name}: {str(e)}")

# Function to monitor performance and memory
def monitor_performance():
    current, peak = tracemalloc.get_traced_memory()
    return current, peak

# Main fuzz testing loop
def fuzz_test(num_tests=100):
    for _ in range(num_tests):
        # Generate random tensor shapes
        shape1 = (random.randint(1, 10), random.randint(1, 10))
        shape2 = (random.randint(1, 10), random.randint(1, 10))

        # Create random tensors
        input1 = generate_random_tensor(shape1)
        input2 = generate_random_tensor(shape2)

        # Test tensor operations
        for op in tensor_operations:
            start_time = time.time()
            fuzz_operation(op, input1, input2)
            end_time = time.time()

            # Check for performance issues
            elapsed_time = end_time - start_time
            if elapsed_time > 0.1:  # threshold for slow operations
                print(f"WARNING: {op} took too long: {elapsed_time:.4f} seconds.")

        # Monitor memory usage
        current, peak = monitor_performance()
        if peak > 10**6:  # arbitrary threshold for peak memory usage in bytes
            print(f"WARNING: High memory usage - Peak: {peak / 10**6:.2f}MB")

# Run the fuzz test
if __name__ == "__main__":
    fuzz_test()
    tracemalloc.stop()

#FIRST SCRIPT TO FIND AN ERROR 
# Traceback (most recent call last):
#   File "/home/stasluke18/TCS/Testing pt4/tensor_1.py", line 71, in <module>
#     fuzz_test()
#   File "/home/stasluke18/TCS/Testing pt4/tensor_1.py", line 50, in fuzz_test
#     input1 = generate_random_tensor(shape1)
#   File "/home/stasluke18/TCS/Testing pt4/tensor_1.py", line 21, in generate_random_tensor
#     return torch.randn(shape, dtype=dtype)
# RuntimeError: "normal_kernel_cpu" not implemented for 'Long'