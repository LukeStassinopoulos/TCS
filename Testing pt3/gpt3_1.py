import torch
import random
import time
import tracemalloc

# Initialize memory tracking
tracemalloc.start()

# Define a list of basic operations to test
operations = [
    'add', 'sub', 'mul', 'div', 'matmul',
    't', 'view', 'reshape', 'permute',
    'mean', 'sum', 'max', 'min', 'cat',
    'stack', 'split', 'unbind', 'slice'
]

# Function to generate random inputs
def generate_random_input(shape):
    return torch.rand(shape)

# Function to perform a fuzz test on an operation
def fuzz_operation(op_name, input1, input2=None):
    try:
        if op_name in ['add', 'sub', 'mul', 'div']:
            result = getattr(torch, op_name)(input1, input2)
        elif op_name in ['matmul']:
            result = torch.matmul(input1, input2)
        else:
            result = getattr(input1, op_name)()
        
        # Check for unexpected outputs
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"WARNING: {op_name} resulted in NaN or Inf.")
        
    except Exception as e:
        print(f"ERROR in {op_name}: {str(e)}")

# Function to monitor performance and memory
def monitor_performance():
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 10**6:.2f}MB; Peak: {peak / 10**6:.2f}MB")

# Main fuzz testing loop
def fuzz_test(num_tests=100):
    for _ in range(num_tests):
        input_shape1 = (random.randint(1, 10), random.randint(1, 10))
        input_shape2 = (random.randint(1, 10), random.randint(1, 10))

        input1 = generate_random_input(input_shape1)
        input2 = generate_random_input(input_shape2)

        for op in operations:
            start_time = time.time()
            fuzz_operation(op, input1, input2)
            end_time = time.time()

            # Check for performance issues
            elapsed_time = end_time - start_time
            if elapsed_time > 0.1:  # threshold for slow operations
                print(f"WARNING: {op} took too long: {elapsed_time:.4f} seconds.")
        
        monitor_performance()

# Run the fuzz test
if __name__ == "__main__":
    fuzz_test()
    tracemalloc.stop()
