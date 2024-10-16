import torch
import random
import time
import tracemalloc

# Initialize memory tracking
tracemalloc.start()

# List of basic operations and modules to test
operations = [
    'add', 'sub', 'mul', 'div', 'matmul', 't', 'view', 'reshape', 'permute',
    'mean', 'sum', 'max', 'min', 'cat', 'stack', 'split', 'unbind', 'slice',
    'nn.functional.relu', 'nn.functional.sigmoid', 'nn.functional.softmax'
]

# Define a simple feedforward neural network for testing nn.Module
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Function to generate random inputs
def generate_random_input(shape):
    return torch.rand(shape)

# Function to perform a fuzz test on an operation
def fuzz_operation(op_name, *inputs):
    try:
        if '.' in op_name:  # Handle module functions
            module, func = op_name.split('.')
            result = getattr(getattr(torch.nn, module), func)(*inputs)
        else:
            result = getattr(torch, op_name)(*inputs)

        # Check for unexpected outputs
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"WARNING: {op_name} returned NaN or Inf.")
        
    except Exception as e:
        if "size mismatch" not in str(e) and "unsupported" not in str(e):
            print(f"ERROR in {op_name}: {str(e)}")

# Function to monitor performance and memory
def monitor_performance():
    current, peak = tracemalloc.get_traced_memory()
    return current, peak

# Main fuzz testing loop
def fuzz_test(num_tests=100):
    for _ in range(num_tests):
        input_shape1 = (random.randint(1, 10), random.randint(1, 10))
        input_shape2 = (random.randint(1, 10), random.randint(1, 10))

        input1 = generate_random_input(input_shape1)
        input2 = generate_random_input(input_shape2)

        # Test tensor operations
        for op in operations:
            start_time = time.time()
            fuzz_operation(op, input1, input2)
            end_time = time.time()

            # Check for performance issues
            elapsed_time = end_time - start_time
            if elapsed_time > 0.1:  # threshold for slow operations
                print(f"WARNING: {op} took too long: {elapsed_time:.4f} seconds.")

        # Test the simple neural network
        model = SimpleNN()
        try:
            output = model(input1)
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"WARNING: Model output returned NaN or Inf.")
        except Exception as e:
            print(f"ERROR in model forward: {str(e)}")

        # Monitor performance
        current, peak = monitor_performance()
        if peak > 10**6:  # arbitrary threshold for peak memory usage in bytes
            print(f"WARNING: High memory usage - Peak: {peak / 10**6:.2f}MB")

# Run the fuzz test
if __name__ == "__main__":
    fuzz_test()
    tracemalloc.stop()
