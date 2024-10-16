import torch
import random
import time
import tracemalloc
import re
import warnings

# Suppress user warnings
warnings.filterwarnings("ignore")

# Initialize memory tracking
tracemalloc.start()

# List of neural network layers, activation functions, and loss functions to test
nn_layers = [
    torch.nn.Linear,
    torch.nn.Conv2d,
    torch.nn.MaxPool2d,
    torch.nn.AvgPool2d,
]

activation_functions = [
    torch.nn.functional.relu,
    torch.nn.functional.sigmoid,
    torch.nn.functional.tanh,
    torch.nn.functional.softmax,
]

loss_functions = [
    torch.nn.functional.cross_entropy,
    torch.nn.functional.mse_loss,
    torch.nn.functional.nll_loss,
]

# Function to generate random input tensors
def generate_random_input_tensor(shape, dtype=torch.float32):
    return torch.randn(shape, dtype=dtype)

# Function to check if an error message is expected
def is_expected_error(error_msg):
    expected_patterns = [
        r"size mismatch",
        r"input",
        r"must match",
        r"expected",
        r"out of bounds",
        r"invalid",
    ]
    return any(re.search(pattern, error_msg) for pattern in expected_patterns)

# Function to perform fuzz test on a neural network layer
def fuzz_layer(layer_class, *inputs):
    try:
        layer = layer_class(*inputs)
        # Generate a random input tensor with a batch size of 1
        input_tensor = generate_random_input_tensor((1, *inputs[:-1]))  # Use inputs except the last one for the batch
        result = layer(input_tensor)
        return layer, result
    except Exception as e:
        if not is_expected_error(str(e)):
            print(f"ERROR initializing layer {layer_class.__name__}: {str(e)}")
        return None

# Function to perform a fuzz test on an activation function
def fuzz_activation(func, input_tensor):
    try:
        result = func(input_tensor)
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"WARNING: Activation function {func.__name__} returned NaN or Inf.")
    except Exception as e:
        if not is_expected_error(str(e)):
            print(f"ERROR in activation function {func.__name__}: {str(e)}")

# Function to perform a fuzz test on a loss function
def fuzz_loss(func, target_tensor, output_tensor):
    try:
        result = func(output_tensor, target_tensor)
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"WARNING: Loss function {func.__name__} returned NaN or Inf.")
    except Exception as e:
        if not is_expected_error(str(e)):
            print(f"ERROR in loss function {func.__name__}: {str(e)}")

# Function to monitor performance and memory
def monitor_performance():
    current, peak = tracemalloc.get_traced_memory()
    return current, peak

# Main fuzz testing loop
def fuzz_test(num_tests=100):
    for _ in range(num_tests):
        # Random shapes for input tensors
        input_shape = (random.randint(1, 10), random.randint(1, 10))
        target_shape = (random.randint(1, 10),)

        # Generate random input tensors
        input_tensor = generate_random_input_tensor(input_shape)
        target_tensor = generate_random_input_tensor(target_shape)

        # Test neural network layers
        for layer in nn_layers:
            if layer == torch.nn.Linear:
                fuzz_layer(layer, input_shape[1], input_shape[1])  # Input and output size
            elif layer == torch.nn.Conv2d:
                fuzz_layer(layer, 1, 16, 3, 1)  # 1 input channel, 16 output channels, kernel size 3
            elif layer in [torch.nn.MaxPool2d, torch.nn.AvgPool2d]:
                fuzz_layer(layer, 2)  # kernel size

        # Test activation functions
        for func in activation_functions:
            fuzz_activation(func, input_tensor)

        # Test loss functions
        output_tensor = generate_random_input_tensor(input_shape)  # Simulated model output
        for func in loss_functions:
            fuzz_loss(func, target_tensor, output_tensor)

        # Monitor memory usage
        current, peak = monitor_performance()
        if peak > 10**6:  # threshold for peak memory usage in bytes
            print(f"WARNING: High memory usage - Peak: {peak / 10**6:.2f}MB")

        # Monitor operation time
        start_time = time.time()
        # Just a dummy operation to measure performance; can replace with relevant operations if needed
        _ = torch.matmul(input_tensor, input_tensor.T)
        end_time = time.time()
        if (end_time - start_time) > 0.1:  # threshold for slow operations
            print(f"WARNING: Operation took too long: {end_time - start_time:.4f} seconds.")

# Run the fuzz test
if __name__ == "__main__":
    fuzz_test()
    tracemalloc.stop()

# No errors