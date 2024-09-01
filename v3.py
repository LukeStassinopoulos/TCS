# Updating the kernel error in v1.py

import torch
import torch.nn as nn
import random

def mutate_value(value, mutation_range=(1.1,2.2)):
    """Mutates a given value by adding a small random number within the mutation range."""
    return max(1, value + random.randint(*mutation_range))  # Ensure value is at least 1 for dimensions

def generate_valid_conv_params(input_height, input_width):
    """Generate valid Conv2d parameters based on input dimensions."""
    kernel_size = random.randint(1, min(input_height, input_width))
    stride = random.randint(1, min(input_height // kernel_size, input_width // kernel_size))
    padding = random.randint(0, (kernel_size - 1) // 2)
    return kernel_size, stride, padding

def fuzz_conv2d():
    try:
        # Generate random base input tensor dimensions
        batch_size = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        height = random.randint(10, 50)
        width = random.randint(10, 50)

        # Generate Conv2d layer parameters that are valid for the given input dimensions
        out_channels = random.randint(1, 20)
        kernel_size, stride, padding = generate_valid_conv_params(height, width)

        # Create a random input tensor with valid dimensions
        input_tensor = torch.randn(batch_size, in_channels, height, width)

        # Initialize Conv2d layer with the valid parameters
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # Perform the forward pass
        output_tensor = conv_layer(input_tensor)

        # Print the input and output tensor shapes and parameters
        print(f"Input Tensor Shape: {input_tensor.shape}")
        print(f"Output Tensor Shape: {output_tensor.shape}")
        print(f"Parameters - Batch Size: {batch_size}, In Channels: {in_channels}, Height: {height}, Width: {width}, "
              f"Out Channels: {out_channels}, Kernel Size: {kernel_size}, Stride: {stride}, Padding: {padding}")
        print("-" * 80)

    except Exception as e:
        print(f"Exception occurred during fuzzing: {e}")

if __name__ == "__main__":
    # Fuzz the Conv2d API multiple times
    for _ in range(10):
        fuzz_conv2d()

# mutation of parameters but everything works (negative dimensions, float etc)