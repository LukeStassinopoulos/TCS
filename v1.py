import torch
import torch.nn as nn
import random

def fuzz_conv2d():
    try:
        # Generate random input tensor dimensions
        batch_size = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        height = random.randint(1, 50)
        width = random.randint(1, 50)

        # Generate random Conv2d layer parameters
        out_channels = random.randint(1, 20)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 5)
        padding = random.randint(0, 5)

        # Create a random input tensor
        input_tensor = torch.randn(batch_size, in_channels, height, width)

        # Initialize Conv2d layer with random parameters
        conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # Perform the forward pass
        output_tensor = conv_layer(input_tensor)

        # Print the input and output tensor shapes
        print(f"Input Tensor Shape: {input_tensor.shape}")
        print(f"Output Tensor Shape: {output_tensor.shape}")

        return output_tensor

    except Exception as e:
        print(f"Exception occurred during fuzzing: {e}")

if __name__ == "__main__":
    # Fuzz the Conv2d API multiple times
    for _ in range(10):
        fuzz_conv2d()

# Exception occurred during fuzzing: Calculated padded input size per channel: (42 x 5). Kernel size: (6 x 6). Kernel size can't be greater than actual input size