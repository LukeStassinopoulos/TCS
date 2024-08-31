import torch
import torch.nn as nn
import random

def mutate_value(value, mutation_range=(-3, 3), min_value=1):
    """Mutates a given value by adding a small random number within the mutation range.
       Ensures the result is at least min_value."""
    mutated_value = value + random.randint(*mutation_range)
    return max(mutated_value, min_value)  # Ensure non-negative and above min_value

def fuzz_conv2d():
    try:
        # Generate random base input tensor dimensions
        batch_size = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        height = random.randint(1, 50)
        width = random.randint(1, 50)

        # Generate random base Conv2d layer parameters
        out_channels = random.randint(1, 20)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 5)
        padding = random.randint(0, 5)
        dilation = random.randint(1, 5)
        groups = random.randint(1, in_channels)  # Must divide in_channels
        bias = random.choice([True, False])

        # Ensure out_channels is divisible by groups
        if out_channels % groups != 0:
            out_channels = (out_channels // groups) * groups

        # Mutate the base parameters
        for _ in range(10):  # Minimum of 10 mutations
            mutated_batch_size = mutate_value(batch_size)
            mutated_in_channels = mutate_value(in_channels)
            mutated_height = mutate_value(height)
            mutated_width = mutate_value(width)

            mutated_out_channels = mutate_value(out_channels)
            mutated_kernel_size = mutate_value(kernel_size)
            mutated_stride = mutate_value(stride)
            mutated_padding = mutate_value(padding)
            mutated_dilation = mutate_value(dilation)
            mutated_groups = max(1, min(mutated_in_channels, mutate_value(groups)))  # Ensure it's within valid range
            mutated_bias = random.choice([True, False])

            # Ensure out_channels is divisible by mutated_groups
            if mutated_groups > 0 and mutated_out_channels % mutated_groups != 0:
                mutated_out_channels = (mutated_out_channels // mutated_groups) * mutated_groups

            # Ensure dimensions are valid and positive
            if mutated_height <= 0 or mutated_width <= 0:
                continue

            # Create a random input tensor with mutated dimensions
            input_tensor = torch.randn(mutated_batch_size, mutated_in_channels, mutated_height, mutated_width)

            # Initialize Conv2d layer with mutated parameters
            conv_layer = nn.Conv2d(
                mutated_in_channels, mutated_out_channels, mutated_kernel_size,
                stride=mutated_stride, padding=mutated_padding, dilation=mutated_dilation,
                groups=mutated_groups, bias=mutated_bias
            )

            # Perform the forward pass
            output_tensor = conv_layer(input_tensor)

            # Print the input and output tensor shapes and mutated parameters
            print(f"Input Tensor Shape: {input_tensor.shape}")
            print(f"Output Tensor Shape: {output_tensor.shape}")
            print(f"Mutated Parameters - Batch Size: {mutated_batch_size}, In Channels: {mutated_in_channels}, "
                  f"Height: {mutated_height}, Width: {mutated_width}, Out Channels: {mutated_out_channels}, "
                  f"Kernel Size: {mutated_kernel_size}, Stride: {mutated_stride}, Padding: {mutated_padding}, "
                  f"Dilation: {mutated_dilation}, Groups: {mutated_groups}, Bias: {mutated_bias}")
            print("-" * 80)

    except Exception as e:
        print(f"Exception occurred during fuzzing: {e}")

if __name__ == "__main__":
    # Fuzz the Conv2d API multiple times
    fuzz_conv2d()

# Exception occurred during fuzzing: Given groups=4, expected weight to be at least 4 at dimension 0, but got weight of size [0, 1, 3, 3] instead
#--------------
import torch
import torch.nn as nn
import random

def mutate_value(value, mutation_range=(-3, 3), min_value=1):
    """Mutates a given value by adding a small random number within the mutation range.
       Ensures the result is at least min_value."""
    mutated_value = value + random.randint(*mutation_range)
    return max(mutated_value, min_value)  # Ensure non-negative and above min_value

def fuzz_conv2d():
    try:
        # Generate random base input tensor dimensions
        batch_size = random.randint(1, 10)
        in_channels = random.randint(1, 10)
        height = random.randint(1, 50)
        width = random.randint(1, 50)

        # Generate random base Conv2d layer parameters
        out_channels = random.randint(1, 20)
        kernel_size = random.randint(1, 10)
        stride = random.randint(1, 5)
        padding = random.randint(0, 5)
        dilation = random.randint(1, 5)
        groups = random.randint(1, in_channels)  # Must divide in_channels
        bias = random.choice([True, False])

        # Ensure `in_channels` is divisible by `groups` and `out_channels` is divisible by `groups`
        in_channels = (in_channels // groups) * groups
        out_channels = (out_channels // groups) * groups

        # Mutate the base parameters
        for _ in range(10):  # Minimum of 10 mutations
            mutated_batch_size = mutate_value(batch_size)
            mutated_in_channels = mutate_value(in_channels)
            mutated_height = mutate_value(height)
            mutated_width = mutate_value(width)

            mutated_out_channels = mutate_value(out_channels)
            mutated_kernel_size = mutate_value(kernel_size)
            mutated_stride = mutate_value(stride)
            mutated_padding = mutate_value(padding)
            mutated_dilation = mutate_value(dilation)
            mutated_groups = max(1, min(mutated_in_channels, mutate_value(groups)))  # Ensure it's within valid range
            mutated_bias = random.choice([True, False])

            # Ensure `mutated_in_channels` is divisible by `mutated_groups` and `mutated_out_channels` is divisible by `mutated_groups`
            if mutated_groups > 0:
                mutated_in_channels = (mutated_in_channels // mutated_groups) * mutated_groups
                mutated_out_channels = (mutated_out_channels // mutated_groups) * mutated_groups

            # Ensure dimensions are valid and positive
            if mutated_height <= 0 or mutated_width <= 0:
                continue

            # Create a random input tensor with mutated dimensions
            input_tensor = torch.randn(mutated_batch_size, mutated_in_channels, mutated_height, mutated_width)

            # Initialize Conv2d layer with mutated parameters
            conv_layer = nn.Conv2d(
                mutated_in_channels, mutated_out_channels, mutated_kernel_size,
                stride=mutated_stride, padding=mutated_padding, dilation=mutated_dilation,
                groups=mutated_groups, bias=mutated_bias
            )

            # Perform the forward pass
            output_tensor = conv_layer(input_tensor)

            # Print the input and output tensor shapes and mutated parameters
            print(f"Input Tensor Shape: {input_tensor.shape}")
            print(f"Output Tensor Shape: {output_tensor.shape}")
            print(f"Mutated Parameters - Batch Size: {mutated_batch_size}, In Channels: {mutated_in_channels}, "
                  f"Height: {mutated_height}, Width: {mutated_width}, Out Channels: {mutated_out_channels}, "
                  f"Kernel Size: {mutated_kernel_size}, Stride: {mutated_stride}, Padding: {mutated_padding}, "
                  f"Dilation: {mutated_dilation}, Groups: {mutated_groups}, Bias: {mutated_bias}")
            print("-" * 80)

    except Exception as e:
        print(f"Exception occurred during fuzzing: {e}")

if __name__ == "__main__":
    # Fuzz the Conv2d API multiple times
    fuzz_conv2d()
