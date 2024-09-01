import torch
import torch.nn as nn
import random

def mutate_value(value, mutation_range=(5,10)):
    """Mutates a given value by adding a small random number within the mutation range."""
    return value + random.randint(*mutation_range)

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

            # Create a random input tensor with mutated dimensions
            input_tensor = torch.randn(mutated_batch_size, mutated_in_channels, mutated_height, mutated_width)

            # Initialize Conv2d layer with mutated parameters
            conv_layer = nn.Conv2d(mutated_in_channels, mutated_out_channels, mutated_kernel_size,
                                   stride=mutated_stride, padding=mutated_padding, dilation=mutated_dilation,
                                   groups=mutated_groups, bias=mutated_bias)

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


# error when input values are differnt 

# mutations of parameters, shows error for everything
