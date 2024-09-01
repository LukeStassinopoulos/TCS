import torch
import torch.nn as nn
import random
import sys
import traceback

def mutate_value(value, mutation_range=(-1,100)):
    """Mutates a given value by adding a small random number within the mutation range."""
    return value + random.randint(*mutation_range)

def create_random_conv_layer():
    """Creates a convolutional layer with randomized parameters and applies mutation."""
    in_channels = random.randint(1, 10)
    out_channels = random.randint(1, 20)
    kernel_size = random.randint(1, 15)
    stride = random.randint(1, 5)
    padding = 0  # Padding will be checked and possibly adjusted

    # Mutate parameters
    in_channels = mutate_value(in_channels)
    out_channels = mutate_value(out_channels)
    kernel_size = mutate_value(kernel_size)
    stride = mutate_value(stride)
    
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), in_channels, out_channels, kernel_size, stride, padding

def generate_random_input(in_channels):
    """Generates a random input tensor with size based on in_channels."""
    batch_size = random.randint(1, 10)
    height = random.randint(1, 50)
    width = random.randint(1, 50)
    return torch.randn(batch_size, in_channels, height, width)

def process_layer(layer, input_tensor):
    """Processes the input tensor through the given convolutional layer."""
    try:
        output_tensor = layer(input_tensor)
        return output_tensor
    except Exception as e:
        # Catch and display exception
        print("\nException occurred during processing:", e)
        if 'padding' in str(e).lower():
            print("Padding issue detected.")
        else:
            print("Error:", e)
        return None

def main():
    for _ in range(10):
        # Create and mutate layer parameters
        conv_layer, in_channels, out_channels, kernel_size, stride, padding = create_random_conv_layer()

        # Ensure that the mutation doesn't exceed practical limits
        if in_channels <= 0 or out_channels <= 0 or kernel_size <= 0 or stride <= 0:
            print("\nError: Invalid parameter values.")
            continue
        
        # Generate random input tensor
        input_tensor = generate_random_input(in_channels)
        
        # Display the parameters and input tensor shape
        print(f"\nInput Tensor Shape: {input_tensor.shape}")
        print(f"Parameters - Batch Size: {input_tensor.size(0)}, In Channels: {in_channels}, Height: {input_tensor.size(2)}, Width: {input_tensor.size(3)}, Out Channels: {out_channels}, Kernel Size: {kernel_size}, Stride: {stride}, Padding: {padding}")
        
        # Process the input tensor through the convolutional layer
        output_tensor = process_layer(conv_layer, input_tensor)
        if output_tensor is not None:
            print(f"Output Tensor Shape: {output_tensor.shape}")
        else:
            print("Processing failed.")

if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("\nError: Stack overflow or excessive memory usage detected.")
    except RuntimeError as e:
        if 'killed' in str(e).lower():
            print("\nError: Process was killed.")
        else:
            print("Runtime error:", e)
    except Exception as e:
        print("\nUnexpected error:", e)

# mutations of parameters and values, properly functioning without padding , with formatting
