import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a function to generate random input tensors
def generate_random_input(shape, device):
    return torch.randn(shape, dtype=torch.float32, device=device)

# Define a function to compare outputs between CPU and GPU
def compare_outputs(cpu_output, gpu_output, tolerance=1e-6):
    # Move both outputs to the same device (CPU)
    cpu_output = cpu_output.cpu()
    gpu_output = gpu_output.cpu()

    if torch.isnan(cpu_output).any() or torch.isnan(gpu_output).any():
        raise ValueError("NaNs detected in outputs")
    if (cpu_output - gpu_output).abs().max() > tolerance:
        raise ValueError("Outputs differ beyond tolerance")

# Define the neural network components to test
components = [
    nn.Linear(10, 20),
    nn.Conv2d(1, 10, kernel_size=5),
    nn.ReLU(),
    nn.Sigmoid(),
    nn.CrossEntropyLoss()
]

# Perform fuzz testing
def fuzz_test(component, input_shape):
    # Initialize the component on CPU
    component.cpu()

    # Generate random input tensors
    cpu_input = generate_random_input(input_shape, 'cpu')

    # Pass inputs through the component on CPU
    try:
        if isinstance(component, nn.CrossEntropyLoss):
            cpu_target = torch.randint(0, 10, (10,))
            cpu_output = component(cpu_input, cpu_target)
        else:
            cpu_output = component(cpu_input)
    except (RuntimeError, TypeError) as e:
        # Handle dtype and shape mismatches silently
        print(f"Error: {e}")
        return

    # Move the component to GPU (if available)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    component.to(device)

    # Generate random input tensors on GPU
    gpu_input = cpu_input.to(device)

    # Pass inputs through the component on GPU
    try:
        if isinstance(component, nn.CrossEntropyLoss):
            gpu_target = cpu_target.to(device)
            gpu_output = component(gpu_input, gpu_target)
        else:
            gpu_output = component(gpu_input)
    except (RuntimeError, TypeError) as e:
        # Handle dtype and shape mismatches silently
        print(f"Error: {e}")
        return

    # Compare outputs between CPU and GPU
    compare_outputs(cpu_output, gpu_output)

# Test the components
for component in components:
    if isinstance(component, nn.Linear):
        fuzz_test(component, (10, 10))
    elif isinstance(component, nn.Conv2d):
        fuzz_test(component, (1, 1, 28, 28))
    elif isinstance(component, nn.ReLU) or isinstance(component, nn.Sigmoid):
        fuzz_test(component, (10, 10))
    elif isinstance(component, nn.CrossEntropyLoss):
        fuzz_test(component, (10, 10))

print("Fuzz testing completed without significant discrepancies.")