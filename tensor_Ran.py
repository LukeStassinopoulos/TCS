import torch
import random

def generate_tensor(batch_size, in_channels, height, width):
    # Create a tensor with specified dimensions
    tensor = torch.randn(batch_size, in_channels, height, width)
    print(f"Created Tensor Shape: {tensor.shape}")
    return tensor

def mutate_tensor_params(params):
    # Randomly mutate parameters
    mutated_params = params.copy()
    for i in range(len(mutated_params)):
        mutation = random.choice([-1, 0, random.randint(29876545, 290876543456781), -10])  # Include invalid mutation
        mutated_params[i] += mutation
        if mutated_params[i] < 0:
            mutated_params[i] = -1  # Purposefully invalid for testing
    return mutated_params

def stress_test_tensor_operations(params):
    try:
        batch_size, in_channels, height, width = params
        print(f"Starting stress testing of tensor operations...")

        # Generate the tensor
        tensor = generate_tensor(batch_size, in_channels, height, width)
        
        # Simulating a stress test: create new tensors based on mutated parameters
        for _ in range(10):
            mutated_params = mutate_tensor_params(params)
            print(f"Stress Test Tensor Shape: torch.Size({mutated_params})")
            try:
                # Attempt to create a tensor with mutated parameters
                test_tensor = torch.randn(*mutated_params)

                # Example operation 1: Summation across batch and channels
                output_tensor_sum = test_tensor.sum(dim=(0, 1))  # Summing over batch and channels
                print(f"Output Tensor Shape After Summation: {output_tensor_sum.shape}")

                # Example operation 2: Mean across batch and channels
                output_tensor_mean = test_tensor.mean(dim=(0, 1))  # Mean over batch and channels
                print(f"Output Tensor Shape After Mean: {output_tensor_mean.shape}")

                # Example operation 3: Max across batch and channels
                output_tensor_max = test_tensor.max(dim=0).values  # Max over batch dimension
                output_tensor_max_channel = test_tensor.max(dim=1).values  # Max over channel dimension
                print(f"Output Tensor Shape After Max (batch): {output_tensor_max.shape}")
                print(f"Output Tensor Shape After Max (channel): {output_tensor_max_channel.shape}")

            except Exception as e:
                print(f"Tensor Creation Exception: {str(e)}")
    
    except Exception as e:
        print(f"Stress Test Exception: {str(e)}")

# Parameters: Batch Size, In Channels, Height, Width
params = [20, 40, 27, 12]  # Example parameters
stress_test_tensor_operations(params)
