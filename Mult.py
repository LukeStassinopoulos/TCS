import torch
import random

def generate_tensor(shape):
    # Create a tensor with specified dimensions
    tensor = torch.randn(*shape)
    print(f"Created Tensor Shape: {tensor.shape}")
    return tensor

def mutate_tensor_params(params):
    # Randomly mutate parameters
    mutated_params = list(params)  # Convert tuple to list for mutability
    for i in range(len(mutated_params)):
        mutation = random.choice([-1, 0, random.randint(-5, 5)])
        mutated_params[i] += mutation
        if mutated_params[i] < 0:
            mutated_params[i] = 0  # Ensure non-negative for dimension sizes
    return tuple(mutated_params)  # Return back to tuple

def mutation(multiply_params):
    # Unpack parameters
    shape1, shape2 = multiply_params
    
    # Generate initial tensors
    tensor1 = generate_tensor(shape1)
    tensor2 = generate_tensor(shape2)
    
    # Display input tensor values
    print("\nInitial Input Tensors:")
    print("Tensor 1 Values:\n", tensor1)
    print("Tensor 2 Values:\n", tensor2)

    # Simulating a stress test: create new tensors based on mutated parameters
    for i in range(10):  # Repeat for 10 outputs
        mutated_shape1 = mutate_tensor_params(shape1)
        mutated_shape2 = mutate_tensor_params(shape2)
        
        print(f"\nOutput {i + 1}:")
        print(f"Stress Test Tensor Shape 1: torch.Size({mutated_shape1})")
        print(f"Stress Test Tensor Shape 2: torch.Size({mutated_shape2})")
        
        # Attempt to create tensors with mutated parameters
        test_tensor1 = torch.randn(*mutated_shape1)
        test_tensor2 = torch.randn(*mutated_shape2)

        # Print mutated tensor values
        print("Mutated Tensor 1 Values:\n", test_tensor1)
        print("Mutated Tensor 2 Values:\n", test_tensor2)

        # Element-wise multiplication
        output_tensor = torch.mul(test_tensor1, test_tensor2)
        print(f"Output Tensor Shape After Multiplication: {output_tensor.shape}")
        print("Output Tensor Values:\n", output_tensor)

# Parameters: Shapes of Tensors to be multiplied
multiply_params = [(1, 2), (1, 2)]  # Example shapes
mutation(multiply_params)
