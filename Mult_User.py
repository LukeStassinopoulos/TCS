import torch
import random

def generate_tensor(shape):
    # Create a tensor with specified dimensions
    tensor = torch.randn(*shape)
    print(f"Created Tensor Shape: {tensor.shape}")
    return tensor

def mutate_tensor_params(params):
    mutated_params = list(params)
    for i in range(len(mutated_params)):
        mutation = random.choice([-1, 0, random.randint(5, 10)])
        mutated_params[i] += mutation
        if mutated_params[i] < 0:
            mutated_params[i] = 0  # Ensure non-negative for dimension sizes
    return tuple(mutated_params)

def stress_test_tensor_operations(multiply_params):
    try:
        shape1, shape2 = multiply_params
        
        tensor1 = generate_tensor(shape1)
        tensor2 = generate_tensor(shape2)

        print("\nInitial Input Tensors:")
        print(f"Tensor 1 Shape: {tensor1.shape}")
        print(f"Tensor 2 Shape: {tensor2.shape}")

        for i in range(10):
            mutated_shape1 = mutate_tensor_params(shape1)
            mutated_shape2 = mutate_tensor_params(shape2)

            print(f"\nOutput {i + 1}:")
            print(f"Stress Test Tensor Shape 1: torch.Size{mutated_shape1}")
            print(f"Stress Test Tensor Shape 2: torch.Size{mutated_shape2}")

            test_tensor1 = torch.randn(*mutated_shape1)
            test_tensor2 = torch.randn(*mutated_shape2)

            try:
                output_tensor = torch.mul(test_tensor1, test_tensor2)
                print(f"Output Tensor Shape After Multiplication: {output_tensor.shape}")
            except Exception as e:
                print(f"Multiplication Exception: {str(e)}")

    except Exception as e:
        print(f"Stress Test Exception: {str(e)}")

# User input for tensor shapes
shape1 = tuple(map(int, input("Enter shape for Tensor 1 (e.g., 2 3): ").split()))
shape2 = tuple(map(int, input("Enter shape for Tensor 2 (e.g., 2 3): ").split()))
stress_test_tensor_operations((shape1, shape2))
