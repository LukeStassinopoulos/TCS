import torch
import random

def generate_random_tensor():
    # Create a random tensor with random shape
    shape = [random.randint(1, 5) for _ in range(4)]  # Random shape between 1 and 5 for each dimension
    tensor = torch.randn(*shape, requires_grad=True)
    print(f"Created Tensor with shape {tensor.shape}")
    return tensor

def mutate_tensor_params(tensor):
    # Randomly mutate tensor by introducing invalid operations
    mutation = random.choice([None, "inf", "nan", "zero_div"])
    if mutation == "inf":
        tensor = tensor + float('inf')  # Introduce infinity
    elif mutation == "nan":
        tensor = tensor + float('nan')  # Introduce NaN
    elif mutation == "zero_div":
        tensor = tensor / 0  # Divide by zero to generate an invalid value
    return tensor

def stress_test_autograd():
    try:
        # Generate random tensor
        tensor = generate_random_tensor()

        for i in range(10):  # Repeat stress test 10 times
            print(f"\nStress Test Iteration {i + 1}:")
            
            # Mutate tensor for testing
            mutated_tensor = mutate_tensor_params(tensor.clone())
            
            # Check if mutation caused an invalid tensor
            if not torch.isfinite(mutated_tensor).all():
                raise ValueError(f"Invalid tensor generated in iteration {i + 1}")

            # Example autograd operation: gradient computation
            output = mutated_tensor.sum()
            output.backward()
            print(f"Gradient of the tensor after sum operation:\n{tensor.grad}")
    
    except Exception as e:
        print(f"Autograd Stress Test Exception: {str(e)}")

# Start stress testing with random tensor mutations
stress_test_autograd()
