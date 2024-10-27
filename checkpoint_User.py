import torch
import torch.utils.checkpoint as checkpoint
import random

def generate_tensor(shape):
    return torch.randn(*shape)

def mutate_tensor_params(shape):
    mutated_shape = list(shape)
    for i in range(len(mutated_shape)):
        mutation = random.choice([-1, 0, random.randint(-5, 5)])
        mutated_shape[i] += mutation
        if mutated_shape[i] < 1:  # Ensure valid dimensions
            mutated_shape[i] = 1
    return tuple(mutated_shape)

def sample_function(x):
    return x * 2

def run_checkpoint_test_user_input():
    try:
        # User input for shape
        shape_input = input("Enter tensor shape as space-separated integers (e.g., '2 2'): ")
        shape = tuple(map(int, shape_input.split()))

        for i in range(10):
            # Generate input tensor
            input_tensor = generate_tensor(shape)

            # Mutate the tensor shape
            mutated_shape = mutate_tensor_params(shape)

            # Generate new tensor with mutated shape
            mutated_tensor = generate_tensor(mutated_shape)
            
            # Run checkpoint function
            output = checkpoint.checkpoint(sample_function, mutated_tensor)

            # Print outputs
            print(f"\nOutput {i + 1}:")
            print(f"Mutated Tensor Shape: {mutated_shape}")
            print(f"Output Shape: {output.shape}")
    except Exception as e:
        print(f"Error: {e}")

# Run the test with user input
run_checkpoint_test_user_input()
