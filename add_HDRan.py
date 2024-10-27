import torch
import random

def generate_random_shape():
    # Generate a random shape for tensors
    return (random.randint(1, 1), random.randint(1, 1))  # Example: (batch_size, features)

def generate_tensor(shape):
    # Create a tensor with specified dimensions
    tensor = torch.randn(*shape)
    print(f"Created Tensor Shape: {tensor.shape}")
    return tensor

def mutate_tensor_params(params):
    # Randomly mutate parameters
    mutated_params = list(params)  # Convert tuple to list for mutability
    for i in range(len(mutated_params)):
        mutation = random.choice([-1, 0, random.randint(5,10)])
        mutated_params[i] += mutation
        if mutated_params[i] < 0:
            mutated_params[i] = 0  # Ensure non-negative for dimension sizes
    return tuple(mutated_params)  # Return back to tuple

def stress_test_tensor_operations(add_params):
    try:
        # Unpack parameters
        shape1, shape2 = add_params
        
        # Generate initial tensors
        tensor1 = generate_tensor(shape1)
        tensor2 = generate_tensor(shape2)

        # Display input tensor values
        print("\nInitial Input Tensors:")
        print(f"{'Tensor 1 Values:':<20}\n{tensor1}\n")
        print(f"{'Tensor 2 Values:':<20}\n{tensor2}\n")

        # Simulating a stress test: create new tensors based on mutated parameters
        for i in range(10):  # Repeat for 10 outputs
            mutated_shape1 = mutate_tensor_params(shape1)
            mutated_shape2 = mutate_tensor_params(shape2)
            
            print(f"{'Output ' + str(i + 1) + ':':<25}")
            print(f"{'Stress Test Tensor Shape 1:':<40} torch.Size{mutated_shape1}")
            print(f"{'Stress Test Tensor Shape 2:':<40} torch.Size{mutated_shape2}")

            # Attempt to create tensors with mutated parameters
            test_tensor1 = torch.randn(*mutated_shape1)
            test_tensor2 = torch.randn(*mutated_shape2)

            # Print mutated tensor values
            print(f"{'Mutated Tensor 1 Values:':<30}\n{test_tensor1}")
            print(f"{'Mutated Tensor 2 Values:':<30}\n{test_tensor2}")

            # Element-wise addition
            try:
                output_tensor = torch.add(test_tensor1, test_tensor2)  # or torch.mul for multiplication
                print(f"{'Output Tensor Shape After Addition:':<40} {output_tensor.shape}")
                print(f"{'Output Tensor Values:':<30}\n{output_tensor}\n")
            except Exception as e:
                print(f"Addition Exception: {str(e)}")

    except Exception as e:
        print(f"Stress Test Exception: {str(e)}")

# Generate random shapes for addition
add_params = (generate_random_shape(), generate_random_shape())  # Generate random shapes
stress_test_tensor_operations(add_params)
