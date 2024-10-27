import torch
import random

def generate_tensor(shape):
    # Create a tensor with specified dimensions
    tensor = torch.randn(*shape)
    print(f"Created Tensor Shape: {tensor.shape}")
    return tensor

def validate_shape(shape):
    # Validate that the shape is correct for the operations
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("Input shape must be of the form (batch_size, features) with batch_size and features > 0.")

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

        # Element-wise addition
        output_tensor = torch.add(tensor1, tensor2)
        print(f"{'Output Tensor Shape After Addition:':<40} {output_tensor.shape}")
        print(f"{'Output Tensor Values:':<30}\n{output_tensor}\n")

    except Exception as e:
        print(f"Stress Test Exception: {str(e)}")

# User input for tensor shapes
for i in range(15):
    try:
        batch_size = int(input("Enter batch size for input tensor (batch_size > 0): "))
        features = int(input("Enter number of features for input tensor (features > 0): "))
        
        # Randomly decide to set an invalid shape purposefully
        if random.choice([True, False]):
            print("Generating an invalid shape for testing...")
            input_shape1 = (random.randint(-5, 0), random.randint(-5, 0))
            input_shape2 = input_shape1
        else:
            input_shape1 = (batch_size, features)
            input_shape2 = (batch_size, features)  # For addition, both shapes should be the same

        validate_shape(input_shape1)
        validate_shape(input_shape2)

        stress_test_tensor_operations((input_shape1, input_shape2))

    except ValueError as ve:
        print(f"Value Error: {ve}")
    except Exception as e:
        print(f"Error: {str(e)}")
