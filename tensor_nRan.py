import torch
import random

def create_tensor(input_values):
    try:
        # Create a tensor from the input values (could be any type, including string)
        tensor = torch.tensor(input_values)
        return tensor
    except Exception as e:
        print(f"Error creating tensor: {e}")
        return None

def mutate_tensor(tensor):
    # Check if tensor is valid
    if tensor is None:
        return None, None, None
    
    # Mutation 1: Add random float values
    random_values = torch.tensor([random.uniform(0.0, 10.0) for _ in range(tensor.numel())])
    mutated_tensor = tensor + random_values
    
    # Mutation 2: Reshape (only if possible)
    try:
        reshaped_tensor = tensor.view(2, 3)  # Reshape to 2x3 if possible
    except RuntimeError:
        reshaped_tensor = "Skipping Reshape: Not enough elements to reshape to 2x3"

    # Mutation 3: Change data type to float
    float_tensor = tensor.to(torch.float32)

    return mutated_tensor, reshaped_tensor, float_tensor

# Main execution
if __name__ == "__main__":
    for i in range(10):
        # Generate a list of random input values (valid floats and integers)
        input_values = [random.choice([random.uniform(1.0, 10.0), random.randint(1, 10)]) for _ in range(5)]

        # Create the tensor
        tensor = create_tensor(input_values)

        # Display the created tensor
        print(f"Output {i + 1}:")
        if tensor is not None:
            print("Created Tensor:")
            print(tensor)

            # Display tensor properties
            print("\nTensor Properties:")
            print(f"Shape: {tensor.shape}")
            print(f"Data Type: {tensor.dtype}")
            print(f"Device: {tensor.device}")

            # Mutations
            mutated, reshaped, float_tensor = mutate_tensor(tensor)

            print("\nMutated Tensor Data:")
            print(mutated)
            print("\nReshaped Tensor Data:")
            print(reshaped)
            print("\nTensor with Changed Data Type (Float):")
            print(float_tensor)
        print("\n" + "=" * 50 + "\n")  # Separator for outputs
