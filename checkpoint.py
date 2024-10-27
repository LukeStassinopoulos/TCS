import torch
import torch.utils.checkpoint as checkpoint

def generate_tensor(shape):
    return torch.randn(*shape)

def sample_function(x):
    return x * 2

def run_checkpoint_test_valid():
    shape = (2, 2)  # Valid tensor shape

    for i in range(10):
        # Generate input tensor
        input_tensor = generate_tensor(shape)

        # Run checkpoint function
        output = checkpoint.checkpoint(sample_function, input_tensor)

        # Print outputs
        print(f"\nOutput {i + 1}:")
        print(f"Input Tensor Shape: {input_tensor.shape}")
        print(f"Output Shape: {output.shape}")

# Run the test with valid inputs
run_checkpoint_test_valid()
