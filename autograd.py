import torch

def generate_valid_tensor():
    # Generate a valid tensor with a predefined shape
    batch_size, channels, height, width = 3, 3, 28, 28  # A typical image tensor shape
    tensor = torch.randn(batch_size, channels, height, width, requires_grad=True)
    print(f"Created Tensor with shape {tensor.shape}")
    return tensor

def run_valid_autograd(num_iterations=10):
    for i in range(num_iterations):
        try:
            print(f"\n=== Iteration {i + 1} ===")
            
            # Generate a valid tensor
            tensor = generate_valid_tensor()

            # Perform a valid autograd operation: gradient computation
            output = tensor.sum()  # Summing all the elements of the tensor
            output.backward()  # Perform backpropagation to calculate the gradients
            print(f"Gradient of the tensor after sum operation:\n{tensor.grad}")

            # Additional operation: mean calculation
            mean_output = tensor.mean()
            print(f"Mean of the tensor: {mean_output.item()}")

        except Exception as e:
            print(f"Autograd Exception: {str(e)}")

# Start the autograd process with valid inputs for 10 iterations
run_valid_autograd(num_iterations=10)




