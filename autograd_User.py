import torch

def generate_tensor_from_user_input():
    # Allow user to input the shape of the tensor
    try:
        batch_size = int(input("Enter batch size: "))
        channels = int(input("Enter number of channels: "))
        height = int(input("Enter height: "))
        width = int(input("Enter width: "))
        
        # Ensure valid dimensions
        if batch_size <= 0 or channels <= 0 or height <= 0 or width <= 0:
            raise ValueError("Tensor dimensions must be positive integers.")
        
        # Create tensor with user-defined shape
        tensor = torch.randn(batch_size, channels, height, width, requires_grad=True)
        print(f"Created Tensor with shape {tensor.shape}")
        return tensor

    except Exception as e:
        raise ValueError(f"Input Error: {str(e)}")

def run_autograd_with_user_input():
    try:
        # Get tensor from user input
        tensor = generate_tensor_from_user_input()

        # Example autograd operation: gradient computation
        output = tensor.sum()
        output.backward()
        print(f"Gradient of the tensor after sum operation:\n{tensor.grad}")

    except Exception as e:
        print(f"Autograd Exception: {str(e)}")

# Start the autograd process with user input
run_autograd_with_user_input()
