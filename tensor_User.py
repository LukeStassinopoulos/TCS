import torch

def generate_tensor(batch_size, in_channels, height, width):
    # Create a tensor with specified dimensions
    tensor = torch.randn(batch_size, in_channels, height, width)
    print(f"Created Tensor Shape: {tensor.shape}")
    return tensor

def stress_test_tensor_operations(params):
    try:
        batch_size, in_channels, height, width = params
        print(f"Starting stress testing of tensor operations...")

        # Generate the tensor
        tensor = generate_tensor(batch_size, in_channels, height, width)

        # Simulating a stress test: create new tensors based on user-defined parameters
        for _ in range(10):
            try:
                # Attempt to create a tensor with parameters
                test_tensor = torch.randn(batch_size, in_channels, height, width)

                # Example operation 1: Summation across batch and channels
                output_tensor_sum = test_tensor.sum(dim=(0, 1))  # Summing over batch and channels
                print(f"Output Tensor Shape After Summation: {output_tensor_sum.shape}")

                # Example operation 2: Mean across batch and channels
                output_tensor_mean = test_tensor.mean(dim=(0, 1))  # Mean over batch and channels
                print(f"Output Tensor Shape After Mean: {output_tensor_mean.shape}")

                # Example operation 3: Max across batch and channels
                output_tensor_max = test_tensor.max(dim=0).values  # Max over batch dimension
                output_tensor_max_channel = test_tensor.max(dim=1).values  # Max over channel dimension
                print(f"Output Tensor Shape After Max (batch): {output_tensor_max.shape}")
                print(f"Output Tensor Shape After Max (channel): {output_tensor_max_channel.shape}")

            except Exception as e:
                print(f"Tensor Creation Exception: {str(e)}")
    
    except Exception as e:
        print(f"Stress Test Exception: {str(e)}")

# User input for tensor parameters
def get_tensor_params():
    batch_size = int(input("Enter batch size (positive integer): "))
    in_channels = int(input("Enter number of input channels (positive integer): "))
    height = int(input("Enter height (positive integer): "))
    width = int(input("Enter width (positive integer): "))
    
    # Check for valid inputs
    if batch_size <= 0 or in_channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("All parameters must be positive integers.")
    
    return [batch_size, in_channels, height, width]

try:
    # Get user input for parameters
    params = get_tensor_params()
    stress_test_tensor_operations(params)
except Exception as e:
    print(f"Input Exception: {str(e)}")
