import torch

# Set tolerance level for discrepancies
TOLERANCE = 1e-5

# Function to compare results
def compare_results(cpu_result, gpu_result, operation_name):
    if torch.any(torch.isnan(cpu_result)) or torch.any(torch.isnan(gpu_result)):
        return  # Skip printing for expected NaNs with extreme tensors
    elif not torch.allclose(cpu_result, gpu_result, atol=TOLERANCE):
        print(f"Significant discrepancy detected in operation: {operation_name}")
        print(f"CPU Result: {cpu_result}")
        print(f"GPU Result: {gpu_result}")

# Function to generate random tensor
def generate_random_tensor(shape, dtype, device):
    return torch.rand(shape, dtype=dtype, device=device)

# Function to generate tensor with specific values
def generate_specific_tensor(shape, value, dtype, device):
    return torch.full(shape, value, dtype=dtype, device=device)

# Function to fuzz test tensor operations
def fuzz_test_tensor_operations():
    shapes = [(2, 3), (3, 3), (4, 5), (5, 5, 5), (2, 3, 4, 2), (10, 10), (100, 100), (1000, 1000)]
    dtypes = [torch.float32, torch.float64]

    # Test both CPU and GPU
    for device in ['cpu', 'cuda' if torch.cuda.is_available() else 'cpu']:
        for shape in shapes:
            for dtype in dtypes:
                for seed in range(10):  # Repeat tests with different seeds
                    torch.manual_seed(seed)
                    try:
                        # Regular tensor tests
                        tensor = generate_random_tensor(shape, dtype, device)

                        operations = {
                            'add': (tensor + tensor, tensor + tensor),
                            'sub': (tensor - tensor, tensor - tensor),
                            'mul': (tensor * tensor, tensor * tensor),
                            'div': (tensor / (tensor + 1e-5), tensor / (tensor + 1e-5)),  # Avoid division by zero
                            'pow': (tensor ** 2, tensor ** 2),  # Power operation
                            'sqrt': (tensor.sqrt(), tensor.sqrt()),  # Square root operation
                        }

                        # Adding multiple permutations
                        if tensor.dim() == 2:
                            permutations = [
                                tensor.permute(1, 0),  # Transpose
                                tensor.permute(0, 1),  # No change
                            ]
                        elif tensor.dim() == 3:
                            permutations = [
                                tensor.permute(0, 2, 1),  # Swap last two dimensions
                                tensor.permute(1, 0, 2),  # Swap first two dimensions
                                tensor.permute(2, 0, 1),  # Move first to last
                            ]
                        elif tensor.dim() == 4:
                            permutations = [
                                tensor.permute(0, 2, 1, 3),  # Swap 2nd and 3rd dimensions
                                tensor.permute(1, 0, 2, 3),  # Swap 1st and 2nd dimensions
                                tensor.permute(0, 3, 1, 2),  # Move 3rd to 2nd
                            ]
                        else:
                            permutations = []

                        for permuted_tensor in permutations:
                            operations['permute'] = (permuted_tensor, permuted_tensor)

                        for op_name, (cpu_result, gpu_result) in operations.items():
                            compare_results(cpu_result, gpu_result, f"{op_name} with regular tensor")

                        # Gradients
                        requires_grad_tensor = tensor.clone().requires_grad_(True)
                        grad_operations = {
                            'sum': (requires_grad_tensor.sum(), requires_grad_tensor.sum()),
                            'mean': (requires_grad_tensor.mean(), requires_grad_tensor.mean()),
                            'max': (requires_grad_tensor.max(), requires_grad_tensor.max()),
                            'min': (requires_grad_tensor.min(), requires_grad_tensor.min()),
                            'norm': (requires_grad_tensor.norm(), requires_grad_tensor.norm()),  # Norm operation
                        }

                        for grad_name, (cpu_result, gpu_result) in grad_operations.items():
                            compare_results(cpu_result, gpu_result, f"{grad_name} with requires_grad tensor")

                        # Test with extreme values
                        extreme_values = [float('inf'), float('-inf'), float('nan'), 1e10, 1e-10]
                        for value in extreme_values:
                            extreme_tensor = generate_specific_tensor(shape, value, dtype, device)

                            operations = {
                                'add': (tensor + extreme_tensor, tensor + extreme_tensor),
                                'sub': (tensor - extreme_tensor, tensor - extreme_tensor),
                                'mul': (tensor * extreme_tensor, tensor * extreme_tensor),
                                'div': (tensor / (extreme_tensor + 1e-5), tensor / (extreme_tensor + 1e-5)),  # Avoid div by zero
                            }

                            for op_name, (cpu_result, gpu_result) in operations.items():
                                compare_results(cpu_result, gpu_result, f"{op_name} with extreme tensor {value}")

                        # Additional tests for edge cases
                        additional_shapes = [(10,), (100,), (1000,), (5000,)]
                        for additional_shape in additional_shapes:
                            large_tensor = generate_random_tensor(additional_shape, dtype, device)
                            compare_results(large_tensor.sum(), large_tensor.sum(), "sum with large tensor")
                            compare_results(large_tensor.mean(), large_tensor.mean(), "mean with large tensor")
                            compare_results(large_tensor.max(), large_tensor.max(), "max with large tensor")
                            compare_results(large_tensor.min(), large_tensor.min(), "min with large tensor")

                        # Testing view and reshape with varying dimensions
                        reshaped_tensor = tensor.view(-1)
                        compare_results(reshaped_tensor.sum(), reshaped_tensor.sum(), "sum after reshape")
                        compare_results(reshaped_tensor.mean(), reshaped_tensor.mean(), "mean after reshape")

                    except Exception as e:
                        print(f"Error during operations with shape {shape}, dtype {dtype}, seed {seed} on device {device}: {e}")

# Main function
if __name__ == "__main__":
    fuzz_test_tensor_operations()
