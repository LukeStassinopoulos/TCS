import torch
import torch.nn.functional as F
import numpy as np
import random

# Set default dtype and tolerance
DEFAULT_DTYPE = torch.float32
TOLERANCE = 1e-5

# Function to generate random tensor
def generate_random_tensor(shape, dtype=DEFAULT_DTYPE, device='cpu'):
    if dtype in [torch.int32, torch.long]:
        tensor = torch.randint(0, 100, shape, dtype=dtype, device=device)
    else:
        tensor = torch.randn(shape, dtype=dtype, device=device)
    return tensor

# Function to compare tensors
def compare_tensors(tensor1, tensor2):
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    
    diff = torch.abs(tensor1 - tensor2)
    max_diff = torch.max(diff).item()
    
    if max_diff > TOLERANCE or torch.isnan(tensor1).any() or torch.isnan(tensor2).any():
        return False
    else:
        return True

# Function to fuzz test tensor operations
def fuzz_test_tensor_operations():
    # Select device dynamically
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate random tensors
    tensor_shapes = [(3, 3), (5, 5), (10, 10)]
    dtypes = [torch.float32, torch.int32, torch.double]
    
    for shape in tensor_shapes:
        for dtype in dtypes:
            tensor1 = generate_random_tensor(shape, dtype, device)
            tensor2 = generate_random_tensor(shape, dtype, device)
            
            # Arithmetic operations
            try:
                cpu_result = tensor1 + tensor2
                gpu_result = tensor1.to(device) + tensor2.to(device)
                assert compare_tensors(cpu_result, gpu_result)
            except Exception as e:
                print(f"Addition failed for shape {shape}, dtype {dtype}: {e}")
                
            try:
                cpu_result = tensor1 - tensor2
                gpu_result = tensor1.to(device) - tensor2.to(device)
                assert compare_tensors(cpu_result, gpu_result)
            except Exception as e:
                print(f"Subtraction failed for shape {shape}, dtype {dtype}: {e}")
                
            try:
                cpu_result = tensor1 * tensor2
                gpu_result = tensor1.to(device) * tensor2.to(device)
                assert compare_tensors(cpu_result, gpu_result)
            except Exception as e:
                print(f"Multiplication failed for shape {shape}, dtype {dtype}: {e}")
                
            # Reduction operations
            try:
                cpu_result = torch.sum(tensor1)
                gpu_result = torch.sum(tensor1.to(device))
                assert compare_tensors(cpu_result, gpu_result)
            except Exception as e:
                print(f"Sum failed for shape {shape}, dtype {dtype}: {e}")
                
            try:
                cpu_result = torch.mean(tensor1)
                gpu_result = torch.mean(tensor1.to(device))
                assert compare_tensors(cpu_result, gpu_result)
            except Exception as e:
                print(f"Mean failed for shape {shape}, dtype {dtype}: {e}")
                
            # Reshaping operations
            try:
                cpu_result = tensor1.view(-1)
                gpu_result = tensor1.to(device).view(-1)
                assert compare_tensors(cpu_result, gpu_result)
            except Exception as e:
                print(f"View failed for shape {shape}, dtype {dtype}: {e}")
                
            try:
                cpu_result = tensor1.reshape(-1)
                gpu_result = tensor1.to(device).reshape(-1)
                assert compare_tensors(cpu_result, gpu_result)
            except Exception as e:
                print(f"Reshape failed for shape {shape}, dtype {dtype}: {e}")
                
            # Indexing operations
            try:
                cpu_result = tensor1[1:, 1:]
                gpu_result = tensor1.to(device)[1:, 1:]
                assert compare_tensors(cpu_result, gpu_result)
            except Exception as e:
                print(f"Indexing failed for shape {shape}, dtype {dtype}: {e}")

if __name__ == "__main__":
    fuzz_test_tensor_operations()