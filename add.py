import torch
import random

def create_tensors():
    # Create two random tensors of the same shape
    shape = (5,)  # Example shape
    tensor1 = torch.tensor([random.uniform(1.0, 10.0) for _ in range(shape[0])], dtype=torch.float32)
    tensor2 = torch.tensor([random.uniform(1.0, 10.0) for _ in range(shape[0])], dtype=torch.float32)
    return tensor1, tensor2

def add_tensors(tensor1, tensor2):
    try:
        result = torch.add(tensor1, tensor2)
        return result
    except Exception as e:
        print(f"Error adding tensors: {e}")
        return None

# Main execution
if __name__ == "__main__":
    tensor1, tensor2 = create_tensors()
    
    print("Tensor 1:")
    print(tensor1)
    print("\nTensor 2:")
    print(tensor2)

    # Test with valid inputs
    result = add_tensors(tensor1, tensor2)
    if result is not None:
        print("\nResult of Addition:")
        print(result)
