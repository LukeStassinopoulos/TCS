import torch
import torch.nn as nn
import numpy as np

# Set the seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to compare gradients
def compare_gradients(cpu_grads, gpu_grads, tolerance=1e-6):
    # Move both gradients to CPU for comparison
    cpu_grads = [grad.cpu() for grad in cpu_grads]
    gpu_grads = [grad.cpu() for grad in gpu_grads]

    # Compare each gradient
    for cpu_grad, gpu_grad in zip(cpu_grads, gpu_grads):
        try:
            # Check for NaNs
            if torch.isnan(cpu_grad).any() or torch.isnan(gpu_grad).any():
                print("NaNs detected in gradients.")
                return False

            # Check for significant discrepancies
            if not torch.allclose(cpu_grad, gpu_grad, atol=tolerance):
                print("Discrepancy detected in gradients.")
                return False
        except (RuntimeError, TypeError):
            # Handle dtype or shape mismatch errors silently
            print("Error comparing gradients. Skipping...")
            return False

    print("Gradients match within tolerance.")
    return True

# Main function
def main():
    # Initialize the model and move it to the desired device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleNN().to(device)

    # Generate random input tensor
    inputs = torch.randn(1, 10).to(device)

    # Perform forward pass
    outputs = model(inputs)
    loss = outputs.mean()

    # Compute gradients
    loss.backward()

    # Get gradients
    cpu_grads = [param.grad.clone() for param in model.parameters()]
    model.zero_grad()  # Reset gradients

    # Move model to GPU (if available) and repeat the process
    if torch.cuda.is_available():
        model = model.to("cuda")
        inputs = inputs.to("cuda")
        outputs = model(inputs)
        loss = outputs.mean()
        loss.backward()
        gpu_grads = [param.grad.clone() for param in model.parameters()]
        compare_gradients(cpu_grads, gpu_grads)

if __name__ == "__main__":
    main()