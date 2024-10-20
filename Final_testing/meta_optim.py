import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set the random seed for reproducibility
torch.manual_seed(42)
random.seed(42)

# Define the model
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def fuzz_test_optimizers(device, model, input_tensor, optimizer_class, optimizer_kwargs):
    # Move model and input tensor to the specified device
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Ensure the input tensor requires gradients
    input_tensor.requires_grad = True

    # Initialize the optimizer
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    # Perform optimization iterations
    for _ in range(10):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_tensor)

        # Calculate loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Optimization step
        optimizer.step()

    # Return the updated model parameters
    return model.state_dict()

def compare_parameters(cpu_params, gpu_params, tolerance=1e-6):
    discrepancies = []
    for (cpu_key, cpu_value), (gpu_key, gpu_value) in zip(cpu_params.items(), gpu_params.items()):
        assert cpu_key == gpu_key

        # Move gpu_value to CPU
        gpu_value = gpu_value.cpu()

        # Check for NaNs
        if torch.isnan(cpu_value).any() or torch.isnan(gpu_value).any():
            discrepancies.append((cpu_key, "NaN"))

        # Check for values that differ beyond the tolerance level
        elif not torch.allclose(cpu_value, gpu_value, atol=tolerance):
            discrepancies.append((cpu_key, torch.max(torch.abs(cpu_value - gpu_value)).item()))

    return discrepancies

def main():
    # Define the model and input tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TestModel()
    input_tensor = torch.randn(1, 10)

    # Move model and input tensor to the selected device
    model.to(device)
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad = True

    # Define the optimizers and their arguments
    optimizers = [
        (optim.SGD, {"lr": 0.01}),
        (optim.Adam, {"lr": 0.01}),
        (optim.RMSprop, {"lr": 0.01}),
    ]

    # Perform fuzz testing
    for optimizer_class, optimizer_kwargs in optimizers:
        print(f"Testing {optimizer_class.__name__}...")

        # Initialize the optimizer
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

        # Perform optimization iterations
        for _ in range(10):
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(input_tensor)

            # Calculate loss
            loss = output.sum()

            # Backward pass
            loss.backward()

            # Optimization step
            optimizer.step()

        # Get the updated model parameters
        updated_params = model.state_dict()

        # Compare with CPU parameters (if testing on GPU)
        if str(device) == "cuda:0":
            # Move model and input tensor to CPU
            model.to("cpu")
            input_tensor = input_tensor.to("cpu")

            # Initialize the optimizer
            cpu_optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

            # Perform optimization iterations
            for _ in range(10):
                # Zero gradients
                cpu_optimizer.zero_grad()

                # Forward pass
                output = model(input_tensor)

                # Calculate loss
                loss = output.sum()

                # Backward pass
                loss.backward()

                # Optimization step
                cpu_optimizer.step()

            # Get the updated CPU model parameters
            cpu_params = model.state_dict()

            # Compare parameters
            discrepancies = compare_parameters(cpu_params, updated_params)

            if discrepancies:
                print(f"Discrepancies found in {optimizer_class.__name__}:")
                for key, value in discrepancies:
                    print(f"  {key}: {value}")
            else:
                print(f"No discrepancies found in {optimizer_class.__name__}.")

        else:
            print(f"No discrepancies found in {optimizer_class.__name__}.")

if __name__ == "__main__":
    main()