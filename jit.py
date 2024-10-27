import torch

# Define a simple neural network
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Create a TorchScript version of the model
scripted_model = torch.jit.script(SimpleNN())

def generate_random_tensor(shape):
    # Generate a tensor with specified dimensions
    tensor = torch.randn(*shape)
    print(f"Created Tensor Shape: {tensor.shape}")
    return tensor

# Valid Input Version
def valid_input_version(num_runs=10):
    print("\n--- Valid Input Version ---")
    for i in range(num_runs):
        print(f"\nValid Input Run {i + 1}:")

        # Use a fixed valid shape
        input_shape = (1, 10)  # Valid shape
        tensor = generate_random_tensor(input_shape)

        # Use TorchScript to run the model
        output = scripted_model(tensor)

        print("Output Shape after TorchScript:", output.shape)

# Execute the valid input version
valid_input_version()
