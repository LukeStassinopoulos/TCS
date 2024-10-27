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

# User Input Version
def user_input_version(num_runs=10):
    print("\n--- User Input Version ---")
    print("Enter tensor shape in the format 'batch_size, features':")
    user_input = input()  # User input for tensor shape
    try:
        user_shape = tuple(map(int, user_input.split(',')))  # Parse input
        
        for i in range(num_runs):
            print(f"\nUser Input Run {i + 1}:")

            # Generate the initial tensor
            tensor = generate_random_tensor(user_shape)

            # Use TorchScript to run the model
            output = scripted_model(tensor)

            print("Output Shape after TorchScript:", output.shape)

    except Exception as e:
        print(f"Invalid input or execution error: {e}")

# Execute the user input version
user_input_version()
