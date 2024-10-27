import torch
import random
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features=10, out_features=5, bias=True)
        self.fc2 = nn.Linear(in_features=5, out_features=2, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def generate_tensor(shape):
    # Create a tensor with specified dimensions
    tensor = torch.randn(*shape)
    print(f"Created Tensor Shape: {tensor.shape}")
    return tensor

def mutate_tensor_params(params):
    # Randomly mutate parameters
    mutated_params = params.copy()
    for i in range(len(mutated_params)):
        mutation = random.choice([-1, 0, random.randint(-5, 5)])
        mutated_params[i] += mutation
        if mutated_params[i] < 0:
            mutated_params[i] = 0  # Ensure non-negative for dimension sizes
    
    # Ensure the second dimension is always 10 for compatibility with the model
    if len(mutated_params) > 1:
        mutated_params[1] = 10
    return mutated_params

def run_hd_techniques(model, input_shape, num_outputs=20):
    for i in range(num_outputs):
        print(f"\nOutput {i + 1}:")

        # Generate initial input tensor
        try:
            tensor = generate_tensor(input_shape)
            print("\nInput Tensor Shape:", tensor.shape)
            print("Input Tensor Values:\n", tensor)

            # Forward pass through the model
            output = model(tensor)
            print("\nOutput Tensor Shape:", output.shape)
            print("Output Tensor Values:\n", output)

        except Exception as e:
            print(f"Stress Test Exception: {str(e)}")

        # Mutate tensor parameters for the next output
        input_shape = mutate_tensor_params(list(input_shape))

# Initialize the model
model = SimpleNN()

# Start the HD techniques process with initial input shape
input_shape = (1, 10)  # Start with a shape of (1, 10)
run_hd_techniques(model, input_shape)
