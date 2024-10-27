import torch
import torch.nn as nn
import random

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation for first layer
        x = self.fc2(x)  # Output layer
        return x

def generate_random_shape():
    # Generate a random shape for input tensor
    return (random.randint(1, 5), random.randint(1, 10))  # Example: (batch_size, features)

def mutate_tensor_params(params):
    # Randomly mutate parameters
    mutated_params = list(params)  # Convert tuple to list for mutability
    for i in range(len(mutated_params)):
        mutation = random.choice([-1, 0, random.randint(-2, 2)])
        mutated_params[i] += mutation
        if mutated_params[i] < 0:
            mutated_params[i] = 0  # Ensure non-negative for dimension sizes
    return tuple(mutated_params)  # Return back to tuple

def stress_test_module_operations(module_class, params):
    try:
        input_size, hidden_size, output_size = params
        
        # Create the neural network module
        model = module_class(input_size, hidden_size, output_size)
        print(f"Model Structure:\n{model}\n")
        
        # Simulating a stress test: create new input tensors based on mutated parameters
        for i in range(10):  # Repeat for 10 outputs
            mutated_input_size = mutate_tensor_params((1, input_size))  # Batch size of 1
            
            # Generate input tensor
            input_tensor = torch.randn(*mutated_input_size)
            print(f"{'Output ' + str(i + 1) + ':':<25}")
            print(f"{'Input Tensor Shape:':<30} {input_tensor.shape}")
            print(f"{'Input Tensor Values:':<30}\n{input_tensor}\n")

            # Perform forward pass through the model
            output_tensor = model(input_tensor)
            print(f"{'Output Tensor Shape:':<30} {output_tensor.shape}")
            print(f"{'Output Tensor Values:':<30}\n{output_tensor}\n")
    
    except Exception as e:
        print(f"Stress Test Exception: {str(e)}")

# Define the parameters for the neural network
nn_params = (10, 5, 2)  # (input_size, hidden_size, output_size)
stress_test_module_operations(SimpleNN, nn_params)
