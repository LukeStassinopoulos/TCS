import torch
import random

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

def mutate_tensor_shape(shape):
    # Randomly mutate parameters
    mutated_shape = list(shape)
    for i in range(len(mutated_shape)):
        mutation = random.choice([-1, 0, random.randint(-5, 5)])  # Adjust mutation values
        mutated_shape[i] += mutation
        if mutated_shape[i] <= 0:
            mutated_shape[i] = random.randint(1, 10)  # Ensure valid size
    return tuple(mutated_shape)

# Randomly Mutated Inputs Version
def random_mutation_version(num_runs=10):
    print("\n--- Random Mutation Version ---")
    for i in range(num_runs):
        print(f"\nRun {i + 1}:")
        
        input_shape = (1, 10)  # Starting shape for valid input
        try:
            # Generate the initial tensor
            tensor = generate_random_tensor(input_shape)

            # Use TorchScript to run the model
            output = scripted_model(tensor)

            print("Output Shape after TorchScript:", output.shape)

            # Mutate tensor shape for the next run
            input_shape = mutate_tensor_shape(input_shape)

        except Exception as e:
            print(f"Exception during model execution: {e}")

# Execute the random mutation version
random_mutation_version()
