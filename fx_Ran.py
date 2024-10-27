import torch
import torch.fx as fx
import random

# Define a simple function to trace
def my_function(x):
    return x * 2

def generate_input():
    # Generate random input
    return torch.randn(10)

def mutate_tensor_params(tensor):
    # Randomly mutate parameters to create valid or invalid inputs
    mutation = random.choice([random.randint(-1042523456234624626246, 1542436345635465460), random.uniform(5,-4235743587245387567645)])  # Reasonable ranges
    tensor += mutation
    return tensor

def run_fx_trace(num_runs=10):
    for i in range(num_runs):
        print(f"\nRun {i + 1}:")
        try:
            # Generate input tensor
            x = generate_input()

            # Mutate tensor parameters randomly
            x_mutated = mutate_tensor_params(x.clone())

            print(f"Input Tensor:\n{x}\n")
            print(f"Mutated Tensor:\n{x_mutated}\n")

            # Create a symbolic trace
            traced = fx.symbolic_trace(my_function)

            # Execute the traced graph
            output = traced(x_mutated)
            print(f"Output Tensor: {output}\n")

        except Exception as e:
            print(f"Exception during trace execution: {e}")

run_fx_trace()
