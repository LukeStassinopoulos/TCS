import torch
import torch.fx as fx
import random

# Define a simple function to trace
def my_function(x):
    return x * 2

def generate_input():
    # Generate input from the user
    value = float(input("Enter a value for the input tensor: "))
    return torch.full((10,), value)

def mutate_tensor_params(tensor, mutation_choice):
    # User-based mutation
    mutation = mutation_choice
    tensor += mutation
    return tensor

def run_fx_trace(num_runs=10):
    for i in range(num_runs):
        print(f"\nRun {i + 1}:")
        try:
            # Generate input tensor
            x = generate_input()

            # User provides mutation
            mutation_choice = float(input("Enter mutation amount: "))
            x_mutated = mutate_tensor_params(x.clone(), mutation_choice)

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
