import torch
import torch.fx as fx

# Define a simple function to trace
def my_function(x):
    return x * 2

def generate_valid_input():
    # Generate only valid inputs
    return torch.randn(10)

def run_fx_trace(num_runs=10):
    for i in range(num_runs):
        print(f"\nRun {i + 1}:")
        try:
            # Generate input tensor
            x = generate_valid_input()
            print(f"Input Tensor:\n{x}\n")

            # Create a symbolic trace
            traced = fx.symbolic_trace(my_function)

            # Execute the traced graph
            output = traced(x)
            print(f"Output Tensor: {output}\n")

        except Exception as e:
            print(f"Exception during trace execution: {e}")

run_fx_trace()
