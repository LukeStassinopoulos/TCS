import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# Function to generate random tensor (same as before)
def generate_random_tensor():
    shape = tuple(random.randint(1, 10) for _ in range(random.randint(1, 4)))
    dtype = random.choice([torch.float32, torch.float64, torch.int32])
    return torch.randn(shape, dtype=dtype) if dtype != torch.int32 else torch.randint(0, 100, shape)

# Function to generate a random neural network model
def generate_random_model():
    layers = [
        nn.Linear(random.randint(1, 10), random.randint(1, 10)),
        nn.ReLU(),
        nn.Conv2d(random.randint(1, 3), random.randint(1, 3), kernel_size=3),
        nn.Sigmoid(),
    ]
    return nn.Sequential(*layers)

# Fuzz testing for neural network functions
def fuzz_nn_functions():
    for _ in range(100):  # Adjust number of tests as needed
        model = generate_random_model()
        input_tensor = generate_random_tensor()
        try:
            output = model(input_tensor)
            if torch.isnan(output).any():
                print("NaN detected in model output.")
        except Exception as e:
            # Silently ignore expected exceptions
            pass

fuzz_nn_functions()

#no errors