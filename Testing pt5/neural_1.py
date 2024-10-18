import torch.nn as nn
import torch.nn.functional as F

def generate_random_model():
    layers = [
        nn.Linear(random.randint(1, 10), random.randint(1, 10)),
        nn.ReLU(),
        nn.Conv2d(random.randint(1, 3), random.randint(1, 3), kernel_size=3),
        nn.Sigmoid(),
    ]
    return nn.Sequential(*layers)

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

#BASIC ERRORS :(