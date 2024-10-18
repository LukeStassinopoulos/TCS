import torch
import random

# Function to generate random tensor
def generate_random_tensor():
    shape = tuple(random.randint(1, 10) for _ in range(random.randint(1, 4)))
    dtype = random.choice([torch.float32, torch.float64, torch.int32])
    return torch.randn(shape, dtype=dtype) if dtype != torch.int32 else torch.randint(0, 100, shape)

# Fuzz testing for optimization algorithms
def fuzz_optimizers():
    for _ in range(100):  # Adjust number of tests as needed
        model = torch.nn.Sequential(
            torch.nn.Linear(random.randint(1, 10), random.randint(1, 10)),
            torch.nn.ReLU()
        )
        optimizer = torch.optim.Adam(model.parameters())
        input_tensor = generate_random_tensor()
        
        output = model(input_tensor)
        loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for param in model.parameters():
            if torch.isnan(param).any():
                print("NaN detected in parameters after optimization.")

if __name__ == "__main__":
    fuzz_optimizers()

# Another error HYPEEEE