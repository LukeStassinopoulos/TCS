import torch
import random
import time
import numpy as np
import psutil

def create_random_tensor(shape, requires_grad=True):
    return torch.randn(shape, requires_grad=requires_grad)

def create_random_nn(input_size, hidden_size, output_size):
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size)
    )

def check_for_nan(tensor):
    return torch.isnan(tensor).any().item()

def monitor_memory():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Return memory in MB

def fuzz_test(num_tests=100):
    for _ in range(num_tests):
        # Randomly select parameters for the test
        input_size = random.randint(1, 100)
        hidden_size = random.randint(1, 100)
        output_size = random.randint(1, 10)
        batch_size = random.randint(1, 10)

        model = create_random_nn(input_size, hidden_size, output_size)
        x = create_random_tensor((batch_size, input_size), requires_grad=True)
        target = create_random_tensor((batch_size, output_size))

        # Monitor memory before running the test
        initial_memory = monitor_memory()

        try:
            # Forward pass
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, target)

            # Backward pass
            loss.backward()

            # Check gradients
            for param in model.parameters():
                if param.grad is not None:
                    if check_for_nan(param.grad):
                        print(f"NaN detected in gradients for model: {model}")
            
        except Exception as e:
            # Ignore standard exceptions
            print(f"An exception occurred: {e}")

        # Monitor memory after the test
        final_memory = monitor_memory()
        memory_usage = final_memory - initial_memory
        if memory_usage > 50:  # Arbitrary threshold for significant memory usage
            print(f"Significant memory spike: {memory_usage:.2f} MB for model: {model}")

        # Measure performance
        start_time = time.time()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        duration = time.time() - start_time
        if duration > 0.1:  # Arbitrary threshold for significant duration
            print(f"Slow operation detected: {duration:.4f} seconds for model: {model}")

if __name__ == "__main__":
    fuzz_test()

#No output (no errors)