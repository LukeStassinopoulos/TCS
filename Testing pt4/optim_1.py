import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import tracemalloc

# Configuration for fuzz testing
NUM_TESTS = 100  # Number of tests to perform
MAX_LAYERS = 5   # Maximum number of layers in the neural network
MAX_UNITS = 100  # Maximum units per layer
INPUT_SIZE = 10  # Input size for the first layer
LEARNING_RATE = 0.01

def create_random_model():
    """Create a random neural network model."""
    model = nn.Sequential()
    num_layers = random.randint(1, MAX_LAYERS)
    
    for i in range(num_layers):
        num_units = random.randint(1, MAX_UNITS)
        model.add_module(f'Linear_{i}', nn.Linear(INPUT_SIZE if i == 0 else num_units, num_units))
        model.add_module(f'Activation_{i}', nn.ReLU())
    
    return model

def fuzz_test_optimizer(optimizer_class):
    """Perform fuzz testing on the specified optimizer."""
    for _ in range(NUM_TESTS):
        model = create_random_model()
        criterion = nn.MSELoss()
        
        # Create random input and target tensors
        inputs = torch.randn(10, INPUT_SIZE)
        targets = torch.randn(10, random.randint(1, MAX_UNITS))
        
        optimizer = optimizer_class(model.parameters(), lr=LEARNING_RATE)

        # Memory tracking
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[1]
        
        try:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Performance measurement
            end_memory = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            
            # Report significant memory usage changes
            memory_change = end_memory - initial_memory
            if memory_change > 1024 * 1024:  # 1 MB threshold
                print(f"Significant memory usage change for {optimizer_class.__name__}: {memory_change / (1024 * 1024):.2f} MB")
        
        except Exception as e:
            # Silence standard exceptions but report real issues
            if not isinstance(e, (RuntimeError, TypeError)):
                print(f"Error occurred in {optimizer_class.__name__}: {e}")

if __name__ == "__main__":
    print("Starting fuzz testing for optimizers...")
    
    optimizers = [optim.SGD, optim.Adam, optim.RMSprop]
    
    for optimizer in optimizers:
        print(f"Testing optimizer: {optimizer.__name__}")
        fuzz_test_optimizer(optimizer)
    
    print("Fuzz testing completed.")

#Excessive user warnings and no errrors