import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from multiprocessing import Process
import time

# Define a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

def init_process(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    # Random input and model sizes
    input_size = random.randint(1, 100)
    output_size = random.randint(1, 100)

    # Create random model and wrap in DistributedDataParallel
    model = SimpleModel(input_size, output_size)
    model = nn.parallel.DistributedDataParallel(model)

    # Generate random data
    data_size = random.randint(1, 1000)
    data = torch.randn(data_size, input_size).to(rank)
    target = torch.randint(0, output_size, (data_size,)).to(rank)

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=random.uniform(0.001, 0.1))

    # Perform a training step
    for _ in range(random.randint(1, 10)):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

    print(f"Process {rank} finished training with loss: {loss.item()}")

def run_fuzz_test(world_size):
    processes = []
    for rank in range(world_size):
        p = Process(target=init_process, args=(rank, world_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    # Number of processes to test
    world_size = 4
    run_fuzz_test(world_size)

#Didn't define MASTER_ADDR and MASTER_PORT