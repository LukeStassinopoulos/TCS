import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys

# Dummy dataset
def get_data(num_samples=1000):
    x = np.random.rand(num_samples, 10).astype(np.float32)
    y = np.random.randint(0, 2, size=(num_samples,)).astype(np.int64)
    return TensorDataset(torch.tensor(x), torch.tensor(y))

# Simple feedforward model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

def train(rank, world_size, num_epochs=5):
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset and DataLoader
    dataset = get_data()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)

    # Initialize model, loss function, and optimizer
    model = SimpleModel().to(device)
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Rank {rank}, Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Check model parameters
    return {name: param.data.clone() for name, param in model.named_parameters()}

def compare_models(model_params_list):
    # Compare model parameters across different processes
    for key in model_params_list[0].keys():
        for i in range(1, len(model_params_list)):
            if not torch.allclose(model_params_list[0][key], model_params_list[i][key], atol=1e-6):
                print(f"Inconsistency found in parameter '{key}' between rank 0 and rank {i}")

def main():
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    rank = 0

    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        rank = dist.get_rank()

    try:
        model_params = train(rank, world_size)
        
        # Gather model parameters across all ranks
        model_params_list = [None] * world_size
        if world_size > 1:
            model_params_list[rank] = model_params
            dist.all_gather_object(model_params_list, model_params)
        else:
            model_params_list[0] = model_params

        # Compare models
        compare_models(model_params_list)

    except Exception as e:
        print(f"Error in process {rank}: {e}", file=sys.stderr)
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    main()
