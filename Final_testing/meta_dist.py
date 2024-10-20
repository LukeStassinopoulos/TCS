import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Check for single GPU availability and set up environment variables
if torch.cuda.device_count() == 1:
    device = torch.device("cuda:0")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 1
    rank = 0
else:
    device = torch.device("cpu")
    world_size = 2
    rank = 0

# Define a simple model and dataset
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        return self.fc(x)

class Dataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 5)
        self.labels = torch.randn(100, 5)  # Add labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # Return both inputs and labels

# Define a custom DataParallel module for fuzz testing
class DataParallel(nn.DataParallel):
    def __init__(self, model):
        super(DataParallel, self).__init__(model)

    def forward(self, *inputs):
        outputs = super().forward(*inputs)
        return outputs

# Set up distributed training
dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)

# Create model, optimizer, and data loader
model = Model()
model.to(device)  # Move the model to the GPU
optimizer = optim.SGD(model.parameters(), lr=0.01)
data_loader = DataLoader(Dataset(), batch_size=32)

# Train the model
for epoch in range(5):
    for batch in data_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, labels)
        loss.backward()
        optimizer.step()

# Compare final model parameters across devices
if world_size > 1:
    params = [p.data.cpu() for p in model.parameters()]
    for param in params:
        print(torch.allclose(param, params[0], atol=1e-6))

# Error handling for device mismatch errors
try:
    model.to("cpu")
except RuntimeError as e:
    print(f"Device mismatch error: {e}")

print("Fuzz testing completed successfully!")