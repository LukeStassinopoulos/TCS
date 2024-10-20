import torch
import torch.nn as nn
import torch.optim as optim

# Set a fixed random seed for reproducibility
torch.manual_seed(42)

# Define a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

def create_random_tensors(device):
    """Create random input tensors."""
    return torch.randn(64, 10, device=device), torch.randn(64, 1, device=device)

def compare_tensors(tensor_cpu, tensor_gpu, tolerance=1e-5):
    """Compare tensors for NaNs or significant discrepancies."""
    discrepancies = {}
    if torch.any(torch.isnan(tensor_cpu)) or torch.any(torch.isnan(tensor_gpu)):
        discrepancies['NaN detected'] = 'At least one tensor contains NaN'
    elif torch.any(torch.abs(tensor_cpu - tensor_gpu) > tolerance):
        discrepancies['Values differ beyond tolerance'] = (tensor_cpu, tensor_gpu)
    return discrepancies

def optimize_model(device, optimizer_class, model, input_tensor, target_tensor, iterations=100):
    """Perform optimization and return updated parameters."""
    optimizer = optimizer_class(model.parameters())
    loss_fn = nn.MSELoss()

    for _ in range(iterations):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()
        optimizer.step()

    return model.state_dict()

def main():
    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    # Initialize model and random input
    model = SimpleModel().to(device)
    input_tensor, target_tensor = create_random_tensors(device)

    # Perform optimization with different optimizers
    optimizers = [optim.SGD, optim.Adam, optim.RMSprop]

    for optimizer_class in optimizers:
        print(f'Testing optimizer: {optimizer_class.__name__}')
        
        # Run optimization on CPU
        model_cpu = SimpleModel().to('cpu')
        input_tensor_cpu, target_tensor_cpu = create_random_tensors('cpu')
        
        # Optimize on CPU
        cpu_params = optimize_model('cpu', optimizer_class, model_cpu, input_tensor_cpu, target_tensor_cpu)

        # Optimize on GPU
        gpu_params = optimize_model(device, optimizer_class, model, input_tensor, target_tensor)

        # Compare parameters
        discrepancies = {}
        discrepancies.update(compare_tensors(cpu_params['fc.weight'], gpu_params['fc.weight'].cpu()))
        discrepancies.update(compare_tensors(cpu_params['fc.bias'], gpu_params['fc.bias'].cpu()))

        if discrepancies:
            print(f'Discrepancies found with {optimizer_class.__name__}: {discrepancies}')
        else:
            print(f'No significant discrepancies found with {optimizer_class.__name__}')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'Error during fuzz testing: {e}')
