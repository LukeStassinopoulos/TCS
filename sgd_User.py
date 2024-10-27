import torch
import torch.optim as optim

def create_optimizer(params):
    model = torch.nn.Linear(10, 2)  # Example model
    optimizer = optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'], weight_decay=params['weight_decay'])
    return optimizer

def run_optimizer_test(params):
    try:
        optimizer = create_optimizer(params)
        print("Optimizer created with parameters:", params)
    except Exception as e:
        print(f"Optimizer Creation Exception: {str(e)}")

# User input for optimizer parameters
def get_optimizer_params():
    lr = float(input("Enter learning rate (must be positive): "))
    momentum = float(input("Enter momentum (0 to 1): "))
    weight_decay = float(input("Enter weight decay (non-negative): "))
    
    # Check for valid inputs
    if lr <= 0:
        raise ValueError("Learning rate must be positive.")
    if momentum < 0 or momentum > 1:
        raise ValueError("Momentum must be between 0 and 1.")
    if weight_decay < 0:
        raise ValueError("Weight decay must be non-negative.")
    
    return {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}

try:
    # Get user input for parameters
    params = get_optimizer_params()
    run_optimizer_test(params)
except Exception as e:
    print(f"Input Exception: {str(e)}")
