import torch
import torch.nn as nn
import torch.optim as optim

# Set a random seed for reproducibility
torch.manual_seed(42)

# Simple neural network definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def test_gradient_calculation(device):
    model = SimpleNN().to(device)
    
    # Generate random input and target tensors
    input_tensor = torch.randn(3, 10, device=device)
    target_tensor = torch.randn(3, 2, device=device)
    
    # Define loss functions
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    # Create a list of criteria to test
    criteria = [mse_criterion, l1_criterion]
    
    for criterion in criteria:
        try:
            # Forward pass
            output = model(input_tensor)

            # Compute the loss
            loss = criterion(output, target_tensor)

            # Backward pass
            model.zero_grad()
            loss.backward()

            # Gather gradients
            gradients = {name: param.grad.clone().detach() for name, param in model.named_parameters()}
            
            # Check for NaN or unusual values
            for name, grad in gradients.items():
                if torch.isnan(grad).any():
                    print(f'NaN detected in gradient of {name} using {criterion.__class__.__name__}!')
                elif torch.max(torch.abs(grad)) > 1e5:  # Example threshold
                    print(f'Unusually large gradient detected in {name} using {criterion.__class__.__name__}: {grad.max().item()}')

            # Check gradients using finite difference method
            check_gradients(model, input_tensor, target_tensor, criterion, device)

        except Exception as e:
            print(f'Error during gradient calculation with {criterion.__class__.__name__}: {e}')

def check_gradients(model, input_tensor, target_tensor, criterion, device, epsilon=1e-5):
    # Ensure model is in evaluation mode
    model.eval()
    
    # Compute baseline loss
    baseline_loss = criterion(model(input_tensor), target_tensor).item()

    # Iterate over model parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Save original value
            original_value = param.data.clone()
            
            # Compute the numerical gradient
            param.data += epsilon
            loss_plus = criterion(model(input_tensor), target_tensor).item()
            
            param.data -= 2 * epsilon
            loss_minus = criterion(model(input_tensor), target_tensor).item()
            
            # Restore original value
            param.data = original_value
            
            # Numerical gradient calculation
            numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)
            analytical_gradient = param.grad.flatten().sum().item()

            # Compare numerical and analytical gradients
            if not torch.isclose(torch.tensor(numerical_gradient), torch.tensor(analytical_gradient), atol=1e-5):
                print(f'Discrepancy detected in {name}: Analytical {analytical_gradient}, Numerical {numerical_gradient}')

if __name__ == '__main__':
    # Dynamic device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # Run the gradient calculation tests
    test_gradient_calculation(device)
