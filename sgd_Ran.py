import torch
import torch.optim as optim
import random

def mutate_optimizer_params(params):
    mutated_params = params.copy()
    for key in mutated_params:
        mutation = random.choice([-0.02, 0, random.uniform(-0.5, 0.5), -1])  # Include negatives and invalids
        mutated_params[key] += mutation

        # Ensure learning rate is positive, momentum and weight decay must be within valid ranges
        if key == 'lr' and mutated_params[key] < 0:
            mutated_params[key] = -1  # Purposefully invalid
        elif key == 'momentum':
            if mutated_params[key] < 0 or mutated_params[key] > 1:
                mutated_params[key] = 2  # Purposefully invalid
        elif key == 'weight_decay' and mutated_params[key] < 0:
            mutated_params[key] = -0.01  # Purposefully invalid
    return mutated_params

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

# Initial parameters
params = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.01}
for _ in range(10):  # Generate 10 mutated optimizers
    mutated_params = mutate_optimizer_params(params)
    print(f"\nMutated Params: {mutated_params}")
    run_optimizer_test(mutated_params)
