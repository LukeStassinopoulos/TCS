import torch
import torch.nn as nn

def device_selection():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_random_input(shape):
    return torch.randn(shape, dtype=torch.float32)

def initialize_weights(layer, seed):
    torch.manual_seed(seed)  # Set seed before initialization
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, a=0.01)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    elif isinstance(layer, nn.Conv2d):
        nn.init.kaiming_uniform_(layer.weight, a=0.01)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

def test_layer(layer, input_tensor):
    try:
        return layer(input_tensor)
    except Exception as e:
        print(f"Error in layer {layer.__class__.__name__}: {e}")
        return None

def compare_outputs(cpu_output, gpu_output, tolerance=1e-5, layer_name=""):
    if cpu_output is None or gpu_output is None:
        return

    if not cpu_output.shape == gpu_output.shape:
        print(f"Shape mismatch in {layer_name}: CPU output {cpu_output.shape}, GPU output {gpu_output.shape}")
        return

    cpu_output = cpu_output.to(gpu_output.device)

    if torch.any(torch.isnan(cpu_output)) or torch.any(torch.isnan(gpu_output)):
        print(f"NaN detected in outputs of {layer_name}.")
    else:
        discrepancy = torch.max(torch.abs(cpu_output - gpu_output))
        if discrepancy > tolerance:
            print(f"Discrepancy detected in {layer_name}: max difference {discrepancy.item()} exceeds tolerance {tolerance}")

def fuzz_test_layer(layer_class, input_shape, seed):
    device = device_selection()
    
    layer = layer_class().to(device)
    layer_cpu = layer_class()

    initialize_weights(layer, seed)
    initialize_weights(layer_cpu, seed)

    layer.eval()
    layer_cpu.eval()

    input_tensor = generate_random_input(input_shape).to(device)
    input_tensor_cpu = input_tensor.cpu()

    cpu_output = test_layer(layer_cpu, input_tensor_cpu)
    gpu_output = test_layer(layer, input_tensor)

    compare_outputs(cpu_output, gpu_output, layer_name=layer.__class__.__name__)

def fuzz_test_activation(activation_class, input_tensor):
    device = device_selection()
    
    activation = activation_class().to(device)
    activation_cpu = activation_class()

    activation.eval()
    activation_cpu.eval()

    cpu_output = test_layer(activation_cpu, input_tensor.cpu())
    gpu_output = test_layer(activation, input_tensor)

    compare_outputs(cpu_output, gpu_output, layer_name=activation.__class__.__name__)

def fuzz_test_loss(loss_class, input_shape, target_shape, seed):
    device = device_selection()
    
    loss_fn = loss_class().to(device)
    loss_fn_cpu = loss_class()

    input_tensor = generate_random_input(input_shape).to(device)
    input_tensor_cpu = input_tensor.cpu()

    # Create target tensor
    if loss_class == nn.CrossEntropyLoss:
        target_tensor = torch.randint(0, target_shape[0], (input_tensor.size(0),), dtype=torch.long).to(device)
    else:
        target_tensor = generate_random_input(target_shape).to(device)

    cpu_loss = loss_fn_cpu(input_tensor_cpu, target_tensor.cpu())
    gpu_loss = loss_fn(input_tensor, target_tensor)

    compare_outputs(cpu_loss.unsqueeze(0), gpu_loss.unsqueeze(0), layer_name=loss_fn.__class__.__name__)

def main():
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    num_classes = 10  # Define the number of classes for classification

    layer_configs = [
        (lambda: nn.Linear(10, 5), (1, 10)),  # Linear layer
        (lambda: nn.Conv2d(3, 6, 3), (1, 3, 24, 24)),  # Conv2d layer
    ]

    for layer_class, input_shape in layer_configs:
        fuzz_test_layer(layer_class, input_shape, seed)

    activations = [nn.ReLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU]
    activation_inputs = generate_random_input((1, 5)).to(device_selection())

    for activation_class in activations:
        fuzz_test_activation(activation_class, activation_inputs)

    loss_functions = [nn.CrossEntropyLoss, nn.MSELoss]
    for loss_class in loss_functions:
        if loss_class == nn.CrossEntropyLoss:
            fuzz_test_loss(loss_class, (1, num_classes), (1,), seed)  # Input for CrossEntropy
        else:
            fuzz_test_loss(loss_class, (1, num_classes), (1, num_classes), seed)  # Input for MSELoss

if __name__ == "__main__":
    main()
