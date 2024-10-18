def fuzz_autograd():
    for _ in range(100):  # Adjust number of tests as needed
        tensor = generate_random_tensor()
        tensor.requires_grad = True
        model = generate_random_model()
        output = model(tensor)
        output.sum().backward()

        if tensor.grad is not None and torch.isnan(tensor.grad).any():
            print("NaN detected in gradients.")

fuzz_autograd()
