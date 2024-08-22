#Based on gpt1_3.py 
# Prompt given of "Generate a script with additional tests, better error handling, and logging to cover more scenarios and edge cases."

# Key Improvements:
# Enhanced Error Handling: Used the log_exception function to handle and log errors with context, making it easier to understand what went wrong and where.

# Additional Tests: Included:

# Testing with different tensor shapes and data types.
# More comprehensive tests for tensor operations, including matrix multiplication with valid shapes.
# Added checks for in-place and out-of-place operations in autograd.
# More detailed neural network layer tests and forward pass checks.
# Various loss functions and their backward passes.
# Logging: Used Python's logging module for structured and detailed logging. This helps track the success or failure of each test and provides context for each step.

# Edge Cases: Included more edge cases and invalid inputs, such as NaN shapes and tensors with Inf values, to ensure the code handles these gracefully.



import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def log_exception(e, context=""):
    logger.error(f"Exception occurred in {context}: {e}")

def test_tensor_creation():
    try:
        logger.info("Testing tensor creation with random shapes and data types...")
        shapes = [
            (random.randint(1, 10), random.randint(1, 10)),
            (random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)),
            (random.randint(1, 10), random.randint(1, 10)),
            (random.randint(1, 10), random.randint(1, 10), random.randint(1, 10), random.randint(1, 10))
        ]
        for shape in shapes:
            try:
                logger.info(f"Creating tensor with shape {shape}")
                tensor = torch.randn(shape)
                logger.info(f"Tensor created: {tensor.shape}")
            except Exception as e:
                log_exception(e, f"tensor creation with shape {shape}")
        
        logger.info("Testing tensors with different data types...")
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64, torch.int8, torch.int16]:
            data = [random.random() for _ in range(random.randint(1, 100))]
            try:
                tensor = torch.tensor(data, dtype=dtype)
                logger.info(f"Tensor created with dtype {dtype}: {tensor.dtype}")
            except Exception as e:
                log_exception(e, f"tensor creation with dtype {dtype}")
        
        logger.info("Testing tensors with invalid shapes...")
        invalid_shapes = [
            (0, 0, 2),  # Example invalid shape
            (-1, 10)    # Example invalid shape
        ]
        for shape in invalid_shapes:
            try:
                logger.info(f"Creating tensor with invalid shape {shape}")
                tensor = torch.randn(shape)
            except Exception as e:
                log_exception(e, f"tensor creation with invalid shape {shape}")
    except Exception as e:
        log_exception(e, "tensor creation")

def test_tensor_operations():
    try:
        logger.info("Testing tensor operations...")
        for shape in [(5, 5), (random.randint(1, 10), random.randint(1, 10))]:
            tensor_a = torch.randn(shape)
            tensor_b = torch.randn(shape)
            
            try:
                torch.add(tensor_a, tensor_b)
                logger.info(f"Addition result: {torch.add(tensor_a, tensor_b)}")
            except Exception as e:
                log_exception(e, f"addition with shape {shape}")
            
            if tensor_a.size(-1) == tensor_b.size(0):  # Ensure matrix multiplication is valid
                try:
                    result = torch.matmul(tensor_a, tensor_b)
                    logger.info(f"Matrix multiplication result: {result}")
                except Exception as e:
                    log_exception(e, f"matrix multiplication with shape {shape}")
            
            try:
                torch.sub(tensor_a, tensor_b)
                torch.mul(tensor_a, tensor_b)
                torch.div(tensor_a, tensor_b)
            except Exception as e:
                log_exception(e, f"operations with shape {shape}")

        logger.info("Testing tensor indexing and slicing...")
        tensor = torch.randn((10, 10))
        try:
            element = tensor[random.randint(0, 9), random.randint(0, 9)]
            logger.info(f"Accessing element: {element}")
            column_slicing = tensor[:, random.randint(0, 9)]
            row_slicing = tensor[random.randint(0, 9), :]
            logger.info(f"Slicing columns: {column_slicing}")
            logger.info(f"Slicing rows: {row_slicing}")
        except Exception as e:
            log_exception(e, "tensor indexing and slicing")

        logger.info("Testing tensor reductions...")
        for shape in [(random.randint(1, 10), random.randint(1, 10))]:
            tensor = torch.randn(shape)
            try:
                tensor_sum = torch.sum(tensor, dim=random.randint(0, len(shape) - 1))
                tensor_mean = torch.mean(tensor, dim=random.randint(0, len(shape) - 1))
                tensor_max = torch.max(tensor, dim=random.randint(0, len(shape) - 1))
                logger.info(f"Sum: {tensor_sum}, Mean: {tensor_mean}, Max: {tensor_max}")
            except Exception as e:
                log_exception(e, f"tensor reductions with shape {shape}")
    except Exception as e:
        log_exception(e, "tensor operations")

def test_nn_layers():
    try:
        logger.info("Testing neural network layers...")
        in_features, out_features = random.randint(1, 100), random.randint(1, 100)
        try:
            nn.Linear(in_features, out_features)
            logger.info(f"Linear layer created with in_features={in_features}, out_features={out_features}")
        except Exception as e:
            log_exception(e, f"Linear layer with in_features={in_features}, out_features={out_features}")
        
        in_channels, out_channels = random.randint(1, 10), random.randint(1, 10)
        try:
            nn.Conv2d(in_channels, out_channels, kernel_size=random.choice([1, 3, 5]))
            logger.info(f"Conv2d layer created with in_channels={in_channels}, out_channels={out_channels}")
        except Exception as e:
            log_exception(e, f"Conv2d layer with in_channels={in_channels}, out_channels={out_channels}")

        logger.info("Testing forward pass...")
        try:
            model = nn.Sequential(nn.Linear(random.randint(1, 10), random.randint(1, 10)), nn.ReLU())
            input_tensor = torch.randn((1, random.randint(1, 10)))
            output = model(input_tensor)
            logger.info(f"Forward pass output: {output}")
        except Exception as e:
            log_exception(e, "forward pass of Linear model")
        
        try:
            model = nn.Sequential(nn.Conv2d(random.randint(1, 10), random.randint(1, 10), kernel_size=random.choice([1, 3, 5])), nn.ReLU())
            input_tensor = torch.randn((1, random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)))
            output = model(input_tensor)
            logger.info(f"Forward pass output: {output}")
        except Exception as e:
            log_exception(e, "forward pass of Conv2d model")
    except Exception as e:
        log_exception(e, "neural network layers")

def test_autograd():
    try:
        logger.info("Testing autograd...")
        x = torch.randn((random.randint(1, 10), random.randint(1, 10)), requires_grad=True)
        y = x**2
        try:
            y.sum().backward()
            logger.info(f"Backward pass successful: {y.grad}")
        except Exception as e:
            log_exception(e, "backward pass after y.sum()")
        
        x = torch.randn((random.randint(1, 10), random.randint(1, 10)), requires_grad=True)
        y = torch.matmul(x, x.T)
        try:
            y.mean().backward()
            logger.info(f"Backward pass successful: {x.grad}")
        except Exception as e:
            log_exception(e, "backward pass after y.mean()")
        
        logger.info("Testing in-place operations...")
        x = torch.randn((5, 5), requires_grad=True)
        try:
            x_out = x.clone().detach()
            x_out += 1  # Out-of-place operation
            x_out.sum().backward()
            logger.info(f"Out-of-place operation successful: {x_out.grad}")
        except Exception as e:
            log_exception(e, "out-of-place operation")

        try:
            x *= 0.5  # In-place operation
            x.sum().backward()
            logger.info(f"In-place operation successful: {x.grad}")
        except Exception as e:
            log_exception(e, "in-place operation")
    except Exception as e:
        log_exception(e, "autograd")

def test_optimizer():
    try:
        logger.info("Testing optimizer steps...")
        model = nn.Linear(random.randint(1, 10), random.randint(1, 10))
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.SGD(model.parameters(), lr=random.random())
        try:
            optimizer.step()
            logger.info("SGD optimizer step executed successfully.")
        except Exception as e:
            log_exception(e, "SGD optimizer step")
        
        model = nn.Linear(random.randint(1, 10), random.randint(1, 10))
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=random.random())
        loss = torch.randn(1, requires_grad=True)
        try:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.info("Adam optimizer step executed successfully.")
        except Exception as e:
            log_exception(e, "Adam optimizer step")
    except Exception as e:
        log_exception(e, "optimizer steps")

def test_loss_functions():
    try:
        logger.info("Testing loss functions...")
        criterion = nn.MSELoss()
        output = torch.randn((random.randint(1, 10), random.randint(1, 10)), requires_grad=True)
        target = torch.randn_like(output)
        try:
            loss = criterion(output, target)
            loss.backward()
            logger.info(f"MSELoss computed successfully: {loss.item()}")
        except Exception as e:
            log_exception(e, "MSELoss computation")
        
        criterion = nn.CrossEntropyLoss()
        output = torch.randn((random.randint(1, 10), random.randint(1, 10)))
        target = torch.randint(0, output.size(1), (output.size(0),))
        try:
            loss = criterion(output, target)
            loss.backward()
            logger.info(f"CrossEntropyLoss computed successfully: {loss.item()}")
        except Exception as e:
            log_exception(e, "CrossEntropyLoss computation")
    except Exception as e:
        log_exception(e, "loss functions")

def test_exceptions():
    try:
        logger.info("Testing exception handling...")
        logger.info("Creating tensor with NaN shape...")
        try:
            tensor = torch.randn((float('nan'), 10))
        except Exception as e:
            log_exception(e, "NaN shape")
        
        logger.info("Creating tensor with Inf value...")
        try:
            tensor = torch.tensor([float('inf')])
            logger.info(f"Tensor with Inf value created: {tensor}")
        except Exception as e:
            log_exception(e, "Inf value")
    except Exception as e:
        log_exception(e, "exception handling")

if __name__ == "__main__":
    test_tensor_creation()
    test_tensor_operations()
    test_nn_layers()
    test_autograd()
    test_optimizer()
    test_loss_functions()
    test_exceptions()

#Output 

# 2024-08-22 21:30:18,393 - INFO - Testing tensor creation with random shapes and data types...
# 2024-08-22 21:30:18,393 - INFO - Creating tensor with shape (5, 3)
# 2024-08-22 21:30:18,394 - INFO - Tensor created: torch.Size([5, 3])
# 2024-08-22 21:30:18,394 - INFO - Creating tensor with shape (38, 50, 86)
# 2024-08-22 21:30:18,397 - INFO - Tensor created: torch.Size([38, 50, 86])
# 2024-08-22 21:30:18,397 - INFO - Creating tensor with shape (9, 5)
# 2024-08-22 21:30:18,398 - INFO - Tensor created: torch.Size([9, 5])
# 2024-08-22 21:30:18,398 - INFO - Creating tensor with shape (2, 8, 2, 3)
# 2024-08-22 21:30:18,398 - INFO - Tensor created: torch.Size([2, 8, 2, 3])
# 2024-08-22 21:30:18,398 - INFO - Testing tensors with different data types...
# 2024-08-22 21:30:18,398 - INFO - Tensor created with dtype torch.float32: torch.float32
# 2024-08-22 21:30:18,398 - INFO - Tensor created with dtype torch.float64: torch.float64
# 2024-08-22 21:30:18,398 - INFO - Tensor created with dtype torch.int32: torch.int32
# 2024-08-22 21:30:18,398 - INFO - Tensor created with dtype torch.int64: torch.int64
# 2024-08-22 21:30:18,399 - INFO - Tensor created with dtype torch.int8: torch.int8
# 2024-08-22 21:30:18,399 - INFO - Tensor created with dtype torch.int16: torch.int16
# 2024-08-22 21:30:18,399 - INFO - Testing tensors with invalid shapes...
# 2024-08-22 21:30:18,399 - INFO - Creating tensor with invalid shape (0, 0, 2)
# 2024-08-22 21:30:18,399 - INFO - Creating tensor with invalid shape (-1, 10)
# 2024-08-22 21:30:18,404 - ERROR - Exception occurred in tensor creation with invalid shape (-1, 10): Trying to create tensor with negative dimension -1: [-1, 10]
# 2024-08-22 21:30:18,404 - INFO - Testing tensor operations...
# 2024-08-22 21:30:18,407 - INFO - Addition result: tensor([[-1.5461, -2.7062, -0.4286, -0.0635, -0.1257],
#         [-1.8575,  0.1190,  0.9568, -0.8371,  0.7848],
#         [ 0.1964,  0.9224,  2.5342,  0.1752, -2.1600],
#         [ 0.8326, -0.3270, -1.0547,  1.3582, -1.1594],
#         [-2.4941, -0.0593,  1.7035, -0.8602,  0.1984]])
# 2024-08-22 21:30:18,409 - INFO - Matrix multiplication result: tensor([[ 2.7737, -0.0376, -2.7629, -0.0867, -0.3537],
#         [ 1.0115,  1.0037,  0.6132, -0.3804,  0.2913],
#         [-3.3417, -0.3615,  0.4446,  0.9387, -0.9711],
#         [-1.9299, -3.0091, -2.5476,  0.0227,  1.0824],
#         [ 3.8309,  4.3830,  1.3799,  1.0330, -3.2318]])
# 2024-08-22 21:30:18,412 - INFO - Addition result: tensor([[ 2.1784,  0.5587,  1.4530,  1.7548],
#         [ 1.2822, -0.1803, -0.0810, -2.5491]])
# 2024-08-22 21:30:18,412 - INFO - Testing tensor indexing and slicing...
# 2024-08-22 21:30:18,412 - INFO - Accessing element: 0.5580993890762329
# 2024-08-22 21:30:18,414 - INFO - Slicing columns: tensor([-1.4141,  0.5322, -1.0686, -1.7911,  0.4663,  1.8079,  1.2730, -0.6068,
#          1.3739, -1.0877])
# 2024-08-22 21:30:18,415 - INFO - Slicing rows: tensor([ 0.1972,  0.5342, -0.5241, -0.3542,  0.0413,  0.2050,  0.0537,  0.2719,
#          0.5322, -2.3355])
# 2024-08-22 21:30:18,415 - INFO - Testing tensor reductions...
# 2024-08-22 21:30:18,417 - INFO - Sum: tensor([-1.6800,  0.8932]), Mean: tensor([-0.5600,  0.2977]), Max: torch.return_types.max(
# values=tensor([-0.2241,  1.1630]),
# indices=tensor([0, 1]))
# 2024-08-22 21:30:18,417 - INFO - Testing neural network layers...
# 2024-08-22 21:30:18,417 - INFO - Linear layer created with in_features=22, out_features=16
# 2024-08-22 21:30:18,418 - INFO - Conv2d layer created with in_channels=9, out_channels=2
# 2024-08-22 21:30:18,418 - INFO - Testing forward pass...
# 2024-08-22 21:30:18,420 - ERROR - Exception occurred in forward pass of Linear model: mat1 and mat2 shapes cannot be multiplied (1x2 and 9x9)
# 2024-08-22 21:30:18,436 - ERROR - Exception occurred in forward pass of Conv2d model: Given groups=1, weight of size [3, 9, 1, 1], expected input[1, 7, 10, 4] to have 9 channels, but got 7 channels instead
# 2024-08-22 21:30:18,436 - INFO - Testing autograd...
# /home/stasluke18/TCS/gpt2_1.py:168: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
#   logger.info(f"Backward pass successful: {y.grad}")
# 2024-08-22 21:30:19,755 - INFO - Backward pass successful: None
# 2024-08-22 21:30:19,758 - INFO - Backward pass successful: tensor([[ 0.2380, -0.0295, -0.0231, -0.0791, -0.0741,  0.0743,  0.1511],
#         [ 0.2380, -0.0295, -0.0231, -0.0791, -0.0741,  0.0743,  0.1511],
#         [ 0.2380, -0.0295, -0.0231, -0.0791, -0.0741,  0.0743,  0.1511],
#         [ 0.2380, -0.0295, -0.0231, -0.0791, -0.0741,  0.0743,  0.1511],
#         [ 0.2380, -0.0295, -0.0231, -0.0791, -0.0741,  0.0743,  0.1511]])
# 2024-08-22 21:30:19,758 - INFO - Testing in-place operations...
# 2024-08-22 21:30:19,787 - ERROR - Exception occurred in out-of-place operation: element 0 of tensors does not require grad and does not have a grad_fn
# 2024-08-22 21:30:19,795 - ERROR - Exception occurred in in-place operation: a leaf Variable that requires grad is being used in an in-place operation.
# 2024-08-22 21:30:19,795 - INFO - Testing optimizer steps...
# 2024-08-22 21:30:20,567 - INFO - SGD optimizer step executed successfully.
# 2024-08-22 21:30:20,568 - INFO - Adam optimizer step executed successfully.
# 2024-08-22 21:30:20,568 - INFO - Testing loss functions...
# 2024-08-22 21:30:20,573 - INFO - MSELoss computed successfully: 1.56183660030365
# 2024-08-22 21:30:20,574 - ERROR - Exception occurred in CrossEntropyLoss computation: element 0 of tensors does not require grad and does not have a grad_fn
# 2024-08-22 21:30:20,574 - INFO - Testing exception handling...
# 2024-08-22 21:30:20,574 - INFO - Creating tensor with NaN shape...
# 2024-08-22 21:30:20,576 - ERROR - Exception occurred in NaN shape: randn(): argument 'size' (position 1) must be tuple of ints, but found element of type float at pos 0
# 2024-08-22 21:30:20,576 - INFO - Creating tensor with Inf value...
# 2024-08-22 21:30:20,576 - INFO - Tensor with Inf value created: tensor([inf])