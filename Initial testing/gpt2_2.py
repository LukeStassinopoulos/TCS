# follow up from gpt 2_1

#Identified Issues and Improvements
# Tensor Creation with Invalid Shapes:

# Issue: The script encountered an error when creating a tensor with a negative dimension ((-1, 10)), which is expected. However, the output indicates a need for better handling or more informative error messages.
# Improvement: Ensure that the error messages are clear and add more robust handling for invalid shapes.
# Neural Network Forward Pass Errors:

# Issue: Errors occurred due to mismatched tensor dimensions during matrix multiplication and Conv2d operations.
# Improvement: Add checks to ensure tensor dimensions are compatible for the operations being tested.
# Autograd and In-Place Operations:

# Issue: Warnings and errors related to gradient computations and in-place operations.
# Improvement: Address the warnings by ensuring tensors with requires_grad=True are used correctly and that in-place operations are handled properly.
# Loss Function Issues:

# Issue: Errors occurred with loss functions due to tensors not having gradients.
# Improvement: Ensure tensors involved in loss calculations have requires_grad=True and are properly initialized.
# Exception Handling for Tensor Creation:

# Issue: Errors with NaN shapes and Inf values.
# Improvement: Improve error handling to provide more descriptive error messages.

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
            (0, 0, 2),  # Valid shape but potentially problematic in some operations
            (-1, 10)    # Invalid shape
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
                result = torch.add(tensor_a, tensor_b)
                logger.info(f"Addition result: {result}")
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
        in_features, out_features = random.randint(1, 10), random.randint(1, 10)
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
            model = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU())
            input_tensor = torch.randn((1, in_features))
            output = model(input_tensor)
            logger.info(f"Forward pass output: {output}")
        except Exception as e:
            log_exception(e, "forward pass of Linear model")
        
        try:
            model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=random.choice([1, 3, 5])), nn.ReLU())
            input_tensor = torch.randn((1, in_channels, random.randint(1, 10), random.randint(1, 10)))
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
            logger.info(f"Backward pass successful: {x.grad}")
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
        model = nn.Linear(10, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        try:
            optimizer.step()
            logger.info("SGD optimizer step executed successfully.")
        except Exception as e:
            log_exception(e, "SGD optimizer step")

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        try:
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

# 2024-08-22 21:37:17,637 - INFO - Testing tensor creation with random shapes and data types...
# 2024-08-22 21:37:17,637 - INFO - Creating tensor with shape (8, 2)
# 2024-08-22 21:37:17,638 - INFO - Tensor created: torch.Size([8, 2])
# 2024-08-22 21:37:17,638 - INFO - Creating tensor with shape (53, 22, 27)
# 2024-08-22 21:37:17,638 - INFO - Tensor created: torch.Size([53, 22, 27])
# 2024-08-22 21:37:17,639 - INFO - Creating tensor with shape (3, 5)
# 2024-08-22 21:37:17,639 - INFO - Tensor created: torch.Size([3, 5])
# 2024-08-22 21:37:17,639 - INFO - Creating tensor with shape (1, 1, 5, 3)
# 2024-08-22 21:37:17,639 - INFO - Tensor created: torch.Size([1, 1, 5, 3])
# 2024-08-22 21:37:17,639 - INFO - Testing tensors with different data types...
# 2024-08-22 21:37:17,639 - INFO - Tensor created with dtype torch.float32: torch.float32
# 2024-08-22 21:37:17,639 - INFO - Tensor created with dtype torch.float64: torch.float64
# 2024-08-22 21:37:17,639 - INFO - Tensor created with dtype torch.int32: torch.int32
# 2024-08-22 21:37:17,639 - INFO - Tensor created with dtype torch.int64: torch.int64
# 2024-08-22 21:37:17,639 - INFO - Tensor created with dtype torch.int8: torch.int8
# 2024-08-22 21:37:17,640 - INFO - Tensor created with dtype torch.int16: torch.int16
# 2024-08-22 21:37:17,640 - INFO - Testing tensors with invalid shapes...
# 2024-08-22 21:37:17,640 - INFO - Creating tensor with invalid shape (0, 0, 2)
# 2024-08-22 21:37:17,640 - INFO - Creating tensor with invalid shape (-1, 10)
# 2024-08-22 21:37:17,641 - ERROR - Exception occurred in tensor creation with invalid shape (-1, 10): Trying to create tensor with negative dimension -1: [-1, 10]
# 2024-08-22 21:37:17,641 - INFO - Testing tensor operations...
# 2024-08-22 21:37:17,643 - INFO - Addition result: tensor([[ 0.6848, -0.6239,  1.4900,  2.8407, -0.0125],
#         [-0.5039,  0.0856, -2.4722,  2.6114,  0.3883],
#         [-2.3209,  0.8474,  0.0285, -0.3279,  1.2587],
#         [ 0.9778, -0.3324, -0.3010, -1.1276,  0.9506],
#         [ 1.2129, -1.8341, -0.3770,  0.7764,  0.4889]])
# 2024-08-22 21:37:17,644 - INFO - Matrix multiplication result: tensor([[-0.0621, -0.7782,  2.1867,  1.1265, -0.7680],
#         [ 2.4979, -0.3564, -0.8337, -6.8535,  0.6264],
#         [ 1.3339,  0.2641, -2.7329,  0.7420,  1.2345],
#         [ 0.0926, -0.6737,  0.4330, -2.2792,  1.7412],
#         [-1.8577, -0.7210,  3.2337, -0.9010, -0.5451]])
# 2024-08-22 21:37:17,645 - INFO - Addition result: tensor([[ 2.8683, -0.9726,  2.7704,  1.2773,  0.5886,  2.1132],
#         [ 1.3064, -0.9529, -2.2863,  0.6495, -0.2426,  0.5026],
#         [-1.0018,  1.6079,  0.0828,  0.5610,  1.2246,  1.3887],
#         [-1.0838,  0.7728, -1.0008,  1.9828, -0.4042,  2.5860],
#         [ 2.0686, -1.0529, -1.4046,  1.9081,  0.9176, -1.8935],
#         [ 3.6191,  1.8908, -0.0570, -0.4060, -0.8055, -0.4912],
#         [-0.3646,  1.1935, -1.2537, -0.1471,  1.0053, -2.7208]])
# 2024-08-22 21:37:17,646 - INFO - Testing tensor indexing and slicing...
# 2024-08-22 21:37:17,646 - INFO - Accessing element: -0.48902198672294617
# 2024-08-22 21:37:17,647 - INFO - Slicing columns: tensor([-1.7735, -0.6821,  0.2189,  1.0376, -1.1551, -0.2432,  1.0648, -2.0605,
#         -0.7399,  0.3193])
# 2024-08-22 21:37:17,648 - INFO - Slicing rows: tensor([ 0.0144, -0.1275,  0.3331, -0.6238, -0.2041, -0.5589,  1.0376, -1.6082,
#         -1.1377, -0.4890])
# 2024-08-22 21:37:17,648 - INFO - Testing tensor reductions...
# 2024-08-22 21:37:17,650 - INFO - Sum: tensor([ 1.8858,  3.9916, -0.5346,  3.9134, -0.6020]), Mean: tensor([-0.1904,  0.7428, -0.4344,  0.2073, -0.3457, -0.0990,  0.5337,  0.7036,
#          0.3546,  0.2586]), Max: torch.return_types.max(
# values=tensor([1.7326, 1.9296, 1.0009, 1.4022, 1.9640]),
# indices=tensor([8, 9, 4, 1, 7]))
# 2024-08-22 21:37:17,650 - INFO - Testing neural network layers...
# 2024-08-22 21:37:17,651 - INFO - Linear layer created with in_features=4, out_features=10
# 2024-08-22 21:37:17,651 - INFO - Conv2d layer created with in_channels=8, out_channels=1
# 2024-08-22 21:37:17,651 - INFO - Testing forward pass...
# 2024-08-22 21:37:17,653 - INFO - Forward pass output: tensor([[0.5874, 0.8155, 0.0491, 0.0000, 0.0000, 0.0000, 0.0880, 0.0000, 0.6082,
#          0.7697]], grad_fn=<ReluBackward0>)
# 2024-08-22 21:37:17,654 - ERROR - Exception occurred in forward pass of Conv2d model: Calculated padded input size per channel: (7 x 3). Kernel size: (5 x 5). Kernel size can't be greater than actual input size
# 2024-08-22 21:37:17,654 - INFO - Testing autograd...
# 2024-08-22 21:37:19,072 - INFO - Backward pass successful: tensor([[ 2.2998, -0.1244, -2.3093,  0.6799, -1.5191,  1.3174, -0.7571,  1.2520,
#          -2.0373, -0.6884],
#         [ 3.4401,  1.1694,  0.2188,  2.6663, -0.0549, -0.0089,  1.2921, -5.2001,
#           2.2214,  0.5342],
#         [-0.4377, -2.3981, -1.8607, -0.9989, -0.1101,  0.9637, -1.4357, -4.6907,
#          -0.7237, -1.3963],
#         [-3.5278, -0.3848,  0.3383,  1.9950, -1.9113,  1.3967,  2.6624, -0.0456,
#          -0.8132, -1.1370]])
# 2024-08-22 21:37:19,080 - INFO - Backward pass successful: tensor([[-0.0641],
#         [-0.0641],
#         [-0.0641],
#         [-0.0641],
#         [-0.0641],
#         [-0.0641],
#         [-0.0641],
#         [-0.0641]])
# 2024-08-22 21:37:19,080 - INFO - Testing in-place operations...
# 2024-08-22 21:37:19,080 - ERROR - Exception occurred in out-of-place operation: element 0 of tensors does not require grad and does not have a grad_fn
# 2024-08-22 21:37:19,081 - ERROR - Exception occurred in in-place operation: a leaf Variable that requires grad is being used in an in-place operation.
# 2024-08-22 21:37:19,081 - INFO - Testing optimizer steps...
# 2024-08-22 21:37:19,837 - INFO - SGD optimizer step executed successfully.
# 2024-08-22 21:37:19,838 - INFO - Adam optimizer step executed successfully.
# 2024-08-22 21:37:19,838 - INFO - Testing loss functions...
# 2024-08-22 21:37:19,839 - INFO - MSELoss computed successfully: 2.4512417316436768
# 2024-08-22 21:37:19,839 - ERROR - Exception occurred in CrossEntropyLoss computation: element 0 of tensors does not require grad and does not have a grad_fn
# 2024-08-22 21:37:19,839 - INFO - Testing exception handling...
# 2024-08-22 21:37:19,839 - INFO - Creating tensor with NaN shape...
# 2024-08-22 21:37:19,839 - ERROR - Exception occurred in NaN shape: randn(): argument 'size' (position 1) must be tuple of ints, but found element of type float at pos 0
# 2024-08-22 21:37:19,839 - INFO - Creating tensor with Inf value...
# 2024-08-22 21:37:19,840 - INFO - Tensor with Inf value created: tensor([inf])