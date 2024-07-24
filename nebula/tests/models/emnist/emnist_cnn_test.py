import torch
import torch.nn as nn
from nebula.core.models.emnist.cnn import EMNISTModelCNN


def test_model_initialization():
    model = EMNISTModelCNN()
    assert model.input_channels == 1, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Incorrect loss function"
    assert isinstance(model.conv1, nn.Conv2d), "First convolutional layer is not initialized correctly"
    assert isinstance(model.conv2, nn.Conv2d), "Second convolutional layer is not initialized correctly"
    assert isinstance(model.l1, nn.Linear), "First linear layer is not initialized correctly"
    assert isinstance(model.l2, nn.Linear), "Second linear layer is not initialized correctly"

def test_forward_pass():
    model = EMNISTModelCNN()
    input_tensor = torch.rand(1, 1, 28, 28)  # Input tensor to match the input size expected by the model
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output tensor has incorrect shape, expected (1, 10) for num_classes"

def test_configure_optimizers():
    model = EMNISTModelCNN()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer type is incorrect"
    # Check if the optimizer configuration is as expected
    assert optimizer.param_groups[0]['betas'] == (0.851436, 0.999689), "Optimizer betas are incorrect"
    assert optimizer.param_groups[0]['amsgrad'] is True, "AMSGrad not set correctly"

def test_output_logits():
    model = EMNISTModelCNN()
    input_tensor = torch.rand(10, 1, 28, 28)  # Larger batch size for more robust test
    output = model(input_tensor)
    assert output.shape == (10, 10), "Logits output shape is incorrect, should be (batch_size, num_classes)"

def test_pooling_layers():
    model = EMNISTModelCNN()
    # Test to ensure that pooling layers are present and configured correctly
    assert isinstance(model.pool1, nn.MaxPool2d), "First pooling layer is not a MaxPool2d"
    assert model.pool1.kernel_size == (2, 2), "First pooling layer kernel size is incorrect"
    assert model.pool1.stride == 2, "First pooling layer stride is incorrect"
    assert isinstance(model.pool2, nn.MaxPool2d), "Second pooling layer is not a MaxPool2d"
    assert model.pool2.kernel_size == (2, 2), "Second pooling layer kernel size is incorrect"
    assert model.pool2.stride == 2, "Second pooling layer stride is incorrect"

