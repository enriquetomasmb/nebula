import torch
from torch import nn
from nebula.core.models.emnist.mlp import EMNISTModelMLP


def test_model_initialization():
    model = EMNISTModelMLP()
    assert model.input_channels == 1, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Incorrect loss function"
    assert isinstance(model.l1, nn.Linear), "First linear layer is not initialized correctly"
    assert isinstance(model.l2, nn.Linear), "Second linear layer is not initialized correctly"
    assert isinstance(model.l3, nn.Linear), "Third linear layer is not initialized correctly"


def test_forward_pass():
    model = EMNISTModelMLP()
    input_tensor = torch.rand(1, 1, 28, 28)  # Input tensor to match the input size expected by the model
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output tensor has incorrect shape, expected (1, 10) for num_classes"
    assert output.requires_grad is True, "Output tensor should have gradients"


def test_configure_optimizers():
    model = EMNISTModelMLP()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer type is incorrect"
    # Ensure that the learning rate is set correctly within the optimizer
    assert optimizer.param_groups[0]["lr"] == 1e-3, "Optimizer learning rate is incorrect"


def test_log_softmax_output():
    model = EMNISTModelMLP()
    input_tensor = torch.rand(10, 1, 28, 28)  # Larger batch size for more robust test
    output = model(input_tensor)
    # Ensure that the output is log-probability, should sum to 1 when exponentiated
    probabilities = torch.exp(output)
    assert torch.allclose(probabilities.sum(dim=1), torch.tensor([1.0] * 10), atol=1e-5), "Outputs are not valid log probabilities"


def test_flattening_and_network_dimensions():
    model = EMNISTModelMLP()
    input_tensor = torch.rand(1, 1, 28, 28)
    batch_size, channels, width, height = input_tensor.size()
    flat_size = width * height * channels
    output = model(input_tensor)
    assert model.l1.in_features == flat_size, "Flattening operation or input dimension to first layer is incorrect"
    assert model.l1.out_features == 256, "Output features of the first layer are incorrect"
    assert model.l3.out_features == model.num_classes, "Output features of the last layer do not match the number of classes"
