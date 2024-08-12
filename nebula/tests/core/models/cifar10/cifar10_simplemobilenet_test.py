import torch
from torch import nn
from nebula.core.models.cifar10.simplemobilenet import SimpleMobileNetV1


def test_model_initialization():
    model = SimpleMobileNetV1()
    assert model.input_channels == 3, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Incorrect loss function"
    assert isinstance(model.fc, nn.Linear), "Final fully connected layer is not initialized properly"


def test_forward_pass():
    model = SimpleMobileNetV1()
    input_tensor = torch.rand(1, 3, 32, 32)
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output tensor has incorrect shape, expected (1, 10)"


def test_configure_optimizers():
    model = SimpleMobileNetV1()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer type is incorrect"
    assert optimizer.param_groups[0]["lr"] == 1e-3, "Learning rate is not set correctly"
