import torch
from nebula.core.models.fashionmnist.mlp import FashionMNISTModelMLP


def test_model_initialization():
    model = FashionMNISTModelMLP()
    assert model.input_channels == 1, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Incorrect loss function"


def test_forward_pass():
    model = FashionMNISTModelMLP()
    input_tensor = torch.rand(1, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output tensor has incorrect shape"


def test_model_output_dimension():
    model = FashionMNISTModelMLP()
    input_tensor = torch.rand(10, 1, 28, 28)  # batch size of 10
    outputs = model(input_tensor)
    assert outputs.shape[1] == model.num_classes, "Model output dimension does not match number of classes"


def test_optimizer_configuration():
    model = FashionMNISTModelMLP()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer type is incorrect"
    # Checking for specific config parameters like 'amsgrad' is omitted as it's not set in this model's optimizer


def test_gradients():
    model = FashionMNISTModelMLP()
    input_tensor = torch.rand(3, 1, 28, 28, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    output = model(input_tensor)
    loss = model.criterion(output, target)
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, f"No gradients for {name}"


def test_loss_calculation():
    model = FashionMNISTModelMLP()
    input_tensor = torch.rand(5, 1, 28, 28)
    target = torch.randint(0, 10, (5,))
    output = model(input_tensor)
    loss = model.criterion(output, target)
    assert loss > 0, "Loss should be greater than zero"
