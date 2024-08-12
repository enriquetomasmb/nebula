import torch
from nebula.core.models.cifar100.cnn import CIFAR100ModelCNN


def test_model_initialization():
    model = CIFAR100ModelCNN()
    assert model.input_channels == 3, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Incorrect loss function"
    # Additional checks can be made for other attributes and configurations


def test_forward_pass():
    model = CIFAR100ModelCNN()
    input_tensor = torch.rand(1, 3, 32, 32)  # Input tensor to match the input size expected by the model
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output tensor has incorrect shape, expected (1, 10) for num_classes"


def test_configure_optimizers():
    model = CIFAR100ModelCNN()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer type is incorrect"
    assert optimizer.param_groups[0]["lr"] == 8.0505e-05, "Learning rate in optimizer is incorrect"
    assert optimizer.param_groups[0]["betas"] == (0.851436, 0.999689), "Beta values are incorrect"
    assert optimizer.param_groups[0]["amsgrad"] is True, "AMSGrad configuration is incorrect"


def test_output_classifier():
    model = CIFAR100ModelCNN()
    input_tensor = torch.rand(10, 3, 32, 32)  # Batch size of 10 for more robust testing
    output = model(input_tensor)
    assert output.shape == (10, 10), "Output dimensions are incorrect for batch processing"
    # Check if output is indeed a tensor of logits
    assert output.requires_grad is True, "Outputs should have gradients for backpropagation"
