import torch
from nebula.core.models.cifar10.fastermobilenet import FasterMobileNet


def test_initialization():
    model = FasterMobileNet()
    assert isinstance(model, FasterMobileNet), "Model instance is incorrect type"
    assert model.input_channels == 3, "Incorrect input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Incorrect loss function type"


def test_forward_pass():
    model = FasterMobileNet()
    input_tensor = torch.rand(1, 3, 32, 32)  # Example with a mini-batch of 1
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output tensor has incorrect shape"


def test_optimizer_configuration():
    model = FasterMobileNet()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be Adam"
    for group in optimizer.param_groups:
        assert group["lr"] == 1e-3, "Learning rate is incorrect"
        assert group["betas"][0] == model.config["beta1"], "Beta1 is incorrect"
        assert group["betas"][1] == model.config["beta2"], "Beta2 is incorrect"
        assert group["amsgrad"] is True, "Amsgrad should be True"


def test_loss_functionality():
    model = FasterMobileNet()
    input_tensor = torch.rand(3, 3, 32, 32)
    target_labels = torch.randint(0, 10, (3,))
    output = model(input_tensor)

    loss = model.criterion(output, target_labels)
    assert loss.item() > 0, "Loss should be greater than zero"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
