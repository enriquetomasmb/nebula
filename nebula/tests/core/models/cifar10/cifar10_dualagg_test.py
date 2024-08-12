import torch

from nebula.core.models.cifar10.dualagg import DualAggModel, ContrastiveLoss


def test_initialization():
    model = DualAggModel()
    assert isinstance(model, DualAggModel), "Model instance is incorrect type"
    assert model.input_channels == 3, "Incorrect input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.criterion, ContrastiveLoss), "Incorrect loss function type"


def test_forward_pass():
    model = DualAggModel()
    input_tensor = torch.rand(2, 3, 32, 32)  # Example with a mini-batch of 2
    outputs = model(input_tensor, mode="local")  # Test in local mode
    assert outputs.shape == (2, 10), "Output shape is incorrect for local mode"
    outputs = model(input_tensor, mode="historical")  # Test in historical mode
    assert outputs.shape == (2, 10), "Output shape is incorrect for historical mode"
    outputs = model(input_tensor, mode="global")  # Test in global mode
    assert outputs.shape == (2, 10), "Output shape is incorrect for global mode"


def test_optimizer_configuration():
    model = DualAggModel()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be Adam"
    for group in optimizer.param_groups:
        assert group["lr"] == 1e-3, "Learning rate is incorrect"
        assert group["betas"][0] == model.config["beta1"], "Beta1 is incorrect"
        assert group["betas"][1] == model.config["beta2"], "Beta2 is incorrect"
        assert group["amsgrad"] is True, "Amsgrad should be True"


def test_loss_functionality():
    model = DualAggModel()
    input_tensor = torch.rand(2, 3, 32, 32)
    target_labels = torch.randint(0, 10, (2,))
    output_local = model(input_tensor, mode="local")
    output_global = model(input_tensor, mode="global")
    output_historical = model(input_tensor, mode="historical")

    loss = model.criterion(output_local, output_global, output_historical, target_labels)
    assert loss.item() > 0, "Loss should be greater than zero"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"


def test_state_dict_operations():
    model = DualAggModel()
    # Simulate a training step to modify the model weights
    optimizer = model.configure_optimizers()
    input_tensor = torch.rand(2, 3, 32, 32)
    target_labels = torch.randint(0, 10, (2,))
    optimizer.zero_grad()
    output = model(input_tensor, mode="local")
    loss = model.criterion(output, output, output, target_labels)  # Using same output for simplicity
    loss.backward()
    optimizer.step()

    # Save and load state dict
    saved_state_dict = model.model.state_dict()
    new_model = DualAggModel()
    new_model.model.load_state_dict(saved_state_dict)

    for param, loaded_param in zip(model.model.parameters(), new_model.model.parameters()):
        assert torch.equal(param, loaded_param), "Parameters did not load correctly"
