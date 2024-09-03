import torch
import pytest

from nebula.core.research.FedProto.models.cifar10.FedProtoCNN import FedProtoCIFAR10ModelCNN


def test_initialization():
    """Test initialization of FedProtoCIFAR10ModelCNN"""
    model = FedProtoCIFAR10ModelCNN()
    assert model.input_channels == 3, "Incorrect input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert model.learning_rate == 1e-3, "Incorrect learning rate"
    assert model.beta == 1, "Incorrect beta value"


def test_forward_pass():
    """Test the forward pass of the model for both training and inference"""
    model = FedProtoCIFAR10ModelCNN()
    input_tensor = torch.rand(1, 3, 32, 32)
    logits, _, _ = model.forward_train(input_tensor, softmax=True, is_feat=True)
    assert logits.shape == (1, 10), "Logits shape incorrect during training"

    # Simulating that global prototypes are available
    model.global_protos = {i: torch.rand(2048) for i in range(10)}
    predictions = model.forward(input_tensor)
    assert predictions.shape == (1,), "Predictions shape incorrect during inference"


def test_loss_calculation():
    """Test the loss calculation during a training step"""
    model = FedProtoCIFAR10ModelCNN()
    input_tensor = torch.rand(10, 3, 32, 32)
    labels = torch.randint(0, 10, (10,))
    batch = (input_tensor, labels)
    loss = model.step(batch, 0, "Validation")
    assert loss >= 0, "Loss calculation failed"


def test_save_and_load_protos():
    """Test saving and loading of prototypes"""
    model = FedProtoCIFAR10ModelCNN()
    # Set some prototypes
    protos = {i: torch.rand(2048) for i in range(10)}
    model.set_protos(protos)

    # Verify that prototypes were set correctly
    loaded_protos = model.get_protos()
    for key, proto in protos.items():
        assert torch.allclose(proto, loaded_protos[key]), "Protos for class {key} did not load correctly"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_assignment(device):
    """Test if the model can be moved to and from GPU correctly"""
    if torch.cuda.is_available() or device == "cpu":
        model = FedProtoCIFAR10ModelCNN()
        model.to(device)
        assert next(model.parameters()).device.type == device, "Model not on {device}"
