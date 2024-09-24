import torch
import pytest

from nebula.core.research.FedProto.models.mnist.FedProtoCNN import FedProtoMNISTModelCNN


def test_initialization():
    """Test initialization of FedProtoMNISTModelCNN"""
    model = FedProtoMNISTModelCNN()
    assert model.input_channels == 1, "Incorrect input channels"
    assert model.num_classes == 10, "Incorrect number of classes"


def test_forward_pass():
    """Test the forward pass of the model for both training and inference"""
    model = FedProtoMNISTModelCNN()
    input_tensor = torch.rand(1, 1, 28, 28)
    logits, _ = model.forward_train(input_tensor)
    assert logits.shape == (1, 10), "Logits shape incorrect during training"

    # Simulating that global prototypes are available
    model.global_protos = {i: torch.rand(2048) for i in range(10)}
    predictions = model.forward(input_tensor)
    assert predictions.shape == (1,), "Predictions shape incorrect during inference"


def test_loss_calculation():
    """Test the loss calculation during a training step"""
    model = FedProtoMNISTModelCNN()
    input_tensor = torch.rand(10, 1, 28, 28)
    labels = torch.randint(0, 10, (10,))
    batch = (input_tensor, labels)
    loss = model.step(batch, 0, "Validation")
    assert loss >= 0, "Loss calculation failed"


def test_save_and_load_protos():
    """Test saving and loading of prototypes"""
    model = FedProtoMNISTModelCNN()
    # Set some prototypes
    protos = {i: torch.rand(2048) for i in range(10)}
    model.set_protos(protos)

    # Verify that prototypes were set correctly
    loaded_protos = model.get_protos()
    for key, proto in protos.items():
        assert torch.allclose(proto, loaded_protos[key]), f"Protos for class {key} did not load correctly"


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_assignment(device):
    """Test if the model can be moved to and from GPU correctly"""
    if torch.cuda.is_available() or device == "cpu":
        model = FedProtoMNISTModelCNN()
        model.to(device)
        assert next(model.parameters()).device.type == device, f"Model not on {device}"


if __name__ == "__main__":
    pytest.main([__file__])
