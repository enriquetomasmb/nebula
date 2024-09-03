from unittest.mock import patch
import torch
from nebula.core.research.FedProto.models.cifar10.FedProtoResnet8 import FedProtoCIFAR10ModelResNet8


def test_model_initialization():
    """Test that the model initializes correctly with expected properties."""
    model = FedProtoCIFAR10ModelResNet8(input_channels=3, num_classes=10, learning_rate=0.01, beta=0.05)
    assert model.input_channels == 3, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert model.learning_rate == 0.01, "Incorrect learning rate"
    assert model.beta == 0.05, "Incorrect beta value"


def test_forward_pass():
    """Test the forward pass of the model."""
    model = FedProtoCIFAR10ModelResNet8(input_channels=3, num_classes=10)
    input_tensor = torch.randn(1, 3, 32, 32)  # Simulate a single CIFAR-10 image input
    logits, features, _ = model.forward_train(input_tensor, softmax=True, is_feat=True)
    assert logits.size(1) == 10, "Output logits should have 10 classes"
    assert features.size(1) == 2048, "Features should be flattened to 2048 dimensions"


def test_prototype_distances():
    """Test the computation of distances to prototypes during inference."""
    model = FedProtoCIFAR10ModelResNet8(input_channels=3, num_classes=10)
    model.set_protos({i: torch.randn(2048) for i in range(10)})  # Set random prototypes for each class
    input_tensor = torch.randn(1, 3, 32, 32)
    prediction = model.forward(input_tensor)
    assert prediction.shape == (1,), "Prediction shape should be (1,) for a single input"


def test_loss_and_prototype_aggregation():
    """Test loss calculation and prototype aggregation during training."""
    model = FedProtoCIFAR10ModelResNet8(input_channels=3, num_classes=10)
    input_tensor = torch.randn(5, 3, 32, 32)  # Simulate a batch of 5 images
    labels = torch.randint(0, 10, (5,))  # Simulate random labels for the batch
    batch = (input_tensor, labels)
    with patch.object(model, "process_metrics") as mocked_process_metrics:
        loss = model.step(batch, 0, "Train")
    assert loss.item() >= 0, "Loss should be non-negative"
    assert len(model.agg_protos_label) > 0, "Prototypes should be aggregated during training"
