from unittest.mock import patch
import torch
from torch import nn
from nebula.core.research.FedProto.models.cifar100.FedProtoResnet18 import FedProtoCIFAR100ModelResNet


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        # Initialize global prototypes with random tensors
        model.global_protos[i] = torch.rand(feature_dim)


def test_model_initialization():
    model = FedProtoCIFAR100ModelResNet()
    assert model.input_channels == 3, "Incorrect number of input channels"
    assert model.num_classes == 100, "Incorrect number of classes"
    assert isinstance(model.criterion_nll, nn.NLLLoss), "Incorrect NLL loss function"
    assert isinstance(model.loss_mse, nn.MSELoss), "Incorrect MSE loss function"
    assert model.learning_rate == 0.01, "Incorrect learning rate set"


def test_forward_train_functionality():
    model = FedProtoCIFAR100ModelResNet()
    input_tensor = torch.rand(32, 3, 32, 32)
    output, _, _ = model.forward_train(input_tensor, softmax=True, is_feat=True)
    assert output.shape == (32, 100), "Output tensor has incorrect shape"


def test_optimizer_configuration():
    model = FedProtoCIFAR100ModelResNet()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer type is incorrect"
    # Assuming beta1 and beta2 are set correctly in model's config dictionary
    assert optimizer.param_groups[0]["lr"] == model.learning_rate, "Learning rate not set correctly"


def test_forward_pass_inference_with_prototypes():
    model = FedProtoCIFAR100ModelResNet()
    create_random_prototypes(model, 100, model.embedding_dim)  # Assuming embedding_dim is defined in the model
    input_tensor = torch.rand(32, 3, 32, 32)
    output = model(input_tensor)
    assert output.ndim == 1 and output.shape[0] == 32, "Output tensor should be 1D with a single class index during inference"


def test_loss_combination_with_prototypes():
    model = FedProtoCIFAR100ModelResNet()
    create_random_prototypes(model, 100, model.embedding_dim)
    input_tensor = torch.randn(5, 3, 32, 32)  # Simulate a batch of 5 images
    labels = torch.randint(0, 10, (5,))  # Simulate random labels for the batch
    batch = (input_tensor, labels)
    with patch.object(model, "process_metrics") as mocked_process_metrics:
        loss = model.step(batch, 0, "Train")
    assert loss.item() >= 0, "Loss should be non-negative"
    assert len(model.agg_protos_label) > 0, "Prototypes should be aggregated during training"
