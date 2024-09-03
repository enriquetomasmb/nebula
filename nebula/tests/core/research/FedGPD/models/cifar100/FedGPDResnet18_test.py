import torch
from nebula.core.research.FedGPD.models.cifar100.FedGPDResnet18 import FedGPDCIFAR100ModelResNet18
from nebula.core.research.FedGPD.models.utils.GlobalPrototypeDistillationLoss import GlobalPrototypeDistillationLoss


# Helper function to create random global prototypes
def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        # Initialize global prototypes with random tensors
        model.global_protos[i] = torch.rand(feature_dim)


def test_model_initialization_cifar100():
    model = FedGPDCIFAR100ModelResNet18()
    assert model.input_channels == 3, "Incorrect number of input channels"
    assert model.num_classes == 100, "Incorrect number of classes"
    assert isinstance(model.criterion_cls, torch.nn.CrossEntropyLoss), "Incorrect classification loss function"
    assert isinstance(model.criterion_gpd, GlobalPrototypeDistillationLoss), "Incorrect GPD loss function"
    assert model.learning_rate == 0.01, "Incorrect learning rate set"


def test_forward_train_functionality_cifar100():
    model = FedGPDCIFAR100ModelResNet18()
    input_tensor = torch.rand(32, 3, 32, 32)
    output, _, _ = model.forward_train(input_tensor, softmax=True, is_feat=True)
    assert output.shape == (32, 100), "Output tensor has incorrect shape"


def test_optimizer_configuration_cifar100():
    model = FedGPDCIFAR100ModelResNet18()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.SGD), "Optimizer type is incorrect"
    for group in optimizer.param_groups:
        assert group["lr"] == 0.01, "Learning rate not set correctly"
        assert group["momentum"] == 0.9, "Momentum not set correctly"
        assert group["weight_decay"] == 0.00001, "Weight decay not set correctly"


def test_forward_pass_inference_with_prototypes_cifar100():
    model = FedGPDCIFAR100ModelResNet18()
    create_random_prototypes(model, 100, model.embedding_dim)  # Assuming the final feature dimension is 512
    input_tensor = torch.rand(32, 3, 32, 32)
    output = model(input_tensor)
    assert output.ndim == 1 and output.shape[0] == 32, "Output tensor should be 1D with a single class index during inference"


def test_loss_combination_with_prototypes_cifar100():
    model = FedGPDCIFAR100ModelResNet18()
    create_random_prototypes(model, 100, model.embedding_dim)
    input_tensor = torch.rand(3, 3, 32, 32, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    logits, features, _ = model.forward_train(input_tensor, softmax=False, is_feat=True)

    loss_ce = model.criterion_cls(logits, target)
    loss_gpd = model.criterion_gpd(model.global_protos, features, target)
    combined_loss = loss_ce + model.lambd * loss_gpd
    assert combined_loss > 0, "Combined loss should be greater than zero"


def test_loss_combination_without_all_prototypes_cifar100():
    model = FedGPDCIFAR100ModelResNet18()
    create_random_prototypes(model, 1, model.embedding_dim)  # Create only one prototype
    input_tensor = torch.rand(3, 3, 32, 32, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    logits, features, _ = model.forward_train(input_tensor, softmax=False, is_feat=True)
    loss_ce = model.criterion_cls(logits, target)
    loss_gpd = model.criterion_gpd(model.global_protos, features, target)
    combined_loss = loss_ce + model.lambd * loss_gpd
    assert combined_loss > 0, "Combined loss should be greater than zero"
