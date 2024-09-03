import torch
from nebula.core.research.FedGPD.models.fashionmnist.FedGPDCNN import FedGPDFashionMNISTModelCNN
from nebula.core.research.FedGPD.models.utils.GlobalPrototypeDistillationLoss import GlobalPrototypeDistillationLoss


# Helper function to create random global prototypes
def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        # Initialize global prototypes with random tensors
        model.global_protos[i] = torch.rand(feature_dim)


def test_model_initialization():
    model = FedGPDFashionMNISTModelCNN()
    create_random_prototypes(model, 10, 2048)  # Assuming 2048 is the dimension of features
    assert model.input_channels == 1, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.criterion_cls, torch.nn.CrossEntropyLoss), "Incorrect classification loss function"
    assert isinstance(model.criterion_gpd, GlobalPrototypeDistillationLoss), "Incorrect GPD loss function"
    assert len(model.global_protos) == 10, "Global prototypes were not initialized correctly"


def test_forward_train_functionality():
    model = FedGPDFashionMNISTModelCNN()
    input_tensor = torch.rand(1, 1, 28, 28)
    output, features, _ = model.forward_train(input_tensor, softmax=False, is_feat=True)
    assert output.shape == (1, 10), "Output tensor has incorrect shape"
    assert features.ndim == 2, "Features tensor should be 2D"


def test_optimizer_configuration():
    model = FedGPDFashionMNISTModelCNN()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.SGD), "Optimizer type is incorrect"
    for group in optimizer.param_groups:
        assert group["lr"] == model.learning_rate, "Learning rate not set correctly"
        assert group["momentum"] == 0.9, "Momentum not set correctly"
        assert group["weight_decay"] == 0.00001, "Weight decay not set correctly"


def test_forward_pass_inference_with_prototypes():
    model = FedGPDFashionMNISTModelCNN()
    create_random_prototypes(model, 10, 2048)
    input_tensor = torch.rand(1, 1, 28, 28)
    output = model(input_tensor)
    assert output.ndim == 1 and output.shape[0] == 1, "Output tensor should be 1D with a single class index during inference"


def test_loss_combination_with_prototypes():
    model = FedGPDFashionMNISTModelCNN()
    create_random_prototypes(model, 10, 2048)
    input_tensor = torch.rand(3, 1, 28, 28, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    logits, features, _ = model.forward_train(input_tensor, softmax=False, is_feat=True)

    assert logits.shape == (3, 10), f"Expected logits shape of (3, 10), got {logits.shape}"
    assert features.shape == (3, 2048), f"Expected feature shape of (3, 2048), got {features.shape}"

    loss_ce = model.criterion_cls(logits, target)
    loss_gpd = model.criterion_gpd(model.global_protos, features, target)
    combined_loss = loss_ce + model.lambd * loss_gpd
    assert combined_loss > 0, "Combined loss should be greater than zero"


def test_loss_combination_without_all_prototypes():
    model = FedGPDFashionMNISTModelCNN()
    create_random_prototypes(model, 1, 2048)
    input_tensor = torch.rand(3, 1, 28, 28, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    logits, features, _ = model.forward_train(input_tensor, softmax=False, is_feat=True)
    loss_ce = model.criterion_cls(logits, target)
    loss_gpd = model.criterion_gpd(model.global_protos, features, target)
    combined_loss = loss_ce + model.lambd * loss_gpd
    assert combined_loss > 0, "Combined loss should be greater than zero"
