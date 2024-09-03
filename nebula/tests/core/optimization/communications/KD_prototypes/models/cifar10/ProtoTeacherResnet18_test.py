import torch
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoTeacherResnet18 import ProtoTeacherCIFAR10ModelResnet18


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        model.global_protos[i] = torch.rand(feature_dim, requires_grad=False).to(model.device)


def test_model_initialization():
    model = ProtoTeacherCIFAR10ModelResnet18(input_channels=3, num_classes=10)
    create_random_prototypes(model, model.num_classes, model.embedding_dim)
    assert model.num_classes == 10, "Incorrect number of classes initialized"
    assert isinstance(model.fc, torch.nn.Linear), "Final layer should be a linear layer"
    assert model.fc.out_features == 10, "Incorrect output features in the final layer"


def test_forward_pass():
    model = ProtoTeacherCIFAR10ModelResnet18(input_channels=3, num_classes=10)
    create_random_prototypes(model, model.num_classes, model.embedding_dim)
    model.eval()  # Set the model to inference mode
    inputs = torch.rand(3, 3, 32, 32)
    outputs = model(inputs)
    assert outputs.size(0) == 3, "Output size should be 3"

    model.train()  # Set the model to training mode
    outputs, dense, features = model.forward_train(inputs, softmax=False, is_feat=True)
    assert len(features) == 5, "Should capture intermediate features from each ResNet block"


def test_optimizer_configuration():
    model = ProtoTeacherCIFAR10ModelResnet18(input_channels=3, num_classes=10)
    create_random_prototypes(model, model.num_classes, model.embedding_dim)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
    assert optimizer.param_groups[0]["lr"] == 1e-3, "Learning rate should be set to 1e-3"


def test_training_step():
    model = ProtoTeacherCIFAR10ModelResnet18(input_channels=3, num_classes=10)
    create_random_prototypes(model, model.num_classes, model.embedding_dim)
    inputs = torch.rand(5, 3, 32, 32)  # Batch of 5 images
    labels = torch.randint(0, 10, (5,))
    batch = (inputs, labels)

    # Assuming a simple loss calculation for demonstration; in practice, replace with actual loss functions
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert loss >= 0, "Loss should be non-negative"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor type"
