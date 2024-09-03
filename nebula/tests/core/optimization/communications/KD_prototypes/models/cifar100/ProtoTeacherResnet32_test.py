import torch
from torch import nn
from nebula.core.optimizations.communications.KD_prototypes.models.cifar100.ProtoTeacherResnet32 import ProtoTeacherCIFAR100ModelResNet32


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        model.global_protos[i] = torch.rand(feature_dim, requires_grad=False).to(model.device)


def test_initialization_proto_teacher_cifar100():
    model = ProtoTeacherCIFAR100ModelResNet32(input_channels=16, num_classes=100)
    assert model.input_channels == 16, "Input channels should be 16"
    assert model.num_classes == 100, "Number of classes should be 100"
    assert isinstance(model.fc, nn.Linear), "Final layer should be a linear layer"
    assert model.fc.out_features == 100, "Output features should match number of classes"


def test_forward_pass_proto_teacher_cifar100():
    model = ProtoTeacherCIFAR100ModelResNet32(input_channels=16, num_classes=100)
    input_tensor = torch.rand(1, 16, 32, 32)
    output = model(input_tensor)
    assert output.shape == (1, 100), "Output shape should be (1, 100) for 100 classes"


def test_loss_and_optimizers_proto_teacher_cifar100():
    model = ProtoTeacherCIFAR100ModelResNet32(input_channels=16, num_classes=100)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"


def test_prototype_integration_proto_teacher_cifar100():
    model = ProtoTeacherCIFAR100ModelResNet32(input_channels=16, num_classes=100)
    create_random_prototypes(model, 100, 512)  # Assuming embedding dimension of 512
    input_tensor = torch.rand(1, 16, 32, 32)
    initial_output = model.forward(input_tensor)
    create_random_prototypes(model, 100, 512)  # Modify prototypes
    modified_output = model.forward(input_tensor)
    assert not torch.equal(initial_output, modified_output), "Outputs should differ after changing prototypes"


def test_loss_combination_proto_teacher_cifar100():
    model = ProtoTeacherCIFAR100ModelResNet32(input_channels=3, num_classes=100)
    create_random_prototypes(model, 100, 512)  # Ensure prototypes are initialized
    input_tensor = torch.rand(5, 3, 32, 32)
    target = torch.randint(0, 100, (5,))
    batch = (input_tensor, target)
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert loss >= 0, "Loss should be non-negative"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor type"
