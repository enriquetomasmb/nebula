import torch
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoTeacherCNN import ProtoTeacherCIFAR10ModelCNN


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        model.global_protos[i] = torch.rand(feature_dim, requires_grad=False).to(model.device)


def test_proto_teacher_cifar10_initialization():
    model = ProtoTeacherCIFAR10ModelCNN(input_channels=3, num_classes=10)
    assert model.input_channels == 3, "Input channels should be set to 3"
    assert model.num_classes == 10, "Number of classes should be 10"


def test_proto_teacher_cifar10_forward_pass():
    model = ProtoTeacherCIFAR10ModelCNN(input_channels=3, num_classes=10)
    model.eval()
    create_random_prototypes(model, 10, 512)  # Assuming prototype feature dimension is 512
    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    assert output.shape[0] == 1, "Output should be batch size of 1"


def test_proto_teacher_cifar10_training_step():
    model = ProtoTeacherCIFAR10ModelCNN(input_channels=3, num_classes=10)
    model.train()
    create_random_prototypes(model, 10, 512)
    inputs = torch.randn(3, 3, 32, 32)
    labels = torch.randint(0, 10, (3,))
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")
    assert loss >= 0, "Loss should be non-negative"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
