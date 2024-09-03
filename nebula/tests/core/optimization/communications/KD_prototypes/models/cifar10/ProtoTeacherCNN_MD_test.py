import torch
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoStudentCNN import ProtoStudentCIFAR10ModelCNN
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoTeacherCNN import MDProtoTeacherCIFAR10ModelCNN


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        model.global_protos[i] = torch.rand(feature_dim, requires_grad=False).to(model.device)


def test_md_proto_teacher_cifar10_initialization():
    model = MDProtoTeacherCIFAR10ModelCNN(input_channels=3, num_classes=10)
    assert model.beta_md == 1000, "Mutual distillation beta should be initialized correctly"


def test_md_proto_teacher_cifar10_forward_pass():
    model = MDProtoTeacherCIFAR10ModelCNN(input_channels=3, num_classes=10)
    student_model = ProtoStudentCIFAR10ModelCNN(input_channels=3, num_classes=10)
    model.set_student_model(student_model)
    model.eval()
    create_random_prototypes(model, 10, 512)
    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    assert output.shape[0] == 1, "Output should have batch size of 1 during inference"


def test_md_proto_teacher_cifar10_training_step():
    model = MDProtoTeacherCIFAR10ModelCNN(input_channels=3, num_classes=10)
    student_model = ProtoStudentCIFAR10ModelCNN(input_channels=3, num_classes=10)
    model.set_student_model(student_model)
    model.train()
    create_random_prototypes(model, 10, 512)
    inputs = torch.randn(3, 3, 32, 32)
    labels = torch.randint(0, 10, (3,))
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")
    assert loss >= 0, "Loss should be non-negative"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
