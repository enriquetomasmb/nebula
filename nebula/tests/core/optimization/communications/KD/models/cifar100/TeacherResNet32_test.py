import torch
from nebula.core.optimizations.communications.KD.models.cifar100.TeacherResnet32 import TeacherCIFAR100ModelResNet32


def test_teacher_cifar100_resnet32_initialization():
    model = TeacherCIFAR100ModelResNet32(input_channels=3, num_classes=100)
    assert model.input_channels == 3, "Incorrect input channels"
    assert model.num_classes == 100, "Incorrect number of classes"
    assert model.learning_rate == 1e-3, "Incorrect learning rate"
    assert isinstance(model.fc, torch.nn.Linear), "Final layer should be a linear layer"


def test_teacher_cifar100_resnet32_forward():
    model = TeacherCIFAR100ModelResNet32(input_channels=3, num_classes=100)
    input_tensor = torch.randn(1, 3, 32, 32)
    output = model(input_tensor)
    assert output.size(1) == 100, "Output size should match number of classes (100)"


def test_teacher_cifar100_resnet32_step():
    model = TeacherCIFAR100ModelResNet32(input_channels=3, num_classes=100)
    input_tensor = torch.randn(5, 3, 32, 32)
    labels = torch.randint(0, 100, (5,))
    batch = (input_tensor, labels)
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be calculated and greater than zero"
