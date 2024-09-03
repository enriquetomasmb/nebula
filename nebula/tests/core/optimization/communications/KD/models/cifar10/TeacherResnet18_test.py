import torch

from nebula.core.optimizations.communications.KD.models.cifar10.TeacherResnet18 import TeacherCIFAR10ModelResNet18


def test_teacher_resnet18_initialization():
    model = TeacherCIFAR10ModelResNet18()
    assert model.input_channels == 3, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert model.learning_rate == 1e-3, "Incorrect learning rate set"


def test_teacher_resnet18_forward_pass():
    model = TeacherCIFAR10ModelResNet18()
    input_tensor = torch.rand(2, 3, 32, 32)  # CIFAR-10 dimensions
    output = model(input_tensor)
    assert output.shape == (2, 10), "Output tensor shape should match (batch_size, num_classes)"


def test_teacher_resnet18_optimizer_configuration():
    model = TeacherCIFAR10ModelResNet18()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"


def test_teacher_resnet18_training_step():
    model = TeacherCIFAR10ModelResNet18()
    images = torch.rand(3, 3, 32, 32)  # Small batch of images
    labels = torch.randint(0, 10, (3,))  # Random class labels
    batch = (images, labels)
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be greater than zero"
    assert isinstance(loss, torch.Tensor), "Return type should be a torch.Tensor"
