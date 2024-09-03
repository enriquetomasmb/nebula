import torch

from nebula.core.optimizations.communications.KD.models.cifar10.StudentCNN import StudentCIFAR10ModelCNN
from nebula.core.optimizations.communications.KD.models.cifar10.TeacherCNN import MDTeacherCIFAR10ModelCNN


def test_mdteacher_model_initialization():
    model = MDTeacherCIFAR10ModelCNN()
    assert model.input_channels == 3, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert model.learning_rate == 1e-3, "Incorrect learning rate set"
    assert isinstance(model.criterion_cls, torch.nn.CrossEntropyLoss), "Incorrect classification loss function"
    assert model.beta == 1000, "Incorrect beta value set"


def test_mdteacher_forward_pass_functionality():
    model = MDTeacherCIFAR10ModelCNN()
    input_tensor = torch.rand(32, 3, 32, 32)
    output = model(input_tensor)
    assert output.shape == (32, 10), "Output tensor has incorrect shape"


def test_mdteacher_optimizer_configuration():
    model = MDTeacherCIFAR10ModelCNN()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
    assert optimizer.param_groups[0]["lr"] == model.learning_rate, "Optimizer learning rate not set correctly"


def test_mdteacher_training_step_with_student():
    teacher = MDTeacherCIFAR10ModelCNN()
    student = StudentCIFAR10ModelCNN()
    teacher.set_student_model(student)
    images = torch.rand(5, 3, 32, 32)  # A small batch of images
    labels = torch.randint(0, 10, (5,))  # Random class labels for each image
    batch = (images, labels)
    loss = teacher.step(batch, batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be greater than zero"
    assert isinstance(loss, torch.Tensor), "Return type should be a torch.Tensor"
    assert hasattr(teacher, "student_model"), "Teacher model does not have a student model set"
