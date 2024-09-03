import torch

from nebula.core.optimizations.communications.KD.models.cifar10.StudentResnet8 import StudentCIFAR10ModelResNet8
from nebula.core.optimizations.communications.KD.models.cifar10.TeacherResnet18 import MDTeacherCIFAR10ModelResNet18, TeacherCIFAR10ModelResNet18


def test_mdteacher_resnet18_initialization():
    model = MDTeacherCIFAR10ModelResNet18()
    assert model.beta == 1000, "Incorrect beta value for distillation"


def test_mdteacher_resnet18_forward_pass_features():
    model = MDTeacherCIFAR10ModelResNet18()
    input_tensor = torch.rand(2, 3, 32, 32)
    logits, features = model(input_tensor, is_feat=True)
    assert logits.shape == (2, 10), "Logits shape should match (batch_size, num_classes)"
    assert len(features) == 4, "Should return four feature maps from each layer"


def test_mdteacher_resnet18_set_student_model():
    model = MDTeacherCIFAR10ModelResNet18()
    student_model = TeacherCIFAR10ModelResNet18()  # Typically, you'd use a student model class
    model.set_student_model(student_model)
    assert model.student_model is not None, "Student model should be set"


def test_mdteacher_resnet18_training_step_with_distillation():
    teacher = MDTeacherCIFAR10ModelResNet18()
    student = StudentCIFAR10ModelResNet8()  # Use a simpler model or a dummy for the test
    teacher.set_student_model(student)
    images = torch.rand(3, 3, 32, 32)
    labels = torch.randint(0, 10, (3,))
    batch = (images, labels)
    loss = teacher.step(batch, batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be greater than zero"
    assert isinstance(loss, torch.Tensor), "Return type should be a torch.Tensor"
