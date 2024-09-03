import torch
from nebula.core.optimizations.communications.KD.models.cifar100.TeacherResnet32 import MDTeacherCIFAR100ModelResNet32
from nebula.core.optimizations.communications.KD.models.cifar100.StudentResnet18 import StudentCIFAR100ModelResNet18


def test_md_teacher_cifar100_resnet32_initialization():
    model = MDTeacherCIFAR100ModelResNet32(input_channels=3, num_classes=100, beta=100)
    assert model.beta == 100, "Beta value should be initialized correctly"


def test_md_teacher_cifar100_resnet32_forward():
    model = MDTeacherCIFAR100ModelResNet32(input_channels=3, num_classes=100, beta=100)
    input_tensor = torch.randn(1, 3, 32, 32)
    logits, features = model(input_tensor, softmax=False, is_feat=True)
    assert logits.shape == (1, 100), "Output logits should match number of classes"
    assert len(features) == 5, "Should return features from all layers"


def test_md_teacher_cifar100_resnet32_set_student():
    teacher = MDTeacherCIFAR100ModelResNet32(input_channels=3, num_classes=100, beta=100)
    student = StudentCIFAR100ModelResNet18(input_channels=3, num_classes=100)
    teacher.set_student_model(student)
    assert hasattr(teacher, "student_model"), "Student model should be set properly"


def test_md_teacher_resnet32_step():
    model = MDTeacherCIFAR100ModelResNet32(input_channels=3, num_classes=100, beta=100)
    student_model = StudentCIFAR100ModelResNet18(input_channels=3, num_classes=100)  # Using a simpler teacher as a student for testing
    model.set_student_model(student_model)
    model.train()  # Ensure the model is in training mode

    # Create a batch of inputs and labels
    inputs = torch.randn(4, 3, 32, 32)  # Batch size of 4
    labels = torch.randint(0, 100, (4,))  # Random class labels

    # Execute the step function with mutual distillation
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")

    # Assertions to ensure the loss is computed and returned correctly
    assert loss is not None, "Loss should not be None"
    assert loss > 0, "Loss should be greater than 0 "
    assert isinstance(loss, torch.Tensor), "Loss should be a Tensor"
