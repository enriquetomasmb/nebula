import torch
from nebula.core.optimizations.communications.KD.models.mnist.StudentCNN import StudentMNISTModelCNN
from nebula.core.optimizations.communications.KD.utils.AT import Attention
from nebula.core.optimizations.communications.KD.models.mnist.TeacherCNN import MDTeacherMNISTModelCNN


def test_md_teacher_model_initialization():
    model = MDTeacherMNISTModelCNN(beta=1000)
    assert model.beta == 1000, "Incorrect beta value"
    assert isinstance(model.criterion_cls, torch.nn.CrossEntropyLoss), "Incorrect classification loss function"
    assert isinstance(model.criterion_kd, Attention), "Incorrect KD loss function"


def test_md_teacher_forward_pass_functionality():
    model = MDTeacherMNISTModelCNN()
    input_tensor = torch.rand(10, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (10, 10), "Output tensor has incorrect shape"
    features, logits = model(input_tensor, is_feat=True)
    assert len(features) == 2, "Should return two feature maps"
    assert logits.shape == (10, 10), "Logits tensor has incorrect shape"


def test_md_teacher_optimizer_configuration():
    model = MDTeacherMNISTModelCNN()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"


def test_md_teacher_training_step():
    model = MDTeacherMNISTModelCNN()
    model.set_student_model(StudentMNISTModelCNN())  # Set a dummy student model
    images = torch.rand(5, 1, 28, 28)
    labels = torch.randint(0, 10, (5,))
    batch = (images, labels)
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be greater than zero"
    assert isinstance(loss, torch.Tensor), "Return type should be a torch.Tensor"
