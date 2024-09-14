import torch

from nebula.core.optimizations.communications.KD_prototypes.models.fashionmnist.ProtoStudentCNN import ProtoStudentFashionMNISTModelCNN
from nebula.core.optimizations.communications.KD_prototypes.models.fashionmnist.ProtoTeacherCNN import MDProtoTeacherFashionMNISTModelCNN


def test_md_proto_teacher_mnist_model_cnn_initialization():
    model = MDProtoTeacherFashionMNISTModelCNN(input_channels=1, num_classes=10, beta_feat=1000)
    assert model.weighting.get_beta() == 1000, "Beta for mutual distillation should be initialized correctly"


def test_md_proto_teacher_mnist_model_cnn_forward_train():
    model = MDProtoTeacherFashionMNISTModelCNN(input_channels=1, num_classes=10, beta_feat=1000)
    input_tensor = torch.randn(5, 1, 28, 28)
    logits, _, _ = model.forward_train(input_tensor, softmax=True, is_feat=True)
    assert logits.shape == (5, 10), "Output from forward_train should match the number of classes"


def test_md_proto_teacher_mnist_model_cnn_step():
    model = MDProtoTeacherFashionMNISTModelCNN(input_channels=1, num_classes=10, beta_feat=1000)
    model.set_student_model(ProtoStudentFashionMNISTModelCNN(input_channels=1, num_classes=10))  # Setup with a dummy student model
    inputs = torch.randn(5, 1, 28, 28)
    labels = torch.randint(0, 10, (5,))
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")
    assert loss >= 0, "Loss should be non-negative and calculated with mutual distillation"


def test_md_proto_teacher_mnist_model_cnn_configure_optimizers():
    model = MDProtoTeacherFashionMNISTModelCNN(input_channels=1, num_classes=10)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer configuration should be correct"
