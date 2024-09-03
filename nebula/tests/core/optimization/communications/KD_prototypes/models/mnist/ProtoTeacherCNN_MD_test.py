import torch

from nebula.core.optimizations.communications.KD_prototypes.models.mnist.ProtoStudentCNN import ProtoStudentMNISTModelCNN
from nebula.core.optimizations.communications.KD_prototypes.models.mnist.ProtoTeacherCNN import MDProtoTeacherMNISTModelCNN


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        # Initialize global prototypes with random tensors
        model.global_protos[i] = torch.rand(feature_dim)


def test_md_proto_teacher_mnist_model_cnn_initialization():
    model = MDProtoTeacherMNISTModelCNN(input_channels=1, num_classes=10, beta_md=1000)
    assert model.beta_md == 1000, "Beta for mutual distillation should be initialized correctly"


def test_md_proto_teacher_mnist_model_cnn_forward_train():
    model = MDProtoTeacherMNISTModelCNN(input_channels=1, num_classes=10, beta_md=1000)
    create_random_prototypes(model, 10, 2048)
    input_tensor = torch.randn(5, 1, 28, 28)
    logits, _, _ = model.forward_train(input_tensor, is_feat=True, softmax=True)
    assert logits.shape == (5, 10), "Output from forward_train should match the number of classes"


def test_md_proto_teacher_mnist_model_cnn_step():
    model = MDProtoTeacherMNISTModelCNN(input_channels=1, num_classes=10, beta_md=1000)
    create_random_prototypes(model, 10, 2048)
    model.set_student_model(ProtoStudentMNISTModelCNN(input_channels=1, num_classes=10))  # Setup with a dummy student model
    create_random_prototypes(model.student_model, 10, 2048)
    inputs = torch.randn(5, 1, 28, 28)
    labels = torch.randint(0, 10, (5,))
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")
    assert loss >= 0, "Loss should be non-negative and calculated with mutual distillation"


def test_md_proto_teacher_mnist_model_cnn_configure_optimizers():
    model = MDProtoTeacherMNISTModelCNN(input_channels=1, num_classes=10)
    create_random_prototypes(model, 10, 2048)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer configuration should be correct"
