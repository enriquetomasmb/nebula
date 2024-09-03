import torch
from nebula.core.optimizations.communications.KD_prototypes.models.fashionmnist.ProtoTeacherCNN import ProtoTeacherFashionMNISTModelCNN


def test_proto_teacher_mnist_model_cnn_initialization():
    model = ProtoTeacherFashionMNISTModelCNN(input_channels=1, num_classes=10)
    assert model.num_classes == 10, "Incorrect number of classes initialized"
    assert model.input_channels == 1, "Incorrect input channels initialized"


def test_proto_teacher_mnist_model_cnn_forward():
    model = ProtoTeacherFashionMNISTModelCNN(input_channels=1, num_classes=10)
    model.eval()  # Set the model to evaluation mode
    input_tensor = torch.randn(1, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output shape should be (1, num_classes)"


def test_proto_teacher_mnist_model_cnn_training_step():
    model = ProtoTeacherFashionMNISTModelCNN(input_channels=1, num_classes=10)
    model.train()  # Ensure the model is in training mode
    inputs = torch.randn(5, 1, 28, 28)
    labels = torch.randint(0, 10, (5,))
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")
    assert loss >= 0, "Loss should be non-negative"


def test_proto_teacher_mnist_model_cnn_configure_optimizers():
    model = ProtoTeacherFashionMNISTModelCNN(input_channels=1, num_classes=10)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
