import torch
from nebula.core.optimizations.communications.KD_prototypes.models.mnist.ProtoTeacherCNN import ProtoTeacherMNISTModelCNN


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        # Initialize global prototypes with random tensors
        model.global_protos[i] = torch.rand(feature_dim)


def test_proto_teacher_mnist_model_cnn_initialization():
    model = ProtoTeacherMNISTModelCNN(input_channels=1, num_classes=10)
    assert model.num_classes == 10, "Incorrect number of classes initialized"
    assert model.input_channels == 1, "Incorrect input channels initialized"


def test_proto_teacher_mnist_model_cnn_forward():
    model = ProtoTeacherMNISTModelCNN(input_channels=1, num_classes=10)
    create_random_prototypes(model, 10, 2048)
    model.eval()  # Set the model to evaluation mode
    input_tensor = torch.randn(1, 1, 28, 28)
    output, _, _ = model.forward_train(input_tensor, is_feat=True, softmax=False)
    assert output.shape == (1, 10), "Output shape should be (1, num_classes)"


def test_proto_teacher_mnist_model_cnn_training_step():
    model = ProtoTeacherMNISTModelCNN(input_channels=1, num_classes=10)
    create_random_prototypes(model, 10, 2048)
    model.train()  # Ensure the model is in training mode
    inputs = torch.randn(5, 1, 28, 28)
    labels = torch.randint(0, 10, (5,))
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")
    assert loss >= 0, "Loss should be non-negative"


def test_proto_teacher_mnist_model_cnn_configure_optimizers():
    model = ProtoTeacherMNISTModelCNN(input_channels=1, num_classes=10)
    create_random_prototypes(model, 10, 2048)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
