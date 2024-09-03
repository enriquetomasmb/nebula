import torch
from nebula.core.optimizations.communications.KD_prototypes.models.mnist.ProtoStudentCNN import ProtoStudentMNISTModelCNN
from nebula.core.optimizations.communications.KD_prototypes.models.mnist.ProtoTeacherCNN import ProtoTeacherMNISTModelCNN


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        # Initialize global prototypes with random tensors
        model.global_protos[i] = torch.rand(feature_dim)


def test_proto_student_mnist_model_cnn_initialization():
    model = ProtoStudentMNISTModelCNN(input_channels=1, num_classes=10)
    create_random_prototypes(model, 10, 2048)  # Assuming 2048 is the dimension for dense features
    assert model.input_channels == 1, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.global_protos, dict), "Global prototypes must be initialized correctly"
    assert len(model.global_protos) == 10, "There should be 10 prototypes initialized"

    assert isinstance(model.teacher_model, ProtoTeacherMNISTModelCNN), "Incorrect teacher model type"


def test_proto_student_mnist_model_cnn_forward_pass():
    model = ProtoStudentMNISTModelCNN(input_channels=1, num_classes=10)
    create_random_prototypes(model, 10, 2048)
    input_tensor = torch.randn(1, 1, 28, 28)

    # Testing training forward pass
    logits, dense, features = model.forward_train(input_tensor, softmax=True, is_feat=True)
    assert logits.shape == (1, 10), "Output logits shape should match the number of classes"
    assert dense.shape[1] == 2048, "Dense layer output should match feature dimension"
    assert len(features) == 2, "Should return two feature tensors from convolutional layers"

    # Testing inference forward pass
    output = model(input_tensor)
    assert output.ndim == 1, "Output tensor should be 1D during inference"


def test_proto_student_mnist_model_cnn_training_step():
    model = ProtoStudentMNISTModelCNN(input_channels=1, num_classes=10)
    create_random_prototypes(model, 10, 2048)
    model.train()  # Make sure the model is in training mode
    inputs = torch.randn(5, 1, 28, 28)  # A batch of 5 images
    labels = torch.randint(0, 10, (5,))  # Random target labels for each image
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be positive"
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"


def test_proto_student_mnist_model_cnn_configure_optimizers():
    model = ProtoStudentMNISTModelCNN(input_channels=1, num_classes=10)
    create_random_prototypes(model, 10, 2048)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
    # Check if the learning rate is set correctly
    for group in optimizer.param_groups:
        assert group["lr"] == model.learning_rate, "Optimizer learning rate should match the model's learning rate"
