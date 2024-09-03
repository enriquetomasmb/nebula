import torch
from nebula.core.optimizations.communications.KD_prototypes.models.fashionmnist.ProtoStudentCNN import ProtoStudentFashionMNISTModelCNN
from nebula.core.optimizations.communications.KD_prototypes.models.fashionmnist.ProtoTeacherCNN import ProtoTeacherFashionMNISTModelCNN


def test_proto_student_mnist_model_cnn_initialization():
    model = ProtoStudentFashionMNISTModelCNN(input_channels=1, num_classes=10)
    assert model.input_channels == 1, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert model.learning_rate == 1e-3, "Incorrect learning rate set"

    # Check if teacher model is set appropriately
    assert isinstance(model.teacher_model, ProtoTeacherFashionMNISTModelCNN), "Incorrect teacher model type"


def test_proto_student_mnist_model_cnn_forward_pass():
    model = ProtoStudentFashionMNISTModelCNN(input_channels=1, num_classes=10)
    model.eval()  # Set model to evaluation mode for this test
    input_tensor = torch.randn(1, 1, 28, 28)

    # Test inference mode forward pass
    output = model(input_tensor)
    assert output.shape[1] == 10, "Output shape should be (batch_size, num_classes)"

    # Test training mode forward pass
    model.train()
    logits, dense, features = model.forward_train(input_tensor, is_feat=True)
    assert logits.shape[1] == 10, "Logits shape should be (batch_size, num_classes)"
    assert len(features) == 2, "Features list should contain two elements for two convolutional layers"


def test_proto_student_mnist_model_cnn_training_step():
    model = ProtoStudentFashionMNISTModelCNN(input_channels=1, num_classes=10)
    model.train()  # Make sure the model is in training mode
    inputs = torch.randn(5, 1, 28, 28)  # A batch of 5 images
    labels = torch.randint(0, 10, (5,))  # Random target labels for each image
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be positive"
    assert isinstance(loss, torch.Tensor), "Loss should be a torch.Tensor"


def test_proto_student_mnist_model_cnn_configure_optimizers():
    model = ProtoStudentFashionMNISTModelCNN(input_channels=1, num_classes=10)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
    # Check if the learning rate is set correctly
    for group in optimizer.param_groups:
        assert group["lr"] == model.learning_rate, "Optimizer learning rate should match the model's learning rate"
