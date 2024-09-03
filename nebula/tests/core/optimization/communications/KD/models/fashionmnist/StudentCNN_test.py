import torch

from nebula.core.optimizations.communications.KD.models.fashionmnist.StudentCNN import StudentFashionMNISTModelCNN
from nebula.core.optimizations.communications.KD.models.fashionmnist.TeacherCNN import TeacherFashionMNISTModelCNN, MDTeacherFashionMNISTModelCNN


def test_student_fashion_mnist_model_initialization():
    model = StudentFashionMNISTModelCNN()
    assert model.input_channels == 1, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert model.learning_rate == 1e-3, "Incorrect learning rate set"
    assert isinstance(model.criterion_cls, torch.nn.CrossEntropyLoss), "Incorrect classification loss function"
    assert isinstance(model.teacher_model, (TeacherFashionMNISTModelCNN, MDTeacherFashionMNISTModelCNN)), "Incorrect teacher model type"


def test_student_fashion_mnist_forward_pass_functionality():
    model = StudentFashionMNISTModelCNN()
    input_tensor = torch.rand(10, 1, 28, 28)
    output = model(input_tensor)
    assert output.shape == (10, 10), "Output tensor has incorrect shape"
    features, logits = model(input_tensor, is_feat=True)
    assert len(features) == 2, "Should return two feature maps"
    assert logits.shape == (10, 10), "Logits tensor has incorrect shape"


def test_student_fashion_mnist_optimizer_configuration():
    model = StudentFashionMNISTModelCNN()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
    assert optimizer.param_groups[0]["lr"] == model.learning_rate, "Optimizer learning rate not set correctly"


def test_student_fashion_mnist_training_step():
    model = StudentFashionMNISTModelCNN(teacher_model=TeacherFashionMNISTModelCNN())  # Ensure a teacher model is set for the test
    images = torch.rand(5, 1, 28, 28)  # A small batch of images
    labels = torch.randint(0, 10, (5,))  # Random class labels for each image
    batch = (images, labels)
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be greater than zero"
    assert isinstance(loss, torch.Tensor), "Return type should be a torch.Tensor"
