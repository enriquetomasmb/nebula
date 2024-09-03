import torch
from nebula.core.optimizations.communications.KD.models.cifar10.TeacherCNN import TeacherCIFAR10ModelCNN


def test_teacher_model_initialization():
    model = TeacherCIFAR10ModelCNN()
    assert model.input_channels == 3, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert model.learning_rate == 1e-3, "Incorrect learning rate set"
    assert isinstance(model.criterion_cls, torch.nn.CrossEntropyLoss), "Incorrect classification loss function"


def test_teacher_forward_pass_functionality():
    model = TeacherCIFAR10ModelCNN()
    input_tensor = torch.rand(32, 3, 32, 32)  # Batch size of 32, CIFAR-10 dimensions
    output = model(input_tensor)
    assert output.shape == (32, 10), "Output tensor has incorrect shape"


def test_teacher_optimizer_configuration():
    model = TeacherCIFAR10ModelCNN()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
    assert optimizer.param_groups[0]["lr"] == model.learning_rate, "Optimizer learning rate not set correctly"


def test_teacher_training_step():
    model = TeacherCIFAR10ModelCNN()
    images = torch.rand(5, 3, 32, 32)  # A small batch of images
    labels = torch.randint(0, 10, (5,))  # Random class labels for each image
    batch = (images, labels)
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be greater than zero"
    assert isinstance(loss, torch.Tensor), "Return type should be a torch.Tensor"
