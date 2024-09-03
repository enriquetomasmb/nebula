import torch
from nebula.core.optimizations.communications.KD.models.cifar100.StudentResnet18 import StudentCIFAR100ModelResNet18


def test_student_resnet18_initialization():
    model = StudentCIFAR100ModelResNet18()
    assert model.input_channels == 3, "Incorrect input channels"
    assert model.num_classes == 100, "Incorrect number of classes"
    assert model.learning_rate == 1e-3, "Incorrect learning rate"
    assert model.beta_kd == 1, "Incorrect beta_kd initialization"
    assert isinstance(model.fc, torch.nn.Linear), "Final layer should be a linear layer"


def test_forward_pass():
    model = StudentCIFAR100ModelResNet18()
    input_tensor = torch.rand(3, 3, 32, 32)  # CIFAR-100 dimensions
    logits = model(input_tensor)
    assert logits.shape == (3, 100), "Logits shape should be (batch_size, num_classes)"

    logits, features = model(input_tensor, softmax=False, is_feat=True)
    assert len(features) == 5, "Should return five feature maps"
    assert all(feature.dim() > 1 for feature in features), "All feature maps should have more than one dimension"


def test_optimizer_configuration():
    model = StudentCIFAR100ModelResNet18()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
    assert optimizer.param_groups[0]["lr"] == model.learning_rate, "Optimizer learning rate should match the initialized value"


def test_training_step():
    model = StudentCIFAR100ModelResNet18()
    images = torch.rand(32, 3, 32, 32)  # A small batch of CIFAR-100 images
    labels = torch.randint(0, 100, (32,))  # Random class labels for CIFAR-100
    batch = (images, labels)
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be greater than zero for valid training step"
    assert isinstance(loss, torch.Tensor), "The type of loss should be a torch.Tensor"
