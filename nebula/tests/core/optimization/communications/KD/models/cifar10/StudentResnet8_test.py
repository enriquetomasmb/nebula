import torch
from nebula.core.optimizations.communications.KD.models.cifar10.StudentResnet8 import StudentCIFAR10ModelResNet8


def test_model_initialization():
    model = StudentCIFAR10ModelResNet8()
    assert model.input_channels == 3, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert model.learning_rate == 1e-3, "Incorrect learning rate set"
    assert model.limit_beta == 0.1, "Incorrect limit beta value"


def test_forward_pass():
    model = StudentCIFAR10ModelResNet8()
    input_tensor = torch.rand(4, 3, 32, 32)  # batch size of 4 for CIFAR-10
    output = model(input_tensor)
    assert output.shape == (4, 10), "Output tensor shape should match (batch_size, num_classes)"

    output, features = model(input_tensor, is_feat=True)
    assert len(features) == 2, "Should return two feature maps from layers"
    assert features[0].shape[1] == 64, "The number of output channels in the first feature map should be 64"
    assert features[1].shape[1] == 128, "The number of output channels in the second feature map should be 128"


def test_optimizer_configuration():
    model = StudentCIFAR10ModelResNet8()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
    assert optimizer.param_groups[0]["lr"] == 1e-3, "Optimizer learning rate should be set correctly"


def test_training_step():
    model = StudentCIFAR10ModelResNet8()
    images = torch.rand(3, 3, 32, 32)  # A small batch of images
    labels = torch.randint(0, 10, (3,))  # Random class labels for each image
    batch = (images, labels)
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert isinstance(loss, torch.Tensor), "The returned type should be a torch.Tensor"
    assert loss.requires_grad, "Loss should require gradients"


def test_load_state_dict():
    model = StudentCIFAR10ModelResNet8()
    state_dict = model.state_dict()
    model.load_state_dict(state_dict)
    assert model.model_updated_flag1, "Model updated flag should be set after loading state dict"


def test_state_dict():
    model = StudentCIFAR10ModelResNet8()
    state = model.state_dict()
    assert isinstance(state, dict), "State dict should be a dictionary"
    assert "conv1.weight" in state, "State dict should contain the initial convolution layer weights"


def test_send_logic():
    model = StudentCIFAR10ModelResNet8(send_logic="mixed_2rounds")
    first_logic = model.send_logic()
    model.send_logic_step()
    second_logic = model.send_logic()
    assert first_logic != second_logic, "Send logic should toggle between calls under mixed_2rounds setting"
