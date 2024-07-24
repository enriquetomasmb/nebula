import torch
import torch.nn as nn

from nebula.core.models.militarysar import _blocks
from nebula.core.models.militarysar.cnn import MilitarySARModelCNN


def test_model_initialization():
    model = MilitarySARModelCNN()
    assert model.input_channels == 2, "Incorrect number of input channels"
    assert model.num_classes == 10, "Incorrect number of classes"
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Incorrect loss function"
    assert isinstance(model.model, nn.Sequential), "Model should be an instance of nn.Sequential"
    assert len(model.model) == 7, "Model should have seven layers"

def test_forward_pass():
    model = MilitarySARModelCNN()
    input_tensor = torch.zeros((1, 2, 128, 128))  # Matching input size specified in the model
    output = model(input_tensor)
    assert output.shape == (1, 10), "Output tensor has incorrect shape, expected (1, 10) for num_classes"

def test_configure_optimizers():
    model = MilitarySARModelCNN()
    optimizers, lr_schedulers = model.configure_optimizers()
    assert isinstance(optimizers[0], torch.optim.SGD), "Optimizer type is incorrect"
    assert isinstance(lr_schedulers[0], torch.optim.lr_scheduler.StepLR), "Learning rate scheduler type is incorrect"
    assert lr_schedulers[0].step_size == 50, "Incorrect step size for learning rate scheduler"

def test_dropout_rate():
    model = MilitarySARModelCNN()
    dropout_layers = [layer for layer in model.model if isinstance(layer, nn.Dropout)]
    assert len(dropout_layers) == 1, "There should be exactly one dropout layer"
    assert dropout_layers[0].p == 0.5, "Dropout rate should be 0.5"

def test_layer_configuration():
    model = MilitarySARModelCNN()
    # Assuming the model is composed of Conv2D blocks and you want to ensure correct configuration
    conv_layers = [layer for layer in model.model if isinstance(layer, _blocks.Conv2DBlock)]
    assert len(conv_layers) == 5, "There should be five convolutional blocks"
