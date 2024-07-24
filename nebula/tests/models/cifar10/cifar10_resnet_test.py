import torch
from nebula.core.models.cifar10.resnet import CIFAR10ModelResNet

def test_model_initialization():
    model = CIFAR10ModelResNet()
    assert isinstance(model.criterion, torch.nn.CrossEntropyLoss), "Incorrect loss function"

def test_metrics_initialization():
    model = CIFAR10ModelResNet()
    # Assuming MetricCollection handles metrics initialization
    assert hasattr(model, 'train_metrics'), "Train metrics not initialized properly"
    assert hasattr(model, 'val_metrics'), "Validation metrics not initialized properly"
    assert hasattr(model, 'test_metrics'), "Test metrics not initialized properly"

def test_forward_pass():
    model = CIFAR10ModelResNet()
    input_tensor = torch.rand(1, 3, 32, 32)
    # Ensuring we are using the correct layers by specifying model.forward
    output = model.forward(input_tensor)
    assert output.shape == (1, 10), "Output tensor has incorrect shape"

def test_process_metrics():
    model = CIFAR10ModelResNet()
    input_tensor = torch.rand(5, 3, 32, 32)
    target = torch.randint(0, 10, (5,))
    # Make sure to use the correct method to obtain outputs
    output = model.forward(input_tensor)
    loss = model.criterion(output, target)
    model.process_metrics('Train', output, target, loss)

def test_configure_optimizers():
    model = CIFAR10ModelResNet()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer type is incorrect"
    assert optimizer.param_groups[0]['lr'] == 1e-2, "Learning rate is not set correctly"

def test_step_function():
    model = CIFAR10ModelResNet()
    input_tensor = torch.rand(10, 3, 32, 32)
    target = torch.randint(0, 10, (10,))
    batch = (input_tensor, target)
    # Again, making sure to use forward correctly with batch data
    loss = model.step(batch, batch_idx=0, phase='Train')

    assert loss >= 0, "Step function did not compute loss correctly"
