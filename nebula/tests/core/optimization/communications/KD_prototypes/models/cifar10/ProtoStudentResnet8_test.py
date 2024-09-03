import torch
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoStudentResnet8 import ProtoStudentCIFAR10ModelResnet8


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        model.global_protos[i] = torch.rand(feature_dim, requires_grad=False).to(model.device)


def test_model_initialization_with_prototypes():
    model = ProtoStudentCIFAR10ModelResnet8(input_channels=3, num_classes=10)
    create_random_prototypes(model, 10, 512)  # Assuming feature dimension is 512 for this model
    assert model.num_classes == 10, "Incorrect number of classes initialized"
    assert len(model.global_protos) == 10, "Incorrect number of prototypes initialized"
    for key, proto in model.global_protos.items():
        assert proto.shape == (512,), "Prototype shape should match the specified feature dimension"


def test_forward_pass_inference_with_prototypes():
    model = ProtoStudentCIFAR10ModelResnet8(input_channels=3, num_classes=10)
    create_random_prototypes(model, 10, 512)  # Ensure prototypes are initialized
    model.eval()
    inputs = torch.rand(1, 3, 32, 32)  # Single input example
    output = model(inputs)
    assert output.shape == (1,), "Output shape should be (1,) for single input in inference mode with prototypes"


def test_training_step_with_prototypes():
    model = ProtoStudentCIFAR10ModelResnet8(input_channels=3, num_classes=10)
    create_random_prototypes(model, 10, 512)
    inputs = torch.rand(5, 3, 32, 32)  # Batch of 5 images
    labels = torch.randint(0, 10, (5,))  # Random class labels
    batch = (inputs, labels)
    initial_loss = model.step(batch, batch_idx=0, phase="Validation")

    assert initial_loss >= 0, "Loss should be non-negative"
    assert isinstance(initial_loss, torch.Tensor), "Loss should be a tensor"
