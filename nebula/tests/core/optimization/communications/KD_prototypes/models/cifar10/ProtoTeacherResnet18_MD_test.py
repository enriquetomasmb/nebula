import torch

from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoStudentResnet8 import ProtoStudentCIFAR10ModelResnet8
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoTeacherResnet18 import MDProtoTeacherCIFAR10ModelResnet18


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        model.global_protos[i] = torch.rand(feature_dim, requires_grad=False).to(model.device)


def test_model_initialization():
    model = MDProtoTeacherCIFAR10ModelResnet18(input_channels=3, num_classes=10, learning_rate=1e-3)
    assert isinstance(model.fc, torch.nn.Linear), "Final FC layer is not initialized correctly."
    assert model.fc.out_features == 10, "Incorrect number of output features."
    assert model.weighting.get_beta() == 1000, "Incorrect beta_md initialization."


def test_model_initialization_with_prototypes():
    model = MDProtoTeacherCIFAR10ModelResnet18(input_channels=3, num_classes=10)
    create_random_prototypes(model, 10, 512)  # Assuming each prototype has a dimension of 512
    assert len(model.global_protos) == 10, "Prototypes were not initialized correctly."
    assert all(model.global_protos[key].shape[0] == 512 for key in model.global_protos), "Prototype dimensions are incorrect."


def test_forward_pass_with_prototypes():
    model = MDProtoTeacherCIFAR10ModelResnet18(input_channels=3, num_classes=10)
    create_random_prototypes(model, 10, 512)
    model.eval()  # Important for models with Dropout or BatchNorm layers
    input_tensor = torch.rand(3, 3, 32, 32)
    output = model(input_tensor)
    print(output.shape)
    assert output.shape == torch.Size([3]), "Output shape is incorrect."


def test_loss_with_prototypes():
    model = MDProtoTeacherCIFAR10ModelResnet18(input_channels=3, num_classes=10)
    model.set_student_model(ProtoStudentCIFAR10ModelResnet8(input_channels=3, num_classes=10))  # Setup with a dummy student model
    create_random_prototypes(model, 10, 512)
    model.train()  # Set model to training mode to include Dropout, etc.
    input_tensor = torch.rand(5, 3, 32, 32)
    labels = torch.randint(0, 10, (5,))
    batch = (input_tensor, labels)
    loss = model.step(batch, batch_idx=0, phase="Validation")
    assert loss > 0, "Loss should be positive."
    assert isinstance(loss, torch.Tensor), "The loss should be a tensor."
