import torch
from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.ProtoStudentCNN import ProtoStudentCIFAR10ModelCNN


def create_random_prototypes(model, num_classes, feature_dim):
    model.global_protos = {}
    for i in range(num_classes):
        # Initialize global prototypes with random tensors
        model.global_protos[i] = torch.rand(feature_dim, requires_grad=False).to(model.device)


def test_proto_student_cifar10_initialization():
    model = ProtoStudentCIFAR10ModelCNN(input_channels=3, num_classes=10)
    assert model.input_channels == 3, "Input channels should be set to 3"
    assert model.num_classes == 10, "Number of classes should be 10"
    assert isinstance(model.criterion_nll, torch.nn.NLLLoss), "NLL Loss should be correctly initialized"


def test_proto_student_cifar10_forward_pass():
    model = ProtoStudentCIFAR10ModelCNN(input_channels=3, num_classes=10)
    model.eval()
    create_random_prototypes(model, 10, 512)  # Assuming 512 is the feature dimension after flatten
    input_tensor = torch.randn(1, 3, 32, 32)

    # Testing training forward functionality
    logits, dense, features = model.forward_train(input_tensor, softmax=True, is_feat=True)
    assert logits.shape == (1, 10), "Output logits shape should match the number of classes"
    assert len(features) == 3, "Should return three feature tensors from convolutional layers"

    # Testing inference functionality
    output = model(input_tensor)
    assert output.dim() == 1, "Inference output should be 1D"
    assert output.shape[0] == 1, "Should return one class index for one input"


def test_proto_student_cifar10_optimizer_configuration():
    model = ProtoStudentCIFAR10ModelCNN(input_channels=3, num_classes=10)
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Adam), "Optimizer should be an instance of Adam"
    assert optimizer.param_groups[0]["lr"] == model.learning_rate, "Learning rate should match model's configuration"


def test_proto_student_cifar10_step_functionality():
    model = ProtoStudentCIFAR10ModelCNN(input_channels=3, num_classes=10)
    model.train()
    create_random_prototypes(model, 10, 512)  # Assuming prototype feature dimension is 512

    inputs = torch.randn(5, 3, 32, 32)
    labels = torch.randint(0, 10, (5,))
    loss = model.step((inputs, labels), batch_idx=0, phase="Validation")

    assert loss >= 0, "Loss should be non-negative and well calculated"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
