import torch
from nebula.core.research.FedGPD.models.utils.GlobalPrototypeDistillationLoss import GlobalPrototypeDistillationLoss


def test_initialization():
    """Test the initialization of GlobalPrototypeDistillationLoss."""
    loss_func = GlobalPrototypeDistillationLoss(temperature=2)
    assert loss_func.temperature == 2, "Temperature initialization failed."


def create_prototypes_and_features(num_classes=10, feature_dim=2048):
    """Helper function to create random global prototypes and sample features."""
    global_protos = {i: torch.rand(feature_dim) for i in range(num_classes)}
    local_features = torch.rand(5, feature_dim)  # Batch size of 5
    labels = torch.tensor([0, 1, 2, 3, 4])
    return global_protos, local_features, labels


def test_loss_calculation():
    """Test the calculation of the global prototype distillation loss."""
    global_protos, local_features, labels = create_prototypes_and_features()
    loss_func = GlobalPrototypeDistillationLoss(temperature=2)
    loss = loss_func(global_protos, local_features, labels)
    assert loss is not None, "Loss calculation failed."
    assert loss >= 0, "Loss should be positive or zero."


def test_loss_with_invalid_labels():
    """Test the loss function with a label that doesn't have a corresponding prototype."""
    global_protos, local_features, labels = create_prototypes_and_features()
    labels[0] = 10  # Invalid label, no corresponding prototype
    loss_func = GlobalPrototypeDistillationLoss(temperature=2)
    loss = loss_func(global_protos, local_features, labels)
    assert loss is not None, "Loss calculation should handle missing prototypes gracefully."


def test_loss_with_missing_prototypes():
    """Test the loss function when some prototypes are intentionally missing."""
    global_protos, local_features, labels = create_prototypes_and_features()
    del global_protos[0]  # Remove prototype for class 0
    loss_func = GlobalPrototypeDistillationLoss(temperature=2)
    loss = loss_func(global_protos, local_features, labels)
    assert loss is not None, "Loss calculation should handle missing prototypes gracefully."


# Optional: Test for the impact of the temperature parameter
def test_temperature_effect():
    """Test the effect of different temperatures on the loss calculation."""
    global_protos, local_features, labels = create_prototypes_and_features()
    low_temp_loss_func = GlobalPrototypeDistillationLoss(temperature=1)
    high_temp_loss_func = GlobalPrototypeDistillationLoss(temperature=10)
    low_temp_loss = low_temp_loss_func(global_protos, local_features, labels)
    high_temp_loss = high_temp_loss_func(global_protos, local_features, labels)
    assert low_temp_loss != high_temp_loss, "Temperature should affect the loss values."


if __name__ == "__main__":
    test_initialization()
    test_loss_calculation()
    test_loss_with_invalid_labels()
    test_loss_with_missing_prototypes()
    test_temperature_effect()
