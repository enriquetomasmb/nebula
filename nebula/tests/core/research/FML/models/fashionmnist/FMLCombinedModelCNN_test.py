from unittest.mock import MagicMock, patch
import torch
from torch.utils.data import DataLoader, TensorDataset

from nebula.core.research.FML.models.fashionmnist.FMLCombinedModel import FMLFashionMNISTCombinedModelCNN
from nebula.core.research.FML.training.FMLlightning import FMLLightning


def test_fml_lightning_training():
    # Create dummy dataset
    batch_size = 32
    num_classes = 10
    num_samples = batch_size * 5  # Total samples
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Instantiate the model
    model = FMLFashionMNISTCombinedModelCNN()
    # Mock the process_metrics method if needed
    model.process_metrics = MagicMock()

    # Create a mock config object with the necessary attributes
    config = MagicMock()
    config.participant = {
        "scenario_args": {"random_seed": 42},  # Provide a seed value
        "device_args": {"accelerator": "cpu", "idx": 0},  # or 'gpu' if you want to test GPU code paths
    }

    # Mock enable_deterministic and logging.getLogger().setLevel()
    with patch("logging.getLogger") as mock_get_logger:
        # Mock setLevel to do nothing
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        mock_logger_instance.setLevel.return_value = None

        # Instantiate the FMLLightning trainer
        trainer_wrapper = FMLLightning(model=model, data=dataloader, config=config, logger=None)

        # Set epochs if needed
        trainer_wrapper.set_epochs(1)

        # Capture initial parameters
        initial_params = [param.clone() for param in model.parameters()]

        # Run training
        trainer_wrapper.train()

        # Get updated parameters
        updated_params = list(model.parameters())

        # Check that at least one parameter has changed
        params_changed = any(not torch.equal(initial_param, updated_param) for initial_param, updated_param in zip(initial_params, updated_params))
        assert params_changed, "Parameters did not change after training"
