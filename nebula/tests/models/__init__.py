# file: tests/test_inference.py
import pytest
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nebula.core.models.mnist.cnn import MNISTModelCNN


@pytest.fixture(scope="module")
def test_dataloader():

    return test_loader


class MNISTInferenceTester(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        return {'preds': preds, 'labels': y}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs], dim=0)
        labels = torch.cat([x['labels'] for x in outputs], dim=0)
        accuracy = torch.sum(preds == labels).float() / len(labels)
        self.log('test_accuracy', accuracy)


@pytest.mark.parametrize("batch_size", [32])
def test_model_inference(test_dataloader, batch_size):
    model = MNISTModelCNN()
    tester = MNISTInferenceTester(model)
    trainer = pl.Trainer(max_epochs=1, logger=False)
    result = trainer.test(tester, test_dataloaders=test_dataloader)
    assert result[0]['test_accuracy'] > 0.8, "The model's accuracy is below 80%"

