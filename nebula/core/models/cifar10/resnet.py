from torch import nn
from torchmetrics import MetricCollection

import lightning as pl
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassRecall,
    MulticlassPrecision,
    MulticlassF1Score,
    MulticlassConfusionMatrix,
)
from torchvision.models import resnet18, resnet34, resnet50

IMAGE_SIZE = 32

BATCH_SIZE = 256 if torch.cuda.is_available() else 64

classifiers = {
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
}


def conv_block(input_channels, num_classes, pool=False):
    layers = [
        nn.Conv2d(input_channels, num_classes, kernel_size=3, padding=1),
        nn.BatchNorm2d(num_classes),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class CIFAR10ModelResNet(pl.LightningModule):
    def process_metrics(self, phase, y_pred, y, loss=None):
        if loss is not None:
            self.log(f"{phase}/Loss", loss, prog_bar=True, logger=True)

        y_pred_classes = torch.argmax(y_pred, dim=1)
        if phase == "Train":
            output = self.train_metrics(y_pred_classes, y)
        elif phase == "Validation":
            output = self.val_metrics(y_pred_classes, y)
        elif phase == "Test":
            output = self.test_metrics(y_pred_classes, y)
        else:
            raise NotImplementedError
        output = {f"{phase}/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output.items()}

        self.log_dict(output, prog_bar=True, logger=True)

        if self.cm is not None:
            self.cm.update(y_pred_classes, y)

    def log_metrics_by_epoch(self, phase, print_cm=False, plot_cm=False):
        print(f"Epoch end: {phase}, epoch number: {self.epoch_global_number[phase]}")
        if phase == "Train":
            output = self.train_metrics.compute()
            self.train_metrics.reset()
        elif phase == "Validation":
            output = self.val_metrics.compute()
            self.val_metrics.reset()
        elif phase == "Test":
            output = self.test_metrics.compute()
            self.test_metrics.reset()
        else:
            raise NotImplementedError

        output = {f"{phase}Epoch/{key.replace('Multiclass', '').split('/')[-1]}": value for key, value in output.items()}

        self.log_dict(output, prog_bar=True, logger=True)

        if self.cm is not None:
            cm = self.cm.compute().cpu()
            print(f"{phase}Epoch/CM\n", cm) if print_cm else None
            if plot_cm:
                import seaborn as sns
                import matplotlib.pyplot as plt

                plt.figure(figsize=(10, 7))
                ax = sns.heatmap(cm.numpy(), annot=True, fmt="d", cmap="Blues")
                ax.set_xlabel("Predicted labels")
                ax.set_ylabel("True labels")
                ax.set_title("Confusion Matrix")
                ax.set_xticks(range(10))
                ax.set_yticks(range(10))
                ax.xaxis.set_ticklabels([i for i in range(10)])
                ax.yaxis.set_ticklabels([i for i in range(10)])
                self.logger.experiment.add_figure(
                    f"{phase}Epoch/CM",
                    ax.get_figure(),
                    global_step=self.epoch_global_number[phase],
                )
                plt.close()

        self.epoch_global_number[phase] += 1

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        implementation="scratch",
        classifier="resnet9",
    ):
        super().__init__()
        if metrics is None:
            metrics = MetricCollection(
                [
                    MulticlassAccuracy(num_classes=num_classes),
                    MulticlassPrecision(num_classes=num_classes),
                    MulticlassRecall(num_classes=num_classes),
                    MulticlassF1Score(num_classes=num_classes),
                ]
            )
        self.train_metrics = metrics.clone(prefix="Train/")
        self.val_metrics = metrics.clone(prefix="Validation/")
        self.test_metrics = metrics.clone(prefix="Test/")

        if confusion_matrix is None:
            self.cm = MulticlassConfusionMatrix(num_classes=num_classes)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.implementation = implementation
        self.classifier = classifier

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate

        self.criterion = torch.nn.CrossEntropyLoss()

        self.model = self._build_model(input_channels, num_classes)

        self.epoch_global_number = {"Train": 0, "Validation": 0, "Test": 0}

    def _build_model(self, input_channels, num_classes):
        if self.implementation == "scratch":
            if self.classifier == "resnet9":
                """
                ResNet9 implementation
                """

                def conv_block(input_channels, num_classes, pool=False):
                    layers = [
                        nn.Conv2d(input_channels, num_classes, kernel_size=3, padding=1),
                        nn.BatchNorm2d(num_classes),
                        nn.ReLU(inplace=True),
                    ]
                    if pool:
                        layers.append(nn.MaxPool2d(2))
                    return nn.Sequential(*layers)

                conv1 = conv_block(input_channels, 64)
                conv2 = conv_block(64, 128, pool=True)
                res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

                conv3 = conv_block(128, 256, pool=True)
                conv4 = conv_block(256, 512, pool=True)
                res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

                self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_classes))

                return dict(
                    conv1=conv1,
                    conv2=conv2,
                    res1=res1,
                    conv3=conv3,
                    conv4=conv4,
                    res2=res2,
                    classifier=self.classifier,
                )

            elif self.implementation in classifiers.keys():
                model = classifiers[self.classifier]
                model.fc = torch.nn.Linear(model.fc.in_features, 10)
                return model
            else:
                raise NotImplementedError()

        elif self.implementation == "timm":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"images must be a torch.Tensor, got {type(x)}")

        if self.implementation == "scratch":
            if self.classifier == "resnet9":
                out = self.model["conv1"](x)
                out = self.model["conv2"](out)
                out = self.model["res1"](out) + out
                out = self.model["conv3"](out)
                out = self.model["conv4"](out)
                out = self.model["res2"](out) + out
                out = self.model["classifier"](out)
                return out
            else:
                return self.model(x)
        elif self.implementation == "timm":
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=1e-4)
        return optimizer

    def step(self, batch, phase):
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)
        y_pred = self.forward(images)
        loss = self.criterion(y_pred, labels)
        self.process_metrics(phase, y_pred, labels, loss)

        return loss

    def training_step(self, batch, batch_id):
        return self.step(batch, "Train")

    def on_train_epoch_end(self):
        self.log_metrics_by_epoch("Train", print_cm=True, plot_cm=True)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "Validation")

    def on_validation_epoch_end(self):
        self.log_metrics_by_epoch("Validation", print_cm=True, plot_cm=True)

    def test_step(self, batch, batch_idx):
        return self.step(batch, "Test")

    def on_test_epoch_end(self):
        self.log_metrics_by_epoch("Test", print_cm=True, plot_cm=True)
