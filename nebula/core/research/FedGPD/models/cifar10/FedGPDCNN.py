import torch
import torch.nn.functional as F
from torch import nn
from nebula.core.research.FedGPD.models.FedGPDnebulamodel import FedGPDNebulaModel
from nebula.core.research.FedGPD.models.utils.GlobalPrototypeDistillationLoss import (
    GlobalPrototypeDistillationLoss,
)


class FedGPDCIFAR10ModelCNN(FedGPDNebulaModel):
    """
    LightningModule for CIFAR-10.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=0.01,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        T=2,
        lambd=0.05,
    ):

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            T,
        )

        self.embedding_dim = 512
        self.example_input_array = torch.zeros(1, 3, 32, 32)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_gpd = GlobalPrototypeDistillationLoss(temperature=T)
        self.lambd = lambd
        # Define layers of the model
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.25),
        )

        self.fc_layer_dense = torch.nn.Sequential(
            torch.nn.Linear(128 * 4 * 4, self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
        )

        self.fc_layer = torch.nn.Linear(self.embedding_dim, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        # Reshape the input tensor
        input_layer = x.view(-1, 3, 32, 32)

        # First convolutional layer
        conv1 = self.layer1(input_layer)

        # Second convolutional layer
        conv2 = self.layer2(conv1)

        # Third convolutional layer
        conv3 = self.layer3(conv2)

        # Flatten the tensor
        flattened = conv3.view(conv3.size(0), -1)

        # Fully connected layers
        dense = self.fc_layer_dense(flattened)
        logits = self.fc_layer(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [conv1, conv2, conv3]
            return logits, dense, [conv1, conv2, conv3]

        del conv1, conv2, conv3

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass for inference the model, if model have prototypes"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        # Reshape the input tensor
        input_layer = x.view(-1, 3, 32, 32)

        conv1 = self.layer1(input_layer)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        flattened = conv3.view(conv3.size(0), -1)  # Flatten the layer
        dense = self.fc_layer_dense(flattened)

        # Calculate distances
        distances = []
        for key, proto in self.global_protos.items():
            # Calculate Euclidean distance
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        # Return the predicted class based on the closest prototype
        return distances.argmin(dim=1)

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.00001)
        return optimizer
