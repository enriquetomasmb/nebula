import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.research.FedProto.models.fedprotonebulamodel import FedProtoNebulaModel


class FedProtoCIFAR10ModelCNN(FedProtoNebulaModel):
    """
    LightningModule for MNIST.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=10,
        learning_rate=1e-3,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        beta=1,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, beta)

        self.example_input_array = torch.rand(1, 3, 32, 32)

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(5, 5), padding="same")
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = nn.Linear(8 * 8 * 64, 2048)  # Adjusted for CIFAR-10 image size
        self.l2 = nn.Linear(2048, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model."""
        # Reshape the input tensor
        input_layer = x.view(-1, 3, 32, 32)

        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        pool2_flat = pool2.reshape(-1, 8 * 8 * 64)

        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        del input_layer, pool1, pool2_flat

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [conv1, conv2]
            return logits, dense, [conv1, conv2]

        del conv1, conv2

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        # Reshape the input tensor
        input_layer = x.view(-1, 3, 32, 32)

        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 8 * 8 * 64)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))

        del input_layer, conv1, pool1, conv2, pool2_flat, pool2
        # Calculate distances
        distances = []
        for key, proto in self.global_protos.items():
            # Calculate Euclidean distance
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        del dense
        # Return the predicted class based on the closest prototype
        return distances.argmin(dim=1)
