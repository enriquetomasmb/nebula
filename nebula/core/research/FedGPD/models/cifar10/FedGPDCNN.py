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

        self.example_input_array = torch.zeros(1, 3, 32, 32)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_gpd = GlobalPrototypeDistillationLoss(temperature=T)
        self.lambd = lambd

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(5, 5), padding="same")
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same")
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = nn.Linear(8 * 8 * 64, 2048)  # Adjusted for CIFAR-10 image size
        self.l2 = nn.Linear(2048, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        input_layer = x.view(-1, 3, 32, 32)

        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        pool2_flat = pool2.reshape(-1, 8 * 8 * 64)

        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [conv1, conv2]
            return logits, dense, [conv1, conv2]

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

    def step(self, batch, batch_idx, phase):

        images, labels_g = batch
        images, labels = images.to(self.device), labels_g.to(self.device)
        logits, features = self.forward_train(images, softmax=False)

        features_copy = features.clone().detach()

        # Compute loss ce
        loss_ce = self.criterion_cls(logits, labels)

        # Compute loss 2
        loss_gpd = self.criterion_gpd(self.global_protos, features_copy, labels)

        # Combine the losses
        loss = loss_ce + self.lambd * loss_gpd

        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Update the prototypes
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(features[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += features[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        return loss
