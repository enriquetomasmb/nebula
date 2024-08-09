import logging

import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.models.nebulamodel import NebulaModel


class ProtoCIFAR10ModelCNN(NebulaModel):
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
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)
        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.beta = beta
        self.global_protos = {}
        self.agg_protos_label = {}
        self.criterion_nll = nn.NLLLoss()
        self.loss_mse = torch.nn.MSELoss()

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

        self.fc_layer_dense = torch.nn.Sequential(torch.nn.Linear(128 * 4 * 4, 512), torch.nn.ReLU(), torch.nn.Dropout(0.5))

        self.fc_layer = torch.nn.Linear(512, num_classes)

    def forward_train(self, x):
        """Forward pass only for train the model."""
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

        return F.log_softmax(logits, dim=1), dense

    def forward(self, x):
        """
        Forward pass of the model.
            is_feat: bool, if True return the features of the model.
        """
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits
        input_layer = x.view(-1, 3, 32, 32)
        conv1 = self.layer1(input_layer)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        flattened = conv3.view(conv3.size(0), -1)  # Flatten the layer
        dense = self.fc_layer_dense(flattened)

        # Calculate the distances
        distances = []
        for key, proto in self.global_protos.items():
            # Calculate euclidean distance
            # send proto to the same device as dense
            proto = proto.to(dense.device)
            dis = torch.norm(dense - proto, dim=1)
            distances.append(dis.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        return distances.argmin(dim=1)

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        y_pred, protos = self.forward_train(images)
        loss1 = self.criterion_nll(y_pred, labels)

        # Compute loss 2 if the model has prototypes
        if len(self.global_protos) == 0:
            loss2 = 0 * loss1
        else:
            protos_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    protos_new[i, :] = self.global_protos[label.item()].data
                i += 1
            # Compute the loss for the prototypes
            loss2 = self.loss_mse(protos, protos_new)
        loss = loss1 + self.beta * loss2
        self.process_metrics(phase, y_pred, labels, loss)

        if phase == "Train":
            # Aggregating the prototypes
            for i in range(len(labels)):
                label = labels[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        return loss

    def get_protos(self):

        if len(self.agg_protos_label) == 0:
            return {k: v.cpu() for k, v in self.global_protos.items()}

        proto = {}
        for label, proto_info in self.agg_protos_label.items():

            if proto_info["count"] > 1:
                proto[label] = (proto_info["sum"] / proto_info["count"]).to("cpu")
            else:
                proto[label] = proto_info["sum"].to("cpu")

        logging.info(f"[ProtoCIFAR10ModelCNN.get_protos] Protos: {proto}")
        return proto

    def set_protos(self, protos):
        self.agg_protos_label = {}
        self.global_protos = {k: v.to(self.device) for k, v in protos.items()}
