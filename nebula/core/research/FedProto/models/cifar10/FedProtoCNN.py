import logging

import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.models.nebulamodel import NebulaModel


class FedProtoCIFAR10ModelCNN(NebulaModel):
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

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [conv1, conv2]
            return logits, dense, [conv1, conv2]

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
