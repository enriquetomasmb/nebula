import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from nebula.core.models.nebulamodel import NebulaModel

import logging
class ProtoFashionMNISTModelCNN(NebulaModel):
    """
    LightningModule for MNIST.
    """


    def __init__(
            self,
            input_channels=1,
            num_classes=10,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None,
            beta=1
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)

        self.example_input_array = torch.zeros(1, 1, 28, 28)
        self.learning_rate = learning_rate
        self.beta = beta
        self.global_protos = dict()
        self.agg_protos_label = dict()
        self.criterion_nll = nn.NLLLoss()
        self.loss_mse = torch.nn.MSELoss()

        self.conv1 = torch.nn.Conv2d(
            in_channels=input_channels, out_channels=32, kernel_size=(5, 5), padding="same"
        )
        self.relu = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(5, 5), padding="same"
        )
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.l1 = torch.nn.Linear(7 * 7 * 64, 2048)
        self.l2 = torch.nn.Linear(2048, num_classes)



    def forward_train(self, x):
        """Forward pass only for train the model."""
        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)

        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))
        logits = self.l2(dense)

        return F.log_softmax(logits, dim=1), dense

    def forward(self, x):
        """Forward pass for inference the model, if model have prototypes"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        # Reshape the input tensor
        input_layer = x.view(-1, 1, 28, 28)

        # First convolutional layer
        conv1 = self.relu(self.conv1(input_layer))
        pool1 = self.pool1(conv1)

        # Second convolutional layer
        conv2 = self.relu(self.conv2(pool1))
        pool2 = self.pool2(conv2)

        # Flatten the tensor
        pool2_flat = pool2.reshape(-1, 7 * 7 * 64)

        # Fully connected layers
        dense = self.relu(self.l1(pool2_flat))

        # Calculate distances
        distances = []
        for key, proto in self.global_protos.items():
            # Calculate Euclidean distance
            # send protos and dense to the same device
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        # Return the predicted class based on the closest prototype
        return distances.argmin(dim=1)

    def configure_optimizers(self):
        """ """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, batch_idx, phase):

        images, labels_g = batch
        images, labels = images.to(self.device), labels_g.to(self.device)
        logits, protos = self.forward_train(images)

        # Compute loss with the logits and the labels nll
        loss1 = self.criterion_nll(logits, labels)

        # Compute loss 2
        if len(self.global_protos) == 0:
            loss2 = 0 * loss1
        else:
            proto_new = protos.clone()
            i = 0
            for label in labels:
                if label.item() in self.global_protos.keys():
                    proto_new[i, :] = self.global_protos[label.item()].data
                i += 1
            # Compute the loss with the global protos
            loss2 = self.loss_mse(proto_new, protos)
        # Compute the final loss
        loss = loss1 + self.beta*loss2
        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Aggregate the protos
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :]), count=0)
                self.agg_protos_label[label]['sum'] += protos[i, :].detach().clone()
                self.agg_protos_label[label]['count'] += 1

        return loss

    def get_protos(self):

        if len(self.agg_protos_label) == 0:
            return {k: v.cpu() for k, v in self.global_protos.items()}

        proto = dict()
        for label, proto_info in self.agg_protos_label.items():

            if proto_info['count'] > 1:
                proto[label] = (proto_info['sum'] / proto_info['count']).to('cpu')
            else:
                proto[label] = proto_info['sum'].to('cpu')

        logging.info(f"[ProtoFashionMNISTModelCNN.get_protos] Protos: {proto}")
        return proto

    def set_protos(self, protos):
        self.agg_protos_label = dict()
        self.global_protos = {k: v.to(self.device) for k, v in protos.items()}


