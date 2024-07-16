
from nebula.core.models.cifar10.prototypes.resnet import CIFAR10ModelResNet8

import torch
import torch.nn.functional as F
import logging

__all__ = ['resnet']

class ProtoCIFAR10ModelResNet8(CIFAR10ModelResNet8):
    """
    LightningModule for CIFAR10.
    """
    def __init__(
            self,
            input_channels=3,
            num_classes=10,
            learning_rate=1e-3,
            metrics=None,
            confusion_matrix=None,
            seed=None,
            depth=8,
            num_filters=[16, 16, 32, 64],
            block_name='BasicBlock',
            beta=1,
    ):


        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed, depth, num_filters, block_name)

        self.beta = beta
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_nll = torch.nn.NLLLoss()
        self.loss_mse = torch.nn.MSELoss()
        self.global_protos = dict()
        self.agg_protos_label = dict()


    def configure_optimizers(self):
        """ Configure the optimizer for training. """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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

        logging.info(f"[ProtoCIFAR10ModelResNet8.get_protos] Protos: {proto}")
        return proto

    def set_protos(self, protos):
        self.agg_protos_label = dict()
        self.global_protos = {k: v.to(self.device) for k, v in protos.items()}

    def forward_train(self, x):
        """Forward pass only for train the model."""

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x, f1_pre = self.layer1(x)
        f1 = x
        x, f2_pre = self.layer2(x)
        f2 = x
        x, f3_pre = self.layer3(x)
        f3 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        dense = self.fc_dense(x)
        logits = self.fc(dense)

        return F.log_softmax(logits, dim=1), dense

    def forward(self, x):
        """Forward pass for the model."""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x, f1_pre = self.layer1(x)  # 32x32
        f1 = x
        x, f2_pre = self.layer2(x)  # 16x16
        f2 = x
        x, f3_pre = self.layer3(x)  # 8x8
        f3 = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        dense = self.fc_dense(x)

        # Calculate the distances
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






