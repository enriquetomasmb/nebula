from abc import ABC, abstractmethod
import torch
from torch import nn
from nebula.core.models.nebulamodel import NebulaModel


class FedProtoNebulaModel(NebulaModel, ABC):
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
        beta=1,
    ):
        super().__init__(input_channels, num_classes, learning_rate, metrics, confusion_matrix, seed)
        self.automatic_optimization = False
        self.config = {"beta1": 0.851436, "beta2": 0.999689, "amsgrad": True}
        self.beta = beta
        self.global_protos = {}
        self.agg_protos_label = {}
        self.criterion_nll = nn.NLLLoss()
        self.loss_mse = torch.nn.MSELoss()

    @abstractmethod
    def forward_train(self, x):
        pass

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer

    def step(self, batch, batch_idx, phase):

        images, labels_g = batch
        if phase == "Train":
            optimizer = self.optimizers()

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
            del proto_new

        # Compute the final loss
        loss = loss1 + self.beta * loss2

        if phase == "Train":
            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()

        self.process_metrics(phase, logits, labels, loss.detach())
        del loss2, loss1, loss

        if phase == "Train":
            # Aggregate the protos
            for i in range(len(labels_g)):
                label = labels_g[i].item()
                if label not in self.agg_protos_label:
                    self.agg_protos_label[label] = dict(sum=torch.zeros_like(protos[i, :].detach()), count=0)
                self.agg_protos_label[label]["sum"] += protos[i, :].detach().clone()
                self.agg_protos_label[label]["count"] += 1

        del labels, labels_g, logits, protos, images

    def get_protos(self):

        if len(self.agg_protos_label) == 0:
            proto = {k: v.detach() for k, v in self.global_protos.items()}

            return proto

        proto = {}
        for label, proto_info in self.agg_protos_label.items():

            if proto_info["count"] > 1:
                proto[label] = (proto_info["sum"] / proto_info["count"]).detach()
            else:
                proto[label] = proto_info["sum"].detach()

        return proto

    def set_protos(self, protos):
        self.agg_protos_label = {}
        self.global_protos = {k: v.to(self.device) for k, v in protos.items()}
