import logging
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock
from nebula.core.models.nebulamodel import NebulaModel


class FedProtoCIFAR10ModelResNet8(NebulaModel):
    """
    LightningModule para CIFAR-100 usando ResNet-18.
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=100,
        learning_rate=0.01,
        metrics=None,
        confusion_matrix=None,
        seed=None,
        beta=0.05,
    ):

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
        )

        self.example_input_array = torch.zeros(1, 3, 32, 32)
        self.beta = beta
        self.global_protos = {}
        self.agg_protos_label = {}
        self.criterion_nll = nn.NLLLoss()
        self.loss_mse = torch.nn.MSELoss()

        # Simplified ResNet-8 architecture
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            BasicBlock(128, 128),
            BasicBlock(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Linear(128, 2048)  # Intermediate layer before classifier
        self.classifier = nn.Linear(2048, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        # Extraer las características intermedias usando ResNet-18
        features = self.features(x)
        features_flat = torch.flatten(features, 1)  # Aplanar para la capa fully connected
        dense = self.fc(features_flat)
        logits = self.classifier(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, features
            return logits, dense, features

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass para la inferencia del modelo."""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        # Obtener las características intermedias
        features = self.features(x)
        features_flat = torch.flatten(features, 1)
        dense = self.fc(features_flat)

        # Calcular distancias a los prototipos globales
        distances = []
        for key, proto in self.global_protos.items():
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        return distances.argmin(dim=1)

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
        loss = loss1 + self.beta * loss2
        self.process_metrics(phase, logits, labels, loss)

        if phase == "Train":
            # Aggregate the protos
            for i in range(len(labels_g)):
                label = labels_g[i].item()
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

        logging.info(f"[ProtoFashionMNISTModelCNN.get_protos] Protos: {proto}")
        return proto

    def set_protos(self, protos):
        self.agg_protos_label = {}
        self.global_protos = {k: v.to(self.device) for k, v in protos.items()}
