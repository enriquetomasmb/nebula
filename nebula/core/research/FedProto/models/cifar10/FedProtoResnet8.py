import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock
from nebula.core.research.FedProto.models.fedprotonebulamodel import FedProtoNebulaModel


class FedProtoCIFAR10ModelResNet8(FedProtoNebulaModel):
    """
    LightningModule para CIFAR-10 usando ResNet-8.
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
            beta,
        )

        self.example_input_array = torch.zeros(1, 3, 32, 32)
        self.beta = beta

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
        # Extraer las características intermedias usando ResNet-8
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
