import torch
from torch import nn
from torch.nn import functional as F

from nebula.core.optimizations.communications.KD_prototypes.models.cifar10.resnet8 import ResNet8
from nebula.core.research.FedGPD.models.FedGPDnebulamodel import FedGPDNebulaModel
from nebula.core.research.FedGPD.models.utils.GlobalPrototypeDistillationLoss import GlobalPrototypeDistillationLoss


class FedGPDCIFAR10ModelResNet8(FedGPDNebulaModel):
    """
    LightningModule para CIFAR-10 usando una implementación personalizada de ResNet-8.
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
            input_channels=input_channels,
            num_classes=num_classes,
            learning_rate=learning_rate,
            metrics=metrics,
            confusion_matrix=confusion_matrix,
            seed=seed,
            T=T,
        )

        self.example_input_array = torch.zeros(1, 3, 32, 32)
        self.criterion_cls = nn.CrossEntropyLoss()
        self.lambd = lambd
        self.criterion_gpd = GlobalPrototypeDistillationLoss(temperature=self.T)
        self.embedding_dim = 512
        self.resnet = ResNet8()
        self.resnet.fc_dense = nn.Linear(self.resnet.fc.in_features, self.embedding_dim)
        self.resnet.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward_train(self, x, softmax=True, is_feat=False):
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        conv1 = x

        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        conv2 = x
        x = self.resnet.layer2(x)
        conv3 = x
        x = self.resnet.layer3(x)
        conv4 = x

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        dense = self.resnet.fc_dense(x)
        logits = self.resnet.fc(dense)

        if is_feat:
            if softmax:
                return (
                    F.log_softmax(logits, dim=1),
                    dense,
                    [conv1, conv2, conv3, conv4],
                )
            return logits, dense, [conv1, conv2, conv3, conv4]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass for inference the model, if model have prototypes"""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        dense = self.resnet.fc_dense(x)

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
    optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=0.00001)
    return optimizer
