import torch
from torch import nn
from torch.nn import functional as F

from nebula.core.research.FedGPD.models.FedGPDnebulamodel import FedGPDNebulaModel
from nebula.core.research.FedGPD.models.utils.GlobalPrototypeDistillationLoss import GlobalPrototypeDistillationLoss


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


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
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.embedding_dim = 128
        # Construcción directa de ResNet-8
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(128 * BasicBlock.expansion, self.embedding_dim)
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward_train(self, x, softmax=True, is_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)

        x = self.avgpool(x2)
        x = torch.flatten(x, 1)
        dense = self.fc_dense(x)
        logits = self.fc(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(logits, dim=1), dense, [x1, x2]
            return logits, dense, [x1, x2]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass para la inferencia del modelo."""
        if len(self.global_protos) == 0:
            logits, _ = self.forward_train(x)
            return logits

        # Obtener las características intermedias
        logits, dense, features = self.forward_train(x, is_feat=True)

        # Calcular distancias a los prototipos globales
        distances = []
        for key, proto in self.global_protos.items():
            proto = proto.to(dense.device)
            dist = torch.norm(dense - proto, dim=1)
            distances.append(dist.unsqueeze(1))
        distances = torch.cat(distances, dim=1)

        return distances.argmin(dim=1)

    def configure_optimizers(self):
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
