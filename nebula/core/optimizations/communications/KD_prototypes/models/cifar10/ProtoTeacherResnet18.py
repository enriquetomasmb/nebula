import torch
from torch import nn
import torch.nn.functional as F

from nebula.core.optimizations.communications.KD.utils.AT import Attention
from nebula.core.optimizations.communications.KD.utils.KD import DistillKL
from nebula.core.optimizations.communications.KD_prototypes.models.prototeachernebulamodel import ProtoTeacherNebulaModel, MDProtoTeacherNebulaModel


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


class ProtoTeacherCIFAR10ModelResnet18(ProtoTeacherNebulaModel):
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
        alpha_kd=1,
        beta_feat=1,
        lambda_proto=1,
        weighting=None,
    ):
        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            alpha_kd,
            beta_feat,
            lambda_proto,
            weighting,
        )

        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.embedding_dim = 512
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Construcción directa de ResNet-14
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(512 * BasicBlock.expansion, self.embedding_dim)
        self.fc = nn.Linear(self.embedding_dim, self.num_classes)

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
        """Forward pass solo para entrenar el modelo.
        is_feat: bool, si es True, devuelve las características del modelo.
        softmax: bool, si es True, aplica softmax a los logits.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        conv1 = x

        x1 = self.layer1(x)  # 32x32
        conv2 = x1
        x2 = self.layer2(x1)  # 16x16
        conv3 = x2
        x3 = self.layer3(x2)  # 8x8
        conv4 = x3
        x4 = self.layer4(x3)  # 4x4
        conv5 = x4

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        dense = self.fc_dense(x)
        logits = self.fc(dense)

        if is_feat:
            if softmax:
                return (
                    F.log_softmax(logits, dim=1),
                    dense,
                    [conv1, conv2, conv3, conv4, conv5],
                )
            return logits, dense, [conv1, conv2, conv3, conv4, conv5]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass para la inferencia del modelo"""
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
        """ """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer


class MDProtoTeacherCIFAR10ModelResnet18(MDProtoTeacherNebulaModel):
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
        T=2,
        p=2,
        alpha_kd=1,
        beta_feat=1000,
        lambda_proto=1,
        weighting=None,
    ):
        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
            T,
            p,
            alpha_kd,
            beta_feat,
            lambda_proto,
            weighting,
        )
        self.embedding_dim = 512
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.learning_rate = learning_rate
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_feat = Attention(self.p)
        self.criterion_kd = DistillKL(self.T)

        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Construcción directa de ResNet-14
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(512 * BasicBlock.expansion, self.embedding_dim)
        self.fc = nn.Linear(self.embedding_dim, self.num_classes)

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
        """Forward pass only for train the model.
        is_feat: bool, if True return the features of the model.
        softmax: bool, if True apply softmax to the logits.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        conv1 = x

        x = self.layer1(x)  # 32x32
        conv2 = x
        x = self.layer2(x)  # 16x16
        conv3 = x
        x = self.layer3(x)  # 8x8
        conv4 = x
        x = self.layer4(x)
        conv5 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        dense = self.fc_dense(x)
        logits = self.fc(dense)

        if is_feat:
            if softmax:
                return (
                    F.log_softmax(logits, dim=1),
                    dense,
                    [conv1, conv2, conv3, conv4, conv5],
                )
            return logits, dense, [conv1, conv2, conv3, conv4, conv5]

        if softmax:
            return F.log_softmax(logits, dim=1), dense
        return logits, dense

    def forward(self, x):
        """Forward pass para la inferencia del modelo"""
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
        """ """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer
