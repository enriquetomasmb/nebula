import copy
import torch
from torch import nn
import torch.multiprocessing
import torch.nn.functional as F
from nebula.core.optimizations.communications.KD.models.teachernebulamodel import TeacherNebulaModel
from nebula.core.optimizations.communications.KD.utils.AT import Attention


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


class TeacherCIFAR10ModelResNet18(TeacherNebulaModel):
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
    ):

        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
        )
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.nn.CrossEntropyLoss()

        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Construcción directa de ResNet-18
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(512 * BasicBlock.expansion, 512)
        self.fc = nn.Linear(512, num_classes)

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

    def forward(self, x):
        """Forward pass para la inferencia del modelo"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.layer4(x)  # 4x4

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        dense = self.fc_dense(x)
        logits = self.fc(dense)

        return logits

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
        images, labels = batch
        logits = self(images)
        loss = self.criterion_cls(logits, labels)
        self.process_metrics(phase, logits, labels, loss)
        return loss


class MDTeacherCIFAR10ModelResNet18(TeacherNebulaModel):
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
        p=2,
        beta=1000,
    ):
        super().__init__(
            input_channels,
            num_classes,
            learning_rate,
            metrics,
            confusion_matrix,
            seed,
        )
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.p = p
        self.beta = beta
        self.criterion_kd = Attention(self.p)

        self.student_model = None

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
        self.fc_dense = nn.Linear(512 * BasicBlock.expansion, 512)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

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

    def forward(self, x, is_feat=False):
        """Forward pass para la inferencia del modelo"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x1 = self.layer1(x)  # 32x32
        x2 = self.layer2(x1)  # 16x16
        x3 = self.layer3(x2)  # 8x8
        x4 = self.layer4(x3)  # 4x4

        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        logits = self.fc(x)

        if is_feat:
            return logits, [x1, x2, x3, x4]
        return logits

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer

    def set_student_model(self, student_model):
        """
        For the mutual distillation, the student model is set in the teacher model.
        """
        self.student_model = copy.deepcopy(student_model)
        if hasattr(self.student_model, "teacher_model"):
            del self.student_model.teacher_model

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        logit_t, feat_t = self(images, is_feat=True)
        loss_cls = self.criterion_cls(logit_t, labels)
        # If the student model is set, the mutual distillation is applied
        if self.student_model is not None:
            with torch.no_grad():
                logit_s, feat_s = self.student_model(images, is_feat=True)
                feat_s = [f.detach() for f in feat_s]
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            # Compute the attention loss
            loss_group = self.criterion_kd(g_t, g_s)
            loss_kd = sum(loss_group)
        else:
            loss_kd = 0 * loss_cls

        loss = loss_cls + self.beta * loss_kd
        self.process_metrics(phase, logit_t, labels, loss)
        return loss
