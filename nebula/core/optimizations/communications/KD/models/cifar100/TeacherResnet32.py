import copy
import torch
from torch import nn
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


class TeacherCIFAR100ModelResNet32(TeacherNebulaModel):
    def __init__(self, input_channels=3, num_classes=100, learning_rate=1e-3, metrics=None, confusion_matrix=None, seed=None):
        super().__init__(
            input_channels=input_channels, num_classes=num_classes, learning_rate=learning_rate, metrics=metrics, confusion_matrix=confusion_matrix, seed=seed
        )
        self.in_planes = 64
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Construcción directa de ResNet-32
        self.layer1 = self._make_layer(BasicBlock, 64, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 5, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(256 * BasicBlock.expansion, 256)
        self.fc = nn.Linear(256, num_classes)

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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x = self.layer1(x)
        f1 = x

        x = self.layer2(x)
        f2 = x

        x = self.layer3(x)
        f3 = x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f4 = x
        dense = self.fc_dense(x)
        x = self.fc(dense)

        if is_feat:
            return x, [f0, f1, f2, f3, f4]
        return x

    def step(self, batch, batch_idx, phase):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion_cls(outputs, labels)

        self.process_metrics(phase, outputs, labels, loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.config["beta1"], self.config["beta2"]),
            amsgrad=self.config["amsgrad"],
        )
        return optimizer


class MDTeacherCIFAR100ModelResNet32(TeacherNebulaModel):
    def __init__(self, input_channels=3, num_classes=100, learning_rate=1e-3, metrics=None, confusion_matrix=None, seed=None, p=2, beta=100):
        super().__init__(
            input_channels=input_channels, num_classes=num_classes, learning_rate=learning_rate, metrics=metrics, confusion_matrix=confusion_matrix, seed=seed
        )
        self.p = p
        self.criterion_cls = torch.nn.CrossEntropyLoss()
        self.criterion_kd = Attention(self.p)
        self.beta = beta
        self.in_planes = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.student_model = None

        # Construcción directa de ResNet-32
        self.layer1 = self._make_layer(BasicBlock, 64, 5, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 5, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 5, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_dense = nn.Linear(256 * BasicBlock.expansion, 256)
        self.fc = nn.Linear(256, num_classes)

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

    def forward(self, x, softmax=True, is_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x
        print(x.shape)
        x = self.layer1(x)
        f1 = x
        print(x.shape)
        x = self.layer2(x)
        f2 = x
        print(x.shape)
        x = self.layer3(x)
        f3 = x
        print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        f4 = x
        print(x.shape)
        dense = self.fc_dense(x)
        x = self.fc(dense)

        if is_feat:
            if softmax:
                return F.log_softmax(x, dim=1), [f0, f1, f2, f3, f4]
            return x, [f0, f1, f2, f3, f4]
        return x

    def configure_optimizers(self):
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
        logits, feat_t = self.forward(images, is_feat=True)

        # Compute loss cross entropy loss
        loss_cls = self.criterion_cls(logits, labels)

        if self.student_model is not None:
            with torch.no_grad():
                student_logits, feat_s = self.student_model(images, is_feat=True)
                feat_s = [f.detach() for f in feat_s]
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            # Compute the mutual distillation loss
            loss_group = self.criterion_kd(g_t, g_s)
            loss_kd = sum(loss_group)
        else:
            loss_kd = 0 * loss_cls

        # Combine the losses
        loss = loss_cls + self.beta * loss_kd
        self.process_metrics(phase, logits, labels, loss)

        return loss

    def set_student_model(self, student_model):
        """
        Para evitar problemas de dependencia cíclica, se crea una copia del modelo de estudiante y se elimina el atributo teacher_model.
        """
        self.student_model = copy.deepcopy(student_model)
        if hasattr(self.student_model, "teacher_model"):
            del self.student_model.teacher_model
